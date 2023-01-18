import collections
import math
from itertools import repeat

import torch
from torch import Tensor
from torch.nn import functional as F, Module, Parameter, init
from typing import TypeVar, Union, Tuple, Optional, List

from torch.nn.modules.utils import _reverse_repeat_tuple


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")

# Template for arguments which can be supplied as a tuple, or which can be a scalar which PyTorch will internally
# broadcast to a tuple.
# Comes in several variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d operations.
T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_2_t = _scalar_or_tuple_2_t[int]


class CubeSpherePadding2D(Module):
    """
    Pads the input cubed sphere tensor according to adjacent panels. The requirements for this layer are as follows:
    - The input data is 5-dimensional (batch, channels, 5, height, width)
    - The last panel is the top panel

    Adapted from CubeSpherePadding2D by @jweyn

    Args:
        padding (int): Width of padding on each cube sphere panel
    """
    __constants__ = ['padding']
    padding: int

    def __init__(self, padding: int) -> None:
        super(CubeSpherePadding2D, self).__init__()
        self.padding = padding

    def forward(self, inputs: Tensor) -> Tensor:
        p = self.padding

        # Pad the equatorial upper/lower boundaries and the polar upper/lower boundaries
        out = [
            # Panel 0
            torch.unsqueeze(
                torch.cat(list(repeat(torch.unsqueeze(inputs[:, :, 0, :, 0], 3), p)) +
                          [inputs[:, :, 0],
                           inputs[:, :, 4, :, :p]], dim=3), 2
            ),
            # Panel 1
            torch.unsqueeze(
                torch.cat(list(repeat(torch.unsqueeze(inputs[:, :, 1, :, 0], 3), p)) +
                          [inputs[:, :, 1],
                           torch.transpose(torch.flip(inputs[:, :, 4, -p:, :], dims=[2]), dim0=2, dim1=3)], dim=3), 2
            ),
            # Panel 2
            torch.unsqueeze(
                torch.cat(list(repeat(torch.unsqueeze(inputs[:, :, 2, :, 0], 3), p)) +
                          [inputs[:, :, 2],
                           torch.flip(inputs[:, :, 4, :, -p:], dims=[2, 3])], dim=3), 2
            ),
            # Panel 3
            torch.unsqueeze(
                torch.cat(list(repeat(torch.unsqueeze(inputs[:, :, 3, :, 0], 3), p)) +
                          [inputs[:, :, 3],
                           torch.transpose(torch.flip(inputs[:, :, 4, :p, :], dims=[3]), dim0=2, dim1=3)], dim=3), 2
            ),
            # Panel 4 (top)
            torch.unsqueeze(
                torch.cat([inputs[:, :, 0, :, -p:],
                           inputs[:, :, 4],
                           torch.flip(inputs[:, :, 2, :, -p:], dims=[2, 3])], dim=3), 2
            )
        ]

        out1 = torch.cat(out, dim=2)
        del out

        # Pad the equatorial periodic lateral boundaries and the polar left/right boundaries
        out = []

        # Panel 0
        out.append(torch.unsqueeze(torch.cat([out1[:, :, 3, -p:, :],
                                              out1[:, :, 0],
                                              out1[:, :, 1, :p, :]], dim=2), 2))
        # Panel 1
        out.append(torch.unsqueeze(torch.cat([out1[:, :, 0, -p:, :],
                                              out1[:, :, 1],
                                              out1[:, :, 2, :p, :]], dim=2), 2))
        # Panel 2
        out.append(torch.unsqueeze(torch.cat([out1[:, :, 1, -p:, :],
                                              out1[:, :, 2],
                                              out1[:, :, 3, :p, :]], dim=2), 2))
        # Panel 3
        out.append(torch.unsqueeze(torch.cat([out1[:, :, 2, -p:, :],
                                              out1[:, :, 3],
                                              out1[:, :, 0, :p, :]], dim=2), 2))
        # Panel 4 (top)
        out.append(torch.unsqueeze(
            torch.cat([torch.transpose(torch.flip(out[3][:, :, 0, :, -2 * p:-p], dims=[2]), dim0=2, dim1=3),
                       out1[:, :, 4],
                       torch.transpose(torch.flip(out[1][:, :, 0, :, -2 * p:-p], dims=[3]), dim0=2, dim1=3)], dim=2),
            2)
        )

        del out1
        outputs = torch.cat(out, dim=2)
        del out

        return outputs


class _ConvNd(Module):
    __constants__ = ['stride', 'padding', 'dilation',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'equatorial_bias': Optional[torch.Tensor], 'polar_bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, equatorial_weight: Tensor, polar_weight: Tensor,
                      equatorial_bias: Optional[Tensor], polar_bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    padding_mode: str
    equatorial_weight: Tensor
    equatorial_bias: Optional[Tensor]
    polar_weight: Tensor
    polar_bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvNd, self).__init__()
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'replicate'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = 1
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                            total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.equatorial_weight = Parameter(torch.empty(
                (in_channels, out_channels, *kernel_size), **factory_kwargs))
            self.polar_weight = Parameter(torch.empty(
                (in_channels, out_channels, *kernel_size), **factory_kwargs))
        else:
            self.equatorial_weight = Parameter(torch.empty(
                (out_channels, in_channels, *kernel_size), **factory_kwargs))
            self.polar_weight = Parameter(torch.empty(
                (out_channels, in_channels, *kernel_size), **factory_kwargs))
        if bias:
            self.equatorial_bias = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.polar_bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('equatorial_bias', None)
            self.register_parameter('polar_bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.equatorial_weight, a=math.sqrt(5))
        if self.equatorial_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.equatorial_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.equatorial_bias, -bound, bound)

        init.kaiming_uniform_(self.polar_weight, a=math.sqrt(5))
        if self.polar_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.polar_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.polar_bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.equatorial_bias is None:
            s += ', equatorial_bias=False'
        if self.polar_bias is None:
            s += ', polar_bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class CubeSphereConv2D(_ConvNd):
    __doc__ = r"""Applies a 2D convolution over an input signal on a cubed sphere. Adapted from PyTorch Conv2d.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, 5, H, W)` and output :math:`(N, C_{\text{out}}, 5, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """ + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input hrtf
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'`` or ``'replicate'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, 5, H_{in}, W_{in})` or :math:`(C_{in}, 5, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, 5, H_{out}, W_{out})` or :math:`(C_{out}, 5, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \text{in\_channels}, 5`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(CubeSphereConv2D, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, equatorial_weight: Tensor, polar_weight: Tensor,
                      equatorial_bias: Optional[Tensor], polar_bias: Optional[Tensor]):
        outputs = []
        if self.padding_mode != 'zeros':
            # Equatorial panels
            for p in range(4):
                outputs.append(
                    torch.unsqueeze(
                        F.conv2d(
                            F.pad(input[:, :, p, :, :], self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            equatorial_weight, equatorial_bias, self.stride,
                            _pair(0), self.dilation, self.groups), 2)
                )

            # Top panel
            outputs.append(
                torch.unsqueeze(
                    F.conv2d(F.pad(input[:, :, 4, :, :], self._reversed_padding_repeated_twice, mode=self.padding_mode),
                             polar_weight, polar_bias, self.stride,
                             _pair(0), self.dilation, self.groups), 2)
            )

        else:
            # Equatorial panels
            for p in range(4):
                outputs.append(
                    torch.unsqueeze(
                        F.conv2d(input[:, :, p, :, :], equatorial_weight, equatorial_bias, self.stride,
                                 self.padding, self.dilation, self.groups), 2)
                )

            # Top panel
            outputs.append(
                torch.unsqueeze(
                    F.conv2d(input[:, :, 4, :, :], polar_weight, polar_bias, self.stride,
                             self.padding, self.dilation, self.groups), 2)
            )

        return torch.cat(outputs, 2)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.equatorial_weight, self.polar_weight,
                                  self.equatorial_bias, self.polar_bias)
