import torch
import torch.nn as nn
import numpy as np
from model.custom_conv import CubeSpherePadding2D, CubeSphereConv2D
nbins = 256

# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py


class ResidualConvBlock(nn.Module):
    """Implements residual conv function.
    Args:
        channels (int): Number of channels in the input.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            CubeSpherePadding2D(1),
            CubeSphereConv2D(channels, channels, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(channels),
            nn.PReLU(),
            CubeSpherePadding2D(1),
            CubeSphereConv2D(channels, channels, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block_1 = nn.Sequential(
            CubeSpherePadding2D(1),
            CubeSphereConv2D(channels, channels * 4, (3, 3), (1, 1))
        )
        self.upsample_block_2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.upsample_block_1(x)
        out = self.upsample_block_2(torch.permute(out1, dims=(0, 2, 1, 3, 4)))

        return torch.permute(out, dims=(0, 2, 1, 3, 4))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (nbin) x 5 x 16 x 16
            CubeSpherePadding2D(1),
            CubeSphereConv2D(nbins, 64, (3, 3), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 5 x 16 x 16
            CubeSpherePadding2D(1),
            CubeSphereConv2D(64, 64, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 5 x 16 x 16
            CubeSpherePadding2D(1),
            CubeSphereConv2D(64, 128, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 5 x 16 x 16
            CubeSpherePadding2D(1),
            CubeSphereConv2D(128, 128, (3, 3), (2, 2), bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),
            CubeSpherePadding2D(1),
            CubeSphereConv2D(128, 256, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 5 x 8 x 8
            CubeSpherePadding2D(1),
            CubeSphereConv2D(256, 256, (3, 3), (2, 2), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),
            CubeSpherePadding2D(1),
            CubeSphereConv2D(256, 512, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 5 x 4 x 4
            CubeSpherePadding2D(1),
            CubeSphereConv2D(512, 512, (3, 3), (2, 2), bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2 * 5, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor) -> None:
        super(Generator, self).__init__()
        self.ngf = 512
        self.num_upsampling_blocks = int(np.log(upscale_factor)/np.log(2))

        # First conv layer.
        self.conv_block1 = nn.Sequential(
            CubeSpherePadding2D(1),
            CubeSphereConv2D(nbins, self.ngf, (3, 3), (1, 1)),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(8):
            trunk.append(ResidualConvBlock(self.ngf))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            CubeSpherePadding2D(1),
            CubeSphereConv2D(self.ngf, self.ngf, (3, 3), (1, 1), bias=False),
            nn.BatchNorm3d(self.ngf),
        )

        # Upscale block
        upsampling = []
        for _ in range(self.num_upsampling_blocks):
            upsampling.append(UpsampleBlock(self.ngf))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv_block3 = nn.Sequential(
            CubeSpherePadding2D(1),
            CubeSphereConv2D(self.ngf, nbins, (3, 3), (1, 1))
        )

        self.classifier = nn.Softplus()

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, CubeSphereConv2D):
                nn.init.kaiming_normal_(module.equatorial_weight)
                nn.init.kaiming_normal_(module.polar_weight)
                if module.equatorial_bias is not None:
                    nn.init.constant_(module.equatorial_bias, 0)
                    nn.init.constant_(module.polar_bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
