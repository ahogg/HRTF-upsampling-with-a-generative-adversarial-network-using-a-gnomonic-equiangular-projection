import os
import pickle
import torch
from torch.utils.data import Dataset


# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/7292452634137d8f5d4478e44727ec1166a89125/dataset.py


class TrainValidHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        hrtf_dir (str): Train/Valid dataset address.
        hrtf_size (int): High resolution hrtf size.
        upscale_factor (int): hrtf up scale factor.
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, hrtf_dir: str, hrtf_size: int, upscale_factor: int, transform=None) -> None:
        super(TrainValidHRTFDataset, self).__init__()
        # Get all hrtf file names in folder
        self.hrtf_file_names = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in os.listdir(hrtf_dir)
                                if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]
        # Specify the high-resolution hrtf size, with equal length and width
        self.hrtf_size = hrtf_size
        # How many times the high-resolution hrtf is the low-resolution hrtf
        self.upscale_factor = upscale_factor
        # transform to be applied to the data
        self.transform = transform

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf = pickle.load(file)

        # hrtf processing operations
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, panels, X, Y)
            hr_hrtf = torch.permute(hrtf, (3, 0, 1, 2))

        # downsample hrtf
        if self.upscale_factor == self.hrtf_size:
            mid_pos = int(self.hrtf_size/2)
            lr_hrtf = hr_hrtf[:, :, mid_pos, mid_pos, None, None]
        else:
            lr_hrtf = torch.nn.functional.interpolate(hr_hrtf, scale_factor=1 / self.upscale_factor)


        return {"lr": lr_hrtf, "hr": hr_hrtf, "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
