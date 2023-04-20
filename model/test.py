import os
import pickle

import scipy
import torch

from model.model import Generator
import shutil
from pathlib import Path


def test(config, val_prefetcher):
    # source: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/test.py
    # Initialize super-resolution model
    ngpu = config.ngpu
    valid_dir = config.valid_path

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = Generator(upscale_factor=config.upscale_factor, nbins=nbins).to(device=device)
    print("Build SRGAN model successfully.")

    # Load super-resolution model weights (always uses the CPU due to HPC having long wait times)
    model.load_state_dict(torch.load(f"{config.model_path}/Gen.pt", map_location=torch.device('cpu')))
    print(f"Load SRGAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_mb = param_size / 1024 ** 2
    size_buffer_mb = buffer_size / 1024 ** 2
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('param size: {:.3f}MB'.format(size_param_mb))
    print('buffer size: {:.3f}MB'.format(size_buffer_mb))
    print('model size: {:.3f}MB'.format(size_all_mb))

    # get list of positive frequencies of HRTF for plotting magnitude spectrum
    all_freqs = scipy.fft.fftfreq(256, 1 / config.hrir_samplerate)
    pos_freqs = all_freqs[all_freqs >= 0]

    # Start the verification mode of the model.
    model.eval()

    # Initialize the data loader and load the first batch of data
    val_prefetcher.reset()
    batch_data = val_prefetcher.next()

    # Clear/Create directories
    shutil.rmtree(Path(valid_dir), ignore_errors=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)

    while batch_data is not None:
        # Transfer in-memory data to CUDA devices to speed up validation
        lr = batch_data["lr"].to(device=device, memory_format=torch.contiguous_format,
                                 non_blocking=True, dtype=torch.float)

        # Use the generator model to generate fake samples
        with torch.no_grad():
            sr = model(lr)

        file_name = '/' + os.path.basename(batch_data["filename"][0])
        with open(valid_dir + file_name, "wb") as file:
            pickle.dump(torch.permute(sr[0], (1, 2, 3, 0)).detach().cpu(), file)

        # Preload the next batch of data
        batch_data = val_prefetcher.next()
