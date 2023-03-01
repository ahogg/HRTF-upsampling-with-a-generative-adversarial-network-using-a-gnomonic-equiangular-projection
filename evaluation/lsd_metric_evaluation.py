from model.util import spectral_distortion_metric
import glob
import torch
import pickle
import os


def run_lsd_evaluation(config, sr_dir):
    sr_data_paths = glob.glob(sr_dir + '/ARI_*')
    sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

    for file_name in sr_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        with open(sr_dir + file_name, "rb") as f:
            sr_hrtf = pickle.load(f)

        generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
        target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))
        error = spectral_distortion_metric(generated, target)
        print('LSD Error: %0.4f' % float(error.detach()))
