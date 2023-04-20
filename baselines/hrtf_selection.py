import pickle
import os
import glob
import re
import torch
import numpy as np
import shutil
from pathlib import Path

from model.util import spectral_distortion_metric

def run_hrtf_selection(config, hrtf_selection_output_path, subject_file=None):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(hrtf_selection_output_path), ignore_errors=True)
    Path(hrtf_selection_output_path).mkdir(parents=True, exist_ok=True)

    # construct dicts of all HRTFs from the training data for left and right ears
    hrtf_dict_left = {}
    hrtf_dict_right = {}
    subj_ids = []
    for file_name in valid_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        # add to dict for right ears
        subj_id = int(re.findall(r'\d+', file_name)[0])
        hrtf_dict_left[subj_id] = torch.permute(torch.tensor(np.array([np.array(hr_hrtf).T[0:config.nbins_hrtf]])), (0, 1, 4, 3, 2))
        hrtf_dict_right[subj_id] = torch.permute(torch.tensor(np.array([np.array(hr_hrtf).T[config.nbins_hrtf:]])), (0, 1, 4, 3, 2))
        subj_ids.append(subj_id)

    # for each subject, compare their HRTF sets to all other subjects' HRTF sets via SD metric
    overall_avg_dict = {}
    for subject_id_ref in subj_ids:
        # the 'reference' HRTFs are considered as candidates for the non-individualized HRTFs
        # essentially, we are trying to find the subject whose HRTF sets are most "average"
        # relative to the rest of the training data
        running_total = 0
        for subject_id in subj_ids:
            if subject_id != subject_id_ref:
                # SD metric for right ear
                sd_right = spectral_distortion_metric(hrtf_dict_right[subject_id_ref], hrtf_dict_right[subject_id])
                # SD metric for left ear
                sd_left = spectral_distortion_metric(hrtf_dict_left[subject_id_ref], hrtf_dict_left[subject_id])
                # average for left & right
                sd_avg = (sd_right + sd_left) / 2.
                # add to running total
                running_total += sd_avg

        # find the average SD metric for each possible 'reference' subject
        overall_avg = running_total / (len(hrtf_dict_right.keys()) - 1.)
        print(f"Average for {subject_id_ref}: {overall_avg}")
        overall_avg_dict[subject_id_ref] = overall_avg

    # find the HR HRTF that minimizes the SD metric relative to all other HR HRTFs
    min_id = min(overall_avg_dict, key=overall_avg_dict.get)
    min_val = overall_avg_dict[min_id]
    with open(f'{hrtf_selection_output_path}/minimum.pickle', "wb") as file:
        with open(f'{config.valid_hrtf_merge_dir}/{config.dataset}_mag_{min_id}.pickle', "rb") as f:
            hr_hrtf = pickle.load(f)
        pickle.dump(hr_hrtf, file)

    print(f"Minimum is {min_id} with average LSD {min_val}")

    max_id = max(overall_avg_dict, key=overall_avg_dict.get)
    max_val = overall_avg_dict[max_id]
    with open(f'{hrtf_selection_output_path}/maximum.pickle', "wb") as file:
        with open(f'{config.valid_hrtf_merge_dir}/{config.dataset}_mag_{max_id}.pickle', "rb") as f:
            hr_hrtf = pickle.load(f)
        pickle.dump(hr_hrtf, file)

    print(f"Maximum is {max_id} with average LSD {max_val}")

    print('Created HRTF selection baseline')

    return
