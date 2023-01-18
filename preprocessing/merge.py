import os
import pickle

import torch

# Data paths
train_dir = "/Users/madsjenkins/PycharmProjects/HRTF-GANs/projected_data/train"
valid_dir = "/Users/madsjenkins/PycharmProjects/HRTF-GANs/projected_data/valid"

merge_train_dir = "/Users/madsjenkins/PycharmProjects/HRTF-GANs/projected_data/train-merge/"
merge_valid_dir = "/Users/madsjenkins/PycharmProjects/HRTF-GANs/projected_data/valid-merge/"


# merge left and right hrtfs and save merged versions
def merge_left_right_hrtfs(input_dir, output_dir):
    hrtf_file_names = [os.path.join(input_dir, hrtf_file_name) for hrtf_file_name in os.listdir(input_dir)]

    hrtf_dict_left = {}
    hrtf_dict_right = {}
    for f in hrtf_file_names:
        with open(f, "rb") as file:
            hrtf = pickle.load(file)

        # add to dict for right ears
        if f[-5:] == 'right':
            subj_id = int(f.split("_")[-1][:-5])
            hrtf_dict_right[subj_id] = hrtf
        # add to dict for left ears
        elif f[-4:] == 'left':
            subj_id = int(f.split("_")[-1][:-4])
            hrtf_dict_left[subj_id] = hrtf

    for subj_id in hrtf_dict_right.keys():
        hrtf_r = hrtf_dict_right[subj_id]
        hrtf_l = hrtf_dict_left[subj_id]
        hrtf_merged = torch.cat((hrtf_l, hrtf_r), dim=3)
        with open(output_dir + "ARI_" + str(subj_id), "wb") as file:
            pickle.dump(hrtf_merged, file)


merge_left_right_hrtfs(train_dir, merge_train_dir)
merge_left_right_hrtfs(valid_dir, merge_valid_dir)
