import argparse
import os
import pickle
import torch
from hrtfdata.torch.full import ARI
import numpy as np

from config import Config
from model.train import train
from model.test import test
from model.util import load_dataset
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, gen_sofa_baseline, \
    load_data, merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories, convert_to_sofa
from model import util
from baselines.barycentric_interpolation import run_barycentric_interpolation
from evaluation.lsd_metric_evaluation import run_lsd_evaluation

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)


def main(mode, tag, using_hpc):
    # Initialise Config object
    config = Config(tag, using_hpc=using_hpc)
    data_dir = config.raw_hrtf_dir / 'ARI'
    print(os.getcwd())
    print(data_dir)

    projection_filename = "projection_coordinates/ARI_projection_" + str(config.hrtf_size)
    if using_hpc:
        projection_filename = "HRTF-GANs/" + projection_filename

    if mode == 'generate_projection':
        # Must be run in this mode once per dataset, finds barycentric coordinates for each point in the cubed sphere

        # No need to load the entire dataset in this case
        ds: ARI = load_data(data_folder=data_dir, load_function=ARI, domain='time', side='left', subject_ids='first')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        generate_euclidean_cube(cs.get_sphere_coords(), projection_filename, edge_len=config.hrtf_size)

    elif mode == 'preprocess':
        # Interpolates data to find HRIRs on cubed sphere, then FFT to obtain HRTF, finally splits data into train and
        # val sets and saves processed data

        ds: ARI = load_data(data_folder=data_dir, load_function=ARI, domain='time', side='both')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        # Clear/Create directories
        clear_create_directories(config)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)

        # collect all train_hrtfs to get mean and sd
        train_hrtfs = torch.empty(size=(2 * train_size, 5, config.hrtf_size, config.hrtf_size, 128))
        j = 0
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")
            clean_hrtf = interpolate_fft(cs, ds[i]['features'], sphere, sphere_triangles, sphere_coeffs, cube,
                                         config.hrtf_size)

            hrtf_original, phase_original, sphere_original = get_hrtf_from_ds(ds, i)

            # save cleaned hrtfdata
            if ds[i]['group'] in train_sample:
                projected_dir = config.train_hrtf_dir
                projected_dir_original = config.train_original_hrtf_dir
                train_hrtfs[j] = clean_hrtf
                j += 1
            else:
                projected_dir = config.valid_hrtf_dir
                projected_dir_original = config.valid_original_hrtf_dir

            subject_id = str(ds[i]['group'])
            side = ds[i]['target']
            with open('%s/ARI_mag_%s%s.pickle' % (projected_dir, subject_id, side), "wb") as file:
                pickle.dump(clean_hrtf, file)

            with open('%s/ARI_mag_%s%s.pickle' % (projected_dir_original, subject_id, side), "wb") as file:
                pickle.dump(hrtf_original, file)

            with open('%s/ARI_phase_%s%s.pickle' % (projected_dir_original, subject_id, side), "wb") as file:
                pickle.dump(phase_original, file)

        if config.merge_flag:
            merge_files(config)

        if config.gen_sofa_flag:
            gen_sofa_preprocess(config, cube, sphere, sphere_original)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        # Trains the GANs, according to the parameters specified in Config
        train_prefetcher, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        util.initialise_folders(tag, overwrite=True)
        train(config, train_prefetcher, overwrite=True)

    elif mode == 'test':
        _, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        util.initialise_folders(tag, overwrite=True)
        test(config, test_prefetcher)

        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        if config.gen_sofa_flag:
            convert_to_sofa(config.valid_path, config, cube, sphere)
            print('Created valid sofa files')

    elif mode == 'baseline':
        no_nodes = str(int(5 * (config.hrtf_size / config.upscale_factor) ** 2))
        no_full_nodes = str(int(5 * config.hrtf_size ** 2))

        barycentric_data_folder = '/barycentric_interpolated_data_%s_%s' % (no_nodes, no_full_nodes)
        cube, sphere = run_barycentric_interpolation(config, barycentric_data_folder, subject_file='ARI_mag_16.pickle')

        if config.gen_sofa_flag:
            gen_sofa_baseline(config, barycentric_data_folder, cube, sphere)
            print('Created barycentric baseline sofa files')

        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
        run_lsd_evaluation(config, barycentric_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-c", "--hpc")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")

    if args.tag:
        tag = args.tag
    else:
        tag = 'test'
    main(args.mode, tag, hpc)

    # main('train', 'localtrain', using_hpc=False)
