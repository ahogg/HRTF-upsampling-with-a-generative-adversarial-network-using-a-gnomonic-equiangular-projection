import pickle
import os
import glob
import numpy as np
import torch
import shutil
from pathlib import Path

from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft
from preprocessing.convert_coordinates import convert_cube_to_sphere
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates

PI_4 = np.pi / 4


def run_barycentric_interpolation(config):
    vaild_data_paths = glob.glob(config.valid_hrtf_merge_dir + '/ARI_*')
    vaild_data_file_names = ['/' + os.path.basename(x) for x in vaild_data_paths]

    no_nodes = str(int(5 * (config.hrtf_size / config.upscale_factor) ** 2))
    no_full_nodes = str(int(5 * config.hrtf_size ** 2))

    barycentric_data_folder = '/barycentric_interpolated_data_%s_%s' % (no_nodes, no_full_nodes)
    barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder

    # Clear/Create directory
    shutil.rmtree(Path(barycentric_output_path), ignore_errors=True)
    Path(barycentric_output_path).mkdir(parents=True, exist_ok=True)

    for file_name in vaild_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        lr_hrtf = torch.permute(
            torch.nn.functional.interpolate(torch.permute(hr_hrtf, (3, 0, 1, 2)), scale_factor=1 / config.upscale_factor),
            (1, 2, 3, 0))

        filename = 'projection_coordinates/ARI_projection_%s' % config.hrtf_size
        with open(filename, "rb") as f:
            (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

        sphere_coords_lr = []
        sphere_coords_lr_index = []
        for panel, x, y in cube_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            i = panel - 1
            j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            if hr_hrtf[i, j, k] in lr_hrtf:
                sphere_coords_lr.append(convert_cube_to_sphere(panel, x, y))
                sphere_coords_lr_index.append([int(i), int(j / config.upscale_factor), int(k / config.upscale_factor)])

        euclidean_sphere_triangles = []
        euclidean_sphere_coeffs = []
        for sphere_coord in sphere_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1], sphere_coords=sphere_coords_lr)
            coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1], closest_points=triangle_vertices)
            euclidean_sphere_triangles.append(triangle_vertices)
            euclidean_sphere_coeffs.append(coeffs)

        cs = CubedSphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)
        barycentric_hr = interpolate_fft(cs, lr_hrtf, sphere_coords, euclidean_sphere_triangles,
                                         euclidean_sphere_coeffs, cube_coords, config.hrtf_size)

        with open(barycentric_output_path + file_name, "wb") as file:
            pickle.dump(barycentric_hr, file)
