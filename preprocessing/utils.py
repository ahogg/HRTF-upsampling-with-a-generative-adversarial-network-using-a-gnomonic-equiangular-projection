import cmath
import pickle
import os

import sofar as sf
import numpy as np
import torch
import scipy
from scipy.signal import hilbert
import scipy.signal as sps
import shutil
from pathlib import Path
import re


from preprocessing.barycentric_calcs import calc_barycentric_coordinates, get_triangle_vertices
from preprocessing.convert_coordinates import convert_cube_to_sphere
from preprocessing.KalmanFilter import KalmanFilter

PI_4 = np.pi / 4


def clear_create_directories(config):
    """Clear/Create directories"""
    shutil.rmtree(Path(config.train_hrtf_dir), ignore_errors=True)
    shutil.rmtree(Path(config.valid_hrtf_dir), ignore_errors=True)
    Path(config.train_hrtf_dir).mkdir(parents=True, exist_ok=True)
    Path(config.valid_hrtf_dir).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(Path(config.train_original_hrtf_dir), ignore_errors=True)
    shutil.rmtree(Path(config.valid_original_hrtf_dir), ignore_errors=True)
    Path(config.train_original_hrtf_dir).mkdir(parents=True, exist_ok=True)
    Path(config.valid_original_hrtf_dir).mkdir(parents=True, exist_ok=True)


def merge_left_right_hrtfs(input_dir, output_dir):
    # Clear/Create directory
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    hrtf_file_names = [os.path.join(input_dir, hrtf_file_name) for hrtf_file_name in os.listdir(input_dir)
                       if os.path.isfile(os.path.join(input_dir, hrtf_file_name))]

    data_dict_left = {}
    data_dict_right = {}
    for f in hrtf_file_names:

        file_ext = re.findall(re.escape(input_dir) + '/(.*)_[0-9]*[a-z]*.pickle$', f)[0]

        with open(f, "rb") as file:
            data = pickle.load(file)

        # add to dict for right ears
        if re.search(re.escape(input_dir)+'/.*_[0-9]*right.pickle$', f):
            subj_id = int(re.findall(re.escape(input_dir)+'/.*_([0-9]*)right.pickle$', f)[0])
            if file_ext not in data_dict_right:
                data_dict_right[file_ext] = {}
            data_dict_right[file_ext][subj_id] = data
        # add to dict for left ears
        elif re.search(re.escape(input_dir)+'/.*_[0-9]*left.pickle$', f):
            subj_id = int(re.findall(re.escape(input_dir)+'/.*_([0-9]*)left.pickle$', f)[0])
            if file_ext not in data_dict_left:
                data_dict_left[file_ext] = {}
            data_dict_left[file_ext][subj_id] = data

    for file_ext in data_dict_right.keys():
        missing_subj_ids =  list(set(data_dict_right[file_ext].keys()) - set(data_dict_left[file_ext].keys()))
        if len(missing_subj_ids) > 0:
            print('Excluding subject IDs where both ears do not exist (IDs: %s)' % ', '.join(map(str, missing_subj_ids)))
            for missing_subj_id in missing_subj_ids:
                data_dict_right[file_ext].pop(missing_subj_id, None)
                data_dict_left[file_ext].pop(missing_subj_id, None)

        for subj_id in data_dict_right[file_ext].keys():
            hrtf_r = data_dict_right[file_ext][subj_id]
            hrtf_l = data_dict_left[file_ext][subj_id]
            dimension = hrtf_r.ndim-1
            hrtf_merged = torch.cat((hrtf_l, hrtf_r), dim=dimension)
            with open('%s/%s_%s.pickle' % (output_dir, file_ext, subj_id), "wb") as file:
                pickle.dump(hrtf_merged, file)


def merge_files(config):
    merge_left_right_hrtfs(config.train_hrtf_dir, config.train_hrtf_merge_dir)
    merge_left_right_hrtfs(config.valid_hrtf_dir, config.valid_hrtf_merge_dir)
    merge_left_right_hrtfs(config.train_original_hrtf_dir, config.train_original_hrtf_merge_dir)
    merge_left_right_hrtfs(config.valid_original_hrtf_dir, config.valid_original_hrtf_merge_dir)


def get_hrtf_from_ds(config, ds, index):
    coordinates = ds.row_angles, ds.column_angles
    position_grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)

    sphere_temp = []
    hrir_temp = []
    for row_idx, row in enumerate(ds.row_angles):
        for column_idx, column in enumerate(ds.column_angles):
            if not any(np.ma.getmaskarray(ds[index]['features'][row_idx][column_idx].flatten())):
                az_temp = np.radians(position_grid[row_idx][column_idx][0])
                el_temp = np.radians(position_grid[row_idx][column_idx][1])
                sphere_temp.append([el_temp, az_temp])
                hrir_temp.append(np.ma.getdata(ds[index]['features'][row_idx][column_idx]).flatten())
    hrtf_temp, phase_temp = calc_hrtf(config, hrir_temp)

    return torch.tensor(np.array(hrtf_temp)), torch.tensor(np.array(phase_temp)), sphere_temp


def add_itd(az, el, hrir, side, fs=48000, r=0.0875, c=343):

    az = np.radians(az)
    el = np.radians(el)
    interaural_azimuth = np.arcsin(np.sin(az) * np.cos(el))
    delay_in_sec = (r / c) * (interaural_azimuth + np.sin(interaural_azimuth))
    fractional_delay = delay_in_sec * fs

    sample_delay = int(abs(fractional_delay))

    if (delay_in_sec > 0 and side == 'right') or (delay_in_sec < 0 and side == 'left'):
        N = len(hrir)
        delayed_hrir = np.zeros(N)
        delayed_hrir[sample_delay:] = hrir[0:N - sample_delay]
        sofa_delay = sample_delay
    else:
        sofa_delay = 0
        delayed_hrir = hrir

    return delayed_hrir, sofa_delay


def gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count, left_phase=None, right_phase=None):
    el = np.degrees(sphere_coords[count][0])
    az = np.degrees(sphere_coords[count][1])
    source_position = [az + 360 if az < 0 else az, el, 1.2]

    if left_phase is None:
        left_hrtf[left_hrtf == 0.0] = 1.0e-08
        left_phase = np.imag(-hilbert(np.log(np.abs(left_hrtf))))
    if right_phase is None:
        right_hrtf[right_hrtf == 0.0] = 1.0e-08
        right_phase = np.imag(-hilbert(np.log(np.abs(right_hrtf))))

    left_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(left_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * left_phase))[:config.nbins_hrtf]
    right_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(right_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * right_phase))[:config.nbins_hrtf]

    left_hrir, left_sample_delay = add_itd(az, el, left_hrir, side='left')
    right_hrir, right_sample_delay = add_itd(az, el, right_hrir, side='right')

    full_hrir = [left_hrir, right_hrir]
    delay = [left_sample_delay, right_sample_delay]

    return source_position, full_hrir, delay


def save_sofa(clean_hrtf, config, cube_coords, sphere_coords, sofa_path_output, phase=None):
    full_hrirs = []
    source_positions = []
    delays = []
    left_full_phase = None
    right_full_phase = None
    if cube_coords is None:
        left_full_hrtf = clean_hrtf[:, :config.nbins_hrtf]
        right_full_hrtf = clean_hrtf[:, config.nbins_hrtf:]

        if phase is not None:
            left_full_phase = phase[:, :config.nbins_hrtf]
            right_full_phase = phase[:, config.nbins_hrtf:]

        for count in range(len(sphere_coords)):
            left_hrtf = np.array(left_full_hrtf[count])
            right_hrtf = np.array(right_full_hrtf[count])

            if phase is None:
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
            else:
                left_phase = np.array(left_full_phase[count])
                right_phase = np.array(right_full_phase[count])
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count, left_phase, right_phase)

            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)

    else:
        left_full_hrtf = clean_hrtf[:, :, :, :config.nbins_hrtf]
        right_full_hrtf = clean_hrtf[:, :, :, config.nbins_hrtf:]

        count = 0
        for panel, x, y in cube_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            i = panel - 1
            j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))

            left_hrtf = np.array(left_full_hrtf[i, j, k])
            right_hrtf = np.array(right_full_hrtf[i, j, k])
            source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)
            count += 1

    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = full_hrirs
    sofa.Data_SamplingRate = config.hrir_samplerate
    sofa.Data_Delay = delays
    sofa.SourcePosition = source_positions
    sf.write_sofa(sofa_path_output, sofa)


def convert_to_sofa(hrtf_dir, config, cube, sphere, phase_ext='_phase', use_phase=False, mag_ext='_mag'):
    if use_phase:
        sofa_path_output = hrtf_dir + '/sofa_with_phase/'
    else:
        sofa_path_output = hrtf_dir + '/sofa_min_phase/'

    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(hrtf_dir)
                        if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name)) and phase_ext not in hrtf_file_name]
    phase_file_names = [phase_file_name for phase_file_name in os.listdir(hrtf_dir)
                        if os.path.isfile(os.path.join(hrtf_dir, phase_file_name)) and phase_ext in phase_file_name]

    # Clear/Create directories
    shutil.rmtree(Path(sofa_path_output), ignore_errors=True)
    Path(sofa_path_output).mkdir(parents=True, exist_ok=True)

    for f in hrtf_file_names:
        with open(os.path.join(hrtf_dir, f), "rb") as hrtf_file:
            hrtf = pickle.load(hrtf_file)
            sofa_filename_output = os.path.basename(hrtf_file.name).replace('.pickle', '.sofa').replace(mag_ext,'')
            sofa_output = sofa_path_output + sofa_filename_output

            if use_phase:
                for f_phase in phase_file_names:
                    f_phase = f_phase.replace(phase_ext, mag_ext)
                    if f_phase == f:
                        with open(os.path.join(hrtf_dir, f), "rb") as phase_file:
                            phase = pickle.load(phase_file)
                            save_sofa(hrtf, config, cube, sphere, sofa_output, phase)
            else:
                save_sofa(hrtf, config, cube, sphere, sofa_output)


def gen_sofa_preprocess(config, cube, sphere, sphere_original):
    convert_to_sofa(config.train_hrtf_merge_dir, config, cube, sphere)
    convert_to_sofa(config.valid_hrtf_merge_dir, config, cube, sphere)
    convert_to_sofa(config.train_original_hrtf_merge_dir, config, cube=None, sphere=sphere_original)
    convert_to_sofa(config.valid_original_hrtf_merge_dir, config, cube=None, sphere=sphere_original)
    convert_to_sofa(config.train_original_hrtf_merge_dir, config, use_phase=True, cube=None, sphere=sphere_original)
    convert_to_sofa(config.valid_original_hrtf_merge_dir, config, use_phase=True, cube=None, sphere=sphere_original)


def generate_euclidean_cube(config, measured_coords, edge_len=16):
    """Calculate barycentric coordinates for projection based on a specified cube sphere edge length and a set of
    measured coordinates, finally save them to the file"""
    cube_coords, sphere_coords = [], []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                cube_coords.append((panel, x_i, y_i))
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))

    euclidean_sphere_triangles = []
    euclidean_sphere_coeffs = []

    for count, p in enumerate(sphere_coords):
        triangle_vertices = get_triangle_vertices(elevation=p[0], azimuth=p[1], sphere_coords=measured_coords)
        coeffs = calc_barycentric_coordinates(elevation=p[0], azimuth=p[1], closest_points=triangle_vertices)
        euclidean_sphere_triangles.append(triangle_vertices)
        euclidean_sphere_coeffs.append(coeffs)

        print(f"Data point {count+1} out of {len(sphere_coords)} ({round(100 * count / len(sphere_coords))}%)")

    # save euclidean_cube, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs
    Path(config.projection_dir).mkdir(parents=True, exist_ok=True)
    filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    with open(filename, "wb") as file:
        pickle.dump((cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs), file)


def save_euclidean_cube(edge_len=16):
    """Save euclidean cube as a txt file for use as input to matlab"""
    sphere_coords = []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))
    with open('../projection_coordinates/generated_coordinates.txt', 'w') as f:
        for coord in sphere_coords:
            print(coord)
            f.write(str(coord[0]))
            f.write(", ")
            f.write(str(coord[1]))
            f.write('\n')


def get_feature_for_point(elevation, azimuth, all_coords, subject_features):
    """For a given point (elevation, azimuth), get the associated feature value"""
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    azimuth_index = int(all_coords_row.azimuth_index)
    elevation_index = int(all_coords_row.elevation_index)
    return subject_features[azimuth_index][elevation_index]


def get_feature_for_point_tensor(elevation, azimuth, all_coords, subject_features):
    """For a given point (elevation, azimuth), get the associated feature value"""
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    return scipy.fft.irfft(np.concatenate((np.array([0.0]), np.array(subject_features[int(all_coords_row.panel-1)][int(all_coords_row.elevation_index)][int(all_coords_row.azimuth_index)]))))


def calc_interpolated_feature(time_domain_flag, triangle_vertices, coeffs, all_coords, subject_features):
    """Calculate the interpolated feature for a given point based on vertices specified by triangle_vertices, features
    specified by subject_features, and barycentric coefficients specified by coeffs"""
    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in triangle_vertices:
        if time_domain_flag:
            features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
            features_no_ITD = remove_itd(features_p, int(len(features_p)*0.04), len(features_p))
            features.append(features_no_ITD)
        else:
            features_p = get_feature_for_point_tensor(p[0], p[1], all_coords, subject_features)
            features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    if len(features) == 3:
        interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]
    else:
        interpolated_feature = features[0]

    return interpolated_feature


def calc_all_interpolated_features(cs, features, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs):
    """Essentially a wrapper function for calc_interpolated_features above, calculated interpolated features for all
    points on the euclidean sphere rather than a single point"""
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            if 'panel_index' in cs.all_coords.columns:
                time_domain_flag = False
            else:
                time_domain_flag = True
            features_p = calc_interpolated_feature(time_domain_flag=time_domain_flag,
                                                   triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=cs.get_all_coords(),
                                                   subject_features=features)

            selected_feature_interpolated.append(features_p)
        else:
            selected_feature_interpolated.append(None)

    return selected_feature_interpolated


def calc_hrtf(config, hrirs):
    """FFT to obtain HRTF from HRIR"""
    magnitudes = []
    phases = []

    for hrir in hrirs:
        # remove value that corresponds to 0 Hz
        hrtf = scipy.fft.rfft(hrir, config.nbins_hrtf*2)[1:]
        magnitude = abs(hrtf)
        phase = [cmath.phase(x) for x in hrtf]
        magnitudes.append(magnitude)
        phases.append(phase)
    return magnitudes, phases


def interpolate_fft(config, cs, features, sphere, sphere_triangles, sphere_coeffs, cube, fs_original, edge_len):
    """Combine all data processing steps into one function

    :param cs: Cubed sphere object associated with dataset
    :param features: All features for a given subject in the dataset, given by ds[i]['features'] from hrtfdata
    :param sphere: A list of locations of the gridded cubed sphere points to be interpolated,
                    given as (elevation, azimuth)
    :param sphere_triangles: A list of lists of triangle vertices for barycentric interpolation, where each list of
                             vertices defines the triangle for the corresponding point in sphere
    :param sphere_coeffs: A list of barycentric coordinates for each location in sphere, corresponding to the triangles
                          described by sphere_triangles
    :param cube: A list of locations of the gridded cubed sphere points to be interpolated, given as (panel, x, y)
    :param edge_len: Edge length of gridded cube
    """

    # interpolated_hrirs is a list of interpolated HRIRs corresponding to the points specified in load_sphere and
    # load_cube, all three lists share the same ordering
    interpolated_hrirs = calc_all_interpolated_features(cs, features, sphere, sphere_triangles, sphere_coeffs)

    # Resample data so that training and validation sets are created at the same fs ('config.hrir_samplerate').
    # number_of_samples = round(np.shape(interpolated_hrirs)[-1] * float(config.hrir_samplerate) / fs_original)
    # interpolated_hrirs_resampled = sps.resample(np.array(interpolated_hrirs).T, number_of_samples).T

    magnitudes, phases = calc_hrtf(config, interpolated_hrirs)

    # create empty list of lists of lists and initialize counter
    magnitudes_raw = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    count = 0

    for panel, x, y in cube:
        # based on cube coordinates, get indices for magnitudes list of lists
        i = panel - 1
        j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
        k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))

        # add to list of lists of lists and increment counter
        magnitudes_raw[i][j][k] = magnitudes[count]
        count += 1

    # convert list of numpy arrays into a single array, such that converting into tensor is faster
    return torch.tensor(np.array(magnitudes_raw))


def trim_hrir(hrir, start, stop):
    if start < 0:
        hrir_padded = np.pad(hrir, (abs(start), 0), mode='constant')
        trimmed_hrir = hrir_padded[:stop]
    else:
        trimmed_hrir = hrir[start:stop]
    return trimmed_hrir


def remove_itd(hrir, pre_window, length):
    """Remove ITD from HRIR using kalman filter"""
    # normalize such that max(abs(hrir)) == 1
    rescaling_factor = 1 / max(np.abs(hrir))
    normalized_hrir = rescaling_factor * hrir

    # initialise Kalman filter
    x = np.array([[0]])  # estimated initial state
    p = np.array([[0]])  # estimated initial variance

    h = np.array([[1]])  # observation model (observation represents internal state directly)

    # r and q may require tuning
    r = np.array([[np.sqrt(400)]])  # variance of the observation noise
    q = np.array([[0.01]])  # variance of the process noise

    hrir_filter = KalmanFilter(x, p, h, q, r)
    f = np.array([[1]])  # F is state transition model
    for i, z in enumerate(normalized_hrir):
        hrir_filter.prediction(f)
        hrir_filter.update(z)
        # find first time post fit residual exceeds some threshold
        if np.abs(hrir_filter.get_post_fit_residual()) > 0.005:
            over_threshold_index = i
            break
    else:
        print("RuntimeWarning: ITD not removed (Kalman filter did not find a time where post fit residual exceeded threshold).")
        return hrir

    # create fade window in order to taper off HRIR towards the beginning and end
    fadeout_len = 50
    fadeout_interval = -1. / fadeout_len
    fadeout = np.arange(1 + fadeout_interval, fadeout_interval, fadeout_interval).tolist()

    fadein_len = 10
    fadein_interval = 1. / fadein_len
    fadein = np.arange(0.0, 1.0, fadein_interval).tolist()

    # trim HRIR based on first time threshold is exceeded
    start = over_threshold_index - pre_window
    if start < 0:
        stop = length
    else:
        stop = start + length

    if len(hrir) >= stop:
        trimmed_hrir = trim_hrir(hrir, start, stop)
        fade_window = fadein + [1] * (length - fadein_len - fadeout_len) + fadeout
        faded_hrir = trimmed_hrir * fade_window
    else:
        trimmed_hrir = trim_hrir(hrir, start, -1)
        fade_window = fadein + [1] * (len(trimmed_hrir) - fadein_len - fadeout_len) + fadeout
        faded_hrir = trimmed_hrir * fade_window
        zero_pad = [0] * (length - len(trimmed_hrir))
        faded_hrir = np.ma.append(faded_hrir, zero_pad)

    return faded_hrir
