import matplotlib.pyplot as plt
import itertools

import numpy as np
import torch
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import LinearLocator

from model.util import spectral_distortion_metric_for_plot
from preprocessing.convert_coordinates import convert_sphere_to_cartesian, convert_cube_to_cartesian, \
    convert_cube_indices_to_spherical
from preprocessing.utils import calc_all_interpolated_features, get_feature_for_point

PI_4 = np.pi / 4


def plot_3d_shape(shape, coordinates, shading=None):
    """Plot points from a sphere or a cubed sphere in 3D

    :param shape: either "sphere" or "cube" to specify the shape to plot
    :param coordinates: A list of coordinates to plot, either (elevation, azimuth) for spheres or
                        (panel, x, y) for cubed spheres
    :param shading: A list of values equal in length to the number of coordinates that is used for shading the points
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Format data.
    if shape == "sphere":
        x, y, z, mask = convert_sphere_to_cartesian(coordinates)
    elif shape == "cube":
        x, y, z, mask = convert_cube_to_cartesian(coordinates)
    else:
        raise RuntimeError("Please provide a valid shape, either 'sphere' or 'cube'.")

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # Plot the surface.
    sc = ax.scatter(x, y, z, c=shading, s=10,
                    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    plt.colorbar(sc)

    plt.show()


def plot_flat_panel(cube_coords, shading=None):
    """Plot points from a single panel of a cubed sphere in its flattened form

    :param cube_coords: A list of coordinates to plot in the form (panel, x, y) for cubed spheres
    :param shading: A list of values equal in length to the number of coordinates that is used for shading the points
    """
    fig, ax = plt.subplots()

    # Format data.
    x, y = [], []
    mask = []

    for panel, p, q in cube_coords:
        if panel == 1:
            if not np.isnan(p) and not np.isnan(q):
                mask.append(True)
                x.append(p)
                y.append(q)
            else:
                mask.append(False)
        else:
            mask.append(False)

    x, y = np.asarray(x), np.asarray(y)

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # Plot the surface.
    sc = ax.scatter(x, y, c=shading, s=50,
                    linewidth=0, antialiased=False, vmin=-0.04, vmax=0.03)
    plt.colorbar(sc)

    fig.tight_layout()
    plt.show()


def plot_flat_cube(cube_coords, shading=None):
    """Plot points from cubed sphere in its flattened form

    :param cube_coords: A list of coordinates to plot in the form (panel, x, y) for cubed spheres
    :param shading: A list of values equal in length to the number of coordinates that is used for shading the points
    """
    fig, ax = plt.subplots()

    # Format data.
    x, y = [], []
    mask = []

    for panel, p, q in cube_coords:
        if not np.isnan(p) and not np.isnan(q):
            mask.append(True)

            if panel == 1:
                x_i, y_i = p, q
            elif panel == 2:
                x_i, y_i = p + np.pi / 2, q
            elif panel == 3:
                x_i, y_i = p + np.pi, q
            elif panel == 4:
                x_i, y_i = p - np.pi / 2, q
            elif panel == 5:
                x_i, y_i = p, q + np.pi / 2
            else:
                x_i, y_i = p, q - np.pi / 2

            x.append(x_i)
            y.append(y_i)
        else:
            mask.append(False)

    x, y = np.asarray(x), np.asarray(y)

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # draw lines outlining cube
    ax.hlines(y=-PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
    ax.hlines(y=PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
    ax.hlines(y=3 * PI_4, xmin=-PI_4, xmax=PI_4, linewidth=2, color="grey")

    ax.vlines(x=-3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=-PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=5 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")

    # Plot the surface.
    sc = ax.scatter(x, y, c=shading, s=10,
                    linewidth=0, antialiased=False)
    plt.colorbar(sc)

    fig.tight_layout()
    fig.set_size_inches(9, 4)
    plt.show()


def plot_polar(sphere_coords, shading=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Format data.
    theta, r = [], []
    mask = []

    for elevation, azimuth in sphere_coords:
        if elevation is not None and azimuth is not None and elevation < 0:
            mask.append(True)
            theta.append(azimuth)
            r.append(np.pi + elevation)
        else:
            mask.append(False)

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    sc = ax.scatter(theta, r, c=shading, s=5)
    plt.colorbar(sc)
    ax.set_rmax(np.pi)
    ax.set_rmin(0)
    ax.set_rticks([2, 2.5, 3])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A plot of the bottom of the sphere", va='bottom')
    plt.show()


def plot_impulse_response(times, title=""):
    """Plot a single impulse response, where sound pressure levels are provided as a list"""
    plt.plot(times)
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Sound Pressure Level", fontsize=14)
    plt.show()


def plot_ir_subplots(hrir1, hrir2, title1="", title2="", suptitle=""):
    """Plot two IRs as subplots"""
    fig, axs = plt.subplots(2)
    fig.suptitle(suptitle, fontsize=16)
    axs[0].plot(hrir1)
    axs[0].set_xlabel('Time (samples)', fontsize=14)
    axs[0].set_title(title1, fontsize=16)
    axs[1].plot(hrir2)
    axs[1].set_xlabel('Time (samples)', fontsize=14)
    axs[1].set_title(title2, fontsize=16)
    fig.supylabel("Sound Pressure Level", fontsize=14)
    plt.subplots_adjust(left=0.15, right=0.95, hspace=0.7, top=0.85)
    plt.show()


def plot_interpolated_features(cs, features, i, euclidean_cube, euclidean_sphere, sphere_triangles, sphere_coeffs):
    """Plot i-th interpolated feature on flatted cubed sphere, 3D cubed sphere, & 3D sphere"""

    interpolated = calc_all_interpolated_features(cs, features, euclidean_sphere, sphere_triangles, sphere_coeffs)
    interpolated = [point[i] for point in interpolated]

    plot_3d_shape("cube", euclidean_cube, shading=interpolated)
    plot_3d_shape("sphere", euclidean_sphere, shading=interpolated)
    plot_flat_cube(euclidean_cube, shading=interpolated)
    plot_flat_panel(euclidean_cube, shading=interpolated)
    plot_polar(euclidean_sphere, shading=interpolated)


def plot_original_features(cs, features, i):
    """Plot i-th original feature on flatted cubed sphere, 3D cubed sphere, & 3D sphere"""

    selected_feature_raw = []
    for p in cs.get_sphere_coords():
        if p[0] is not None:
            features_p = get_feature_for_point(p[0], p[1], cs.get_all_coords(), features)
            selected_feature_raw.append(features_p[i])
        else:
            selected_feature_raw.append(None)

    plot_3d_shape("sphere", cs.get_sphere_coords(), shading=selected_feature_raw)
    plot_3d_shape("cube", cs.get_cube_coords(), shading=selected_feature_raw)
    plot_flat_cube(cs.get_cube_coords(), shading=selected_feature_raw)
    plot_flat_panel(cs.get_cube_coords(), shading=selected_feature_raw)
    plot_polar(cs.get_sphere_coords(), shading=selected_feature_raw)


def plot_padded_panels(panel_tensors, edge_len, pad_width, label_cells, title):
    """Plot panels with padding, indicating on the plot which areas are padded vs. not

    Useful for verifying that padding has been performed correctly
    """

    # panel tensor must be of shape (5, n, n) where n = edge_len + padding
    fig, axs = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(9, 5))

    plot_locs = [(1, 1), (1, 2), (1, 3), (1, 0), (0, 1)]
    for panel in range(5):
        row, col = plot_locs[panel]
        plot_tensor = torch.flip(panel_tensors[panel].T, [0])
        axs[row, col].imshow(plot_tensor, vmin=torch.min(panel_tensors), vmax=torch.max(panel_tensors))

        # Create a Rectangle patch to outline panel and separate padded area
        rect = patches.Rectangle((0.5 + (pad_width - 1), 0.5 + (pad_width - 1)), edge_len, edge_len,
                                 linewidth=1, edgecolor='white', facecolor='none', hatch='/')
        # Add the patch to the Axes
        axs[row, col].add_patch(rect)

        if label_cells:
            for i in range(edge_len + 2 * pad_width):
                for j in range(edge_len + 2 * pad_width):
                    axs[row, col].text(j, i, round(1000 * plot_tensor[i][j].item(), 1), ha="center", va="center",
                                       color="w")

    axs[0, 0].axis('off')
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')

    # Show all ticks and label them with the respective list entries
    axs[1, 0].set_xticks([])
    axs[1, 1].set_xticks([])
    axs[1, 2].set_xticks([])
    axs[1, 3].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_panel(lr, sr, hr, batch_index, epoch, path, ncol, freq_index):
    """Based on the input data to the GAN and the output from the generator, plot a single panel for the first 4 HRTFs
    in the batch, at a given freq_index
    """
    lr_selected = lr.detach().cpu()[:ncol, freq_index, 0, :, :]
    sr_selected = sr.detach().cpu()[:ncol, freq_index, 0, :, :]
    hr_selected = hr.detach().cpu()[:ncol, freq_index, 0, :, :]
    min_magnitude = min((torch.min(lr_selected), torch.min(sr_selected), torch.min(hr_selected)))
    max_magnitude = max((torch.max(lr_selected), torch.max(sr_selected), torch.max(hr_selected)))

    fig, axs = plt.subplots(3, ncol, subplot_kw={'xticks': [], 'yticks': []})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.8, hspace=0.5, wspace=0.1)

    for n, lr_hrtf in enumerate(lr_selected):
        ax = plt.subplot(3, ncol, n + 1)
        ax.imshow(lr_hrtf, vmin=min_magnitude, vmax=max_magnitude)
        ax.set_title("LR " + str(n))

    for n, sr_hrtf in enumerate(sr_selected):
        ax = plt.subplot(3, ncol, n + 1 + ncol)
        ax.imshow(sr_hrtf, vmin=min_magnitude, vmax=max_magnitude)
        ax.set_title("SR " + str(n))

    for n, hr_hrtf in enumerate(hr_selected):
        ax = plt.subplot(3, ncol, n + 1 + (2 * ncol))
        temp = ax.imshow(hr_hrtf, vmin=min_magnitude, vmax=max_magnitude)
        ax.set_title("HR " + str(n))

    fig.colorbar(temp, ax=axs, shrink=0.7)
    fig.suptitle("Comparison of LR magnitudes, their generated SR counterparts, \nand HR ground truth")

    plt.savefig(f'{path}/{epoch}_{batch_index}_slices.png')
    plt.close(fig)


def plot_losses(train_losses_1, train_losses_2, label_1, label_2, color_1, color_2,
                path, filename, title="Loss Curves"):
    """Plot the discriminator and generator loss over time"""
    params = {
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [6, 4.5]
    }
    plt.rcParams.update(params)

    plt.figure()
    plt.grid(ls='dashed', axis='y', color='0.8')

    # loss_1 = [x for x in train_losses_1]
    # loss_2 = [x for x in train_losses_2]
    plt.plot(train_losses_1, label=label_1, linewidth=2, color=color_1)
    plt.plot(train_losses_2, label=label_2, linewidth=2, color=color_2)
    # plt.ylim(bottom=0)

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt_legend = plt.legend()
    frame = plt_legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    plt.savefig(f'{path}/{filename}.png')


def plot_magnitude_spectrums(frequencies, magnitudes_real, magnitudes_interpolated, ear, mode, label, path,
                             log_scale_magnitudes=True):
    fig, axs = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(9, 9))

    sdm = spectral_distortion_metric_for_plot(magnitudes_interpolated, magnitudes_real)
    sdm = round(sdm, 5)
    title = f"Magnitude spectrum, horizontal plane ({ear} ear) \n ({mode} data, spectral distortion metric = {sdm})"

    # keys refer to the locations of the subplots, values are the indices in the cubed sphere
    plot_locs = {(0, 0): (1, 0, 8), (0, 1): (0, 8, 8), (0, 2): (0, 0, 8),
                 (1, 0): (1, 8, 8), (1, 2): (3, 8, 8),
                 (2, 0): (2, 0, 8), (2, 1): (2, 8, 8), (2, 2): (3, 0, 8)}

    for subplot, indices in plot_locs.items():
        row, col = subplot
        spherical_coordinates = convert_cube_indices_to_spherical(indices[0], indices[1], indices[2], 16)
        azimuth = (spherical_coordinates[1] / np.pi) * 180
        elevation = (spherical_coordinates[0] / np.pi) * 180
        if log_scale_magnitudes:
            magnitudes_real_plot = 20 * np.log10(magnitudes_real[indices[0]][indices[1]][indices[2]])
            magnitudes_interpolated_plot = 20 * np.log10(magnitudes_interpolated[indices[0]][indices[1]][indices[2]])
        else:
            magnitudes_real_plot = magnitudes_real[indices[0]][indices[1]][indices[2]]
            magnitudes_interpolated_plot = magnitudes_interpolated[indices[0]][indices[1]][indices[2]]

        axs[row, col].plot(frequencies, magnitudes_real_plot, label="Real HRTF")
        axs[row, col].plot(frequencies, magnitudes_interpolated_plot, label="GAN interpolated HRTF")

        axs[row, col].set(title=f"(az={round(azimuth)}\u00B0, el={round(elevation)}\u00B0)",
                          xlabel='Frequency in Hz', ylabel='Amplitude in dB')
        axs[row, col].label_outer()
        axs[row, col].set_xscale('log')

    axs[0, 1].legend(loc=(0, -0.5))
    axs[1, 1].axis('off')
    fig.suptitle(title)
    plt.savefig(f'{path}/magnitude_spectrum_{label}.png')


def plot_grad_flow(named_parameters, path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.figure('grad_flow', figsize=(18, 10))
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if len(n) < 12:
                layers.append(n)
            else:
                layers.append(n[:6] + "..." + n[-6:])
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.05)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(f'{path}/grad_flow.png')

