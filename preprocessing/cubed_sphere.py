import itertools

import numpy as np
import pandas as pd

from preprocessing.convert_coordinates import convert_sphere_to_cube


class CubedSphere(object):
    # diagram of unfolded cube, with panel indices
    #             _______
    #            |       |
    #            |   4   |
    #     _______|_______|_______ _______
    #    |       |       |       |       |
    #    |   3   |   0   |   1   |   2   |
    #    |_______|_______|_______|_______|
    # In all cases, low values of x and y are situated in lower left of the unfolded sphere

    def __init__(self, mask=None, row_angles=None, column_angles=None, sphere_coords=None, indices=None):
        # initiate two lists of tuples, one will store (elevation, azimuth) for every measurement point
        # the other will store (elevation_index, azimuth_index) for every measurement point
        self.sphere_coords = []
        self.indices = []

        if indices is None:
            # at this stage, we can simplify by acting as if there are the same number of elevation measurement points at
            # every azimuth angle
            def elevation_validate(a, b): return None if b else a
            num_elevation_measurements = len(column_angles)
            elevation_indices = list(range(num_elevation_measurements))
            elevation = column_angles * np.pi / 180

            # loop through all azimuth positions
            for azimuth_index, azimuth in enumerate(row_angles):
                # convert degrees to radians by multiplying by a factor of pi/180
                azimuth = azimuth * np.pi / 180
                if type(mask) is np.bool_:
                    if not mask:
                        elevation_valid = elevation
                else:
                    elevation_valid = list(map(elevation_validate, list(elevation), [x.flatten().any() for x in mask[azimuth_index]]))

                # sphere_coords is stored as (elevation, azimuth). Ultimately, we're creating a list of (elevation,
                # azimuth) pairs for every measurement position in the sphere
                self.sphere_coords += list(zip(elevation_valid, [azimuth] * num_elevation_measurements))
                self.indices += list(zip(elevation_indices, [azimuth_index] * num_elevation_measurements))

            # self.cube_coords is created from measurement_positions, such that order is the same
            self.cube_coords = list(itertools.starmap(convert_sphere_to_cube, self.sphere_coords))

            # create pandas dataframe containing all coordinate data (spherical and cubed sphere)
            # this can be useful for debugging
            self.all_coords = pd.concat([pd.DataFrame(self.indices, columns=["elevation_index", "azimuth_index"]),
                                         pd.DataFrame(self.sphere_coords, columns=["elevation", "azimuth"]),
                                         pd.DataFrame(self.cube_coords, columns=["panel", "x", "y"])], axis="columns")

        else:
            self.sphere_coords = sphere_coords
            self.indices = indices

            # self.cube_coords is created from measurement_positions, such that order is the same
            self.cube_coords = list(itertools.starmap(convert_sphere_to_cube, self.sphere_coords))

            # create pandas dataframe containing all coordinate data (spherical and cubed sphere)
            # this can be useful for debugging
            self.all_coords = pd.concat(
                [pd.DataFrame(self.indices, columns=["panel_index", "elevation_index", "azimuth_index"]),
                 pd.DataFrame(self.sphere_coords, columns=["elevation", "azimuth"]),
                 pd.DataFrame(self.cube_coords, columns=["panel", "x", "y"])], axis="columns")

    def get_sphere_coords(self):
        return self.sphere_coords

    def get_cube_coords(self):
        return self.cube_coords

    def get_all_coords(self):
        return self.all_coords
