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

    def __init__(self, sphere_coords):
        # initiate two lists of tuples, one will store (elevation, azimuth) for every measurement point
        # the other will store (elevation_index, azimuth_index) for every measurement point
        self.sphere_coords = []
        self.indices = []

        # at this stage, we can simplify by acting as if there are the same number of elevation measurement points at
        # every azimuth angle
        num_elevation_measurements = sphere_coords[0].shape[0]
        elevation_indices = list(range(num_elevation_measurements))

        # loop through all azimuth positions
        for azimuth_index, azimuth in enumerate(sphere_coords.keys()):
            # convert degrees to radians by multiplying by a factor of pi/180
            elevation = sphere_coords[azimuth] * np.pi / 180
            azimuth = azimuth * np.pi / 180

            # sphere_coords is stored as (elevation, azimuth). Ultimately, we're creating a list of (elevation,
            # azimuth) pairs for every measurement position in the sphere
            self.sphere_coords += list(zip(elevation.tolist(), [azimuth] * num_elevation_measurements))
            self.indices += list(zip(elevation_indices, [azimuth_index] * num_elevation_measurements))

        # self.cube_coords is created from measurement_positions, such that order is the same
        self.cube_coords = list(itertools.starmap(convert_sphere_to_cube, self.sphere_coords))

        # create pandas dataframe containing all coordinate data (spherical and cubed sphere)
        # this can be useful for debugging
        self.all_coords = pd.concat([pd.DataFrame(self.indices, columns=["elevation_index", "azimuth_index"]),
                                     pd.DataFrame(self.sphere_coords, columns=["elevation", "azimuth"]),
                                     pd.DataFrame(self.cube_coords, columns=["panel", "x", "y"])], axis="columns")

    def get_sphere_coords(self):
        return self.sphere_coords

    def get_cube_coords(self):
        return self.cube_coords

    def get_all_coords(self):
        return self.all_coords
