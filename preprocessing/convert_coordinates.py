import numpy as np

PI_4 = np.pi / 4


def calc_offset(quadrant):
    """Based on sphere quadrant, calculate the offset that is used with the azimuth"""
    return ((quadrant - 1) / 2) * np.pi


def calc_panel(elevation, azimuth):
    """Based on the elevation and azimuth of a point on a sphere, determine which panel it would fall in on a cubed
    sphere

    Note that in this function, the panels are numbered 1 through 5 to be consistent with existing literature,
    however later in the data processing the panels are zero-indexed for convenience """
    # use the azimuth to determine the quadrant of the sphere
    if azimuth < np.pi / 4:
        quadrant = 1
    elif azimuth < 3 * np.pi / 4:
        quadrant = 2
    elif azimuth < 5 * np.pi / 4:
        quadrant = 3
    else:
        quadrant = 4

    offset = calc_offset(quadrant)
    threshold_val = np.tan(elevation) / np.cos(azimuth - offset)

    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -1 <= threshold_val < 1:
        panel = quadrant
    # above a certain elevation, in panel 5
    elif threshold_val >= 1:
        panel = 5
    # below a certain elevation, in panel 6
    else:
        panel = 6

    return panel


def convert_sphere_to_cube(elevation, azimuth):
    """used for obtaining corresponding cubed sphere coordinates from a pair of spherical coordinates"""
    if elevation is None or azimuth is None:
        # if this position was not measured in the sphere, keep as np.nan in cube
        panel, x, y = np.nan, np.nan, np.nan
    else:
        # shift the range of azimuth angles such that it works with conversion equations
        if azimuth < -np.pi / 4:
            azimuth += 2 * np.pi
        panel = calc_panel(elevation, azimuth)

        if panel <= 4:
            offset = calc_offset(panel)
            x = azimuth - offset
            y = np.arctan(np.tan(elevation) / np.cos(azimuth - offset))
        elif panel == 5:
            x = np.arctan(np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
        else:
            x = np.arctan(-np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
    return panel, x, y


def convert_cube_to_sphere(panel, x, y):
    """used for obtaining spherical coordinates from corresponding cubed sphere coordinates"""
    if panel <= 4:
        offset = calc_offset(panel)
        azimuth = x + offset
        elevation = np.arctan(np.tan(y) * np.cos(x))
    elif panel == 5:
        # if tan(x) is 0, handle as a special case
        if np.tan(x) == 0:
            azimuth = np.arctan(0)
            elevation = np.pi/2
        else:
            azimuth = np.arctan(-np.tan(x) / np.tan(y))
            elevation = np.arctan(np.sin(azimuth) / np.tan(x))
            if elevation < 0:
                elevation *= -1
                azimuth += np.pi
    # not including panel 6 for now, as it is being excluded from this data
    elif panel == 6:
        pass

    # ensure azimuth is in range -pi to +pi
    while azimuth > np.pi:
        azimuth -= 2 * np.pi
    while azimuth <= -np.pi:
        azimuth += 2 * np.pi

    return elevation, azimuth


def convert_cube_indices_to_spherical(panel, i, j, edge_len):
    # offset panel to be compatible with earlier functions that used 1-indexing for panels
    panel += 1
    # use edge length to determine spacing between points on euclidean cubed sphere
    spacing = (np.pi / 2) / edge_len
    # find the lowest value for each panel's x and y (same in both dimensions)
    start = (-np.pi/4) + (spacing/2)
    # get x and y values from start, spacing, and index
    x_i = start + i*spacing
    y_j = start + j*spacing
    return convert_cube_to_sphere(panel, x_i, y_j)


def convert_sphere_to_cartesian(coordinates):
    """For a list of spherical coordinates of the form (elevation, azimuth), convert to (x, y, z) cartesian
    coordinates for plotting purposes """
    x, y, z = [], [], []
    mask = []

    for elevation, azimuth in coordinates:
        if elevation is not None and azimuth is not None:
            mask.append(True)
            # convert to cartesian coordinates
            x_i = np.cos(elevation) * np.cos(azimuth)
            y_i = np.cos(elevation) * np.sin(azimuth)
            z_i = np.sin(elevation)

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask


def convert_cube_to_cartesian(coordinates):
    """For a list of cube sphere coordinates of the form (panel, x, y), convert to (x, y, z) cartesian
    coordinates for plotting purposes """
    x, y, z = [], [], []
    mask = []

    for panel, p, q in coordinates:
        if not np.isnan(p) and not np.isnan(q):
            mask.append(True)
            if panel == 1:
                x_i, y_i, z_i = PI_4, p, q
            elif panel == 2:
                x_i, y_i, z_i = -p, PI_4, q
            elif panel == 3:
                x_i, y_i, z_i = -PI_4, -p, q
            elif panel == 4:
                x_i, y_i, z_i = p, -PI_4, q
            elif panel == 5:
                x_i, y_i, z_i = -q, p, PI_4
            else:
                x_i, y_i, z_i = q, p, -PI_4

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask
