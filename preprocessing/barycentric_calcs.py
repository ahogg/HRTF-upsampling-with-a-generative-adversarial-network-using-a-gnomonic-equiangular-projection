import numpy as np

from preprocessing.convert_coordinates import convert_sphere_to_cartesian


def calc_dist_haversine(elevation1, azimuth1, elevation2, azimuth2):
    """Calculates the haversine distance between two points on a sphere

    adapted from CalculateDistance_HaversineFormula in the 3DTune-In toolkit
    https://github.com/3DTune-In/3dti_AudioToolkit/blob/master/3dti_Toolkit/BinauralSpatializer/HRTF.cpp#L1052
    """
    increment_azimuth = azimuth1 - azimuth2
    increment_elevation = elevation1 - elevation2
    sin2_inc_elev = np.sin(increment_elevation / 2) ** 2
    cos_elev1 = np.cos(elevation1)
    cos_elev2 = np.cos(elevation2)
    sin2_inc_azi = np.sin(increment_azimuth / 2) ** 2
    raiz = sin2_inc_elev + (cos_elev1 * cos_elev2 * sin2_inc_azi)
    sqrt_distance = raiz ** 0.5
    distance = 2 * np.arcsin(sqrt_distance)
    return distance


def calc_spherical_excess(elevation1, azimuth1, elevation2, azimuth2, elevation3, azimuth3):
    """Calculates the spherical excess of a spherical triangle based on the position of the triangle's vertices"""
    dist12 = calc_dist_haversine(elevation1, azimuth1, elevation2, azimuth2)
    dist13 = calc_dist_haversine(elevation1, azimuth1, elevation3, azimuth3)
    dist23 = calc_dist_haversine(elevation2, azimuth2, elevation3, azimuth3)
    semiperimeter = 0.5 * (dist12 + dist13 + dist23)
    inner = (np.tan(0.5 * semiperimeter) *
             np.tan(0.5 * (semiperimeter - dist12)) *
             np.tan(0.5 * (semiperimeter - dist13)) *
             np.tan(0.5 * (semiperimeter - dist23)))
    excess = 4 * np.arctan(np.sqrt(inner))
    return excess


def calc_all_distances(elevation, azimuth, sphere_coords):
    """Calculates the distances from a given point to every measurement point on the sphere

    Returns a list of tuples of the form (elevation, azimuth, distance) which provides the location and distance of
    every measurement point in the sphere, sorted from the closest point to the farthest point
    """
    distances = []
    for elev, azi in sphere_coords:
        if elev is not None and azi is not None:
            dist = calc_dist_haversine(elevation1=elevation, azimuth1=azimuth,
                                       elevation2=elev, azimuth2=azi)
            distances.append((elev, azi, dist))

    # sorted list of (elevation, azimuth, distance) for all points
    return sorted(distances, key=lambda x: x[2])


def get_possible_triangles(max_vertex_index, point_distances):
    """Using the closest max_vertex_index points as possible vertices, find all possible triangles

    Return possible triangles as a list of (vertex_0, vertex_1, vertex_2, total_distance), sorted from smallest
    total_distance to the largest possible distance

    :param max_vertex_index: how many possible vertices to consider
    :param point_distances: a list of measured points in the form (elevation, azimuth, distance), where distance refers
                            to their distance from the interpolation point of interest. Sorted based on this distance.
    """
    possible_triangles = []
    for v0 in range(max_vertex_index - 1):
        for v1 in range(v0 + 1, max_vertex_index):
            for v2 in range(v1 + 1, max_vertex_index + 1):
                total_dist = point_distances[v0][2] + point_distances[v1][2] + point_distances[v2][2]
                possible_triangles.append((v0, v1, v2, total_dist))

    return sorted(possible_triangles, key=lambda x: x[3])


def triangle_encloses_point(elevation, azimuth, triangle_coordinates):
    """Determine whether the spherical triangle defined by triangle_coordinates encloses the point located at
    (elevation, azimuth)"""
    # convert point of interest to cartesian coordinates and add to array
    x, y, z, _ = convert_sphere_to_cartesian([[elevation, azimuth]])
    point = np.array([x, y, z])
    # convert triangle coordinates to cartesian and add to array
    x_triangle, y_triangle, z_triangle, _ = convert_sphere_to_cartesian(triangle_coordinates)
    triangle_points = np.array([x_triangle, y_triangle, z_triangle])

    # check if matrix is singular
    if np.linalg.matrix_rank(triangle_points) < 3:
        return False

    # solve system of equations
    solution = np.linalg.solve(triangle_points, point)

    # this checks that a point lies in a spherical triangle by checking that the vector formed from the center of the
    # sphere to the point of interest intersects the plane formed by the triangle's vertices
    # check that constraints are satisfied
    solution_sum = np.sum(solution)
    solution_lambda = 1. / solution_sum

    return solution_lambda > 0 and np.all(solution > 0)


def get_triangle_vertices(elevation, azimuth, sphere_coords):
    """For a given point (elevation, azimuth), find the best possible triangle for barycentric interpolation.

    The best triangle is defined as the triangle with the minimum total distance from vertices to the point of interest
    that also encloses the point of interest"""
    # get distances from point of interest to every other point
    point_distances = calc_all_distances(elevation=elevation, azimuth=azimuth, sphere_coords=sphere_coords)

    # first try triangle formed by closest points
    triangle_vertices = [point_distances[0][:2], point_distances[1][:2], point_distances[2][:2]]
    if triangle_encloses_point(elevation, azimuth, triangle_vertices):
        selected_triangle_vertices = triangle_vertices
    else:
        # failing that, examine all possible triangles
        # possible triangles is sorted from shortest total distance to longest total distance
        # possible_triangles = get_possible_triangles(len(point_distances) - 1, point_distances)
        possible_triangles = get_possible_triangles(500, point_distances)
        for v0, v1, v2, _ in possible_triangles:
            triangle_vertices = [point_distances[v0][:2], point_distances[v1][:2], point_distances[v2][:2]]

            # for each triangle, check if it encloses the point
            if triangle_encloses_point(elevation, azimuth, triangle_vertices):
                selected_triangle_vertices = triangle_vertices
                break
        else:
            raise RuntimeError(f"No enclosing triangle found for elevation {elevation}, azimuth {azimuth}.")

    # if no triangles enclose the point, this will return none
    return selected_triangle_vertices


def calc_barycentric_coordinates(elevation, azimuth, closest_points):
    """Calculate alpha, beta, and gamma coefficients for barycentric interpolation (modified for spherical triangle)"""
    # not zero indexing var names in order to match equations in 3D Tune-In Toolkit paper
    elev1, elev2, elev3 = closest_points[0][0], closest_points[1][0], closest_points[2][0]
    azi1, azi2, azi3 = closest_points[0][1], closest_points[1][1], closest_points[2][1]

    # modified calculations to suit spherical triangle
    denominator = calc_spherical_excess(elev1, azi1, elev2, azi2, elev3, azi3)

    alpha = calc_spherical_excess(elevation, azimuth, elev2, azi2, elev3, azi3) / denominator
    beta = calc_spherical_excess(elev1, azi1, elevation, azimuth, elev3, azi3) / denominator
    gamma = 1 - alpha - beta

    return {"alpha": alpha, "beta": beta, "gamma": gamma}
