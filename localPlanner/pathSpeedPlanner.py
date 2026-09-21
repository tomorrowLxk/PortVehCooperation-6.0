"""为空间path附加每点期望速度上限。"""

import numpy as np

from localPlanner.pathGeometry import clean_xy, path_curvatures
from parameters import (
    PATH_CURVATURE_MIN_SPEED,
    PATH_CURVATURE_SPEED_COEFFICIENT,
    PATH_CURVATURE_WINDOW,
)


def build_speed_limit_path(path, maximum_speed, terminal_stop=False):
    points = clean_xy(path)
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    curvature = path_curvatures(points, PATH_CURVATURE_WINDOW)
    speeds = np.full(len(points), max(0.0, float(maximum_speed)), dtype=float)
    curved = curvature > 1e-6
    if np.any(curved):
        curve_limit = np.maximum(
            PATH_CURVATURE_MIN_SPEED,
            1.0 / (
                    PATH_CURVATURE_SPEED_COEFFICIENT
                    * np.maximum(curvature[curved], 1e-6)
            ),
        )
        speeds[curved] = np.minimum(speeds[curved], curve_limit)
    if terminal_stop:
        speeds[-1] = 0.0
    return np.column_stack((points, speeds))


def stationary_path(position):
    point = np.asarray(position, dtype=float)[:2]
    return np.asarray([
        [point[0], point[1], 0.0],
        [point[0], point[1], 0.0],
    ])
