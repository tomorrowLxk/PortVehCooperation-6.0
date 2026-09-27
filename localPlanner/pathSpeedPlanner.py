"""为空间path附加每点期望速度上限。"""

import numpy as np

from localPlanner.pathGeometry import clean_xy, path_curvatures
from parameters import (
    PATH_CURVATURE_WINDOW,
    VEHICLE_MAX_LATERAL_ACCELERATION,
)


def build_speed_limit_path(path, maximum_speed, terminal_stop=False):
    points = clean_xy(path)
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    curvature = path_curvatures(points, PATH_CURVATURE_WINDOW)
    speeds = np.full(len(points), max(0.0, float(maximum_speed)), dtype=float)
    curved = curvature > 1e-6
    if np.any(curved):
        curve_limit = np.sqrt(
            VEHICLE_MAX_LATERAL_ACCELERATION
            / np.maximum(curvature[curved], 1e-6)
        )
        speeds[curved] = np.minimum(speeds[curved], curve_limit)
    if terminal_stop:
        speeds[-1] = 0.0
    return np.column_stack((points, speeds))


def build_speed_envelope(
        path, current_speed, max_acceleration, max_deceleration):
    """在既有速度上限内生成满足纵向加减速度约束的速度包络。"""
    output = np.asarray(path, dtype=float).copy()
    if len(output) == 0:
        return output

    segment_lengths = np.linalg.norm(
        np.diff(output[:, :2], axis=0), axis=1)
    output[:, 2] = speed_envelope_values(
        output[:, 2],
        segment_lengths,
        current_speed,
        max_acceleration,
        max_deceleration,
    )
    return output


def speed_envelope_values(
        speed_limits, segment_lengths, current_speed,
        max_acceleration, max_deceleration):
    """用等价的向量累计最小运算完成前向、后向、前向三次约束。"""
    speed_squared = np.square(np.maximum(
        0.0, np.asarray(speed_limits, dtype=float)))
    speed_squared[0] = min(
        speed_squared[0], max(0.0, float(current_speed)) ** 2)
    distances = np.asarray(segment_lengths, dtype=float)

    def forward(values, intervals, acceleration):
        increments = np.concatenate((
            [0.0],
            np.cumsum(2.0 * max(1e-6, float(acceleration)) * intervals),
        ))
        return increments + np.minimum.accumulate(values - increments)

    speed_squared = forward(
        speed_squared, distances, max_acceleration)
    speed_squared = forward(
        speed_squared[::-1], distances[::-1], max_deceleration)[::-1]
    speed_squared = forward(
        speed_squared, distances, max_acceleration)
    return np.sqrt(np.maximum(0.0, speed_squared))


def stationary_path(position):
    point = np.asarray(position, dtype=float)[:2]
    return np.asarray([
        [point[0], point[1], 0.0],
        [point[0], point[1], 0.0],
    ])
