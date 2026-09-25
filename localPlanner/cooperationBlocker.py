"""仅根据futurePath空间接近关系生成协同连通块。"""

import math

import numpy as np

from localPlanner.pathGeometry import clean_xy, resample_polyline
from parameters import (
    COOPERATION_LATERAL_MARGIN,
    COOPERATION_PATH_POINT_SPACING,
    COOPERATION_PATH_SIMPLIFICATION_MARGIN,
)


def group_by_future_paths(
        future_paths,
        vehicle_static_info,
        vehicle_dynamic_info):
    """建立车辆无向接近图并返回连通分量，保留单车块。"""
    vehicle_ids = list(future_paths)
    count = len(vehicle_ids)
    if count == 0:
        return []

    paths = []
    lower_bounds = []
    upper_bounds = []
    lateral_radii = np.empty(count, dtype=float)

    for index, vehicle_id in enumerate(vehicle_ids):
        static_info = vehicle_static_info[vehicle_id]
        dynamic_info = vehicle_dynamic_info[vehicle_id]
        path = _occupied_centerline(
            future_paths[vehicle_id],
            float(static_info["length"]),
            float(dynamic_info["longitudinalUncertainty"]),
            float(dynamic_info["heading"]),
        )
        paths.append(path)
        if len(path):
            lower_bounds.append(np.min(path, axis=0))
            upper_bounds.append(np.max(path, axis=0))
        else:
            lower_bounds.append(np.asarray([np.inf, np.inf]))
            upper_bounds.append(np.asarray([-np.inf, -np.inf]))
        lateral_radii[index] = (
            0.5 * float(static_info["width"])
            + float(dynamic_info["lateralUncertainty"])
            + COOPERATION_LATERAL_MARGIN
        )

    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    parent = np.arange(count)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        parent[second_root] = first_root
        return True

    pair_thresholds = (
        lateral_radii[:, None]
        + lateral_radii[None, :]
        + COOPERATION_PATH_SIMPLIFICATION_MARGIN
    )
    axis_gap = np.maximum(
        0.0,
        np.maximum(
            lower_bounds[:, None, :] - upper_bounds[None, :, :],
            lower_bounds[None, :, :] - upper_bounds[:, None, :],
        ),
    )
    valid = np.asarray([len(path) > 0 for path in paths])
    coarse_candidates = (
        np.sum(axis_gap * axis_gap, axis=2) <= pair_thresholds ** 2
    ) & valid[:, None] & valid[None, :]

    component_count = count
    for first, second in np.argwhere(np.triu(coarse_candidates, k=1)):
        first = int(first)
        second = int(second)
        if find(first) == find(second):
            continue
        threshold_squared = pair_thresholds[first, second] ** 2
        if (
                _has_close_vertices(
                    paths[first], paths[second], threshold_squared)
                or _polyline_min_distance_squared(
                    paths[first], paths[second]) <= threshold_squared):
            if union(first, second):
                component_count -= 1
                if component_count == 1:
                    break

    blocks = {}
    for index, vehicle_id in enumerate(vehicle_ids):
        blocks.setdefault(find(index), []).append(vehicle_id)
    return list(blocks.values())


def _occupied_centerline(path, vehicle_length, longitudinal_uncertainty,
                         heading):
    """按车辆纵向占用延长首尾，再生成分块专用的低密度折线。"""
    points = clean_xy(path)
    if len(points) == 0:
        return points

    extension = (
        0.5 * max(0.0, vehicle_length)
        + max(0.0, longitudinal_uncertainty)
    )
    if len(points) == 1:
        tangent_start = np.asarray([
            math.cos(heading), math.sin(heading)
        ])
        tangent_end = tangent_start
    else:
        tangent_start = _unit(points[1] - points[0])
        tangent_end = _unit(points[-1] - points[-2])

    extended = np.vstack((
        points[0] - extension * tangent_start,
        points,
        points[-1] + extension * tangent_end,
    ))
    return resample_polyline(
        extended,
        COOPERATION_PATH_POINT_SPACING,
    )


def _has_close_vertices(first, second, threshold_squared):
    delta = first[:, None, :] - second[None, :, :]
    return bool(np.any(np.sum(delta * delta, axis=2) <= threshold_squared))

def _polyline_min_distance_squared(first, second):
    """连续二维折线之间的精确最小距离平方。"""
    first = clean_xy(first)
    second = clean_xy(second)
    if len(first) == 1 and len(second) == 1:
        delta = first[0] - second[0]
        return float(delta @ delta)
    if len(first) == 1:
        return float(np.min(_points_to_segments_squared(
            first, second[:-1], second[1:])))
    if len(second) == 1:
        return float(np.min(_points_to_segments_squared(
            second, first[:-1], first[1:])))

    first_start = first[:-1]
    first_end = first[1:]
    second_start = second[:-1]
    second_end = second[1:]
    if np.any(_segment_intersection_matrix(
            first_start, first_end, second_start, second_end)):
        return 0.0

    return float(min(
        np.min(_points_to_segments_squared(
            first_start, second_start, second_end)),
        np.min(_points_to_segments_squared(
            first_end, second_start, second_end)),
        np.min(_points_to_segments_squared(
            second_start, first_start, first_end)),
        np.min(_points_to_segments_squared(
            second_end, first_start, first_end)),
    ))


def _points_to_segments_squared(points, segment_start, segment_end):
    points = np.asarray(points, dtype=float)
    segment_start = np.asarray(segment_start, dtype=float)
    segment_end = np.asarray(segment_end, dtype=float)
    segment = segment_end - segment_start
    length_squared = np.sum(segment * segment, axis=1)
    relative = points[:, None, :] - segment_start[None, :, :]
    ratio = np.divide(
        np.sum(relative * segment[None, :, :], axis=2),
        length_squared[None, :],
        out=np.zeros((len(points), len(segment_start)), dtype=float),
        where=length_squared[None, :] > 1e-12,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    residual = (
        relative - ratio[:, :, None] * segment[None, :, :]
    )
    return np.sum(residual * residual, axis=2)


def _segment_intersection_matrix(
        first_start, first_end, second_start, second_end):
    first_vector = first_end - first_start
    second_vector = second_end - second_start

    first_start_3d = first_start[:, None, :]
    first_end_3d = first_end[:, None, :]
    second_start_3d = second_start[None, :, :]
    second_end_3d = second_end[None, :, :]

    first_vector_3d = first_vector[:, None, :]
    second_vector_3d = second_vector[None, :, :]
    first_to_second_start = second_start_3d - first_start_3d
    first_to_second_end = second_end_3d - first_start_3d
    second_to_first_start = first_start_3d - second_start_3d
    second_to_first_end = first_end_3d - second_start_3d

    orientation_1 = _cross(first_vector_3d, first_to_second_start)
    orientation_2 = _cross(first_vector_3d, first_to_second_end)
    orientation_3 = _cross(second_vector_3d, second_to_first_start)
    orientation_4 = _cross(second_vector_3d, second_to_first_end)

    lower_overlap = np.maximum(
        np.minimum(first_start_3d, first_end_3d),
        np.minimum(second_start_3d, second_end_3d),
    )
    upper_overlap = np.minimum(
        np.maximum(first_start_3d, first_end_3d),
        np.maximum(second_start_3d, second_end_3d),
    )
    boxes_overlap = np.all(
        lower_overlap <= upper_overlap + 1e-12,
        axis=2,
    )
    return (
        orientation_1 * orientation_2 <= 1e-12
    ) & (
        orientation_3 * orientation_4 <= 1e-12
    ) & boxes_overlap


def _cross(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _unit(vector):
    length = float(np.linalg.norm(vector))
    return vector / length

