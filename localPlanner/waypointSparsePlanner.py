"""第十二阶段：从waypointWhole中选取固定四点waypointSparse。"""

import numpy as np

from localPlanner.pathGeometry import heading_at_s, point_at_s
from parameters import (
    WAYPOINT_SPARSE_COUNT,
    WAYPOINT_SPARSE_FIRST_SECOND_MIN_SPACING,
    WAYPOINT_SPARSE_MIN_SPACING,
)


def build_waypoint_sparse(
        waypoint_whole, dynamic_info,
        junction_entrance=None, centerline_dict=None):
    """返回四个[x, y, heading, speed]，运动点均来自waypointWhole。"""
    if len(waypoint_whole.points) == 1:
        return _stationary_sparse(
            waypoint_whole,
            dynamic_info,
            junction_entrance,
            centerline_dict,
        )

    indices = _select_sparse_indices(waypoint_whole.futurePathS)
    rows = []
    for index in indices:
        path_s = float(waypoint_whole.futurePathS[index])
        rows.append([
            waypoint_whole.points[index, 0],
            waypoint_whole.points[index, 1],
            heading_at_s(waypoint_whole.futurePath, path_s),
            waypoint_whole.pointSpeeds[index],
        ])
    return np.asarray(rows, dtype=float)


def build_hv_waypoint_sparse(waypoint_whole, dynamic_info):
    "从本帧HV匀速预测中等距抽取四点，不使用AGV交叉口与复用规则。"
    total = float(waypoint_whole.motionS[-1])
    speed = max(0.0, float(dynamic_info['speed']))
    if total <= 1e-9 or speed <= 1e-9:
        point = np.asarray(dynamic_info['position'], dtype=float)[:2]
        row = np.asarray([
            point[0], point[1], float(dynamic_info['heading']), 0.0
        ])
        return np.repeat(row.reshape(1, 4), WAYPOINT_SPARSE_COUNT, axis=0)

    rows = []
    for path_s in np.linspace(0.0, total, WAYPOINT_SPARSE_COUNT):
        point = point_at_s(waypoint_whole.futurePath, path_s)
        rows.append([
            point[0],
            point[1],
            heading_at_s(waypoint_whole.futurePath, path_s),
            waypoint_whole.speed_at_s(path_s),
        ])
    return np.asarray(rows, dtype=float)


def _stationary_sparse(
        waypoint_whole, dynamic_info,
        junction_entrance, centerline_dict):
    if junction_entrance is None:
        point = waypoint_whole.points[0]
        heading = float(dynamic_info['heading'])
    else:
        point = np.asarray(
            junction_entrance['entrancePoint'], dtype=float)[:2]
        lane_id = junction_entrance['junctionLaneId']
        heading = heading_at_s(centerline_dict[lane_id], 0.0)

    row = np.asarray([point[0], point[1], heading, 0.0], dtype=float)
    return np.repeat(row.reshape(1, 4), WAYPOINT_SPARSE_COUNT, axis=0)


def _select_sparse_indices(future_path_s):
    values = np.asarray(future_path_s, dtype=float)
    first = 1
    last = len(values) - 1
    total = float(values[last] - values[first])

    # 首末两个有效点都无法满足一般间距时，四点全部合并到末点。
    if total < WAYPOINT_SPARSE_MIN_SPACING - 1e-9:
        return np.full(WAYPOINT_SPARSE_COUNT, last, dtype=int)

    second_target = values[first] + total / 3.0
    second = _first_at_or_after(values, second_target, first + 1, last)
    third_target = values[second] + (values[last] - values[second]) / 2.0
    third = _first_at_or_after(values, third_target, second + 1, last)

    selected = [first]
    for index in (second, third):
        if index >= last or index == selected[-1]:
            continue
        if values[index] - values[selected[-1]] < WAYPOINT_SPARSE_MIN_SPACING:
            continue
        if values[last] - values[index] < WAYPOINT_SPARSE_MIN_SPACING:
            continue
        selected.append(index)
    selected.append(last)

    if (len(selected) >= 2
            and values[selected[1]] - values[selected[0]]
            <= WAYPOINT_SPARSE_FIRST_SECOND_MIN_SPACING):
        return np.full(WAYPOINT_SPARSE_COUNT, last, dtype=int)

    selected.extend([last] * (WAYPOINT_SPARSE_COUNT - len(selected)))
    return np.asarray(selected[:WAYPOINT_SPARSE_COUNT], dtype=int)


def _first_at_or_after(values, target, start, last):
    if start >= last:
        return last
    relative = int(np.searchsorted(
        values[start:last + 1], target, side='left'))
    return min(start + relative, last)
