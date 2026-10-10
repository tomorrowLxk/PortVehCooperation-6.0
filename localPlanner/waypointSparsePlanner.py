"""第十二阶段：从waypointWhole中选取固定四点waypointSparse。"""

import numpy as np

from localPlanner.pathGeometry import (
    clean_xy,
    heading_at_s,
    point_at_s,
    polyline_length,
    project_to_polyline,
)
from parameters import (
    JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE,
    JUNCTION_EXIT_PATH_EXTENSION,
    VEHICLE_LANE_LOCATE_MAX_DISTANCE,
    WAYPOINT_SPARSE_COUNT,
    WAYPOINT_SPARSE_FIRST_SECOND_MIN_SPACING,
    WAYPOINT_SPARSE_MIN_SPACING,
)


def build_waypoint_sparse(
        waypoint_whole, dynamic_info,
        junction_entrance=None, centerline_dict=None,
        junction_lane_ids=None):
    """返回四个[x, y, heading, speed, arrivalTime]。"""
    if len(waypoint_whole.points) == 1:
        return _stationary_sparse(
            waypoint_whole,
            dynamic_info,
            junction_entrance,
            centerline_dict,
        )

    junction_exit_s = _terminal_junction_exit_s(
        waypoint_whole.futurePath,
        centerline_dict,
        junction_lane_ids,
    )
    indices = _select_sparse_indices(waypoint_whole.futurePathS)
    indices = _keep_terminal_junction_exit(
        indices, waypoint_whole.futurePathS, junction_exit_s)
    rows = []
    for index in indices:
        path_s = float(waypoint_whole.futurePathS[index])
        rows.append([
            waypoint_whole.points[index, 0],
            waypoint_whole.points[index, 1],
            heading_at_s(waypoint_whole.futurePath, path_s),
            waypoint_whole.pointSpeeds[index],
            waypoint_whole.arrivalTimes[index],
        ])
    return np.asarray(rows, dtype=float)


def build_hv_waypoint_sparse(waypoint_whole, dynamic_info):
    "从本帧HV匀速预测中等距抽取四点，不使用AGV交叉口与复用规则。"
    total = float(waypoint_whole.motionS[-1])
    speed = max(0.0, float(dynamic_info['speed']))
    if total <= 1e-9 or speed <= 1e-9:
        point = np.asarray(dynamic_info['position'], dtype=float)[:2]
        row = np.asarray([
            point[0], point[1], float(dynamic_info['heading']), 0.0, 0.0
        ])
        return np.repeat(row.reshape(1, 5), WAYPOINT_SPARSE_COUNT, axis=0)

    rows = []
    for path_s in np.linspace(0.0, total, WAYPOINT_SPARSE_COUNT):
        point = point_at_s(waypoint_whole.futurePath, path_s)
        rows.append([
            point[0],
            point[1],
            heading_at_s(waypoint_whole.futurePath, path_s),
            waypoint_whole.speed_at_s(path_s),
            waypoint_whole.time_at_s(path_s),
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

    row = np.asarray([point[0], point[1], heading, 0.0, 0.0], dtype=float)
    return np.repeat(row.reshape(1, 5), WAYPOINT_SPARSE_COUNT, axis=0)


def _select_sparse_indices(future_path_s):
    values = np.asarray(future_path_s, dtype=float)
    first = 1
    last = len(values) - 1
    total = float(values[last] - values[first])

    # 一般间距不足时，仅在实际候选点中确实存在第二个合法点时保留两点。
    if total < WAYPOINT_SPARSE_MIN_SPACING - 1e-9:
        hasSecondPoint = any(
            values[index] - values[first]
            > WAYPOINT_SPARSE_FIRST_SECOND_MIN_SPACING
            for index in range(first + 1, last + 1)
        )
        if hasSecondPoint:
            return np.asarray([first, last, last, last], dtype=int)
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


def _terminal_junction_exit_s(
        future_path, centerline_dict, junction_lane_ids):
    """Return the terminal junction exit retained before the path tail."""
    path = clean_xy(future_path)
    if (len(path) < 2
            or not centerline_dict
            or not junction_lane_ids):
        return None

    total = polyline_length(path)
    candidates = []
    path_min = np.min(path, axis=0)
    path_max = np.max(path, axis=0)
    for lane_id in junction_lane_ids:
        if lane_id not in centerline_dict:
            continue
        lane = clean_xy(centerline_dict[lane_id])
        if len(lane) < 2:
            continue
        if (np.any(np.max(lane, axis=0) < path_min)
                or np.any(np.min(lane, axis=0) > path_max)):
            continue

        projection = project_to_polyline(lane[-1], path)
        remaining = total - float(projection['s'])
        if (projection['distance'] <= 1e-3
                and remaining
                > JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE + 1e-6
                and remaining
                <= JUNCTION_EXIT_PATH_EXTENSION + 1e-3):
            candidates.append(float(projection['s']))

    return max(candidates) if candidates else None


def _keep_terminal_junction_exit(selected, future_path_s, junction_exit_s):
    """Keep first, junction-exit and post-turn tail within four points."""
    if junction_exit_s is None:
        return np.asarray(selected, dtype=int)

    values = np.asarray(future_path_s, dtype=float)
    last = len(values) - 1
    mandatory = int(np.argmin(np.abs(values - junction_exit_s)))
    if (abs(float(values[mandatory]) - float(junction_exit_s)) > 1e-3
            or mandatory <= 0
            or mandatory >= last):
        return np.asarray(selected, dtype=int)

    first = min(1, last)
    anchors = sorted(set((first, mandatory, last)))
    if len(anchors) < WAYPOINT_SPARSE_COUNT:
        optional = sorted(set(
            int(index) for index in np.asarray(selected, dtype=int)
            if first < int(index) < mandatory
            and int(index) not in anchors
        ))
        if optional:
            midpoint = 0.5 * (values[first] + values[mandatory])
            anchors.append(min(
                optional,
                key=lambda index: abs(values[index] - midpoint),
            ))
            anchors.sort()

    anchors.extend([last] * (WAYPOINT_SPARSE_COUNT - len(anchors)))
    return np.asarray(anchors[:WAYPOINT_SPARSE_COUNT], dtype=int)
