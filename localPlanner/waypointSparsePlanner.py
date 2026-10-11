"""Stage 12: organize the complete sparse waypoint sequence for output."""

import math

import numpy as np

from localPlanner.pathGeometry import (
    clean_xy, heading_at_s, point_at_s, polyline_length,
    project_to_polyline,
)
from parameters import (
    JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE,
    JUNCTION_EXIT_PATH_EXTENSION,
    WAYPOINT_OUTPUT_FORWARD_TIME,
    WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
    WAYPOINT_SPARSE_MIN_SPACING,
    VEHICLE_LANE_LOCATE_MAX_DISTANCE,
)


def build_waypoint_sparse(
        waypoint_whole, dynamic_info, base_path, base_candidates,
        junction_entrance=None, centerline_dict=None,
        junction_lane_ids=None):
    """Build an unrestricted sparse sequence over the complete basePath."""
    del junction_entrance
    path = clean_xy(base_path)
    position = np.asarray(dynamic_info['position'], dtype=float)[:2]
    heading = float(dynamic_info['heading'])
    speed = max(0.0, float(dynamic_info['speed']))
    forward_distance = max(
        WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
        WAYPOINT_OUTPUT_FORWARD_TIME * speed,
    )
    if len(path) < 2 or polyline_length(path) <= 1e-9:
        return _forward_stop(position, heading, forward_distance)

    total = polyline_length(path)
    current_s = float(project_to_polyline(position, path)['s'])
    minimum_output_s = current_s + forward_distance
    stop_s = float(project_to_polyline(
        waypoint_whole.points[-1], path)['s'])
    stop_s = float(np.clip(stop_s, current_s, total))
    candidate_s = _candidate_s_values(base_candidates, path, total)
    junction_intervals = _junction_intervals(
        path, centerline_dict, junction_lane_ids)

    mandatory = []
    _add_mandatory(mandatory, total, 0)

    output_stop_s = _legal_output_s(
        stop_s, current_s, junction_intervals)
    if output_stop_s >= minimum_output_s - 1e-9:
        _add_mandatory(mandatory, output_stop_s, 1)

    for entrance_s, exit_s in junction_intervals:
        if exit_s < current_s - 1e-9:
            continue
        if entrance_s >= minimum_output_s - 1e-9:
            _add_mandatory(mandatory, entrance_s, 2)
        if (exit_s >= minimum_output_s - 1e-9
                and not _terminal_merges_with_exit(total, exit_s)):
            _add_mandatory(mandatory, exit_s, 2)
        post_exit_s = min(exit_s + JUNCTION_EXIT_PATH_EXTENSION, total)
        post_exit_s = _legal_output_s(
            post_exit_s, current_s, junction_intervals)
        if (post_exit_s > exit_s + 1e-6
                and post_exit_s >= minimum_output_s - 1e-9):
            _add_mandatory(mandatory, post_exit_s, 3)

    first_forward_s = _first_forward_s(
        candidate_s, mandatory, minimum_output_s, total)
    if first_forward_s is not None:
        _add_mandatory(mandatory, first_forward_s, 4)

    selected_s = _add_supplemental_points(
        mandatory, candidate_s, minimum_output_s)
    if len(selected_s) == 0:
        return _forward_stop(position, heading, forward_distance)

    rows = []
    stop_relative_s = max(0.0, stop_s - current_s)
    for path_s in selected_s:
        point = point_at_s(path, path_s)
        relative_s = max(0.0, path_s - current_s)
        if relative_s <= stop_relative_s + 1e-6:
            arrival = waypoint_whole.time_at_s(relative_s)
            point_speed = waypoint_whole.speed_at_s(relative_s)
            if (waypoint_whole.stopped
                    and abs(relative_s - stop_relative_s) <= 1e-6):
                point_speed = 0.0
        else:
            arrival = float('inf')
            point_speed = 0.0
        rows.append([
            point[0], point[1], heading_at_s(path, path_s),
            max(0.0, float(point_speed)), float(arrival),
        ])
    return np.asarray(rows, dtype=float)


def build_hv_waypoint_sparse(waypoint_whole, dynamic_info, base_path):
    """Build a fresh complete-range sparse result for an HV prediction."""
    return build_waypoint_sparse(
        waypoint_whole=waypoint_whole,
        dynamic_info=dynamic_info,
        base_path=base_path,
        base_candidates=None,
    )


def _forward_stop(position, heading, forward_distance):
    """A failed plan is represented by one zero-speed point ahead."""
    point = position + forward_distance * np.asarray([
        math.cos(heading), math.sin(heading)
    ])
    return np.asarray([[
        point[0], point[1], heading, 0.0, 0.0
    ]], dtype=float)


def _candidate_s_values(base_candidates, path, total):
    if base_candidates is None:
        count = max(int(math.ceil(total / WAYPOINT_SPARSE_MIN_SPACING)), 1)
        values = np.linspace(0.0, total, count + 1)
    else:
        candidates = np.asarray(base_candidates, dtype=float)
        if candidates.ndim == 2 and candidates.shape[1] >= 3:
            values = candidates[:, 2]
        elif candidates.ndim == 2 and len(candidates):
            values = np.asarray([
                project_to_polyline(point[:2], path)['s']
                for point in candidates
            ], dtype=float)
        else:
            values = np.empty(0, dtype=float)
    values = np.r_[np.asarray(values, dtype=float), total]
    return np.unique(np.clip(values, 0.0, total))


def _junction_intervals(path, centerline_dict, junction_lane_ids):
    if len(path) < 2 or not centerline_dict or not junction_lane_ids:
        return []
    intervals = []
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
        entrance = project_to_polyline(lane[0], path)
        exit_point = project_to_polyline(lane[-1], path)
        entrance_s = float(entrance['s'])
        exit_s = float(exit_point['s'])
        if (entrance['distance'] <= VEHICLE_LANE_LOCATE_MAX_DISTANCE
                and exit_point['distance'] <= VEHICLE_LANE_LOCATE_MAX_DISTANCE
                and exit_s > entrance_s + 1e-6):
            intervals.append((entrance_s, exit_s))
    intervals.sort()
    unique = []
    for interval in intervals:
        if (not unique
                or abs(interval[0] - unique[-1][0]) > 1e-6
                or abs(interval[1] - unique[-1][1]) > 1e-6):
            unique.append(interval)
    return unique


def _legal_output_s(value, current_s, junction_intervals):
    value = float(value)
    for entrance_s, exit_s in junction_intervals:
        if entrance_s + 1e-6 < value < exit_s - 1e-6:
            return entrance_s if current_s < entrance_s else exit_s
    return value


def _terminal_merges_with_exit(total, exit_s):
    return total - exit_s <= JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE + 1e-6


def _add_mandatory(anchors, value, priority):
    value = float(value)
    for index, (existing, existing_priority) in enumerate(anchors):
        if abs(existing - value) <= 1e-6:
            if priority < existing_priority:
                anchors[index] = (value, priority)
            return
    anchors.append((value, priority))


def _first_forward_s(candidate_s, mandatory, minimum_output_s, total):
    pool = list(np.asarray(candidate_s, dtype=float))
    pool.extend(value for value, _ in mandatory)
    forward = [
        value for value in pool
        if value >= minimum_output_s - 1e-9
    ]
    if forward:
        return float(min(forward))
    return float(total)


def _add_supplemental_points(mandatory, candidate_s, minimum_output_s):
    selected = sorted(value for value, _ in mandatory)
    for value in np.asarray(candidate_s, dtype=float):
        value = float(value)
        if value < minimum_output_s - 1e-9:
            continue
        if all(
                abs(value - existing)
                >= WAYPOINT_SPARSE_MIN_SPACING - 1e-9
                for existing in selected):
            selected.append(value)
            selected.sort()
    return np.asarray(selected, dtype=float)
