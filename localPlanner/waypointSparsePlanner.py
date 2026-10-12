"""Stage 12: organize the complete sparse waypoint sequence for output."""

import math

import numpy as np

from localPlanner.pathGeometry import (
    clean_xy, heading_at_s, point_at_s, polyline_length,
    project_to_polyline,
)
from localPlanner.waypointCandidatePlanner import (
    WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE,
    WAYPOINT_CANDIDATE_JUNCTION_EXIT,
    WAYPOINT_CANDIDATE_PATH_END,
)
from parameters import (
    WAYPOINT_MANDATORY_MERGE_DISTANCE,
    WAYPOINT_OUTPUT_FORWARD_TIME,
    WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
    WAYPOINT_SPARSE_MIN_SPACING,
)


_MANDATORY_STOP = 8
_MANDATORY_FIRST_FORWARD = 16


def build_waypoint_sparse(
        waypoint_whole, dynamic_info, base_path, base_candidates):
    """Build an unrestricted sparse sequence over the complete basePath."""
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
    candidate_s, candidate_types = _candidate_data(
        base_candidates, path, total)
    junction_intervals = _candidate_junction_intervals(
        candidate_s, candidate_types)

    mandatory = []
    _add_mandatory(
        mandatory, total, 0, WAYPOINT_CANDIDATE_PATH_END)

    output_stop_s = _legal_output_s(
        stop_s, current_s, junction_intervals)
    if output_stop_s >= minimum_output_s - 1e-9:
        _add_mandatory(mandatory, output_stop_s, 1, _MANDATORY_STOP)

    for path_s, candidate_type in zip(candidate_s, candidate_types):
        if path_s < minimum_output_s - 1e-9:
            continue
        priority = _candidate_priority(candidate_type, path_s, total)
        if priority is not None:
            _add_mandatory(
                mandatory, path_s, priority, int(candidate_type))

    first_forward_s = _first_forward_s(
        candidate_s, mandatory, minimum_output_s, total)
    if first_forward_s is not None:
        _add_mandatory(
            mandatory, first_forward_s, 4, _MANDATORY_FIRST_FORWARD)

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


def _candidate_data(base_candidates, path, total):
    if base_candidates is None:
        count = max(int(math.ceil(total / WAYPOINT_SPARSE_MIN_SPACING)), 1)
        values = np.linspace(0.0, total, count + 1)
        candidate_types = np.zeros(len(values), dtype=int)
    else:
        candidates = np.asarray(base_candidates, dtype=float)
        if candidates.ndim == 2 and candidates.shape[1] >= 3:
            values = candidates[:, 2]
            candidate_types = (
                np.rint(candidates[:, 3]).astype(int)
                if candidates.shape[1] >= 4
                else np.zeros(len(candidates), dtype=int)
            )
        elif candidates.ndim == 2 and len(candidates):
            values = np.asarray([
                project_to_polyline(point[:2], path)['s']
                for point in candidates
            ], dtype=float)
            candidate_types = np.zeros(len(candidates), dtype=int)
        else:
            values = np.empty(0, dtype=float)
            candidate_types = np.empty(0, dtype=int)

    merged = {}
    for value, candidate_type in zip(values, candidate_types):
        key = round(float(np.clip(value, 0.0, total)), 6)
        merged[key] = int(merged.get(key, 0)) | int(candidate_type)
    end_key = round(float(total), 6)
    merged[end_key] = int(merged.get(end_key, 0)) | WAYPOINT_CANDIDATE_PATH_END
    sorted_s = np.asarray(sorted(merged), dtype=float)
    sorted_types = np.asarray([merged[value] for value in sorted_s], dtype=int)
    return sorted_s, sorted_types


def _candidate_junction_intervals(candidate_s, candidate_types):
    entrances = []
    intervals = []
    for value, candidate_type in zip(candidate_s, candidate_types):
        if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE:
            entrances.append(float(value))
        if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_EXIT:
            for index, entrance_s in enumerate(entrances):
                if value > entrance_s + 1e-6:
                    intervals.append((entrance_s, float(value)))
                    entrances.pop(index)
                    break
    return intervals


def _candidate_priority(candidate_type, path_s, total):
    candidate_type = int(candidate_type)
    if candidate_type & WAYPOINT_CANDIDATE_PATH_END:
        return 0
    if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE:
        return 2
    if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_EXIT:
        return 2
    return None


def _legal_output_s(value, current_s, junction_intervals):
    value = float(value)
    for entrance_s, exit_s in junction_intervals:
        if entrance_s + 1e-6 < value < exit_s - 1e-6:
            return entrance_s if current_s < entrance_s else exit_s
    return value


def _add_mandatory(anchors, value, priority, semantic):
    value = float(value)
    semantic = int(semantic)
    for index, (existing, existing_priority, existing_semantic) in enumerate(
            anchors):
        if abs(existing - value) <= WAYPOINT_MANDATORY_MERGE_DISTANCE + 1e-9:
            if _protected_junction_pair(existing_semantic, semantic):
                continue
            if priority < existing_priority:
                existing = value
                existing_priority = priority
            anchors[index] = (
                existing,
                existing_priority,
                int(existing_semantic) | semantic,
            )
            return
    anchors.append((value, priority, semantic))


def _protected_junction_pair(first_semantic, second_semantic):
    first_entrance = bool(
        int(first_semantic) & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE)
    first_exit = bool(
        int(first_semantic) & WAYPOINT_CANDIDATE_JUNCTION_EXIT)
    second_entrance = bool(
        int(second_semantic) & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE)
    second_exit = bool(
        int(second_semantic) & WAYPOINT_CANDIDATE_JUNCTION_EXIT)
    return bool(
        (first_entrance and second_exit)
        or (first_exit and second_entrance)
    )


def _first_forward_s(candidate_s, mandatory, minimum_output_s, total):
    pool = list(np.asarray(candidate_s, dtype=float))
    pool.extend(value for value, _, _ in mandatory)
    forward = [
        value for value in pool
        if value >= minimum_output_s - 1e-9
    ]
    if forward:
        return float(min(forward))
    return float(total)


def _add_supplemental_points(mandatory, candidate_s, minimum_output_s):
    selected = sorted(value for value, _, _ in mandatory)
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
