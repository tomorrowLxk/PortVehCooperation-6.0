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
    WAYPOINT_CANDIDATE_LANE_CHANGE_ENTRANCE,
    WAYPOINT_CANDIDATE_LANE_CHANGE_EXIT,
    WAYPOINT_CANDIDATE_PATH_END,
)
from parameters import (
    WAYPOINT_JUNCTION_ENTRANCE_UPSTREAM_OFFSET,
    WAYPOINT_MANDATORY_MERGE_DISTANCE,
    WAYPOINT_MANDATORY_MIN_FORWARD_SPACING,
    WAYPOINT_OUTPUT_FORWARD_TIME,
    WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
    WAYPOINT_SPARSE_MIN_SPACING,
)


_MANDATORY_STOP = 8


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
    lane_change_intervals = _candidate_lane_change_intervals(
        candidate_s, candidate_types)

    mandatory = []
    _add_mandatory(
        mandatory, total, 0, WAYPOINT_CANDIDATE_PATH_END)

    output_stop_s = _legal_output_s(
        stop_s, current_s, junction_intervals)
    if output_stop_s >= minimum_output_s - 1e-9:
        _add_mandatory(mandatory, output_stop_s, 1, _MANDATORY_STOP)

    for entrance_s, exit_s in lane_change_intervals:
        effective_entrance_s = max(entrance_s, minimum_output_s)
        if effective_entrance_s < exit_s - 1e-6:
            _add_mandatory(
                mandatory,
                effective_entrance_s,
                2,
                WAYPOINT_CANDIDATE_LANE_CHANGE_ENTRANCE,
            )

    for path_s, candidate_type in zip(candidate_s, candidate_types):
        if path_s < minimum_output_s - 1e-9:
            continue
        priority = _candidate_priority(candidate_type, path_s, total)
        if priority is not None:
            mandatory_s = path_s
            if _is_shiftable_junction_entrance(candidate_type):
                mandatory_s = _shift_junction_entrance_s(
                    path_s, mandatory, minimum_output_s)
            _add_mandatory(
                mandatory, mandatory_s, priority, int(candidate_type))

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
    return _candidate_intervals(
        candidate_s,
        candidate_types,
        WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE,
        WAYPOINT_CANDIDATE_JUNCTION_EXIT,
        initial_exit=True,
    )


def _candidate_lane_change_intervals(candidate_s, candidate_types):
    return _candidate_intervals(
        candidate_s,
        candidate_types,
        WAYPOINT_CANDIDATE_LANE_CHANGE_ENTRANCE,
        WAYPOINT_CANDIDATE_LANE_CHANGE_EXIT,
    )


def _candidate_intervals(
        candidate_s, candidate_types, entrance_type, exit_type,
        initial_exit=False):
    entrances = []
    intervals = []
    for value, candidate_type in zip(candidate_s, candidate_types):
        if candidate_type & entrance_type:
            entrances.append(float(value))
        if candidate_type & exit_type:
            for index, entrance_s in enumerate(entrances):
                if value > entrance_s + 1e-6:
                    intervals.append((entrance_s, float(value)))
                    entrances.pop(index)
                    break
            else:
                if (initial_exit and not entrances and not intervals
                        and value > 1e-6):
                    intervals.append((0.0, float(value)))
    return intervals


def _candidate_priority(candidate_type, path_s, total):
    candidate_type = int(candidate_type)
    if candidate_type & WAYPOINT_CANDIDATE_PATH_END:
        return 0
    if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE:
        return 2
    if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_EXIT:
        return 2
    if candidate_type & WAYPOINT_CANDIDATE_LANE_CHANGE_ENTRANCE:
        return 2
    if candidate_type & WAYPOINT_CANDIDATE_LANE_CHANGE_EXIT:
        return 2
    return None


def _is_shiftable_junction_entrance(candidate_type):
    candidate_type = int(candidate_type)
    return bool(
        candidate_type & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE
        and not candidate_type & WAYPOINT_CANDIDATE_JUNCTION_EXIT
        and not candidate_type & WAYPOINT_CANDIDATE_PATH_END
    )


def _shift_junction_entrance_s(value, mandatory, minimum_output_s):
    value = float(value)
    lower_bound = max(
        value - WAYPOINT_JUNCTION_ENTRANCE_UPSTREAM_OFFSET,
        float(minimum_output_s),
    )
    previous = [
        float(minimum_output_s),
        *(existing for existing, _, _ in mandatory
          if existing < value - 1e-9),
    ]
    previous_s = max(previous)
    if previous_s < value - 1e-9:
        lower_bound = max(
            lower_bound,
            previous_s + WAYPOINT_MANDATORY_MIN_FORWARD_SPACING,
        )
    return min(value, lower_bound)


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
            if _protected_boundary_pair(existing_semantic, semantic):
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


def _protected_boundary_pair(first_semantic, second_semantic):
    first_entrance = bool(
        int(first_semantic) & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE)
    first_exit = bool(
        int(first_semantic) & WAYPOINT_CANDIDATE_JUNCTION_EXIT)
    second_entrance = bool(
        int(second_semantic) & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE)
    second_exit = bool(
        int(second_semantic) & WAYPOINT_CANDIDATE_JUNCTION_EXIT)
    junction_pair = bool(
        (first_entrance and second_exit)
        or (first_exit and second_entrance)
    )
    first_lane_change_entrance = bool(
        int(first_semantic) & WAYPOINT_CANDIDATE_LANE_CHANGE_ENTRANCE)
    first_lane_change_exit = bool(
        int(first_semantic) & WAYPOINT_CANDIDATE_LANE_CHANGE_EXIT)
    second_lane_change_entrance = bool(
        int(second_semantic) & WAYPOINT_CANDIDATE_LANE_CHANGE_ENTRANCE)
    second_lane_change_exit = bool(
        int(second_semantic) & WAYPOINT_CANDIDATE_LANE_CHANGE_EXIT)
    lane_change_pair = bool(
        (first_lane_change_entrance and second_lane_change_exit)
        or (first_lane_change_exit and second_lane_change_entrance)
    )
    return junction_pair or lane_change_pair



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
