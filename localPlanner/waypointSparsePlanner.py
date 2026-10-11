"""Stage 12: build the 1-4 waypoints that are actually sent downstream."""

import math
import numpy as np

from localPlanner.pathGeometry import (
    clean_xy, heading_at_s, point_at_s, polyline_length,
    project_to_polyline,
)
from parameters import (
    JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE,
    JUNCTION_EXIT_PATH_EXTENSION,
    WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
    WAYPOINT_OUTPUT_MIN_SPACING,
    WAYPOINT_SPARSE_COUNT,
)


def build_waypoint_sparse(
        waypoint_whole, dynamic_info, base_path, base_candidates,
        junction_entrance=None, centerline_dict=None,
        junction_lane_ids=None):
    """Build a complete-range sparse result while retaining the internal stop."""
    del junction_entrance
    path = clean_xy(base_path)
    position = np.asarray(dynamic_info['position'], dtype=float)[:2]
    heading = float(dynamic_info['heading'])
    if len(path) < 2 or polyline_length(path) <= 1e-9:
        return _forward_stop(position, heading)

    total = polyline_length(path)
    current_s = float(project_to_polyline(position, path)['s'])
    stop_s = float(project_to_polyline(
        waypoint_whole.points[-1], path)['s'])
    stop_s = float(np.clip(stop_s, current_s, total))
    candidate_s = _candidate_s_values(base_candidates, path, total)
    forward_s = candidate_s[
        candidate_s >= current_s + WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE - 1e-9
    ]

    anchors = []
    _add_anchor(anchors, total, 0)
    if stop_s >= current_s + WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE - 1e-9:
        _add_anchor(anchors, stop_s, 1)
    junction_exit_s = _terminal_junction_exit_s(
        path, centerline_dict, junction_lane_ids)
    if (junction_exit_s is not None
            and junction_exit_s
            >= current_s + WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE - 1e-9):
        _add_anchor(anchors, junction_exit_s, 2)
    if len(forward_s):
        _add_anchor(anchors, float(forward_s[0]), 3)
    _fill_balanced_anchors(anchors, forward_s)

    selected_s = np.asarray(sorted(value for value, _ in anchors), dtype=float)
    if len(selected_s) == 0:
        return _forward_stop(position, heading)
    rows = []
    stop_relative_s = max(0.0, stop_s - current_s)
    for path_s in selected_s:
        point = point_at_s(path, path_s)
        relative_s = max(0.0, path_s - current_s)
        if relative_s <= stop_relative_s + 1e-6:
            arrival = waypoint_whole.time_at_s(relative_s)
            speed = waypoint_whole.speed_at_s(relative_s)
            if (waypoint_whole.stopped
                    and abs(relative_s - stop_relative_s) <= 1e-6):
                speed = 0.0
        else:
            arrival = float('inf')
            speed = 0.0
        rows.append([
            point[0], point[1], heading_at_s(path, path_s),
            max(0.0, float(speed)), float(arrival),
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


def _forward_stop(position, heading):
    """A failed one-point plan is sent as one zero-speed point ahead."""
    point = position + WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE * np.asarray([
        math.cos(heading), math.sin(heading)
    ])
    return np.asarray([[
        point[0], point[1], heading, 0.0, 0.0
    ]], dtype=float)


def _candidate_s_values(base_candidates, path, total):
    if base_candidates is None:
        values = np.linspace(0.0, total, max(WAYPOINT_SPARSE_COUNT * 4, 2))
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


def _add_anchor(anchors, value, priority):
    value = float(value)
    for index, (existing, existing_priority) in enumerate(anchors):
        if abs(existing - value) <= WAYPOINT_OUTPUT_MIN_SPACING + 1e-9:
            if priority < existing_priority:
                anchors[index] = (value, priority)
            return
    if len(anchors) < WAYPOINT_SPARSE_COUNT:
        anchors.append((value, priority))


def _fill_balanced_anchors(anchors, candidates):
    available = [
        float(value) for value in np.asarray(candidates, dtype=float)
        if all(
            abs(float(value) - existing)
            > WAYPOINT_OUTPUT_MIN_SPACING + 1e-9
            for existing, _ in anchors
        )
    ]
    while available and len(anchors) < WAYPOINT_SPARSE_COUNT:
        ordered = sorted(value for value, _ in anchors)
        best = max(
            available,
            key=lambda value: min(abs(value - existing) for existing in ordered),
        )
        _add_anchor(anchors, best, 4)
        available = [
            value for value in available
            if all(
                abs(value - existing)
                > WAYPOINT_OUTPUT_MIN_SPACING + 1e-9
                for existing, _ in anchors
            )
        ]


def _terminal_junction_exit_s(path, centerline_dict, junction_lane_ids):
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
