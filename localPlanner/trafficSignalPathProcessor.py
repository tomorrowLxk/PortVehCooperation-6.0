"""??path??????????????"""

import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths,
    clean_xy,
    heading_at_s,
    polyline_length,
    project_points_to_polyline,
    truncate_polyline,
)
from localPlanner.pathSpeedPlanner import build_speed_limit_path
from localPlanner.spatialPathPlanner import PathCandidate
from parameters import (
    TRAFFIC_SIGNAL_HEADING_AFTER_WINDOW,
    TRAFFIC_SIGNAL_HEADING_WINDOW,
    TRAFFIC_SIGNAL_MATCH_DISTANCE,
    TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG,
    TRAFFIC_SIGNAL_UTURN_THRESHOLD_DEG,
)


RIGHT_TURN = "rightTurn"
LEFT_TURN = "leftTurn"
U_TURN = "UTurn"
STRAIGHT = "straight"
RED = "RED"


def apply_traffic_signals_to_base_paths(
        base_paths, planning_max_speed, centerline_dict, connector_dict,
        lane_signals, eligible_lane_ids=None):
    """对一辆车的basePath集合应用本帧信控，不修改输入集合。"""
    output = []
    for base_path in base_paths:
        base = np.asarray(base_path, dtype=float)
        candidate = PathCandidate(clean_xy(base), 0.0, (), 0.0)
        processed = apply_traffic_signal(
            candidate,
            centerline_dict,
            connector_dict,
            lane_signals,
            eligible_lane_ids,
        )
        if processed.terminalReason == "RED_SIGNAL":
            output.append(build_speed_limit_path(
                processed.xy,
                planning_max_speed,
                True,
            ))
        else:
            output.append(base.copy())
    return output


def apply_traffic_signal(
        candidate, centerline_dict, connector_dict, lane_signals,
        eligible_lane_ids=None):
    """????path??????????????path?"""
    if candidate is None or not lane_signals:
        return candidate
    path = clean_xy(candidate.xy)
    if len(path) < 2:
        return candidate

    lanes = _available_lanes(centerline_dict, eligible_lane_ids)
    if not lanes:
        return candidate
    point_lanes = _classify_points(
        path, centerline_dict, connector_dict, lanes)
    path_arc = arc_lengths(path)
    checked_non_red_lanes = set()

    for index, lane_id in enumerate(point_lanes):
        if lane_id is None or lane_id in checked_non_red_lanes:
            continue
        states = lane_signals.get(lane_id)
        if states is None:
            continue
        centerline = clean_xy(centerline_dict[lane_id])
        if np.linalg.norm(
                path[index] - centerline[-1]) > TRAFFIC_SIGNAL_MATCH_DISTANCE:
            continue

        direction = _movement_direction(path, path_arc[index])
        state = states.get(direction)
        if state != RED:
            checked_non_red_lanes.add(lane_id)
            continue

        truncated = truncate_polyline(path, path_arc[index])
        if len(truncated) == 0:
            truncated = path[:1].copy()
        return PathCandidate(
            truncated,
            candidate.score,
            candidate.lineSequence,
            candidate.maxCurvature,
            True,
            "RED_SIGNAL",
            candidate.degraded,
            candidate.degradationReason,
        )
    return candidate


def _available_lanes(centerline_dict, eligible_lane_ids):
    lane_ids = (
        centerline_dict.keys()
        if eligible_lane_ids is None else eligible_lane_ids
    )
    return sorted({
        str(lane_id) for lane_id in lane_ids
        if lane_id is not None and str(lane_id) in centerline_dict
    })


def _classify_points(path, centerline_dict, connector_dict, lanes):
    choices = [[] for _ in range(len(path))]
    path_segments = np.diff(path, axis=0)
    path_headings = np.append(
        np.arctan2(path_segments[:, 1], path_segments[:, 0]),
        math.atan2(path_segments[-1, 1], path_segments[-1, 0]),
    )

    for lane_id in lanes:
        centerline = clean_xy(centerline_dict[lane_id])
        if len(centerline) < 2:
            continue
        projection = project_points_to_polyline(path, centerline)
        lane_headings = np.arctan2(
            projection["tangent"][:, 1],
            projection["tangent"][:, 0],
        )
        heading_errors = np.abs(_normalize_angles(
            path_headings - lane_headings))
        valid = (
            (projection["distance"] <= TRAFFIC_SIGNAL_MATCH_DISTANCE)
            & (heading_errors <= math.pi / 2.0)
        )
        for index in np.flatnonzero(valid):
            choices[index].append((
                float(projection["distance"][index]
                      + heading_errors[index]),
                lane_id,
            ))

    result = []
    previous = None
    for point_choices in choices:
        point_choices.sort(key=lambda item: (item[0], item[1]))
        if not point_choices:
            result.append(None)
            previous = None
            continue
        selected = point_choices[0]
        by_lane = {lane_id: score for score, lane_id in point_choices}
        if previous in by_lane and by_lane[previous] <= selected[0] + 0.5:
            selected = (by_lane[previous], previous)
        elif previous is not None:
            connected = next(
                (item for item in point_choices
                 if _connected(previous, item[1], connector_dict)),
                None,
            )
            if connected is not None:
                selected = connected
        previous = selected[1]
        result.append(previous)
    return result


def _connected(first, second, connector_dict):
    if first == second:
        return True
    info = connector_dict[first]
    return (
        info.get("left_lane", info.get("leftLane")) == second
        or info.get("right_lane", info.get("rightLane")) == second
        or second in info["successor"]
        or second in info["predecessor"]
    )


def _movement_direction(path, signal_s):
    total = polyline_length(path)
    before = max(0.0, signal_s - TRAFFIC_SIGNAL_HEADING_WINDOW)
    after = min(total, signal_s + TRAFFIC_SIGNAL_HEADING_AFTER_WINDOW)
    delta = _normalize_angle(
        heading_at_s(path, after) - heading_at_s(path, before))
    if abs(delta) >= math.radians(TRAFFIC_SIGNAL_UTURN_THRESHOLD_DEG):
        return U_TURN
    if delta > math.radians(TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG):
        return LEFT_TURN
    if delta < -math.radians(TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG):
        return RIGHT_TURN
    return STRAIGHT


def _normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _normalize_angles(values):
    return np.arctan2(np.sin(values), np.cos(values))
