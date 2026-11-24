"""Optional spatial crossing filter applied only to newly issued AGV paths."""

import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths, clean_xy, project_points_to_polyline,
    project_to_polyline, slice_polyline,
)
from parameters import (
    NONCROSS_CURVE_PROXIMITY_M,
    NONCROSS_RESPONSE_REUSE_MAX_LENGTH_DIFFERENCE,
    NONCROSS_RESPONSE_REUSE_MAX_SPATIAL_DEVIATION,
    NONCROSS_RESPONSE_REUSE_MAX_SPEED_DIFFERENCE,
    NONCROSS_RESPONSE_REUSE_MAX_SEGMENT_TIME_DIFFERENCE,
    WAYPOINT_OUTPUT_FORWARD_TIME,
    WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
)


def same_noncross_response(old_paths, new_paths, old_waypoints,
                           new_waypoints, current_position,
                           old_planned_path=None, new_planned_path=None):
    """Compare actual issued paths with the noncrossing output tolerances."""
    if old_paths == new_paths:
        return True
    if not old_paths or not new_paths:
        return False
    if (len(old_paths) == len(new_paths) == 1
            and old_waypoints is not None and new_waypoints is not None):
        old = np.asarray(old_waypoints, dtype=float)
        new = np.asarray(new_waypoints, dtype=float)
        if (old.ndim == new.ndim == 2 and len(old) == len(new) == 1
                and float(old_paths[0].get('speedLimit') or 0.0) == 0.0
                and float(new_paths[0].get('speedLimit') or 0.0) == 0.0
                and bool(old_paths[0].get('isFinalDestination'))
                == bool(new_paths[0].get('isFinalDestination'))
                and np.linalg.norm(old[0, :2] - new[0, :2])
                <= NONCROSS_RESPONSE_REUSE_MAX_LENGTH_DIFFERENCE):
            return True
    if len(old_paths) != len(new_paths):
        return False

    position = np.asarray(current_position, dtype=float)[:2]
    old_curve = _remaining_curve(
        np.asarray(old_waypoints, dtype=float),
        {'position': position}, old_planned_path)
    new_curve = _remaining_curve(
        np.asarray(new_waypoints, dtype=float),
        {'position': position}, new_planned_path)
    if len(old_curve) < 2 or len(new_curve) < 2:
        return False
    tolerance = NONCROSS_RESPONSE_REUSE_MAX_SPATIAL_DEVIATION
    if (np.linalg.norm(old_curve[-1] - new_curve[-1]) > tolerance
            or not _curves_within_distance(old_curve, new_curve, tolerance)
            or not _curves_within_distance(new_curve, old_curve, tolerance)):
        return False
    for index, (old_path, new_path) in enumerate(zip(old_paths, new_paths)):
        if (old_path.keys() != new_path.keys()
                or any(old_path[key] != new_path[key]
                       for key in old_path
                       if key not in ('speedLimit', 'points', 'length'))):
            return False
        old_points = old_path['points']
        new_points = new_path['points']
        if (np.linalg.norm(np.asarray((old_points[-1]['x'], old_points[-1]['y']))
                           - np.asarray((new_points[-1]['x'], new_points[-1]['y'])))
                > tolerance):
            return False
        if index and (np.linalg.norm(
                np.asarray((old_points[0]['x'], old_points[0]['y']))
                - np.asarray((new_points[0]['x'], new_points[0]['y'])))
                > tolerance):
            return False
        old_speed = max(0.0, float(old_path.get('speedLimit') or 0.0))
        new_speed = max(0.0, float(new_path.get('speedLimit') or 0.0))
        if not (math.isfinite(old_speed) and math.isfinite(new_speed)):
            return False
        if (old_speed <= 1e-9) != (new_speed <= 1e-9):
            return False
        if old_speed <= 1e-9:
            continue

        # Compare the common remaining portion, including the first segment.
        old_start = project_to_polyline(
            (old_points[0]['x'], old_points[0]['y']), new_curve)['s']
        new_start = project_to_polyline(
            (new_points[0]['x'], new_points[0]['y']), new_curve)['s']
        old_end = project_to_polyline(
            (old_points[-1]['x'], old_points[-1]['y']), new_curve)['s']
        new_end = project_to_polyline(
            (new_points[-1]['x'], new_points[-1]['y']), new_curve)['s']
        length = max(0.0, min(old_end, new_end) - max(old_start, new_start))
        if length <= 1e-9:
            continue
        old_time = length / old_speed
        new_time = length / new_speed
        if (abs(new_speed - old_speed)
                > NONCROSS_RESPONSE_REUSE_MAX_SPEED_DIFFERENCE
                and abs(new_time - old_time)
                > NONCROSS_RESPONSE_REUSE_MAX_SEGMENT_TIME_DIFFERENCE):
            return False
    return True


def _curves_within_distance(source, target, tolerance):
    samples = np.vstack((source, (source[:-1] + source[1:]) / 2.0))
    return bool(np.all(
        project_points_to_polyline(samples, target)['distance']
        <= tolerance + 1e-9))


def is_current_position_stop(waypoint_whole, dynamic):
    """The planner has no reachable motion beyond the vehicle position."""
    return bool(
        waypoint_whole.stopped
        and len(waypoint_whole.points)
        and np.linalg.norm(
            np.asarray(waypoint_whole.points[-1], dtype=float)[:2]
            - np.asarray(dynamic['position'], dtype=float)[:2]
        ) <= 1e-3
    )


def select_noncrossing_responses(
        candidate_waypoints, priority_decisions, cached_responses,
        vehicle_dynamic_info, vehicle_static_info, stop_clearance,
        fallback_waypoints=None, candidate_paths=None, cached_paths=None):
    """Return prefixes to issue; None means keep the old response, if any."""
    selected = {}
    forced_updates = set()
    stop_fallbacks = set()
    fallback_waypoints = fallback_waypoints or {}
    candidate_paths = candidate_paths or {}
    cached_paths = cached_paths or {}
    for decision in priority_decisions:
        higher_curves = []
        for vehicle_id in decision.priorityOrder:
            if vehicle_static_info[vehicle_id].get('vehicleType') != 'AGV':
                continue
            dynamic = vehicle_dynamic_info[vehicle_id]
            if vehicle_id in candidate_waypoints:
                path = candidate_paths.get(vehicle_id)
                waypoints = np.asarray(
                    candidate_waypoints[vehicle_id], dtype=float)
                prefix, conflict = _trim_to_safe_prefix(
                    waypoints, dynamic, path, higher_curves,
                    stop_clearance)
                if prefix is None and conflict:
                    prefix = _forward_stop_waypoint(
                        fallback_waypoints.get(vehicle_id, waypoints), dynamic)
                    if prefix is not None:
                        stop_fallbacks.add(vehicle_id)
                selected[vehicle_id] = prefix
                active = prefix
                if active is None:
                    active = cached_responses.get(vehicle_id)
                    path = cached_paths.get(vehicle_id)
            else:
                active = cached_responses.get(vehicle_id)
                path = cached_paths.get(vehicle_id)
                if active is not None and len(active) and higher_curves:
                    minimum = max(
                        WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
                        max(0.0, float(dynamic['speed']))
                        * WAYPOINT_OUTPUT_FORWARD_TIME)
                    remaining = _forward_waypoints(active, dynamic, minimum)
                    if len(remaining):
                        prefix, conflict = _trim_to_safe_prefix(
                            remaining, dynamic, path, higher_curves,
                            stop_clearance)
                        if conflict:
                            if prefix is None:
                                prefix = _forward_stop_waypoint(
                                    active,
                                    dynamic)
                                if prefix is not None:
                                    stop_fallbacks.add(vehicle_id)
                            if prefix is not None and not np.array_equal(
                                    prefix, np.asarray(active, dtype=float)):
                                selected[vehicle_id] = prefix
                                forced_updates.add(vehicle_id)
                                active = prefix
                    else:
                        old_curve = _remaining_curve(active, dynamic, path)
                        conflict = any(
                            math.isfinite(_first_curve_proximity(old_curve, high))
                            for high in higher_curves)
                        if conflict:
                            prefix = _forward_stop_waypoint(
                                active,
                                dynamic)
                            if prefix is not None and not np.array_equal(
                                    prefix, np.asarray(active, dtype=float)):
                                selected[vehicle_id] = prefix
                                forced_updates.add(vehicle_id)
                                stop_fallbacks.add(vehicle_id)
                                active = prefix
            if active is not None and len(active):
                curve = _remaining_curve(active, dynamic, path)
                if len(curve) >= 2:
                    higher_curves.append(curve)

    # Vehicles outside a cooperation block have no higher-priority peer.
    for vehicle_id, waypoints in candidate_waypoints.items():
        if vehicle_id not in selected:
            selected[vehicle_id] = np.asarray(waypoints, dtype=float).copy()
    return selected, forced_updates, stop_fallbacks


def _forward_waypoints(waypoints, dynamic, minimum_distance):
    values = np.asarray(waypoints, dtype=float)
    if len(values) == 0:
        return values
    position = np.asarray(dynamic['position'], dtype=float)[:2]
    points = values[:, :2]
    if len(points) == 1:
        heading = float(dynamic['heading'])
        tangent = np.asarray([math.cos(heading), math.sin(heading)])
        forward = float((points[0] - position) @ tangent)
        return values.copy() if forward >= minimum_distance - 1e-9 else values[:0]
    path_s = arc_lengths(points)
    current_s = float(project_to_polyline(position, points)['s'])
    segments = np.diff(points, axis=0)
    nonzero = np.flatnonzero(np.linalg.norm(segments, axis=1) > 1e-9)
    tangent = (segments[int(nonzero[0])] if len(nonzero)
               else np.asarray([math.cos(float(dynamic['heading'])),
                                math.sin(float(dynamic['heading']))]))
    length = float(np.linalg.norm(tangent))
    offset = (max(0.0, float((points[0] - position) @ tangent / length))
              if current_s <= 1e-9 and length > 1e-9 else 0.0)
    eligible = np.flatnonzero(
        path_s - current_s + offset >= minimum_distance - 1e-9)
    return values[int(eligible[0]):].copy() if len(eligible) else values[:0]


def _forward_stop_waypoint(waypoints, dynamic):
    minimum = max(
        WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
        max(0.0, float(dynamic['speed'])) * WAYPOINT_OUTPUT_FORWARD_TIME,
    )
    remaining = _forward_waypoints(waypoints, dynamic, minimum)
    if len(remaining) == 0:
        return None
    stop = remaining[:1].copy()
    stop[0, 3:] = 0.0
    return stop


def moving_waypoint_prefix(waypoints, dynamic):
    values = np.asarray(waypoints, dtype=float)
    if len(values) == 0:
        return None
    if len(values) == 1:
        return _forward_stop_waypoint(values, dynamic)
    last_moving_index = None
    for index in range(len(values) - 1):
        distance = float(np.linalg.norm(
            values[index + 1, :2] - values[index, :2]))
        if distance <= 1e-9:
            continue
        start_time, end_time = values[index, 4], values[index + 1, 4]
        duration = (float(end_time - start_time)
                    if np.isfinite(start_time) and np.isfinite(end_time)
                    else math.inf)
        if not math.isfinite(duration) or duration <= 1e-9:
            return (values[:last_moving_index + 1].copy()
                    if last_moving_index is not None
                    else _forward_stop_waypoint(values[:1], dynamic))
        last_moving_index = index + 1
    return (values[:last_moving_index + 1].copy()
            if last_moving_index is not None
            else _forward_stop_waypoint(values[:1], dynamic))


def _trim_to_safe_prefix(
        waypoints, dynamic, path, higher_curves, stop_clearance):
    if len(waypoints) == 0:
        return None, False
    if not higher_curves:
        return waypoints.copy(), False
    low = _remaining_curve(waypoints, dynamic, path)
    if len(low) < 2:
        return None, False
    first_conflict = min(
        _first_curve_proximity(low, high) for high in higher_curves)
    if not math.isfinite(first_conflict):
        return waypoints.copy(), False
    safe_end = first_conflict - max(0.0, float(stop_clearance))
    base = clean_xy(path)
    start_s = float(project_to_polyline(dynamic['position'], base)['s'])
    waypoint_s = np.asarray([
        float(project_to_polyline(point, base)['s']) - start_s
        for point in waypoints[:, :2]
    ])
    legal = np.flatnonzero(
        (waypoint_s <= safe_end + 1e-9)
        & (waypoint_s > 1e-9))
    if len(legal) == 0:
        return None, True
    prefix = waypoints[:int(legal[-1]) + 1].copy()
    if len(prefix) == 1:
        prefix[0, 3:] = 0.0
    return prefix, True


def _remaining_curve(waypoints, dynamic, path):
    if path is None or len(waypoints) == 0:
        return np.empty((0, 2), dtype=float)
    base = clean_xy(path)
    if len(base) < 2:
        return base
    start = float(project_to_polyline(dynamic['position'], base)['s'])
    end = float(project_to_polyline(waypoints[-1, :2], base)['s'])
    return slice_polyline(base, start, end)


def _first_curve_proximity(low, high):
    """Arc length of the first low-curve sample near a high-curve sample."""
    if len(low) < 2 or len(high) < 2:
        return math.inf
    threshold = NONCROSS_CURVE_PROXIMITY_M
    threshold_sq = threshold * threshold
    high_min = np.min(high, axis=0) - threshold
    high_max = np.max(high, axis=0) + threshold
    progress = arc_lengths(low)
    for index, point in enumerate(low):
        if np.any(point < high_min) or np.any(point > high_max):
            continue
        delta = high - point
        if np.any(np.sum(delta * delta, axis=1) < threshold_sq):
            return float(progress[index])
    return math.inf
