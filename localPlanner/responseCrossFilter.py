"""Optional spatial crossing filter applied only to newly issued AGV paths."""

import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths, clean_xy, project_to_polyline, resample_polyline,
    slice_polyline,
)
from localPlanner.waypointWholePlanner import _overlap_matrix
from parameters import (
    COOPERATION_LATERAL_MARGIN,
    WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE,
    WAYPOINT_WHOLE_REUSE_MAX_SPEED_DIFFERENCE,
    WAYPOINT_WHOLE_REUSE_MAX_SEGMENT_TIME_DIFFERENCE,
    WAYPOINT_OUTPUT_FORWARD_TIME,
    WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
)


_SAMPLE_SPACING = 2.0


def same_noncross_response(old_paths, new_paths, old_waypoints,
                           new_waypoints, current_position):
    """Compare actual issued paths with the speed-planning reuse tolerances."""
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
                <= WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE):
            return True
    if len(old_paths) != len(new_paths):
        return False

    position = np.asarray(current_position, dtype=float)[:2]
    for index, (old_path, new_path) in enumerate(zip(old_paths, new_paths)):
        if (old_path.keys() != new_path.keys()
                or any(old_path[key] != new_path[key]
                       for key in old_path if key != 'speedLimit')):
            return False
        old_speed = max(0.0, float(old_path.get('speedLimit') or 0.0))
        new_speed = max(0.0, float(new_path.get('speedLimit') or 0.0))
        if not (math.isfinite(old_speed) and math.isfinite(new_speed)):
            return False
        if (old_speed <= 1e-9) != (new_speed <= 1e-9):
            return False
        if old_speed <= 1e-9:
            continue

        # A segment ahead of the vehicle is not the currently executed one.
        if index == 0:
            points = old_path['points']
            start = np.asarray((points[0]['x'], points[0]['y']), dtype=float)
            end = np.asarray((points[-1]['x'], points[-1]['y']), dtype=float)
            direction = end - start
            length_squared = float(direction @ direction)
            if length_squared > 1e-18:
                progress = float((position - start) @ direction
                                 / length_squared)
                lateral = np.linalg.norm(
                    position - start - np.clip(progress, 0.0, 1.0)
                    * direction)
                if (0.0 <= progress < 1.0
                        and lateral <= WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE):
                    continue

        length = float(old_path.get('length') or 0.0)
        old_time = length / old_speed
        new_time = length / new_speed
        if (abs(new_speed - old_speed)
                > WAYPOINT_WHOLE_REUSE_MAX_SPEED_DIFFERENCE
                and abs(new_time - old_time)
                > WAYPOINT_WHOLE_REUSE_MAX_SEGMENT_TIME_DIFFERENCE):
            return False
    return True


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
        fallback_waypoints=None):
    """Return prefixes to issue; None means keep the old response, if any."""
    selected = {}
    forced_updates = set()
    stop_fallbacks = set()
    fallback_waypoints = fallback_waypoints or {}
    for decision in priority_decisions:
        higher_geometries = []
        for vehicle_id in decision.priorityOrder:
            if vehicle_static_info[vehicle_id].get('vehicleType') != 'AGV':
                continue
            dynamic = vehicle_dynamic_info[vehicle_id]
            static = vehicle_static_info[vehicle_id]
            if vehicle_id in candidate_waypoints:
                waypoints = np.asarray(
                    candidate_waypoints[vehicle_id], dtype=float)
                prefix, conflict = _trim_to_safe_prefix(
                    waypoints, dynamic, static, higher_geometries,
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
            else:
                active = cached_responses.get(vehicle_id)
                if active is not None and len(active) and higher_geometries:
                    minimum = max(
                        WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE,
                        max(0.0, float(dynamic['speed']))
                        * WAYPOINT_OUTPUT_FORWARD_TIME)
                    remaining = _forward_waypoints(active, dynamic, minimum)
                    if len(remaining):
                        prefix, conflict = _trim_to_safe_prefix(
                            remaining, dynamic, static, higher_geometries,
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
                        old_geometry = _remaining_geometry(active, dynamic, static)
                        conflict = any(
                            _boxes_intersect(old_geometry['bounds'], high['bounds'])
                            and np.any(_overlap_matrix(old_geometry, high))
                            for high in higher_geometries)
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
                geometry = _remaining_geometry(active, dynamic, static)
                if geometry is not None:
                    higher_geometries.append(geometry)

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
        waypoints, dynamic, static, higher_geometries, stop_clearance):
    if len(waypoints) == 0:
        return None, False
    if not higher_geometries:
        return waypoints.copy(), False
    position = np.asarray(dynamic['position'], dtype=float)[:2]
    points = np.vstack((position, waypoints[:, :2]))
    waypoint_s = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(points, axis=0), axis=1))]
    low = _remaining_geometry(waypoints, dynamic, static)
    if low is None:
        return None, False
    first_conflict = math.inf
    for high in higher_geometries:
        if not _boxes_intersect(low['bounds'], high['bounds']):
            continue
        overlaps = _overlap_matrix(low, high)
        indices = np.flatnonzero(np.any(overlaps, axis=1))
        if len(indices):
            index = int(indices[0])
            # Use the preceding sample as a conservative entry bound.
            first_conflict = min(
                first_conflict, float(low['sampleS'][max(0, index - 1)]))
    if not math.isfinite(first_conflict):
        return waypoints.copy(), False
    safe_end = first_conflict - max(0.0, float(stop_clearance))
    legal = np.flatnonzero(
        (waypoint_s[1:] <= safe_end + 1e-9)
        & (waypoint_s[1:] > 1e-9))
    if len(legal) == 0:
        return None, True
    prefix = waypoints[:int(legal[-1]) + 1].copy()
    if len(prefix) == 1:
        prefix[0, 3:] = 0.0
    return prefix, True


def _remaining_geometry(waypoints, dynamic, static):
    path = clean_xy(np.asarray(waypoints, dtype=float)[:, :2])
    if len(path) == 0:
        return None
    position = np.asarray(dynamic['position'], dtype=float)[:2]
    if len(path) >= 2:
        progress = project_to_polyline(position, path)['s']
        path = slice_polyline(path, progress, arc_lengths(path)[-1])
    if np.linalg.norm(path[0] - position) > 1e-9:
        path = np.vstack((position, path))
    samples = resample_polyline(path, _SAMPLE_SPACING)
    sample_s = arc_lengths(samples)
    if len(samples) == 1:
        heading = float(dynamic['heading'])
        tangents = np.asarray([[math.cos(heading), math.sin(heading)]])
    else:
        segments = np.diff(samples, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        tangents = np.vstack((
            segments / np.maximum(lengths[:, None], 1e-12),
            segments[-1] / max(float(lengths[-1]), 1e-12),
        ))
    half_length = (
        0.5 * float(static['length'])
        + float(dynamic.get('longitudinalUncertainty', 0.0)))
    half_width = (
        0.5 * float(static['width'])
        + float(dynamic.get('lateralUncertainty', 0.0))
        + COOPERATION_LATERAL_MARGIN)
    radius = math.hypot(half_length, half_width)
    return {
        'sampleS': sample_s,
        'centers': samples,
        'tangents': tangents,
        'normals': np.column_stack((-tangents[:, 1], tangents[:, 0])),
        'halfLength': half_length,
        'halfWidth': half_width,
        'bounds': (np.min(samples, axis=0) - radius,
                   np.max(samples, axis=0) + radius),
    }


def _boxes_intersect(first, second):
    return bool(np.all(first[0] <= second[1])
                and np.all(second[0] <= first[1]))
