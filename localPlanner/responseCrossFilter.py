"""Optional spatial crossing filter applied only to newly issued AGV paths."""

import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths, clean_xy, project_to_polyline, resample_polyline,
    slice_polyline,
)
from localPlanner.waypointWholePlanner import _overlap_matrix
from parameters import COOPERATION_LATERAL_MARGIN


_SAMPLE_SPACING = 2.0


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
        vehicle_dynamic_info, vehicle_static_info, stop_clearance):
    """Return prefixes to issue; None means keep the old response, if any."""
    selected = {}
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
                selected[vehicle_id] = _trim_to_safe_prefix(
                    waypoints, dynamic, static, higher_geometries,
                    stop_clearance)
                active = selected[vehicle_id]
                if active is None:
                    active = cached_responses.get(vehicle_id)
            else:
                active = cached_responses.get(vehicle_id)
            if active is not None and len(active):
                geometry = _remaining_geometry(active, dynamic, static)
                if geometry is not None:
                    higher_geometries.append(geometry)

    # Vehicles outside a cooperation block have no higher-priority peer.
    for vehicle_id, waypoints in candidate_waypoints.items():
        if vehicle_id not in selected:
            selected[vehicle_id] = np.asarray(waypoints, dtype=float).copy()
    return selected


def _trim_to_safe_prefix(
        waypoints, dynamic, static, higher_geometries, stop_clearance):
    if len(waypoints) == 0:
        return None
    if not higher_geometries:
        return waypoints.copy()
    position = np.asarray(dynamic['position'], dtype=float)[:2]
    points = np.vstack((position, waypoints[:, :2]))
    waypoint_s = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(points, axis=0), axis=1))]
    low = _remaining_geometry(waypoints, dynamic, static)
    if low is None:
        return None
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
        return waypoints.copy()
    safe_end = first_conflict - max(0.0, float(stop_clearance))
    legal = np.flatnonzero(
        (waypoint_s[1:] <= safe_end + 1e-9)
        & (waypoint_s[1:] > 1e-9))
    if len(legal) == 0:
        return None
    return waypoints[:int(legal[-1]) + 1].copy()


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
