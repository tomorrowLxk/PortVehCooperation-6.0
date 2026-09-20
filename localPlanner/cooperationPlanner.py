"""Stages 6--8: grouping, default path choice and physical precedence."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from localPlanner.pathSpeedPlanner import stationary_path
from tools.default_params import DEFAULT_PLANNING_PARAMS
from tools.path_geometry import (
    as_xy, extend_polyline_ends, polyline_length, polyline_min_distance,
    project_to_polyline)


def group_by_future_paths(
    future_paths: Mapping[str, np.ndarray],
    vehicle_static_info: Mapping[str, Mapping[str, Any]],
    vehicle_dynamic_info: Mapping[str, Mapping[str, Any]],
) -> List[List[str]]:
    """第六阶段入口；复用几何分组，同时保留已有调用名称。"""
    return group_by_paths(future_paths, vehicle_static_info, vehicle_dynamic_info)


def group_by_paths(
    paths: Mapping[str, np.ndarray],
    vehicle_static_info: Mapping[str, Mapping[str, Any]],
    vehicle_dynamic_info: Mapping[str, Mapping[str, Any]],
) -> List[List[str]]:
    """按输入路径的XY分组；同时支持Nx2 futurePath与Nx3 selectedPath。"""
    params = DEFAULT_PLANNING_PARAMS
    vehicle_ids = list(paths)
    safety_margin = max(0.0, float(params["cooperationSafetyMargin"]))
    longitudinal_margin = max(0.0, float(params["cooperationStopBuffer"]))
    default_length = float(params["vehicleLength"])
    default_width = float(params["vehicleWidth"])
    corridor_paths = {}
    lateral_radii = {}
    path_aabbs = {}
    for vehicle_id in vehicle_ids:
        path_xy = as_xy(paths[vehicle_id])
        if len(path_xy) == 0:
            path_aabbs[vehicle_id] = None
            continue
        static = vehicle_static_info.get(vehicle_id, {})
        dynamic = vehicle_dynamic_info.get(vehicle_id, {})
        longitudinal_radius = (
            0.5*max(0.0, _number(static.get("length"), default_length))
            + _longitudinal_uncertainty(dynamic)
            + 0.5*longitudinal_margin)
        lateral_radius = (
            0.5*max(0.0, _number(static.get("width"), default_width))
            + _lateral_uncertainty(dynamic)
            + 0.5*safety_margin)
        corridor_path = extend_polyline_ends(
            path_xy, longitudinal_radius, longitudinal_radius,
            _number(dynamic.get("heading"), 0.0))
        corridor_paths[vehicle_id] = corridor_path
        lateral_radii[vehicle_id] = lateral_radius
        path_aabbs[vehicle_id] = (
            np.min(corridor_path, axis=0),
            np.max(corridor_path, axis=0),
        )
    parent = {vehicle_id: vehicle_id for vehicle_id in vehicle_ids}

    def find(vehicle_id):
        while parent[vehicle_id] != vehicle_id:
            parent[vehicle_id] = parent[parent[vehicle_id]]
            vehicle_id = parent[vehicle_id]
        return vehicle_id

    def union(first, second):
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first_index, first_id in enumerate(vehicle_ids):
        for second_id in vehicle_ids[first_index+1:]:
            first_path = corridor_paths[first_id]
            second_path = corridor_paths[second_id]
            threshold = lateral_radii[first_id]+lateral_radii[second_id]
            first_box = path_aabbs[first_id]
            second_box = path_aabbs[second_id]
            if first_box is None or second_box is None:
                continue
            first_lower, first_upper = first_box
            second_lower, second_upper = second_box
            axis_gap = np.maximum(
                0.0,
                np.maximum(first_lower-second_upper, second_lower-first_upper))
            if float(np.linalg.norm(axis_gap)) > threshold:
                continue

            if polyline_min_distance(first_path, second_path) <= threshold:
                union(first_id, second_id)

    blocks: Dict[str, List[str]] = {}
    for vehicle_id in vehicle_ids:
        blocks.setdefault(find(vehicle_id), []).append(vehicle_id)
    return list(blocks.values())


def select_default_paths(
    cooperation_blocks: Sequence[Sequence[str]],
    agv_candidate_paths: Mapping[str, Sequence[np.ndarray]],
    hv_local_paths: Mapping[str, np.ndarray],
    vehicle_dynamic_info: Mapping[str, Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """Stage-7 placeholder: choose the first candidate or the fixed HV path."""
    selected: Dict[str, np.ndarray] = {}
    for block in cooperation_blocks:
        for vehicle_id in block:
            if vehicle_id in agv_candidate_paths:
                candidates = list(agv_candidate_paths[vehicle_id])
                path = candidates[0] if candidates else None
            elif vehicle_id in hv_local_paths:
                path = hv_local_paths[vehicle_id]
            else:
                path = None
            if path is None or not _valid_speed_path(path):
                position = vehicle_dynamic_info.get(vehicle_id, {}).get("position")
                path = stationary_path(position)
            selected[vehicle_id] = np.asarray(path, dtype=float).copy()
    return selected


def determine_front_rear_priorities(
    cooperation_blocks: Sequence[Sequence[str]],
    selected_paths: Mapping[str, np.ndarray],
    vehicle_static_info: Mapping[str, Mapping[str, Any]],
    vehicle_dynamic_info: Mapping[str, Mapping[str, Any]],
) -> List[Tuple[str, str]]:
    """Return ``(front/higher id, rear/lower id)`` physical priorities."""
    params = DEFAULT_PLANNING_PARAMS
    output: List[Tuple[str, str]] = []
    seen = set()
    for block in cooperation_blocks:
        members = list(block)
        for first_index, first_id in enumerate(members):
            for second_id in members[first_index+1:]:
                first_ahead = _ahead_progress(
                    first_id, second_id, selected_paths,
                    vehicle_static_info, vehicle_dynamic_info, params)
                second_ahead = _ahead_progress(
                    second_id, first_id, selected_paths,
                    vehicle_static_info, vehicle_dynamic_info, params)
                relation = None
                if first_ahead is not None and second_ahead is None:
                    relation = (first_id, second_id)
                elif second_ahead is not None and first_ahead is None:
                    relation = (second_id, first_id)
                elif first_ahead is not None and second_ahead is not None:
                    minimum = float(params["frontRearMinProgress"])
                    if first_ahead > second_ahead+minimum:
                        relation = (first_id, second_id)
                    elif second_ahead > first_ahead+minimum:
                        relation = (second_id, first_id)
                if relation is not None and relation not in seen:
                    seen.add(relation)
                    output.append(relation)
    return output


def _ahead_progress(
    candidate_front_id,
    candidate_rear_id,
    selected_paths,
    static_info,
    dynamic_info,
    params,
):
    front_dynamic = dynamic_info.get(candidate_front_id, {})
    front_position = np.asarray(
        front_dynamic.get("position", [np.nan, np.nan]), dtype=float)[:2]
    rear_path = as_xy(selected_paths.get(candidate_rear_id))
    if (front_position.shape != (2,) or not np.all(np.isfinite(front_position))
            or len(rear_path) < 2 or polyline_length(rear_path) <= 1e-8):
        return None

    projection = project_to_polyline(front_position, rear_path)
    minimum_progress = float(params["frontRearMinProgress"])
    if float(projection["s"]) <= minimum_progress:
        return None
    # Reject a nearest endpoint created by clamping an out-of-segment
    # projection; the physical projection must lie on the selected path.
    residual = front_position-np.asarray(projection["point"], dtype=float)
    longitudinal_residual = abs(float(np.dot(residual, projection["tangent"])))
    if longitudinal_residual > minimum_progress:
        return None

    default_width = float(params["vehicleWidth"])
    front_width = max(0.0, _number(
        static_info.get(candidate_front_id, {}).get("width"), default_width))
    rear_width = max(0.0, _number(
        static_info.get(candidate_rear_id, {}).get("width"), default_width))
    threshold = (
        0.5*front_width
        + 0.5*rear_width
        + _lateral_uncertainty(front_dynamic)
        + _lateral_uncertainty(dynamic_info.get(candidate_rear_id, {})))
    if float(np.linalg.norm(residual)) > threshold:
        return None

    path_heading = math.atan2(
        float(projection["tangent"][1]), float(projection["tangent"][0]))
    vehicle_heading = _number(front_dynamic.get("heading"), 0.0)
    heading_error = abs(math.atan2(
        math.sin(vehicle_heading-path_heading),
        math.cos(vehicle_heading-path_heading)))
    if heading_error > math.radians(
            float(params["frontRearHeadingThresholdDeg"])):
        return None
    return float(projection["s"])


def _valid_speed_path(path):
    array = np.asarray(path, dtype=float)
    return (array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 3
            and np.all(np.isfinite(array[:, :3])))


def _lateral_uncertainty(dynamic):
    return max(0.0, _number(
        dynamic.get("lateralUncertainty"), 0.0))




def _longitudinal_uncertainty(dynamic):
    return max(0.0, _number(
        dynamic.get("longitudinalUncertainty"), 0.0))

def _number(value, default):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)
