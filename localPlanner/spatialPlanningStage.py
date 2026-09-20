"""Stage 5: future paths and per-vehicle spatial candidate paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np

from localPlanner.hvPathPredictor import predict_hv_path
from localPlanner.pathSpeedPlanner import (
    build_speed_limit_path, stationary_path)
from localPlanner.spatialPathPlanner import (
    AGVPathPlanningJob, plan_spatial_paths_batch)
from tools.default_params import DEFAULT_PLANNING_PARAMS
from tools.obstacle_selection import select_vehicle_obstacles
from tools.path_geometry import (
    as_xy, extract_future_path, polyline_length, truncate_polyline)
from tools.reverse_point_processing import reverse_point_reached


@dataclass
class SpatialPlanningStageOutput:
    # All published paths are simple arrays. Candidate/HV paths are N x 3
    # [x, y, vMax]; future paths are N x 2.
    hvLocalPaths: Dict[str, np.ndarray] = field(default_factory=dict)
    agvCandidatePaths: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    futurePaths: Dict[str, np.ndarray] = field(default_factory=dict)
    # 仅供诊断/可视化：复制本帧规划过程中已经计算出的局部可行域。
    drivableLines: Dict[str, tuple[np.ndarray, ...]] = field(default_factory=dict)


def prepare_spatial_paths(
    requests: Sequence[Any],
    final_result: List[Any],
    vehicle_route_caches: Mapping[str, Any],
    vehicle_dynamic_caches: Mapping[str, Dict[str, Any]],
    vehicle_static_caches: Mapping[str, Dict[str, Any]],
    drivable_area_planners: Mapping[str, Any],
    reverse_point_caches: MutableMapping[str, Any],
    common_obstacles: Sequence[Any],
    directional_signals: Sequence[Any],
    route_progress_caches: MutableMapping[str, Any] | None = None,
) -> SpatialPlanningStageOutput:
    """Update final_result failures in place and return only spatial arrays."""
    params = DEFAULT_PLANNING_PARAMS
    if route_progress_caches is None:
        route_progress_caches = {}
    output = SpatialPlanningStageOutput()
    jobs: List[AGVPathPlanningJob] = []
    job_context: Dict[str, tuple] = {}
    future_length = float(params["futurePathLength"])
    future_spacing = float(params["futurePathSpacing"])

    for request_index, request in enumerate(requests):
        vehicle_id = str(_field(request, "deviceId"))
        vehicle_type = str(_field(request, "deviceType"))
        if vehicle_type == "AGV":
            output.drivableLines[vehicle_id] = ()
        dynamic = vehicle_dynamic_caches.get(vehicle_id, {})
        static = vehicle_static_caches.get(vehicle_id, {})
        response = (final_result[request_index]
                    if request_index < len(final_result) else None)
        position = None
        dynamic["inDxmArea"] = False
        try:
            long_success = _success(response)
            # 缺失BSM属于输入不完整：只更新该车response并退出，不生成占位
            # futurePath/path/waypoint，也不让该车进入后续协同分组与优先级计算。
            dynamic_valid = bool(dynamic) and bool(dynamic.get("valid", False))
            if not dynamic_valid:
                _mark_failure(
                    response,
                    "未获取到该车辆有效BSM动态状态",
                    code="MISSING_VEHICLE_DYNAMIC_INFO",
                )
                continue
            route = vehicle_route_caches.get(vehicle_id) if long_success else None
            position = _vehicle_position(dynamic, request)
            if position is None:
                raise ValueError(
                    f"车辆{vehicle_id}既没有有效BSM位置，也没有有效request起点")

            area_planner = drivable_area_planners.get(vehicle_id)
            heading = _number(dynamic.get("heading"), 0.0)
            speed = max(0.0, _number(dynamic.get("speed"), 0.0))
            reverse_point = reverse_point_caches.get(vehicle_id)
            drivable_lines = None
            if vehicle_type == "AGV" and area_planner is not None and long_success:
                if reverse_point_reached(
                        position, speed, reverse_point,
                        float(params["reversePointReachTolerance"]),
                        float(params["reversePointStopSpeed"])):
                    area_planner.set_progress_at_point(reverse_point)
                    reverse_point_caches.pop(vehicle_id, None)
                    reverse_point = None
                drivable_lines = area_planner.plan(
                    position, currentHeading=heading,
                    currentSpeed=speed, reversePoint=reverse_point)
                output.drivableLines[vehicle_id] = tuple(
                    np.asarray(line, dtype=float).copy()
                    for line in drivable_lines)
            dynamic["inDxmArea"] = bool(
                vehicle_type == "AGV" and area_planner is not None
                and str(getattr(area_planner, "lastLaneId", "")).endswith(
                    ("__DXM_FORWARD", "__DXM_REVERSE")))
            route_s = getattr(area_planner, "lastRouteS", None)
            future_path = extract_future_path(
                route=route,
                current_position=position,
                horizon=future_length,
                spacing=future_spacing,
                fallback_heading=heading,
                fallback_speed=speed,
                fallback_prediction_horizon=float(params["hvPredictionHorizon"]),
                route_s_hint=route_s,
                route_search_window=(0.0 if vehicle_type == "AGV" and route_s is not None
                                     else max(10.0, 2.0*speed)),
                initial_search_length=50.0,
                progress_cache=(route_progress_caches.setdefault(vehicle_id, {})
                                if vehicle_type != "AGV" else None),
            )
            future_path = _two_row_xy(future_path, position)
            output.futurePaths[vehicle_id] = future_path

            visible_obstacles = select_vehicle_obstacles(
                common_obstacles, directional_signals, future_path,
                heading_window=float(params["trafficSignalHeadingWindow"]),
                heading_after_window=float(params["trafficSignalHeadingAfterWindow"]),
                turn_threshold_deg=float(params["trafficSignalTurnThresholdDeg"]),
                uturn_threshold_deg=float(params["trafficSignalUTurnThresholdDeg"]),
                signal_group_radius=float(params["trafficSignalGroupRadius"]),
            )
            longitudinal_uncertainty = max(0.0, _number(
                dynamic.get("longitudinalUncertainty"), 0.0))
            lateral_uncertainty = max(0.0, _number(
                dynamic.get("lateralUncertainty"), 0.0))

            if vehicle_type != "AGV":
                prediction_horizon = float(params["hvPredictionHorizon"])
                expected_distance = speed*prediction_horizon
                expected_path = truncate_polyline(future_path, expected_distance)
                predicted = predict_hv_path(
                    future_path=future_path,
                    speed=speed,
                    prediction_horizon=prediction_horizon,
                    obstacles=visible_obstacles,
                    vehicle_length=float(static.get(
                        "length", params["vehicleLength"])),
                    vehicle_width=float(static.get(
                        "width", params["vehicleWidth"])),
                    longitudinal_uncertainty=longitudinal_uncertainty,
                    lateral_uncertainty=lateral_uncertainty,
                    safety_margin=float(params["pathObstacleSafetyMargin"]),
                    collision_check_spacing=float(params["pathCollisionCheckSpacing"]),
                )
                stopped = (speed <= 1e-8 or
                           polyline_length(predicted)+future_spacing
                           < polyline_length(expected_path))
                output.hvLocalPaths[vehicle_id] = build_speed_limit_path(
                    predicted if len(as_xy(predicted)) else position.reshape(1, 2),
                    static, terminal_stop=stopped, stationary=speed <= 1e-8,
                    params=params)
                continue

            if not long_success:
                _record_agv_stop(output, vehicle_id, position)
                continue
            route_xy = as_xy(route)
            if area_planner is None or len(route_xy) == 0:
                _mark_failure(response, "NO_ROUTE_OR_DRIVABLE_AREA_PLANNER")
                _record_agv_stop(output, vehicle_id, position)
                continue

            stop_at_route_end = (
                np.linalg.norm(future_path[-1]-route_xy[-1])
                <= future_spacing+1e-6)
            jobs.append(AGVPathPlanningJob(
                vehicleId=vehicle_id,
                drivableAreaPlanner=area_planner,
                position=position,
                heading=heading,
                speed=speed,
                globalReferenceLine=future_path,
                obstacles=visible_obstacles,
                vehicleStaticInfo=static,
                longitudinalUncertainty=longitudinal_uncertainty,
                lateralUncertainty=lateral_uncertainty,
                reversePoint=reverse_point,
                stopAtRouteEnd=stop_at_route_end,
                drivableLines=drivable_lines,
            ))
            job_context[vehicle_id] = (static, position, response)
        except Exception as error:
            _mark_failure(response, f"空间规划失败: {error}", code="SPATIAL_PLANNING_FAILED")
            if position is not None and np.all(np.isfinite(position)):
                output.futurePaths[vehicle_id] = np.tile(position, (2, 1))
                if vehicle_type == "AGV":
                    _record_agv_stop(output, vehicle_id, position)
                else:
                    output.hvLocalPaths[vehicle_id] = stationary_path(position)

    results = plan_spatial_paths_batch(jobs, params)
    for vehicle_id, result in results.items():
        static, position, response = job_context[vehicle_id]
        try:
            paths = [build_speed_limit_path(
                candidate.xy, static,
                terminal_stop=candidate.terminalStop,
                stationary=len(as_xy(candidate.xy)) < 2, params=params,
            ) for candidate in result.candidates]
            output.agvCandidatePaths[vehicle_id] = paths or [stationary_path(position)]
            if not result.success:
                _mark_failure(response, result.failureReason or "NO_SPATIAL_PATH")
        except Exception as error:
            _record_agv_stop(output, vehicle_id, position)
            _mark_failure(response, f"空间限速处理失败: {error}")
    return output


def _record_agv_stop(output, vehicle_id, position):
    output.agvCandidatePaths[vehicle_id] = [stationary_path(position)]


def _two_row_xy(path, fallback_position):
    points = as_xy(path)
    if len(points) == 0:
        points = np.asarray(fallback_position, dtype=float).reshape(1, 2)
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    return points


def _vehicle_position(dynamic, request):
    position = np.asarray(
        dynamic.get("position", [np.nan, np.nan]), dtype=float).reshape(-1)[:2]
    if position.shape == (2,) and np.all(np.isfinite(position)):
        return position
    start = _field(_field(request, "startPosition"), "position")
    if start is None:
        return None
    fallback = np.asarray([
        _number(_field(start, "x"), float("nan")),
        _number(_field(start, "y"), float("nan")),
    ], dtype=float)
    return fallback if np.all(np.isfinite(fallback)) else None


def _mark_failure(response, reason, code="SHORT_PATH_PLANNING_FAILED"):
    if response is None:
        return
    previous = _field(response, "message")
    if not _success(response) and previous:
        reason = str(previous) + "; " + str(reason)
    if isinstance(response, dict):
        response["success"] = False
        response["code"] = str(code)
        response["message"] = str(reason)
    else:
        response.success = False
        response.code = str(code)
        response.message = str(reason)


def _success(value):
    if isinstance(value, dict):
        return bool(value.get("success", False))
    return bool(getattr(value, "success", False))


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default) if value is not None else default


def _number(value, default):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)
