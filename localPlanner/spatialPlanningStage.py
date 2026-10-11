"""第五阶段的单车空间规划入口。循环由PlanningEngine负责。"""

from dataclasses import dataclass, field
import math

import numpy as np

from localPlanner.pathCollision import (
    first_polygon_stop_s,
    point_in_polygon,
    polygon_xy,
    vehicle_polygon,
)
from localPlanner.pathGeometry import (
    clean_xy,
    point_at_s,
    polyline_length,
    project_points_to_polyline,
    project_to_polyline,
    resample_polyline,
    slice_polyline,
    tangent_at_s,
    truncate_polyline,
)
from localPlanner.pathSpeedPlanner import (
    build_speed_limit_path,
    stationary_path,
)
from localPlanner.spatialPathPlanner import plan_spatial_path
from parameters import (
    FUTURE_PATH_LENGTH,
    FUTURE_PATH_POINT_SPACING,
    HV_PREDICTION_HORIZON,
    PATH_CURVE_SAMPLE_SPACING,
    PATH_CONTINUATION_COMPARE_DISTANCE,
    PATH_CONTINUATION_MAX_DEVIATION,
    PATH_STATIC_STOP_DISTANCE,
    REVERSE_POINT_REACH_TOLERANCE,
    REVERSE_POINT_STOP_SPEED,
)


@dataclass
class VehicleSpatialResult:
    success: bool
    candidatePaths: list[np.ndarray] = field(default_factory=list)
    drivableLines: tuple[np.ndarray, ...] = ()
    searchNodePaths: tuple[np.ndarray, ...] = ()
    failureReason: str | None = None
    reversePointReached: bool = False


def append_response_message(response, message):
    """保留已有信息，只追加当前阶段的处理结果。"""
    message = str(message).strip()
    if not message:
        return
    previous = str(response.get("message") or "").strip()
    if message in previous:
        return
    response["message"] = f"{previous}；{message}" if previous else message


def mark_response_failure(response, code, message):
    """失败状态只能由成功变为失败，已有错误码不被后续阶段覆盖。"""
    if response.get("success", True):
        response["success"] = False
        response["code"] = code
    append_response_message(response, message)


def build_obstacle_polygons(
        obstacle_map, vehicle_obstacle_caches,
        vehicle_dynamic_caches, vehicle_static_caches):
    """整理本帧全部电子围栏和已确认死车polygon。"""
    polygons = [
        polygon_xy(obstacle)
        for obstacle in obstacle_map.values()
    ]

    for vehicle_id, obstacle_state in vehicle_obstacle_caches.items():
        vehicle_id = str(vehicle_id)
        if not obstacle_state.get("dead", False):
            continue
        dynamic = vehicle_dynamic_caches[vehicle_id]
        static = vehicle_static_caches[vehicle_id]
        polygons.append(vehicle_polygon(
            dynamic["position"], dynamic["heading"],
            static["length"], static["width"],
            dynamic.get("longitudinalUncertainty", 0.0),
            dynamic.get("lateralUncertainty", 0.0),
        ))

    return [polygon for polygon in polygons if len(polygon) >= 3]


def exclude_vehi_polygons(polygons, dynamic_info):
    """排除包含当前车辆位置的polygon。"""
    current = np.asarray(dynamic_info["position"], dtype=float)[:2]
    return [
        polygon for polygon in polygons
        if not point_in_polygon(current, polygon)
    ]


def plan_hv_spatial(
        dynamic_info, static_info, planning_max_speed, route, polygons,
        route_progress_cache):
    """有route时按固定空间长度预测，否则按当前速度预测固定时长。"""
    position = np.asarray(dynamic_info["position"], dtype=float)[:2]
    heading = float(dynamic_info["heading"])
    speed = max(0.0, float(dynamic_info["speed"]))

    base_path, reaches_route_end = _build_hv_base_path(
        position, heading, speed, route, route_progress_cache)
    base_path, terminal_stop = _apply_hv_obstacles(
        base_path, polygons, static_info, reaches_route_end)
    speed_path = build_speed_limit_path(
        base_path,
        planning_max_speed,
        terminal_stop,
    )
    return VehicleSpatialResult(
        success=True,
        candidatePaths=[speed_path],
    )


def plan_agv_spatial(
        dynamic_info, static_info, planning_max_speed,
        drivable_area_planner, reverse_point, polygons,
        previous_base_path=None, prefer_previous_path=False):
    """生成一辆AGV的basePath候选，不写接口response。"""
    position = np.asarray(dynamic_info["position"], dtype=float)[:2]
    heading = float(dynamic_info["heading"])
    speed = max(0.0, float(dynamic_info["speed"]))

    if drivable_area_planner is None:
        reason = "NO_DRIVABLE_AREA_PLANNER"
        return VehicleSpatialResult(
            success=False,
            candidatePaths=[stationary_path(position)],
            failureReason=reason,
        )

    reached = _reverse_point_reached(position, speed, reverse_point)
    active_reverse_point = reverse_point
    if reached:
        drivable_area_planner.set_progress_at_point(reverse_point)
        active_reverse_point = None

    try:
        drivable_lines = drivable_area_planner.plan(
            position,
            currentHeading=heading,
            currentSpeed=speed,
            reversePoint=active_reverse_point,
        )
    except Exception:
        reason = "DRIVABLE_AREA_PLANNING_FAILED"
        return VehicleSpatialResult(
            success=False,
            candidatePaths=[stationary_path(position)],
            failureReason=reason,
            reversePointReached=reached,
        )

    reference_line, stop_at_route_end = _build_agv_reference_line(
        position, drivable_area_planner, active_reverse_point)
    if not drivable_lines:
        reason = "NO_DRIVABLE_LINE"
        return VehicleSpatialResult(
            success=False,
            candidatePaths=[stationary_path(position)],
            failureReason=reason,
            reversePointReached=reached,
        )

    try:
        path_result = plan_spatial_path(
            drivable_lines=drivable_lines,
            current_position=position,
            current_heading=heading,
            reference_line=reference_line,
            polygons=polygons,
            vehicle_static_info=static_info,
            vehicle_dynamic_info=dynamic_info,
            reverse_point=active_reverse_point,
            stop_at_route_end=stop_at_route_end,
            continuation_path=previous_base_path,
        )
    except Exception:
        reason = "SPATIAL_PATH_SEARCH_FAILED"
        return VehicleSpatialResult(
            success=False,
            candidatePaths=[stationary_path(position)],
            drivableLines=tuple(
                np.asarray(line, dtype=float).copy()
                for line in drivable_lines
            ),
            failureReason=reason,
            reversePointReached=reached,
        )

    searchNodePaths = tuple(
        np.asarray(nodePath, dtype=float).copy()
        for nodePath in path_result.searchNodePaths
    )
    candidates = [
        build_speed_limit_path(
            candidate.xy,
            planning_max_speed,
            candidate.terminalStop,
        )
        for candidate in path_result.candidates
    ]
    if prefer_previous_path:
        candidates = _prefer_continuous_candidates(
            candidates, previous_base_path, position)
    if not candidates:
        candidates = [stationary_path(position)]
    return VehicleSpatialResult(
        success=path_result.success,
        candidatePaths=candidates,
        drivableLines=tuple(
            np.asarray(line, dtype=float).copy()
            for line in drivable_lines
        ),
        searchNodePaths=searchNodePaths,
        failureReason=path_result.failureReason,
        reversePointReached=reached,
    )


def _prefer_continuous_candidates(candidates, previous_base_path, position):
    "历史路径将执行完时，优先保留仍沿其未执行换道段前进的安全候选。"
    if previous_base_path is None or len(candidates) < 2:
        return candidates

    previous = clean_xy(previous_base_path)
    if len(previous) < 2:
        return candidates
    start_s = project_to_polyline(position, previous)['s']
    remaining = slice_polyline(
        previous, start_s, polyline_length(previous))
    compare_length = min(
        PATH_CONTINUATION_COMPARE_DISTANCE,
        polyline_length(remaining),
    )
    if compare_length <= 1e-6:
        return candidates
    reference = slice_polyline(remaining, 0.0, compare_length)

    continuous = []
    others = []
    for candidate in candidates:
        candidate_xy = clean_xy(candidate)
        candidate_prefix = slice_polyline(
            candidate_xy,
            0.0,
            min(compare_length, polyline_length(candidate_xy)),
        )
        max_deviation = float(np.max(
            project_points_to_polyline(
                candidate_prefix, reference)['distance']
        ))
        if max_deviation <= PATH_CONTINUATION_MAX_DEVIATION + 1e-6:
            continuous.append(candidate)
        else:
            others.append(candidate)
    return continuous + others if continuous else candidates


def _build_agv_reference_line(position, planner, reverse_point):
    route = clean_xy(planner.referLine)
    route_length = polyline_length(route)
    start_s = float(planner.lastRouteS)
    phase_end_s = route_length
    if reverse_point is not None and not planner.passedReversePoint:
        phase_end_s = min(
            phase_end_s,
            float(planner.routeStageBounds[planner.reverseStageIndex][1]),
        )

    end_s = min(start_s + FUTURE_PATH_LENGTH, phase_end_s)
    for stage_index, (_, stage_end_s) in enumerate(planner.routeStageBounds):
        if end_s > float(stage_end_s) + 1e-6:
            continue
        if any(
                lane_id in planner.junctionLaneIds
                for lane_id in planner.routeStages[stage_index]):
            end_s = min(float(stage_end_s), phase_end_s)
        break

    reference = slice_polyline(route, start_s, end_s)
    if len(reference) == 0:
        reference = position.reshape(1, 2)
    reference = resample_polyline(
        reference, FUTURE_PATH_POINT_SPACING)
    stop_at_route_end = (
            reverse_point is None
            and end_s >= route_length - 1e-6
    )
    return reference, stop_at_route_end


def _build_hv_base_path(
        position, heading, speed, route, route_progress_cache):
    route_xy = clean_xy(route) if route is not None else np.empty((0, 2))
    if len(route_xy) >= 2:
        projection = project_to_polyline(position, route_xy)
        start_s = max(
            float(projection["s"]),
            float(route_progress_cache.get("routeS", 0.0)),
        )
        route_progress_cache["routeS"] = start_s
        end_s = min(
            start_s + FUTURE_PATH_LENGTH,
            polyline_length(route_xy),
        )
        sample_s = np.r_[
            np.arange(start_s, end_s, FUTURE_PATH_POINT_SPACING),
            end_s,
        ]
        lateral = float(projection["d"])
        future = []
        for value in sample_s:
            center = point_at_s(route_xy, value)
            tangent = tangent_at_s(route_xy, value)
            normal = np.asarray([-tangent[1], tangent[0]])
            future.append(center + lateral * normal)
        future = np.asarray(future, dtype=float)
        future[0] = position
        return future, end_s >= polyline_length(route_xy) - 1e-6

    prediction_distance = speed * HV_PREDICTION_HORIZON
    if prediction_distance <= 1e-9:
        return position.reshape(1, 2), False

    direction = np.asarray([math.cos(heading), math.sin(heading)])
    future = np.vstack((
        position,
        position + prediction_distance * direction,
    ))
    return resample_polyline(future, FUTURE_PATH_POINT_SPACING), False


def _apply_hv_obstacles(
        base_path, polygons, static_info, reaches_route_end):
    stop_s = first_polygon_stop_s(
        base_path,
        polygons,
        static_info["length"],
        0.0,
        PATH_STATIC_STOP_DISTANCE,
        PATH_CURVE_SAMPLE_SPACING,
    )
    if stop_s is None:
        return base_path, reaches_route_end
    return truncate_polyline(base_path, stop_s), True


def _reverse_point_reached(position, speed, reverse_point):
    return (
            reverse_point is not None
            and np.linalg.norm(
        position - np.asarray(reverse_point, dtype=float)[:2]
    ) <= REVERSE_POINT_REACH_TOLERANCE
            and speed <= REVERSE_POINT_STOP_SPEED
    )
