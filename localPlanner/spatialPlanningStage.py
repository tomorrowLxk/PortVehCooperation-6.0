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
    polyline_length,
    project_to_polyline,
    resample_polyline,
    slice_polyline,
    truncate_polyline,
)
from localPlanner.pathSpeedPlanner import (
    build_speed_limit_path,
    stationary_path,
)
from localPlanner.spatialPathPlanner import PathCandidate, plan_spatial_path
from localPlanner.trafficSignalPathProcessor import apply_traffic_signal
from parameters import (
    FUTURE_PATH_LENGTH,
    FUTURE_PATH_POINT_SPACING,
    HV_PREDICTION_HORIZON,
    PATH_CURVE_SAMPLE_SPACING,
    PATH_STATIC_STOP_DISTANCE,
    REVERSE_POINT_REACH_TOLERANCE,
    REVERSE_POINT_STOP_SPEED,
)


@dataclass
class VehicleSpatialResult:
    success: bool
    futurePath: np.ndarray
    candidatePaths: list[np.ndarray] = field(default_factory=list)
    drivableLines: tuple[np.ndarray, ...] = ()
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
        response, dynamic_info, static_info, planning_max_speed, route, polygons,
        route_progress_cache, route_lane_ids, lane_signals,
        centerline_dict, connector_dict):
    """生成一辆HV的futurePath和唯一预测path。"""
    position = np.asarray(dynamic_info["position"], dtype=float)[:2]
    heading = float(dynamic_info["heading"])
    speed = max(0.0, float(dynamic_info["speed"]))

    if not response.get("success", True):
        route_xy = clean_xy(route) if route is not None else np.empty((0, 2))
        if len(route_xy) >= 2:
            append_response_message(
                response, "短路径降级：启用历史长路径。")
        else:
            append_response_message(
                response, "短路径降级：无历史长路径，启用匀速预测。")

    future_path, future_reaches_route_end = _build_hv_future_path(
        position, heading, speed, route, route_progress_cache)
    local_path, terminal_stop = _build_hv_local_path(
        future_path, speed, polygons, static_info,
        future_reaches_route_end)
    signal_candidate = apply_traffic_signal(
        PathCandidate(
            local_path, 0.0, (), 0.0, terminal_stop,
            "ROUTE_END_OR_STATIC_OBSTACLE" if terminal_stop else None,
        ),
        centerline_dict,
        connector_dict,
        lane_signals,
        route_lane_ids,
    )
    if speed <= 1e-9:
        speed_path = stationary_path(position)
    else:
        speed_path = build_speed_limit_path(
            signal_candidate.xy,
            planning_max_speed,
            signal_candidate.terminalStop,
        )
    return VehicleSpatialResult(True, future_path, [speed_path])


def plan_agv_spatial(
        response, dynamic_info, static_info, planning_max_speed,
        drivable_area_planner, reverse_point, polygons, lane_signals):
    """生成一辆AGV的futurePath和空间path候选，并原地追加本阶段状态。"""
    position = np.asarray(dynamic_info["position"], dtype=float)[:2]
    heading = float(dynamic_info["heading"])
    speed = max(0.0, float(dynamic_info["speed"]))

    if drivable_area_planner is None:
        reason = "NO_DRIVABLE_AREA_PLANNER"
        mark_response_failure(
            response, reason, "空间规划失败：无可行域规划器。")
        return VehicleSpatialResult(
            False, position.reshape(1, 2),
            [stationary_path(position)],
            failureReason=reason)

    degraded = not response.get("success", True)
    if degraded:
        append_response_message(
            response, "短路径降级：启用历史可行域规划上下文。")

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
    except Exception as error:
        reason = "DRIVABLE_AREA_PLANNING_FAILED"
        mark_response_failure(
            response, reason, f"局部可行域规划失败：{error}")
        return VehicleSpatialResult(
            False, position.reshape(1, 2),
            [stationary_path(position)],
            failureReason=reason,
            reversePointReached=reached)

    future_path, stop_at_route_end = _build_agv_future_path(
        position, drivable_area_planner, active_reverse_point)
    if not drivable_lines:
        reason = "NO_DRIVABLE_LINE"
        mark_response_failure(
            response, reason, "局部可行域规划失败：未生成可行域线")
        return VehicleSpatialResult(
            False,
            future_path,
            [stationary_path(position)],
            failureReason=reason,
            reversePointReached=reached)

    try:
        path_result = plan_spatial_path(
            drivable_lines=drivable_lines,
            current_position=position,
            current_heading=heading,
            reference_line=future_path,
            polygons=polygons,
            vehicle_static_info=static_info,
            vehicle_dynamic_info=dynamic_info,
            reverse_point=active_reverse_point,
            stop_at_route_end=stop_at_route_end,
        )
    except Exception as error:
        reason = "SPATIAL_PATH_SEARCH_FAILED"
        mark_response_failure(
            response, reason, f"空间路径搜索失败：{error}")
        return VehicleSpatialResult(
            False,
            future_path,
            [stationary_path(position)],
            tuple(np.asarray(line, dtype=float).copy()
                  for line in drivable_lines),
            reason,
            reached,
        )

    if not path_result.success:
        mark_response_failure(
            response,
            "SPATIAL_PATH_SEARCH_FAILED",
            f"空间路径搜索失败：{path_result.failureReason or 'NO_SPATIAL_PATH'}",
        )

    signal_candidates = [
        apply_traffic_signal(
            candidate,
            drivable_area_planner.centerline_dict,
            drivable_area_planner.connector_dict,
            lane_signals,
        )
        for candidate in path_result.candidates
    ]
    candidates = [
        build_speed_limit_path(
            candidate.xy,
            planning_max_speed,
            candidate.terminalStop,
        )
        for candidate in signal_candidates
    ]
    if not candidates:
        candidates = [stationary_path(position)]
    return VehicleSpatialResult(
        path_result.success,
        future_path,
        candidates,
        tuple(np.asarray(line, dtype=float).copy()
              for line in drivable_lines),
        path_result.failureReason,
        reached,
    )

def _build_agv_future_path(position, planner, reverse_point):
    route = clean_xy(planner.referLine)
    start_s = float(planner.lastRouteS)
    end_s = min(start_s + FUTURE_PATH_LENGTH, polyline_length(route))
    if reverse_point is not None and not planner.passedReversePoint:
        reverse_s = planner.routeStageBounds[planner.reverseStageIndex][1]
        end_s = min(end_s, reverse_s)
    future = slice_polyline(route, start_s, end_s)
    if len(future) == 0:
        future = position.reshape(1, 2)
    elif np.linalg.norm(future[0] - position) > 1e-6:
        future = np.vstack((position, future))
    future = resample_polyline(future, FUTURE_PATH_POINT_SPACING)
    stop_at_route_end = (
        reverse_point is None
        and end_s >= polyline_length(route) - 1e-6
    )
    return future, stop_at_route_end


def _build_hv_future_path(
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
        future = slice_polyline(route_xy, start_s, end_s)
        if np.linalg.norm(future[0] - position) > 1e-6:
            future = np.vstack((position, future))
        return (
            resample_polyline(future, FUTURE_PATH_POINT_SPACING),
            end_s >= polyline_length(route_xy) - 1e-6,
        )

    distance = max(speed * HV_PREDICTION_HORIZON,
                   FUTURE_PATH_POINT_SPACING)
    direction = np.asarray([math.cos(heading), math.sin(heading)])
    return np.vstack((position, position + distance * direction)), False


def _build_hv_local_path(
        future_path, speed, polygons, static_info,
        future_reaches_route_end):
    expected_distance = speed * HV_PREDICTION_HORIZON
    local_path = truncate_polyline(future_path, expected_distance)
    terminal_stop = (
        future_reaches_route_end
        and expected_distance >= polyline_length(future_path) - 1e-6
    )
    stop_s = first_polygon_stop_s(
        local_path,
        polygons,
        static_info["length"],
        0.0,
        PATH_STATIC_STOP_DISTANCE,
        PATH_CURVE_SAMPLE_SPACING,
    )
    if stop_s is None:
        return local_path, terminal_stop
    return truncate_polyline(local_path, stop_s), True


def _reverse_point_reached(position, speed, reverse_point):
    return (
        reverse_point is not None
        and np.linalg.norm(
            position - np.asarray(reverse_point, dtype=float)[:2]
        ) <= REVERSE_POINT_REACH_TOLERANCE
        and speed <= REVERSE_POINT_STOP_SPEED
    )
