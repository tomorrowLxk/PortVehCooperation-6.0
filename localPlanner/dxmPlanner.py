"""调箱门任务新增局部车道，并修复route最后一次转弯。"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from parameters import (
    DEFAULT_AGV_STATIC_INFO,
    DXM_CURVATURE_TOLERANCE,
    DXM_CURVE_CHECK_MIN_POINTS,
    DXM_CURVE_CHECK_SPACING,
    DXM_REVERSE_SEARCH_COUNT,
    DXM_REVERSE_SEARCH_MIN_STEP,
    DXM_REVERSE_SEARCH_RADIUS_RATIO,
    DXM_ROUTE_POINT_SPACING,
    DXM_TURN_LANE_THRESHOLD_DEG,
)

DEFAULT_MIN_TURNING_RADIUS = DEFAULT_AGV_STATIC_INFO["minTurningRadius"]
DEFAULT_TURN_THRESHOLD = math.radians(DXM_TURN_LANE_THRESHOLD_DEG)
DEFAULT_SPACING = DXM_ROUTE_POINT_SPACING


def fixed_dxm(
        vehLaneID: Sequence[str],
        vehRoute: Any,
        connector_dict: Mapping[str, Mapping[str, Any]],
        centerline_dict: Mapping[str, Any],
        vehicleStaticInfo: Optional[Mapping[str, Any]] = None,
        turnThreshold: float = DEFAULT_TURN_THRESHOLD,
        spacing: float = DEFAULT_SPACING,
):
    """用弱地图依据选择调箱门方案，并新增独立的前进、倒车车道。

    从route最后一个转弯向上游搜索。目标出口存在直行前驱时采用
    出口直线方案；否则存在反向转弯前驱时采用进口直线方案。所有
    转弯都没有依据时，在最后一个转弯处强制采用出口直线方案。
    地图车道只用于选择方案，不复用其中心线生成临时车道。
    """
    lane_ids = [str(lane_id) for lane_id in (vehLaneID or []) if lane_id is not None]
    route = np.asarray(vehRoute, dtype=float)[:, :2].copy()
    if not lane_ids:
        raise ValueError("调箱门route缺少laneId")
    if spacing <= 0.0 or not math.isfinite(float(spacing)):
        raise ValueError("spacing必须为正有限数")

    new_connector = {str(k): deepcopy(dict(v)) for k, v in connector_dict.items()}
    new_centerlines = {
        str(k): np.asarray(v, dtype=float)[:, :2].copy()
        for k, v in centerline_dict.items()
    }

    turns = _find_turn_lanes(
        lane_ids, route, new_centerlines, float(turnThreshold))
    if not turns:
        raise ValueError("长路径中没有找到可替换的转向lane")

    selected_turn = None
    selected_scheme = None
    for candidate_turn in turns:
        scheme = _map_supported_scheme(
            candidate_turn, lane_ids, new_connector,
            new_centerlines, float(turnThreshold))
        if scheme is not None:
            selected_turn = candidate_turn
            selected_scheme = scheme
            break

    # 所有转弯均无弱地图依据时，回到最后一个转弯强制执行旧方案。
    if selected_turn is None:
        selected_turn = turns[0]
        selected_scheme = "EXIT_STRAIGHT"

    turn_index, turn_lane_id, turn_line, theta0, theta1 = selected_turn
    p0 = turn_line[0].copy()
    p1 = turn_line[-1].copy()
    t0 = _unit(theta0)
    t1 = _unit(theta1)
    static_info = vehicleStaticInfo or {}
    min_turning_radius = _positive(
        static_info.get("minTurningRadius"), DEFAULT_MIN_TURNING_RADIUS
    )

    original_turn_angle = _wrap_angle(theta1 - theta0)
    removed_curve_length = _polyline_length(turn_line)
    if selected_scheme == "EXIT_STRAIGHT":
        forward_line, reverse_point = _build_forward_curve(
            p0=p0,
            p1=p1,
            t0=t0,
            t1=t1,
            t_reverse_heading=-t1,
            original_turn_angle=original_turn_angle,
            removed_curve_length=removed_curve_length,
            min_turning_radius=min_turning_radius,
            spacing=float(spacing),
        )
        reverse_line = _sample_straight(
            reverse_point, p1, float(spacing))
    else:
        forward_line, reverse_line, reverse_point = _build_entry_straight(
            p0=p0,
            p1=p1,
            t0=t0,
            t1=t1,
            original_turn_angle=original_turn_angle,
            removed_curve_length=removed_curve_length,
            min_turning_radius=min_turning_radius,
            spacing=float(spacing),
        )

    forward_lane_id = f"{turn_lane_id}__DXM_FORWARD"
    reverse_lane_id = f"{turn_lane_id}__DXM_REVERSE"
    old_info = deepcopy(new_connector.get(turn_lane_id, {}))
    predecessors = _lane_list(old_info.get("predecessor"))
    successors = _lane_list(old_info.get("successor"))
    if turn_index > 0 and lane_ids[turn_index - 1] not in predecessors:
        predecessors.append(lane_ids[turn_index - 1])
    if turn_index + 1 < len(lane_ids) and lane_ids[turn_index + 1] not in successors:
        successors.append(lane_ids[turn_index + 1])

    # 原lane与原拓扑全部保留；仅新增两条lane，route最后一次转弯改走它们。
    new_centerlines[forward_lane_id] = forward_line
    new_centerlines[reverse_lane_id] = reverse_line
    new_connector[forward_lane_id] = {
        "predecessor": [lane for lane in predecessors if lane in new_connector],
        "successor": [reverse_lane_id],
        "left_lane": None,
        "right_lane": None,
    }
    new_connector[reverse_lane_id] = {
        "predecessor": [forward_lane_id],
        "successor": [lane for lane in successors if lane in new_connector],
        "left_lane": None,
        "right_lane": None,
    }
    for predecessor in new_connector[forward_lane_id]["predecessor"]:
        _append_unique(new_connector[predecessor], "successor", forward_lane_id)
    for successor in new_connector[reverse_lane_id]["successor"]:
        _append_unique(new_connector[successor], "predecessor", reverse_lane_id)

    # 先沿laneId顺序定位到所选转弯前，不能用全局最近点误修第一次经过的位置。
    route_cursor = 0
    for lane_id in lane_ids[:turn_index]:
        line = new_centerlines[lane_id]
        route_cursor += _nearest_index(route[route_cursor:], line[-1])
    route_p0_index, route_p1_index = _ordered_route_indices(
        route, p0, p1, start_index=route_cursor)
    prefix = route[:route_p0_index]
    suffix = route[route_p1_index + 1:]
    new_route = _join_lines(prefix, forward_line, reverse_line, suffix)
    new_lane_ids = (
            lane_ids[:turn_index]
            + [forward_lane_id, reverse_lane_id]
            + lane_ids[turn_index + 1:]
    )

    return new_lane_ids, new_route, new_connector, new_centerlines, reverse_point.copy()


def _find_turn_lanes(lane_ids, route, centerlines, threshold):
    """按route末端到起点的顺序返回全部转弯车道。"""
    turns = []
    for index in range(len(lane_ids) - 1, -1, -1):
        lane_id = lane_ids[index]
        line = centerlines.get(lane_id)
        if line is None or len(line) < 3:
            continue
        oriented = _orient_line_to_route(line, route)
        theta0 = _endpoint_heading(oriented, at_start=True)
        theta1 = _endpoint_heading(oriented, at_start=False)
        if abs(_wrap_angle(theta1 - theta0)) >= threshold:
            turns.append((index, lane_id, oriented, theta0, theta1))
    return turns


def _map_supported_scheme(turn, lane_ids, connectors, centerlines, threshold):
    """仅根据目标出口的其他前驱转向类型选择方案，不复用其几何。"""
    turn_index, turn_lane_id, _, theta0, theta1 = turn
    turn_info = connectors.get(turn_lane_id, {})
    if turn_index + 1 < len(lane_ids):
        exit_lane_id = lane_ids[turn_index + 1]
    else:
        successors = _lane_list(turn_info.get("successor"))
        exit_lane_id = successors[0] if successors else None
    if exit_lane_id is None:
        return None

    exit_info = connectors.get(exit_lane_id, {})
    candidates = _lane_list(exit_info.get("predecessor"))
    original_angle = _wrap_angle(theta1 - theta0)
    has_opposite_turn = False
    for lane_id in candidates:
        if lane_id == turn_lane_id:
            continue
        line = centerlines.get(lane_id)
        if line is None or len(line) < 2:
            continue
        candidate_theta0 = _endpoint_heading(line, at_start=True)
        candidate_theta1 = _endpoint_heading(line, at_start=False)
        candidate_angle = _wrap_angle(candidate_theta1 - candidate_theta0)
        if abs(candidate_angle) < threshold:
            return "EXIT_STRAIGHT"
        if (abs(candidate_angle) < math.radians(150.0)
                and candidate_angle * original_angle < 0.0):
            has_opposite_turn = True
    return "ENTRY_STRAIGHT" if has_opposite_turn else None


def _orient_line_to_route(line, route):
    line = np.asarray(line, dtype=float)[:, :2]
    first_index = _nearest_index(route, line[0])
    last_index = _nearest_index(route, line[-1])
    return line if first_index <= last_index else line[::-1].copy()


def _endpoint_heading(line, at_start):
    """用端部局部曲线拟合估计切向，避免用长弦系统性低估转角。"""
    line = np.asarray(line, dtype=float)[:, :2]
    points = line[:min(5, len(line))] if at_start else line[-min(5, len(line)):]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-8))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("转向lane端部无法计算航向")
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    degree = min(3, len(points) - 1)
    coefficient_x = np.polyfit(s, points[:, 0], degree)
    coefficient_y = np.polyfit(s, points[:, 1], degree)
    target_s = 0.0 if at_start else float(s[-1])
    tangent = np.asarray([
        np.polyval(np.polyder(coefficient_x), target_s),
        np.polyval(np.polyder(coefficient_y), target_s),
    ])
    if np.linalg.norm(tangent) < 1e-9:
        tangent = points[1] - points[0] if at_start else points[-1] - points[-2]
    if np.linalg.norm(tangent) < 1e-9:
        raise ValueError("转向lane端部无法计算航向")
    return math.atan2(float(tangent[1]), float(tangent[0]))


def _build_forward_curve(
        p0, p1, t0, t1, t_reverse_heading, original_turn_angle,
        removed_curve_length, min_turning_radius, spacing,
):
    """粗搜索满足两端零曲率和最小转弯半径的五次Bezier曲线。"""
    projection_distance = max(0.0, float(np.dot(p1 - p0, t1)))
    minimum_reverse_distance = min(
        float(removed_curve_length),
        projection_distance + 0.5 * min_turning_radius,
    )
    initial_distance = max(spacing, minimum_reverse_distance)
    distance_step = max(
        DXM_REVERSE_SEARCH_MIN_STEP,
        DXM_REVERSE_SEARCH_RADIUS_RATIO * min_turning_radius,
    )
    search_count = int(DXM_REVERSE_SEARCH_COUNT)
    check_spacing = float(DXM_CURVE_CHECK_SPACING)
    check_min_points = int(DXM_CURVE_CHECK_MIN_POINTS)
    curvature_tolerance = float(DXM_CURVATURE_TOLERANCE)
    curvature_limit = 1.0 / max(min_turning_radius, 1e-6)

    # 每个R只检查少量对称/非对称控制柄，避免原来的二维细网格搜索。
    handle_scale_pairs = (
        (0.18, 0.18),
        (0.25, 0.25),
        (0.35, 0.35),
        (0.50, 0.50),
        (0.30, 0.13),
        (0.13, 0.30),
        (0.40, 0.20),
        (0.20, 0.40),
    )

    for attempt in range(search_count):
        reverse_distance = initial_distance + attempt * distance_step
        reverse_point = p1 - reverse_distance * t1
        chord = float(np.linalg.norm(reverse_point - p0))
        if chord < spacing:
            continue

        accepted = []
        for start_scale, end_scale in handle_scale_pairs:
            start_handle = max(
                0.5 * min_turning_radius, start_scale * chord)
            end_handle = max(
                0.5 * min_turning_radius, end_scale * chord)

            # B0/B1/B2位于起点切线上且等间距，B3/B4/B5位于终点
            # 切线上且等间距，因此两端二阶导数为0，端点曲率严格为0。
            controls = np.vstack((
                p0,
                p0 + start_handle * t0,
                p0 + 2.0 * start_handle * t0,
                reverse_point - 2.0 * end_handle * t_reverse_heading,
                reverse_point - end_handle * t_reverse_heading,
                reverse_point,
            ))
            sample_count = max(
                check_min_points,
                int(math.ceil(chord / check_spacing)),
            )
            dense = _sample_quintic_bezier(controls, sample_count)
            max_curvature, first_turn_sign = _curve_metrics(dense)
            if (not math.isfinite(max_curvature)
                    or max_curvature > curvature_limit * curvature_tolerance):
                continue
            if abs(original_turn_angle) < math.radians(150.0):
                original_sign = math.copysign(1.0, original_turn_angle)
                if first_turn_sign == 0.0 or first_turn_sign * original_sign >= 0.0:
                    continue
            if _has_self_intersection(dense):
                continue
            curve = _resample_polyline(dense, spacing)
            accepted.append((_polyline_length(curve), curve))

        if accepted:
            accepted.sort(key=lambda item: item[0])
            return accepted[0][1], reverse_point

    raise ValueError("无法在最小转弯半径和端点零曲率约束下生成调箱门前进曲线")


def _build_entry_straight(
        p0, p1, t0, t1, original_turn_angle,
        removed_curve_length, min_turning_radius, spacing,
):
    """方案二：从进口向前画直线，再生成R到出口的倒车曲线。"""
    # 复用方案一的曲线搜索：先按P1->R生成，再反转为R->P1。
    reversed_curve, reverse_point = _build_forward_curve(
        p0=p1,
        p1=p0,
        t0=-t1,
        t1=-t0,
        t_reverse_heading=t0,
        original_turn_angle=-original_turn_angle,
        removed_curve_length=removed_curve_length,
        min_turning_radius=min_turning_radius,
        spacing=spacing,
    )
    forward_line = _sample_straight(p0, reverse_point, spacing)
    reverse_curve = reversed_curve[::-1].copy()
    return forward_line, reverse_curve, reverse_point


def _sample_quintic_bezier(controls, sample_count):
    t = np.linspace(0.0, 1.0, max(6, sample_count))[:, None]
    omt = 1.0 - t
    return (
            omt ** 5 * controls[0]
            + 5.0 * omt ** 4 * t * controls[1]
            + 10.0 * omt ** 3 * t ** 2 * controls[2]
            + 10.0 * omt ** 2 * t ** 3 * controls[3]
            + 5.0 * omt * t ** 4 * controls[4]
            + t ** 5 * controls[5]
    )


def _curve_metrics(line):
    line = np.asarray(line, dtype=float)[:, :2]
    if len(line) < 3:
        return float("inf"), 0.0
    first = np.gradient(line, axis=0)
    second = np.gradient(first, axis=0)
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    speed = np.linalg.norm(first, axis=1)
    valid = speed > 1e-7
    curvature = np.full(len(line), np.inf)
    curvature[valid] = np.abs(cross[valid]) / np.maximum(speed[valid] ** 3, 1e-12)
    signed = np.zeros(len(line))
    signed[valid] = cross[valid] / np.maximum(speed[valid] ** 3, 1e-12)
    early = signed[1:max(3, min(len(line), len(line) // 5))]
    nonzero = early[np.abs(early) > 1e-6]
    first_sign = 0.0 if len(nonzero) == 0 else float(math.copysign(1.0, np.median(nonzero)))
    return float(np.max(curvature[valid])), first_sign


def _has_self_intersection(line):
    line = np.asarray(line, dtype=float)[:, :2]
    # 自交只需排除明显回折，最多取约25个点，避免候选搜索退化为高开销O(n²)。
    step = max(1, int(math.ceil((len(line) - 1) / 24.0)))
    sampled = line[::step]
    if not np.allclose(sampled[-1], line[-1]):
        sampled = np.vstack((sampled, line[-1]))
    for i in range(len(sampled) - 1):
        for j in range(i + 2, len(sampled) - 1):
            if i == 0 and j == len(sampled) - 2:
                continue
            if _segments_intersect(sampled[i], sampled[i + 1], sampled[j], sampled[j + 1]):
                return True
    return False


def _segments_intersect(a, b, c, d):
    def orient(p, q, r):
        qx, qy = float(q[0] - p[0]), float(q[1] - p[1])
        rx, ry = float(r[0] - p[0]), float(r[1] - p[1])
        return qx * ry - qy * rx

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 < -1e-9 and o3 * o4 < -1e-9


def _sample_straight(start, end, spacing):
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(distance / spacing)) + 1)
    return np.linspace(start, end, count)


def _resample_polyline(line, spacing):
    line = np.asarray(line, dtype=float)[:, :2]
    segment_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.vstack((line[0], line[-1]))
    sample_count = max(2, int(math.ceil(total / spacing)) + 1)
    targets = np.linspace(0.0, total, sample_count)
    x = np.interp(targets, cumulative, line[:, 0])
    y = np.interp(targets, cumulative, line[:, 1])
    return np.column_stack((x, y))


def _ordered_route_indices(route, p0, p1, start_index=0):
    first = start_index + _nearest_index(route[start_index:], p0)
    tail = route[first:]
    second = first + _nearest_index(tail, p1)
    if second < first:
        raise ValueError("无法在route中定位转向lane出口")
    return first, second


def _nearest_index(line, point):
    distances = np.linalg.norm(np.asarray(line)[:, :2] - np.asarray(point)[:2], axis=1)
    return int(np.argmin(distances))


def _join_lines(*parts):
    output = []
    for part in parts:
        if part is None:
            continue
        array = np.asarray(part, dtype=float)
        if array.size == 0:
            continue
        array = array.reshape((-1, array.shape[-1]))[:, :2]
        for point in array:
            if not output or np.linalg.norm(output[-1] - point) > 1e-8:
                output.append(point.copy())
    if len(output) < 2:
        raise ValueError("调箱门修复后的route不足两个点")
    return np.asarray(output, dtype=float)


def _append_unique(info, field, lane_id):
    values = _lane_list(info.get(field))
    if lane_id not in values:
        values.append(lane_id)
    info[field] = values


def _lane_list(values):
    return [str(value) for value in (values or []) if value is not None]


def _polyline_length(line):
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(line)[:, :2], axis=0), axis=1)))


def _unit(angle):
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=float)


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _positive(value, default):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) and result > 0.0 else default
