"""将一辆车的完整basePath转换为本帧信控处理后的futurePath。"""

import math

import numpy as np

from localPlanner.pathGeometry import (
    clean_xy,
    heading_at_s,
    polyline_length,
    project_to_polyline,
    project_points_to_polyline,
)
from localPlanner.pathSpeedPlanner import (
    build_speed_envelope,
    speed_envelope_values,
)
from parameters import (
    TRAFFIC_SIGNAL_GREEN_ETA_MARGIN,
    TRAFFIC_SIGNAL_HEADING_AFTER_WINDOW,
    TRAFFIC_SIGNAL_HEADING_WINDOW,
    TRAFFIC_SIGNAL_MATCH_DISTANCE,
    TRAFFIC_SIGNAL_STOP_DISTANCE,
    TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG,
    TRAFFIC_SIGNAL_UTURN_THRESHOLD_DEG,
)


RIGHT_TURN = "rightTurn"
LEFT_TURN = "leftTurn"
U_TURN = "UTurn"
STRAIGHT = "straight"
RED = "RED"
GREEN = "GREEN"


def build_future_path(
        base_path, current_position, current_speed,
        max_acceleration, max_deceleration,
        centerline_dict, connector_dict, lane_signals,
        base_path_lane_ids=None, junction_lane_ids=None):
    """裁剪已驶过部分、处理第一个信控并生成最终速度包络。"""
    path, point_lane_ids = _trim_passed_path(
        base_path,
        current_position,
        base_path_lane_ids,
    )
    if len(path) >= 2 and lane_signals:
        signal = _first_signal(
            path,
            centerline_dict,
            connector_dict,
            lane_signals,
            point_lane_ids,
            junction_lane_ids,
        )
        if signal is not None:
            stop_s, state = signal
            color = state['color']
            if color in (RED, GREEN):
                countdown = max(0.0, _signal_time(state))
                eta = _estimate_arrival_time(
                    path,
                    stop_s,
                    current_speed,
                    max_acceleration,
                    max_deceleration,
                )
                arrival_time = (
                    eta + TRAFFIC_SIGNAL_GREEN_ETA_MARGIN
                    if color == GREEN else eta
                )
                arrival_color = (
                    color if arrival_time <= countdown
                    else (GREEN if color == RED else RED)
                )
                if arrival_color == RED:
                    path = _slice_path(path, 0.0, stop_s)
                    path[-1, 2] = 0.0

    return build_speed_envelope(
        path,
        current_speed,
        max_acceleration,
        max_deceleration,
    )


def classify_path_lane_ids(path, centerline_dict, connector_dict):
    """在basePath生成时一次性确定每个离散点所属的车道。"""
    path_xy = clean_xy(path)
    if len(path_xy) < 2:
        return np.full(len(path_xy), None, dtype=object)
    return np.asarray(_classify_points(
        path_xy,
        centerline_dict,
        connector_dict,
        sorted(centerline_dict),
    ), dtype=object)


def _signal_time(state):
    value = state.get('time')
    if value is None:
        return float('inf')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('inf')


def _first_signal(
        path, centerline_dict, connector_dict, lane_signals,
        point_lane_ids=None, junction_lane_ids=None):
    """按path行驶顺序返回第一个信控的停车弧长和方向状态。"""
    path_xy = clean_xy(path)
    point_lanes = (
        classify_path_lane_ids(path_xy, centerline_dict, connector_dict)
        if point_lane_ids is None
        else np.asarray(point_lane_ids, dtype=object)
    )
    candidates = []
    for lane_id in lane_signals:
        indices = np.flatnonzero(point_lanes == lane_id)
        if len(indices) == 0:
            continue
        centerline = clean_xy(centerline_dict[lane_id])
        distances = np.linalg.norm(
            path_xy[indices] - centerline[-1], axis=1)
        matched = np.flatnonzero(
            distances <= TRAFFIC_SIGNAL_MATCH_DISTANCE)
        if len(matched):
            candidates.append((int(indices[matched[0]]), lane_id, centerline))

    if candidates:
        point_index, lane_id, centerline = min(
            candidates, key=lambda item: item[0])
        signal_s = project_to_polyline(centerline[-1], path_xy)['s']
        direction = _junction_movement_direction(
            point_lanes, point_index, lane_id, centerline,
            centerline_dict, junction_lane_ids,
        )
        if direction is None:
            direction = _movement_direction(path_xy, signal_s)
        state = lane_signals[lane_id].get(direction)
        if state is None:
            return None
        return max(0.0, signal_s - TRAFFIC_SIGNAL_STOP_DISTANCE), state
    return None


def _junction_movement_direction(
        point_lanes, point_index, incoming_lane_id, incoming_centerline,
        centerline_dict, junction_lane_ids):
    "按本次路径所经虚拟车道的完整转向判断信控方向。"
    if junction_lane_ids is None:
        return None
    junction_lanes = junction_lane_ids
    if not isinstance(junction_lanes, (set, frozenset)):
        junction_lanes = set(junction_lanes)
    for lane_id in point_lanes[point_index + 1:]:
        if lane_id is None or lane_id == incoming_lane_id:
            continue
        if lane_id not in junction_lanes:
            return None
        virtual_line = centerline_dict.get(lane_id)
        if virtual_line is None:
            return None
        virtual_xy = clean_xy(virtual_line)
        if len(virtual_xy) < 2:
            return None
        delta = _normalize_angle(
            heading_at_s(virtual_xy, polyline_length(virtual_xy))
            - heading_at_s(incoming_centerline,
                           polyline_length(incoming_centerline)))
        if abs(delta) >= math.radians(TRAFFIC_SIGNAL_UTURN_THRESHOLD_DEG):
            return U_TURN
        if delta > math.radians(TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG):
            return LEFT_TURN
        if delta < -math.radians(TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG):
            return RIGHT_TURN
        return STRAIGHT
    return None


def _trim_passed_path(
        path, current_position, base_path_lane_ids=None):
    values = np.asarray(path, dtype=float)
    position = np.asarray(current_position, dtype=float)[:2]
    path_length = polyline_length(values)
    if len(values) < 2 or path_length <= 1e-9:
        lane_ids = (
            None if base_path_lane_ids is None
            else np.asarray([base_path_lane_ids[0]], dtype=object)
        )
        return np.asarray([[position[0], position[1], 0.0]]), lane_ids

    progress = project_to_polyline(position, values)['s']
    output = _slice_path(values, progress, path_length)
    output[0, :2] = position
    lane_ids = (
        None if base_path_lane_ids is None
        else _slice_lane_ids(values, base_path_lane_ids, progress, path_length)
    )
    return output, lane_ids


def _slice_lane_ids(path, lane_ids, start_s, end_s):
    """按照与_slice_path完全相同的切片规则同步裁剪逐点laneId。"""
    values = np.asarray(path, dtype=float)
    labels = np.asarray(lane_ids, dtype=object)
    lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    start = float(np.clip(start_s, 0.0, arc[-1]))
    end = float(np.clip(end_s, start, arc[-1]))

    def label_at(target):
        index = min(
            int(np.searchsorted(arc, target, side='right') - 1),
            len(values) - 2,
        )
        ratio = (
            (target - arc[index]) / lengths[index]
            if lengths[index] > 1e-9 else 0.0
        )
        return labels[index] if ratio <= 0.5 else labels[index + 1]

    # 裁剪新增的当前位置点固定继承下游相邻点的laneId，避免车辆压过
    # 停车线后仍因起点保留进口道laneId而重复响应已经驶过的信号灯。
    start_index = min(
        int(np.searchsorted(arc, start, side='right') - 1),
        len(values) - 2,
    )
    output = [labels[start_index + 1]]
    output.extend(
        labels[index]
        for index in range(1, len(values) - 1)
        if start + 1e-9 < arc[index] < end - 1e-9
    )
    if end > start + 1e-9:
        output.append(label_at(end))
    return np.asarray(output, dtype=object)


def _estimate_arrival_time(
        path, stop_s, current_speed,
        max_acceleration, max_deceleration):
    prefix = _slice_path(path, 0.0, stop_s)
    if len(prefix) < 2:
        return 0.0

    distances = np.linalg.norm(np.diff(prefix[:, :2], axis=0), axis=1)
    speeds = np.maximum(0.0, prefix[:, 2].copy())
    speeds[0] = min(speeds[0], max(0.0, float(current_speed)))
    speeds = speed_envelope_values(
        speeds,
        distances,
        current_speed,
        max_acceleration,
        max_deceleration,
    )

    segment_speed_sums = speeds[:-1] + speeds[1:]
    if np.any(segment_speed_sums <= 1e-9):
        return math.inf
    return float(np.sum(2.0 * distances / segment_speed_sums))


def _slice_path(path, start_s, end_s):
    """按xy弧长切片，同时保留并插值path的其余列。"""
    values = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    start = float(np.clip(start_s, 0.0, arc[-1]))
    end = float(np.clip(end_s, start, arc[-1]))

    def row_at(target):
        index = min(
            int(np.searchsorted(arc, target, side='right') - 1),
            len(values) - 2,
        )
        ratio = (
            (target - arc[index]) / lengths[index]
            if lengths[index] > 1e-9 else 0.0
        )
        return values[index] + ratio * (values[index + 1] - values[index])

    output = [row_at(start)]
    output.extend(
        values[index].copy()
        for index in range(1, len(values) - 1)
        if start + 1e-9 < arc[index] < end - 1e-9
    )
    if end > start + 1e-9:
        output.append(row_at(end))
    return np.asarray(output, dtype=float)


def _classify_points(path, centerline_dict, connector_dict, lanes):
    scores = np.full((len(lanes), len(path)), np.inf, dtype=float)
    path_min = np.min(path, axis=0) - TRAFFIC_SIGNAL_MATCH_DISTANCE
    path_max = np.max(path, axis=0) + TRAFFIC_SIGNAL_MATCH_DISTANCE
    path_segments = np.diff(path, axis=0)
    path_headings = np.append(
        np.arctan2(path_segments[:, 1], path_segments[:, 0]),
        math.atan2(path_segments[-1, 1], path_segments[-1, 0]),
    )

    for lane_index, lane_id in enumerate(lanes):
        centerline = clean_xy(centerline_dict[lane_id])
        if (np.any(np.max(centerline, axis=0) < path_min)
                or np.any(np.min(centerline, axis=0) > path_max)):
            continue
        projection = project_points_to_polyline(path, centerline)
        lane_headings = np.arctan2(
            projection['tangent'][:, 1],
            projection['tangent'][:, 0],
        )
        heading_errors = np.abs(_normalize_angles(
            path_headings - lane_headings))
        valid = (
            (projection['distance'] <= TRAFFIC_SIGNAL_MATCH_DISTANCE)
            & (heading_errors <= math.pi / 2.0)
        )
        scores[lane_index, valid] = (
            projection['distance'][valid] + heading_errors[valid])

    result = []
    previous_index = None
    for point_index in range(len(path)):
        point_scores = scores[:, point_index]
        selected_index = int(np.argmin(point_scores))
        selected_score = point_scores[selected_index]
        if not np.isfinite(selected_score):
            result.append(None)
            previous_index = None
            continue

        if previous_index is not None:
            previous_score = point_scores[previous_index]
            if (np.isfinite(previous_score)
                    and previous_score <= selected_score + 0.5):
                selected_index = previous_index
            else:
                connected_indices = [
                    index for index, lane_id in enumerate(lanes)
                    if np.isfinite(point_scores[index])
                    and _connected(
                        lanes[previous_index], lane_id, connector_dict)
                ]
                if connected_indices:
                    selected_index = min(
                        connected_indices,
                        key=lambda index: (point_scores[index], lanes[index]),
                    )
        previous_index = selected_index
        result.append(lanes[selected_index])
    return result


def _connected(first, second, connector_dict):
    if first == second:
        return True
    info = connector_dict[first]
    return (
        info.get('left_lane', info.get('leftLane')) == second
        or info.get('right_lane', info.get('rightLane')) == second
        or second in info['successor']
        or second in info['predecessor']
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
