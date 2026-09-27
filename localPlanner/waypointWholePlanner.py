"""第十一阶段：在合法waypointCandidate上顺序生成带时间的waypointWhole。"""

from dataclasses import dataclass
import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths,
    clean_xy,
    point_at_s,
    project_to_polyline,
    slice_polyline,
    tangent_at_s,
)
from localPlanner.pathSpeedPlanner import speed_envelope_values
from parameters import (
    COOPERATION_AEB_STATIC_GAP,
    COOPERATION_CONFLICT_SAMPLE_SPACING,
    COOPERATION_CONFLICT_SEGMENT_MERGE_DISTANCE,
    COOPERATION_EMERGENCY_DECELERATION,
    COOPERATION_LATERAL_MARGIN,
    COOPERATION_STATIC_GAP,
    COOPERATION_STOP_SPEED_THRESHOLD,
    COOPERATION_TIME_CLEARANCE,
    WAYPOINT_CANDIDATE_SPACING,
    WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE,
    WAYPOINT_WHOLE_REUSE_MAX_TIME_DIFFERENCE,
)


@dataclass(frozen=True)
class WaypointWhole:
    """一辆车本轮实际准备执行的候选点前缀及其时间、速度。"""

    vehicleId: str
    points: np.ndarray
    futurePathS: np.ndarray
    arrivalTimes: np.ndarray
    segmentSpeeds: np.ndarray
    pointSpeeds: np.ndarray
    terminalSpeed: float
    stopped: bool
    conflictPointSegments: tuple
    futurePath: np.ndarray
    motionS: np.ndarray
    motionTimes: np.ndarray
    motionPointSpeeds: np.ndarray
    motionSegmentSpeeds: np.ndarray
    motionSpeedLimits: np.ndarray
    candidateNodeIndices: np.ndarray

    def time_at_s(self, value):
        """按waypointWhole相邻点间的恒定等效速度插值到达时间。"""
        if len(self.motionS) <= 1:
            return 0.0
        target = float(np.clip(value, self.motionS[0], self.motionS[-1]))
        index = min(
            int(np.searchsorted(self.motionS, target, side='right') - 1),
            len(self.motionS) - 2,
        )
        index = max(index, 0)
        speed = float(self.motionSegmentSpeeds[index])
        if speed <= 1e-9:
            return float('inf')
        return float(
            self.motionTimes[index]
            + (target - self.motionS[index]) / speed
        )

    def speed_at_s(self, value):
        """返回包含指定futurePath位置的waypointWhole区间速度。"""
        if len(self.motionSegmentSpeeds) == 0:
            return float(self.terminalSpeed)
        target = float(np.clip(value, self.motionS[0], self.motionS[-1]))
        index = min(
            int(np.searchsorted(self.motionS, target, side='right') - 1),
            len(self.motionSegmentSpeeds) - 1,
        )
        return float(self.motionSegmentSpeeds[max(index, 0)])

    def state_at_time(self, value):
        """Return path position and interval speed at a relative time."""
        if len(self.motionS) <= 1:
            return float(self.motionS[0]), float(self.terminalSpeed)

        target = max(0.0, float(value))
        finite_times = np.flatnonzero(np.isfinite(self.motionTimes))
        if len(finite_times) == 0:
            return float(self.motionS[0]), 0.0
        last_finite = int(finite_times[-1])
        if target >= self.motionTimes[last_finite] - 1e-9:
            if last_finite >= len(self.motionS) - 1:
                return float(self.motionS[-1]), float(self.terminalSpeed)
            if not np.isfinite(self.motionTimes[last_finite + 1]):
                return float(self.motionS[last_finite]), 0.0

        index = int(np.searchsorted(
            self.motionTimes, target, side='right') - 1)
        index = max(0, min(index, len(self.motionSegmentSpeeds) - 1))
        speed = float(self.motionSegmentSpeeds[index])
        position = float(np.clip(
            self.motionS[index]
            + (target - self.motionTimes[index]) * speed,
            self.motionS[index],
            self.motionS[index + 1],
        ))
        return position, speed

    def as_array(self):
        """调试数组列为[x, y, futurePathS, arrivalTime, pointSpeed]。"""
        if len(self.points) == 0:
            return np.empty((0, 5), dtype=float)
        return np.column_stack((
            self.points,
            self.futurePathS,
            self.arrivalTimes,
            self.pointSpeeds,
        ))

    def as_dict(self):
        """返回不限定点数、明确区分点时间与区间速度的waypointWhole。"""
        return {
            'points': self.points.copy(),
            'futurePathS': self.futurePathS.copy(),
            'arrivalTimes': self.arrivalTimes.copy(),
            'segmentSpeeds': self.segmentSpeeds.copy(),
            'terminalSpeed': float(self.terminalSpeed),
            'stopped': bool(self.stopped),
            'conflictPointSegments': tuple(self.conflictPointSegments),
        }


def plan_block_waypoint_whole(
        priority_decision, future_paths, waypoint_candidates,
        vehicle_static_info, vehicle_dynamic_info):
    """按一个协同块的完整优先级由高到低生成waypointWhole。"""
    accepted = {}
    predecessors = _priority_predecessors(
        priority_decision.members,
        priority_decision.priorityEdges,
    )

    for vehicle_id in priority_decision.priorityOrder:
        prepared = _prepare_motion(
            vehicle_id,
            future_paths[vehicle_id],
            waypoint_candidates.get(vehicle_id),
            vehicle_static_info[vehicle_id],
            vehicle_dynamic_info[vehicle_id],
        )
        if vehicle_static_info[vehicle_id].get('vehicleType') != 'AGV':
            accepted[vehicle_id] = _build_hv_prediction(
                prepared,
                vehicle_static_info[vehicle_id],
                vehicle_dynamic_info[vehicle_id],
            )
            continue

        constraints = _build_candidate_constraints(
            vehicle_id,
            prepared,
            {
                high_id: accepted[high_id]
                for high_id in predecessors[vehicle_id]
                if high_id in accepted
            },
            vehicle_static_info,
            vehicle_dynamic_info,
        )
        accepted[vehicle_id] = _apply_candidate_constraints(
            prepared,
            constraints,
            vehicle_static_info[vehicle_id],
            vehicle_dynamic_info[vehicle_id],
        )

    return accepted


def _priority_predecessors(members, priority_edges):
    """返回每辆车在优先级图中的全部上游车辆；无优先级关系的车辆互不可见。"""
    reverse_graph = {str(vehicle_id): set() for vehicle_id in members}
    for high, low in priority_edges:
        high, low = str(high), str(low)
        if high in reverse_graph and low in reverse_graph:
            reverse_graph[low].add(high)

    output = {}
    for vehicle_id in reverse_graph:
        stack = list(reverse_graph[vehicle_id])
        predecessors = set()
        while stack:
            high = stack.pop()
            if high in predecessors:
                continue
            predecessors.add(high)
            stack.extend(reverse_graph[high])
        output[vehicle_id] = predecessors
    return output


def align_old_waypoint_whole(old_whole, current_position):
    "把历史waypointWhole临时裁剪到当前位置，并将剩余时间重新归零。"
    current_s = (
        project_to_polyline(current_position, old_whole.futurePath)['s']
        if len(old_whole.futurePath) >= 2 else 0.0
    )
    current_s = min(current_s, float(old_whole.motionS[-1]))
    current_t = old_whole.time_at_s(current_s)

    future_path = slice_polyline(
        old_whole.futurePath,
        current_s,
        float(old_whole.motionS[-1]),
    )
    current_xy = np.asarray(current_position, dtype=float)[:2]
    future_path[0] = current_xy

    motion_keep = old_whole.motionS > current_s + 1e-9
    motion_s = np.r_[0.0, old_whole.motionS[motion_keep] - current_s]
    motion_times = np.r_[
        0.0,
        old_whole.motionTimes[motion_keep] - current_t,
    ]
    current_speed = old_whole.speed_at_s(current_s)
    motion_point_speeds = np.r_[
        current_speed,
        old_whole.motionPointSpeeds[motion_keep],
    ]
    current_limit = float(np.interp(
        current_s,
        old_whole.motionS,
        old_whole.motionSpeedLimits,
    ))
    motion_speed_limits = np.r_[
        current_limit,
        old_whole.motionSpeedLimits[motion_keep],
    ]
    if len(motion_s) > 1:
        motion_segment_speeds = np.divide(
            np.diff(motion_s),
            np.diff(motion_times),
            out=np.zeros(len(motion_s) - 1, dtype=float),
            where=np.diff(motion_times) > 1e-9,
        )
    else:
        motion_segment_speeds = np.empty(0, dtype=float)

    point_keep = old_whole.futurePathS > current_s + 1e-9
    points = np.vstack((current_xy, old_whole.points[point_keep]))
    future_path_s = np.r_[
        0.0,
        old_whole.futurePathS[point_keep] - current_s,
    ]
    arrival_times = np.r_[
        0.0,
        old_whole.arrivalTimes[point_keep] - current_t,
    ]
    point_speeds = np.r_[
        current_speed,
        old_whole.pointSpeeds[point_keep],
    ]
    if len(future_path_s) > 1:
        segment_speeds = np.divide(
            np.diff(future_path_s),
            np.diff(arrival_times),
            out=np.zeros(len(future_path_s) - 1, dtype=float),
            where=np.diff(arrival_times) > 1e-9,
        )
    else:
        segment_speeds = np.empty(0, dtype=float)
    candidate_nodes = np.searchsorted(
        motion_s, future_path_s, side='left')
    candidate_nodes = np.minimum(candidate_nodes, len(motion_s) - 1)

    return WaypointWhole(
        vehicleId=old_whole.vehicleId,
        points=points,
        futurePathS=future_path_s,
        arrivalTimes=arrival_times,
        segmentSpeeds=segment_speeds,
        pointSpeeds=point_speeds,
        terminalSpeed=float(point_speeds[-1]),
        stopped=old_whole.stopped,
        conflictPointSegments=(),
        futurePath=future_path,
        motionS=motion_s,
        motionTimes=motion_times,
        motionPointSpeeds=motion_point_speeds,
        motionSegmentSpeeds=motion_segment_speeds,
        motionSpeedLimits=motion_speed_limits,
        candidateNodeIndices=candidate_nodes,
    )


def should_replace_waypoint_whole(aligned_old, new_whole):
    "比较同一起点、同一时间原点的新旧结果，判断是否需要重新下发。"
    if aligned_old.stopped != new_whole.stopped:
        return True

    old_length = float(aligned_old.motionS[-1])
    new_length = float(new_whole.motionS[-1])
    if abs(old_length - new_length) > WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE:
        return True

    common_length = min(old_length, new_length)
    sample_s = np.unique(np.r_[
        0.0,
        aligned_old.motionS[aligned_old.motionS <= common_length + 1e-9],
        new_whole.motionS[new_whole.motionS <= common_length + 1e-9],
        common_length,
    ])
    old_times = np.asarray([
        aligned_old.time_at_s(value) for value in sample_s
    ])
    new_times = np.asarray([
        new_whole.time_at_s(value) for value in sample_s
    ])
    if np.any(np.isfinite(old_times) != np.isfinite(new_times)):
        return True
    finite = np.isfinite(old_times) & np.isfinite(new_times)
    return bool(
        np.any(np.abs(old_times[finite] - new_times[finite])
               > WAYPOINT_WHOLE_REUSE_MAX_TIME_DIFFERENCE)
    )


def _build_hv_prediction(prepared, static, dynamic):
    "按HV当前速度生成时空占用预测，不接受协同降速或停车约束。"
    speed = max(0.0, min(
        float(dynamic['speed']),
        float(static['maxSpeed']),
    ))
    limits = np.full(len(prepared['nodeS']), speed, dtype=float)
    return _build_whole(prepared, limits, static, dynamic, ())


def _prepare_motion(vehicle_id, future_path, candidates, static, dynamic):
    path, source_s, source_limits = _prepare_speed_path(
        future_path, float(static['maxSpeed']))
    position = np.asarray(dynamic['position'], dtype=float)[:2]
    if len(path) == 0:
        path = np.asarray([[position[0], position[1]]], dtype=float)
        source_s = np.asarray([0.0])
        source_limits = np.asarray([0.0])

    raw = np.asarray(candidates, dtype=float) if candidates is not None else np.empty((0, 3))
    if raw.ndim != 2 or raw.shape[1] < 3:
        raw = np.empty((0, 3), dtype=float)
    if len(raw):
        raw = raw[np.all(np.isfinite(raw[:, :3]), axis=1)]
        raw = raw[np.argsort(raw[:, 2])]
        keep = np.concatenate((
            [True],
            (np.diff(raw[:, 2]) > 1e-6)
            | (np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1) > 1e-6),
        ))
        raw = raw[keep]

    if len(raw) == 0:
        raw = np.asarray([[position[0], position[1], 0.0]], dtype=float)
        node_s = np.asarray([0.0])
        node_xy = raw[:, :2].copy()
        candidate_nodes = np.asarray([0], dtype=int)
    else:
        raw[:, 2] = np.clip(raw[:, 2], 0.0, source_s[-1])
        motion_end = float(raw[-1, 2])
        node_s = np.unique(np.r_[
            0.0,
            source_s[source_s <= motion_end + 1e-9],
            raw[:, 2],
        ])
        node_xy = np.column_stack((
            np.interp(node_s, source_s, path[:, 0]),
            np.interp(node_s, source_s, path[:, 1]),
        ))
        candidate_nodes = np.searchsorted(node_s, raw[:, 2])

    limits = np.interp(node_s, source_s, source_limits)
    limits = np.minimum(limits, float(static['maxSpeed']))
    return {
        'vehicleId': str(vehicle_id),
        'futurePath': path,
        'candidates': raw,
        'candidateS': raw[:, 2].copy(),
        'nodeS': node_s,
        'nodeXY': node_xy,
        'candidateNodes': candidate_nodes,
        'speedLimits': np.maximum(0.0, limits),
    }


def _prepare_speed_path(path, maximum_speed):
    array = np.asarray(path, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    finite = np.all(np.isfinite(array[:, :2]), axis=1)
    array = array[finite]
    if len(array) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    keep = np.concatenate((
        [True], np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1) > 1e-9
    )) if len(array) > 1 else np.asarray([True])
    array = array[keep]
    xy = array[:, :2].copy()
    values = (
        np.maximum(0.0, array[:, 2])
        if array.shape[1] >= 3
        else np.full(len(array), maximum_speed, dtype=float)
    )
    return xy, arc_lengths(xy), values


def _build_candidate_constraints(
        low_id, prepared, accepted, static_info, dynamic_info):
    count = len(prepared['candidates'])
    required = np.full(count, -np.inf, dtype=float)
    speed_caps = np.full(count, np.inf, dtype=float)
    all_segments = []
    aeb_checks = []

    for high_id, high_whole in accepted.items():
        high_geometry = _prepare_collision_geometry(high_whole.futurePath)
        pair_required = np.full(count, np.nan, dtype=float)
        pair_speed_caps = np.full(count, np.nan, dtype=float)
        candidate_s = prepared['candidateS']
        segment_ranges = (
            [(0, 0)] if count == 1
            else [(index - 1, index) for index in range(1, count)]
        )
        for segment_number, (left, right) in enumerate(segment_ranges):
            start_s = float(candidate_s[left])
            end_s = float(candidate_s[right])
            if left == right:
                sample_s = np.asarray([start_s])
            else:
                segment_count = max(
                    1, int(math.ceil(
                        (end_s - start_s) / WAYPOINT_CANDIDATE_SPACING)))
                sample_s = np.linspace(
                    start_s, end_s, segment_count + 1)
                if segment_number > 0:
                    sample_s = sample_s[1:]

            for value in sample_s:
                low_point = point_at_s(prepared['futurePath'], value)
                low_heading = _path_heading(prepared['futurePath'], value)
                conflict = _first_collision_exit(
                    high_whole,
                    high_geometry,
                    low_point,
                    low_heading,
                    static_info[high_id],
                    dynamic_info[high_id],
                    static_info[low_id],
                    dynamic_info[low_id],
                    COOPERATION_STATIC_GAP,
                )
                if conflict is None:
                    continue

                exit_s, blocked = conflict
                high_speed = high_whole.speed_at_s(exit_s)
                required_time = (
                    float('inf') if blocked
                    else high_whole.time_at_s(exit_s)
                    + COOPERATION_TIME_CLEARANCE
                )
                if blocked or high_speed <= 1e-9:
                    left_required = required_time
                    right_required = required_time
                else:
                    left_required = max(
                        0.0,
                        required_time - (value - start_s) / high_speed,
                    )
                    right_required = (
                        required_time + (end_s - value) / high_speed)

                pair_required[left] = _maximum_optional(
                    pair_required[left], left_required)
                pair_required[right] = _maximum_optional(
                    pair_required[right], right_required)
                pair_speed_caps[left] = _minimum_optional(
                    pair_speed_caps[left], high_speed)
                pair_speed_caps[right] = _minimum_optional(
                    pair_speed_caps[right], high_speed)
                aeb_checks.append({
                    'highWhole': high_whole,
                    'highGeometry': high_geometry,
                    'highStatic': static_info[high_id],
                    'highDynamic': dynamic_info[high_id],
                    'lowS': float(value),
                    'stopIndex': int(
                        max(0, left - 1)
                        if value <= start_s + 1e-9 else left),
                })

        raw_mask = ~np.isnan(pair_required)
        if not np.any(raw_mask):
            continue
        segments = _conflict_point_segments(
            raw_mask, prepared['candidateS'])
        all_segments.extend(segments)
        _fill_segment_constraints(
            pair_required, pair_speed_caps,
            prepared['candidateS'], segments,
        )
        known = ~np.isnan(pair_required)
        required[known] = np.maximum(required[known], pair_required[known])
        capped = ~np.isnan(pair_speed_caps)
        speed_caps[capped] = np.minimum(
            speed_caps[capped], pair_speed_caps[capped])

    segments = _merge_point_segments(
        all_segments, prepared['candidateS'])
    return {
        'requiredTimes': required,
        'speedCaps': speed_caps,
        'segments': tuple(segments),
        'aebChecks': tuple(aeb_checks),
    }


def _maximum_optional(current, value):
    return (
        float(value) if np.isnan(current)
        else max(float(current), float(value))
    )


def _minimum_optional(current, value):
    return (
        float(value) if np.isnan(current)
        else min(float(current), float(value))
    )


def _conflict_point_segments(mask, candidate_s):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    starts = [int(indices[0])]
    ends = []
    for previous, current in zip(indices[:-1], indices[1:]):
        if current != previous + 1:
            ends.append(int(previous))
            starts.append(int(current))
    ends.append(int(indices[-1]))
    return _merge_point_segments(
        list(zip(starts, ends)), candidate_s)


def _merge_point_segments(segments, candidate_s):
    if not segments:
        return []
    ordered = sorted((int(start), int(end)) for start, end in segments)
    output = [list(ordered[0])]
    for start, end in ordered[1:]:
        previous = output[-1]
        gap = max(0.0, float(
            candidate_s[start] - candidate_s[previous[1]]))
        if (start <= previous[1] + 1
                or gap <= COOPERATION_CONFLICT_SEGMENT_MERGE_DISTANCE):
            previous[1] = max(previous[1], end)
        else:
            output.append([start, end])
    return [tuple(value) for value in output]


def _fill_segment_constraints(required, speed_caps, candidate_s, segments):
    raw_required = required.copy()
    raw_caps = speed_caps.copy()
    for start, end in segments:
        indices = np.arange(start, end + 1)
        known = indices[~np.isnan(raw_required[indices])]
        if len(known) == 0:
            continue
        required[known] = raw_required[known]
        missing = indices[np.isnan(raw_required[indices])]
        if len(known) == 1:
            required[indices] = raw_required[known[0]]
        elif len(missing):
            if np.all(np.isfinite(raw_required[known])):
                required[missing] = np.interp(
                    candidate_s[missing],
                    candidate_s[known],
                    raw_required[known],
                )
            else:
                for index in missing:
                    nearest = known[int(np.argmin(
                        np.abs(candidate_s[known] - candidate_s[index])
                    ))]
                    required[index] = raw_required[nearest]

        cap_known = indices[~np.isnan(raw_caps[indices])]
        if len(cap_known) == 1:
            speed_caps[indices] = raw_caps[cap_known[0]]
        elif len(cap_known) > 1:
            speed_caps[indices] = np.interp(
                candidate_s[indices],
                candidate_s[cap_known],
                raw_caps[cap_known],
            )


def _apply_candidate_constraints(prepared, constraints, static, dynamic):
    limits = prepared['speedLimits'].copy()
    candidate_nodes = prepared['candidateNodes']
    required = constraints['requiredTimes']
    speed_caps = constraints['speedCaps']
    segments = constraints['segments']

    for index, cap in enumerate(speed_caps):
        if np.isfinite(cap):
            limits[candidate_nodes[index]] = min(
                limits[candidate_nodes[index]], max(0.0, cap))

    whole = _build_whole(prepared, limits, static, dynamic, segments)
    for index, required_time in enumerate(required):
        if required_time == -np.inf:
            continue
        if (np.isinf(required_time)
                or (np.isfinite(speed_caps[index])
                    and speed_caps[index] < COOPERATION_STOP_SPEED_THRESHOLD)):
            return _stop_before_constraint(
                prepared, limits, index, segments, static, dynamic)
        if whole.arrivalTimes[index] >= required_time - 1e-6:
            continue

        node_index = int(candidate_nodes[index])
        lowest_limits = limits.copy()
        first_adjustable = 1 if node_index > 0 else 0
        lowest_limits[first_adjustable:node_index + 1] = np.minimum(
            lowest_limits[first_adjustable:node_index + 1],
            COOPERATION_STOP_SPEED_THRESHOLD,
        )
        lowest = _build_whole(
            prepared, lowest_limits, static, dynamic, segments)
        if lowest.arrivalTimes[index] < required_time - 1e-6:
            return _stop_before_constraint(
                prepared, limits, index, segments, static, dynamic)

        lower = COOPERATION_STOP_SPEED_THRESHOLD
        upper = float(np.max(limits[first_adjustable:node_index + 1]))
        selected_limits = lowest_limits
        selected = lowest
        for _ in range(18):
            middle = 0.5 * (lower + upper)
            candidate_limits = limits.copy()
            candidate_limits[first_adjustable:node_index + 1] = np.minimum(
                candidate_limits[first_adjustable:node_index + 1], middle)
            candidate = _build_whole(
                prepared, candidate_limits, static, dynamic, segments)
            if candidate.arrivalTimes[index] >= required_time:
                selected_limits = candidate_limits
                selected = candidate
                lower = middle
            else:
                upper = middle
        limits = selected_limits
        whole = selected

    unsafe_stop_index = _first_aeb_unsafe_stop_index(
        whole, prepared, constraints['aebChecks'], static, dynamic)
    if unsafe_stop_index is not None:
        return _stop_at_candidate(
            prepared,
            whole.motionSpeedLimits,
            unsafe_stop_index,
            segments,
            static,
            dynamic,
        )
    return whole


def _first_aeb_unsafe_stop_index(
        low_whole, prepared, checks, low_static, low_dynamic):
    for check in sorted(checks, key=lambda value: value['lowS']):
        low_s = float(check['lowS'])
        if low_s > low_whole.motionS[-1] + 1e-9:
            continue
        arrival_time = low_whole.time_at_s(low_s)
        if not np.isfinite(arrival_time):
            continue

        high_whole = check['highWhole']
        high_s, high_speed = high_whole.state_at_time(arrival_time)
        low_speed = low_whole.speed_at_s(low_s)
        delta_v = max(0.0, low_speed - high_speed)
        aeb_gap = (
            delta_v * delta_v
            / (2.0 * COOPERATION_EMERGENCY_DECELERATION)
            + COOPERATION_AEB_STATIC_GAP
        )
        if aeb_gap <= COOPERATION_STATIC_GAP + 1e-9:
            continue

        low_point = point_at_s(prepared['futurePath'], low_s)
        low_heading = _path_heading(prepared['futurePath'], low_s)
        if _vehicle_rectangles_overlap(
                high_whole.futurePath,
                high_s,
                check['highStatic'],
                check['highDynamic'],
                low_point,
                low_heading,
                low_static,
                low_dynamic,
                aeb_gap,
                check['highGeometry']):
            return int(check['stopIndex'])
    return None


def _stop_before_constraint(
        prepared, limits, constrained_index, segments, static, dynamic):
    # 候选点按顺序逐个判定：下一个点不可达时，在当前已可达候选点停车。
    # 第一个候选点即不可达时，没有更早候选点，只能在第一个点停车。
    stop_index = max(0, int(constrained_index) - 1)
    return _stop_at_candidate(
        prepared, limits, stop_index, segments, static, dynamic)


def _stop_at_candidate(
        prepared, limits, stop_index, segments, static, dynamic):
    stop_index = max(0, min(
        int(stop_index), len(prepared['candidates']) - 1))
    stop_node = int(prepared['candidateNodes'][stop_index])
    truncated = dict(prepared)
    truncated['candidates'] = prepared['candidates'][:stop_index + 1].copy()
    truncated['candidateS'] = prepared['candidateS'][:stop_index + 1].copy()
    truncated['nodeS'] = prepared['nodeS'][:stop_node + 1].copy()
    truncated['nodeXY'] = prepared['nodeXY'][:stop_node + 1].copy()
    truncated['candidateNodes'] = prepared['candidateNodes'][:stop_index + 1].copy()
    stop_limits = limits[:stop_node + 1].copy()
    stop_limits[-1] = 0.0
    retained_segments = tuple(
        (start, min(end, stop_index))
        for start, end in segments if start <= stop_index
    )
    return _build_whole(
        truncated, stop_limits, static, dynamic,
        retained_segments, stopped=True)


def _build_whole(
        prepared, speed_limits, static, dynamic,
        conflict_segments, stopped=False):
    node_s = prepared['nodeS']
    distances = np.diff(node_s)
    if len(node_s) == 1:
        speeds = np.asarray([
            0.0 if stopped else min(
                float(dynamic['speed']), float(speed_limits[0]))
        ])
        segment_speeds = np.empty(0, dtype=float)
        times = np.asarray([0.0])
    else:
        speeds = speed_envelope_values(
            speed_limits,
            distances,
            float(dynamic['speed']),
            float(static['maxAcceleration']),
            float(static['maxDeceleration']),
        )
        segment_speeds = 0.5 * (speeds[:-1] + speeds[1:])
        durations = np.divide(
            distances,
            segment_speeds,
            out=np.full(len(distances), np.inf, dtype=float),
            where=segment_speeds > 1e-9,
        )
        times = np.r_[0.0, np.cumsum(durations)]

    nodes = prepared['candidateNodes']
    points = prepared['candidates'][:, :2].copy()
    candidate_s = prepared['candidateS'].copy()
    arrival = times[nodes]
    point_speeds = speeds[nodes]
    if len(nodes) > 1:
        output_segment_speeds = np.asarray([
            segment_speeds[nodes[index]]
            for index in range(len(nodes) - 1)
        ], dtype=float)
    else:
        output_segment_speeds = np.empty(0, dtype=float)
    terminal_speed = float(point_speeds[-1]) if len(point_speeds) else 0.0
    return WaypointWhole(
        vehicleId=prepared['vehicleId'],
        points=points,
        futurePathS=candidate_s,
        arrivalTimes=arrival,
        segmentSpeeds=output_segment_speeds,
        pointSpeeds=point_speeds,
        terminalSpeed=terminal_speed,
        stopped=bool(stopped or terminal_speed <= 1e-9),
        conflictPointSegments=tuple(conflict_segments),
        futurePath=prepared['futurePath'],
        motionS=node_s.copy(),
        motionTimes=times,
        motionPointSpeeds=speeds,
        motionSegmentSpeeds=segment_speeds,
        motionSpeedLimits=np.asarray(speed_limits, dtype=float).copy(),
        candidateNodeIndices=nodes.copy(),
    )


def _first_collision_exit(
        high_whole, high_geometry, low_point, low_heading,
        high_static, high_dynamic, low_static, low_dynamic,
        longitudinal_gap):
    end_s = float(high_whole.motionS[-1])
    if end_s <= 1e-9:
        collides = _vehicle_rectangles_overlap(
            high_whole.futurePath, 0.0,
            high_static, high_dynamic,
            low_point, low_heading, low_static, low_dynamic,
            longitudinal_gap,
            high_geometry,
        )
        return (0.0, True) if collides else None

    spacing = max(0.1, COOPERATION_CONFLICT_SAMPLE_SPACING)
    samples = np.arange(0.0, end_s, spacing)
    samples = np.r_[samples, end_s]
    collision = np.asarray([
        _vehicle_rectangles_overlap(
            high_whole.futurePath, value,
            high_static, high_dynamic,
            low_point, low_heading, low_static, low_dynamic,
            longitudinal_gap,
            high_geometry,
        )
        for value in samples
    ], dtype=bool)
    entered = np.flatnonzero(collision)
    if len(entered) == 0:
        return None
    first = int(entered[0])
    exits = np.flatnonzero(~collision[first + 1:])
    if len(exits) == 0:
        return end_s, True

    upper_index = first + 1 + int(exits[0])
    lower = float(samples[upper_index - 1])
    upper = float(samples[upper_index])
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        if _vehicle_rectangles_overlap(
                high_whole.futurePath, middle,
                high_static, high_dynamic,
                low_point, low_heading, low_static, low_dynamic,
                longitudinal_gap,
                high_geometry):
            lower = middle
        else:
            upper = middle
    return upper, False


def _vehicle_rectangles_overlap(
        high_path, high_s, high_static, high_dynamic,
        low_point, low_heading, low_static, low_dynamic,
        longitudinal_gap, high_geometry=None):
    geometry = (
        high_geometry
        if high_geometry is not None
        else _prepare_collision_geometry(high_path)
    )
    high_center, high_tangent = _collision_state_at_s(
        geometry, high_s)
    high_heading = math.atan2(high_tangent[1], high_tangent[0])
    high_axes = _vehicle_axes(high_heading)
    low_axes = _vehicle_axes(low_heading)
    high_half = np.asarray([
        0.5 * float(high_static['length'])
        + float(high_dynamic.get('longitudinalUncertainty', 0.0)),
        0.5 * float(high_static['width'])
        + float(high_dynamic.get('lateralUncertainty', 0.0))
        + 0.5 * COOPERATION_LATERAL_MARGIN,
    ])
    low_half = np.asarray([
        0.5 * float(low_static['length'])
        + float(low_dynamic.get('longitudinalUncertainty', 0.0))
        + float(longitudinal_gap),
        0.5 * float(low_static['width'])
        + float(low_dynamic.get('lateralUncertainty', 0.0))
        + 0.5 * COOPERATION_LATERAL_MARGIN,
    ])
    difference = np.asarray(low_point, dtype=float)[:2] - high_center
    for axis in (
            high_axes[0], high_axes[1], low_axes[0], low_axes[1]):
        center_distance = abs(float(np.dot(difference, axis)))
        high_radius = (
            high_half[0] * abs(float(np.dot(high_axes[0], axis)))
            + high_half[1] * abs(float(np.dot(high_axes[1], axis)))
        )
        low_radius = (
            low_half[0] * abs(float(np.dot(low_axes[0], axis)))
            + low_half[1] * abs(float(np.dot(low_axes[1], axis)))
        )
        if center_distance > high_radius + low_radius + 1e-9:
            return False
    return True


def _prepare_collision_geometry(path):
    points = clean_xy(path)
    return points, arc_lengths(points)


def _collision_state_at_s(geometry, value):
    points, arc = geometry
    if len(points) < 2:
        return points[0].copy(), np.asarray([1.0, 0.0])
    target = float(np.clip(value, 0.0, arc[-1]))
    index = min(
        int(np.searchsorted(arc, target, side='right') - 1),
        len(points) - 2,
    )
    segment = points[index + 1] - points[index]
    length = float(arc[index + 1] - arc[index])
    tangent = segment / length
    ratio = (target - arc[index]) / length
    return points[index] + ratio * segment, tangent


def _vehicle_axes(heading):
    tangent = np.asarray([math.cos(heading), math.sin(heading)])
    normal = np.asarray([-tangent[1], tangent[0]])
    return tangent, normal


def _path_heading(path, value):
    tangent = tangent_at_s(path, float(value))
    return float(math.atan2(tangent[1], tangent[0]))
