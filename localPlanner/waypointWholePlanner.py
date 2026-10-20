"""第十一阶段：在合法waypointCandidate上顺序生成带时间的waypointWhole。"""

from dataclasses import dataclass
import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths,
    clean_xy,
    project_to_polyline,
    slice_polyline,
)
from localPlanner.pathSpeedPlanner import speed_envelope_values
from parameters import (
    COOPERATION_AEB_STATIC_GAP,
    COOPERATION_CONFLICT_SEGMENT_MERGE_DISTANCE,
    COOPERATION_EMERGENCY_DECELERATION,
    COOPERATION_LATERAL_MARGIN,
    COOPERATION_STATIC_GAP,
    COOPERATION_STOP_SPEED_THRESHOLD,
    COOPERATION_TIME_CLEARANCE,
    WAYPOINT_CANDIDATE_SPACING,
    WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE,
    WAYPOINT_WHOLE_REUSE_MAX_SPATIAL_DEVIATION,
    WAYPOINT_WHOLE_REUSE_MAX_TIME_DIFFERENCE,
)


_TERMINAL_STOP_SPEED_THRESHOLD = 0.1
_ARC_LENGTH_MERGE_TOLERANCE = 1e-6


def _unique_sorted_values(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) <= 1:
        return ordered
    return ordered[np.r_[
        True,
        np.diff(ordered) > _ARC_LENGTH_MERGE_TOLERANCE,
    ]]


def _nearest_sorted_indices(values, queries):
    values = np.asarray(values, dtype=float)
    queries = np.asarray(queries, dtype=float)
    right = np.minimum(np.searchsorted(values, queries), len(values) - 1)
    left = np.maximum(right - 1, 0)
    use_left = (
        np.abs(queries - values[left])
        <= np.abs(values[right] - queries)
    )
    return np.where(use_left, left, right).astype(int)

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
        vehicle_static_info, vehicle_dynamic_info,
        select_effective_whole):
    """逐车生成新结果，立即选出供后车约束的实际结果。"""
    planned = {}
    accepted = {}
    accepted_geometry = {}
    predecessors = _priority_predecessors(
        priority_decision.members,
        priority_decision.priorityEdges,
    )
    constraining_vehicles = {
        high_id for high_ids in predecessors.values()
        for high_id in high_ids
    }

    for vehicle_id in priority_decision.priorityOrder:
        prepared = _prepare_motion(
            vehicle_id,
            future_paths[vehicle_id],
            waypoint_candidates.get(vehicle_id),
            vehicle_static_info[vehicle_id],
            vehicle_dynamic_info[vehicle_id],
        )
        if vehicle_static_info[vehicle_id].get('vehicleType') != 'AGV':
            new_whole = _build_hv_prediction(
                prepared,
                vehicle_static_info[vehicle_id],
                vehicle_dynamic_info[vehicle_id],
            )
        else:
            constraints = _build_candidate_constraints(
                prepared,
                {
                    high_id: accepted[high_id]
                    for high_id in predecessors[vehicle_id]
                    if high_id in accepted
                },
                {
                    high_id: accepted_geometry[high_id]
                    for high_id in predecessors[vehicle_id]
                    if high_id in accepted_geometry
                },
            )
            new_whole = _apply_candidate_constraints(
                prepared,
                constraints,
                vehicle_static_info[vehicle_id],
                vehicle_dynamic_info[vehicle_id],
            )

        planned[vehicle_id] = new_whole
        effective_whole, old_path_s = select_effective_whole(vehicle_id, new_whole)
        accepted[vehicle_id] = effective_whole
        if vehicle_id not in constraining_vehicles:
            continue
        geometry = prepared['spatialGeometry']
        if effective_whole is not new_whole:
            old_path = effective_whole.futurePath
            geometry = _prepare_spatial_geometry(
                old_path, old_path_s,
                effective_whole.futurePathS,
                vehicle_static_info[vehicle_id],
                vehicle_dynamic_info[vehicle_id],
            )
        accepted_geometry[vehicle_id] = _truncate_spatial_geometry(
            geometry, effective_whole.motionS[-1],
        )

    return planned, accepted


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


def align_old_waypoint_whole(old_whole, current_position, with_path_s=False):
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
    path_s = arc_lengths(future_path)
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
    candidate_nodes = _nearest_sorted_indices(motion_s, future_path_s)

    aligned = WaypointWhole(
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
    return (aligned, path_s) if with_path_s else aligned


def should_replace_waypoint_whole(aligned_old, new_whole):
    "比较同一起点、同一时间原点的新旧结果，判断是否需要重新下发。"
    if aligned_old.stopped != new_whole.stopped:
        return True

    old_length = float(aligned_old.motionS[-1])
    new_length = float(new_whole.motionS[-1])
    if abs(old_length - new_length) > WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE:
        return True

    common_length = min(old_length, new_length)
    sample_s = _unique_sorted_values(np.r_[
        0.0,
        aligned_old.motionS[aligned_old.motionS <= common_length + 1e-9],
        new_whole.motionS[new_whole.motionS <= common_length + 1e-9],
        common_length,
    ])
    old_times, _ = _whole_states_at_s(aligned_old, sample_s)
    new_times, _ = _whole_states_at_s(new_whole, sample_s)
    if np.any(np.isfinite(old_times) != np.isfinite(new_times)):
        return True
    finite = np.isfinite(old_times) & np.isfinite(new_times)
    return bool(
        np.any(np.abs(old_times[finite] - new_times[finite])
               > WAYPOINT_WHOLE_REUSE_MAX_TIME_DIFFERENCE)
    )


def _path_within_distance(source, target, max_distance):
    """Check vertices and short interpolated sections against another polyline."""
    source_s = arc_lengths(source)
    samples = np.unique(np.r_[
        source_s,
        np.arange(0.0, source_s[-1], max_distance),
    ])
    points, _ = _path_states_at_s(source, source_s, samples)
    if len(target) == 1:
        return bool(np.all(np.sum((points - target[0]) ** 2, axis=1)
                           <= max_distance ** 2 + 1e-9))

    starts = target[:-1]
    segments = target[1:] - starts
    lengths_sq = np.sum(segments ** 2, axis=1)
    relative = points[:, None, :] - starts[None, :, :]
    fractions = np.divide(
        np.sum(relative * segments[None, :, :], axis=2),
        lengths_sq[None, :],
        out=np.zeros((len(points), len(segments)), dtype=float),
        where=lengths_sq[None, :] > 1e-12,
    )
    closest = starts[None, :, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * segments
    distances_sq = np.sum((points[:, None, :] - closest) ** 2, axis=2)
    return bool(np.all(np.min(distances_sq, axis=1)
                       <= max_distance ** 2 + 1e-9))


def _spatially_consistent_whole_paths(
        old_whole, new_whole, current_position):
    """Compare complete futurePath geometry, independent of speed stop points."""
    old_path = clean_xy(old_whole.futurePath)
    new_path = clean_xy(new_whole.futurePath)
    if len(old_path) == 0 or len(new_path) == 0:
        return False
    if len(old_path) >= 2:
        current_s = project_to_polyline(current_position, old_path)['s']
        old_path = slice_polyline(old_path, current_s, arc_lengths(old_path)[-1])
    old_path[0] = np.asarray(current_position, dtype=float)[:2]
    tolerance = WAYPOINT_WHOLE_REUSE_MAX_SPATIAL_DEVIATION
    if len(old_path) == len(new_path) and np.all(
            np.sum((old_path - new_path) ** 2, axis=1)
            <= tolerance ** 2 + 1e-9):
        # Corresponding segments remain within tolerance under interpolation.
        return True
    return (
        _path_within_distance(old_path, new_path, tolerance)
        and _path_within_distance(new_path, old_path, tolerance)
    )


def select_effective_waypoint_whole(
        new_whole, old_whole, has_sparse_cache,
        must_dispatch, must_not_dispatch, current_position):
    """沿用下发判据，在本车规划后立即确定后车实际看到的结果。"""
    has_history = old_whole is not None and has_sparse_cache
    if must_dispatch or not has_history:
        return new_whole, None
    if not _spatially_consistent_whole_paths(
            old_whole, new_whole, current_position):
        return new_whole, None
    if must_not_dispatch and has_history:
        return align_old_waypoint_whole(
            old_whole, current_position, with_path_s=True,
        )
    aligned_old, old_path_s = align_old_waypoint_whole(
        old_whole, current_position, with_path_s=True,
    )
    if should_replace_waypoint_whole(aligned_old, new_whole):
        return new_whole, None
    return aligned_old, old_path_s


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
        motion_end = 0.0
    else:
        raw[:, 2] = np.clip(raw[:, 2], 0.0, source_s[-1])
        motion_end = float(raw[-1, 2])
        node_s = _unique_sorted_values(np.r_[
            0.0,
            source_s[source_s <= motion_end + 1e-9],
            raw[:, 2],
        ])
        node_xy = np.column_stack((
            np.interp(node_s, source_s, path[:, 0]),
            np.interp(node_s, source_s, path[:, 1]),
        ))
        candidate_nodes = _nearest_sorted_indices(node_s, raw[:, 2])


    limits = np.interp(node_s, source_s, source_limits)
    limits = np.minimum(limits, float(static['maxSpeed']))
    prepared = {
        'vehicleId': str(vehicle_id),
        'futurePath': path,
        'candidates': raw,
        'candidateS': raw[:, 2].copy(),
        'nodeS': node_s,
        'nodeXY': node_xy,
        'candidateNodes': candidate_nodes,
        'speedLimits': np.maximum(0.0, limits),
        'terminalStop': bool(
            source_limits[-1] <= 1e-9
            and motion_end >= source_s[-1] - 1e-3
        ),
    }
    prepared['spatialGeometry'] = _prepare_spatial_geometry(
        path, source_s, prepared['candidateS'], static, dynamic)
    return prepared


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


def _prepare_spatial_geometry(path, path_s, candidate_s, static, dynamic):
    sample_s, left, right, starts, ends = _constraint_sample_layout(candidate_s)
    centers, tangents = _path_states_at_s(path, path_s, sample_s)
    return {
        'path': path,
        'pathS': path_s,
        'sampleS': sample_s,
        'centers': centers,
        'tangents': tangents,
        'normals': np.column_stack((-tangents[:, 1], tangents[:, 0])),
        'leftCandidate': left,
        'rightCandidate': right,
        'segmentStartS': starts,
        'segmentEndS': ends,
        'halfLength': (
            0.5 * float(static['length'])
            + float(dynamic.get('longitudinalUncertainty', 0.0))
        ),
        'halfWidth': (
            0.5 * float(static['width'])
            + float(dynamic.get('lateralUncertainty', 0.0))
            + 0.5 * COOPERATION_LATERAL_MARGIN
        ),
    }


def _constraint_sample_layout(candidate_s):
    values, lefts, rights, starts, ends = [], [], [], [], []
    if len(candidate_s) <= 1:
        value = float(candidate_s[0]) if len(candidate_s) else 0.0
        return (
            np.asarray([value]),
            np.asarray([0], dtype=int),
            np.asarray([0], dtype=int),
            np.asarray([value]),
            np.asarray([value]),
        )

    for right in range(1, len(candidate_s)):
        left = right - 1
        start_s = float(candidate_s[left])
        end_s = float(candidate_s[right])
        segment_count = max(
            1, int(math.ceil(
                (end_s - start_s) / WAYPOINT_CANDIDATE_SPACING)))
        segment_samples = np.linspace(start_s, end_s, segment_count + 1)
        if right > 1:
            segment_samples = segment_samples[1:]
        values.extend(segment_samples)
        lefts.extend([left] * len(segment_samples))
        rights.extend([right] * len(segment_samples))
        starts.extend([start_s] * len(segment_samples))
        ends.extend([end_s] * len(segment_samples))
    return (
        np.asarray(values, dtype=float),
        np.asarray(lefts, dtype=int),
        np.asarray(rights, dtype=int),
        np.asarray(starts, dtype=float),
        np.asarray(ends, dtype=float),
    )


def _path_states_at_s(path, path_s, values):
    targets = np.clip(np.asarray(values, dtype=float), path_s[0], path_s[-1])
    if len(path) == 1:
        centers = np.repeat(path[:1], len(targets), axis=0)
        tangents = np.tile(np.asarray([[1.0, 0.0]]), (len(targets), 1))
        return centers, tangents

    indices = np.searchsorted(path_s, targets, side='right') - 1
    indices = np.clip(indices, 0, len(path) - 2)
    segments = path[indices + 1] - path[indices]
    lengths = path_s[indices + 1] - path_s[indices]
    ratios = np.divide(
        targets - path_s[indices], lengths,
        out=np.zeros(len(targets), dtype=float),
        where=lengths > 1e-12,
    )
    centers = path[indices] + ratios[:, None] * segments
    segment_norms = np.linalg.norm(segments, axis=1)
    tangents = np.divide(
        segments, segment_norms[:, None],
        out=np.tile(np.asarray([[1.0, 0.0]]), (len(targets), 1)),
        where=segment_norms[:, None] > 1e-12,
    )
    return centers, tangents


def _truncate_spatial_geometry(geometry, end_s):
    end_s = float(np.clip(end_s, geometry['pathS'][0], geometry['pathS'][-1]))
    keep = geometry['sampleS'] <= end_s + 1e-9
    sample_s = geometry['sampleS'][keep]
    centers = geometry['centers'][keep]
    tangents = geometry['tangents'][keep]
    normals = geometry['normals'][keep]
    if len(sample_s) == 0 or abs(sample_s[-1] - end_s) > 1e-9:
        end_center, end_tangent = _path_states_at_s(
            geometry['path'], geometry['pathS'], [end_s])
        sample_s = np.r_[sample_s, end_s]
        centers = np.vstack((centers, end_center))
        tangents = np.vstack((tangents, end_tangent))
        normals = np.vstack((
            normals,
            np.asarray([[-end_tangent[0, 1], end_tangent[0, 0]]]),
        ))
    return {
        'path': geometry['path'],
        'pathS': geometry['pathS'],
        'sampleS': sample_s,
        'centers': centers,
        'tangents': tangents,
        'normals': normals,
        'halfLength': geometry['halfLength'],
        'halfWidth': geometry['halfWidth'],
    }


def _build_candidate_constraints(prepared, accepted, accepted_geometry):
    count = len(prepared['candidates'])
    required = np.full(count, -np.inf, dtype=float)
    speed_caps = np.full(count, np.inf, dtype=float)
    all_segments = []
    aeb_checks = []

    low_geometry = prepared['spatialGeometry']
    for high_id, high_whole in accepted.items():
        high_geometry = accepted_geometry[high_id]
        pair_required = np.full(count, np.nan, dtype=float)
        pair_speed_caps = np.full(count, np.nan, dtype=float)
        conflict_s, blocked_mask = _last_conflicting_positions(
            low_geometry,
            high_geometry,
            high_whole.stopped,
            COOPERATION_STATIC_GAP,
        )
        conflict_indices = np.flatnonzero(np.isfinite(conflict_s))
        conflict_times, conflict_speeds = _whole_states_at_s(
            high_whole, conflict_s[conflict_indices])
        conflict_times += COOPERATION_TIME_CLEARANCE
        conflict_times[blocked_mask[conflict_indices]] = np.inf
        for position, sample_index in enumerate(conflict_indices):
            value = float(low_geometry['sampleS'][sample_index])
            left = int(low_geometry['leftCandidate'][sample_index])
            right = int(low_geometry['rightCandidate'][sample_index])
            start_s = float(low_geometry['segmentStartS'][sample_index])
            end_s = float(low_geometry['segmentEndS'][sample_index])
            blocked = bool(blocked_mask[sample_index])
            high_speed = float(conflict_speeds[position])
            required_time = float(conflict_times[position])
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
                'lowS': value,
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


def _whole_states_at_s(whole, values):
    targets = np.asarray(values, dtype=float)
    if len(whole.motionSegmentSpeeds) == 0:
        return (
            np.zeros(len(targets), dtype=float),
            np.full(len(targets), whole.terminalSpeed, dtype=float),
        )

    targets = np.clip(targets, whole.motionS[0], whole.motionS[-1])
    indices = np.searchsorted(whole.motionS, targets, side='right') - 1
    indices = np.clip(indices, 0, len(whole.motionSegmentSpeeds) - 1)
    speeds = whole.motionSegmentSpeeds[indices]
    times = whole.motionTimes[indices] + np.divide(
        targets - whole.motionS[indices],
        speeds,
        out=np.full(len(targets), np.inf, dtype=float),
        where=speeds > 1e-9,
    )
    return times, speeds


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
        lowest_arrival = _candidate_arrival_time(
            prepared, lowest_limits, static, dynamic, index)
        if lowest_arrival < required_time - 1e-6:
            return _stop_before_constraint(
                prepared, limits, index, segments, static, dynamic)

        lower = COOPERATION_STOP_SPEED_THRESHOLD
        upper = float(np.max(limits[first_adjustable:node_index + 1]))
        selected_limits = lowest_limits
        for _ in range(18):
            middle = 0.5 * (lower + upper)
            candidate_limits = limits.copy()
            candidate_limits[first_adjustable:node_index + 1] = np.minimum(
                candidate_limits[first_adjustable:node_index + 1], middle)
            candidate_arrival = _candidate_arrival_time(
                prepared, candidate_limits, static, dynamic, index)
            if candidate_arrival >= required_time:
                selected_limits = candidate_limits
                lower = middle
            else:
                upper = middle
        limits = selected_limits
        whole = _build_whole(
            prepared, limits, static, dynamic, segments)

    unsafe_stop_index = _first_aeb_unsafe_stop_index(
        whole, prepared, constraints['aebChecks'])
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


def _first_aeb_unsafe_stop_index(low_whole, prepared, checks):
    for check in sorted(checks, key=lambda value: value['lowS']):
        low_s = float(check['lowS'])
        if low_s > low_whole.motionS[-1] + 1e-9:
            continue
        arrival_time = low_whole.time_at_s(low_s)
        if not np.isfinite(arrival_time):
            continue

        high_whole = check['highWhole']
        if (not high_whole.stopped
                and arrival_time > high_whole.motionTimes[-1] + 1e-9):
            continue
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

        if _overlap_at_s(
                prepared['spatialGeometry'], low_s,
                check['highGeometry'], high_s,
                aeb_gap):
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
    nodes = prepared['candidateNodes']
    points = prepared['candidates'][:, :2].copy()
    candidate_s = prepared['candidateS'].copy()
    terminal_stop = bool(
        stopped
        or (len(speed_limits) > 0 and float(speed_limits[-1]) <= 1e-9)
        or prepared['terminalStop']
    )
    effective_speed_limits = np.asarray(speed_limits, dtype=float).copy()
    if terminal_stop and len(effective_speed_limits):
        effective_speed_limits[-1] = 0.0
    speeds, segment_speeds, times = _motion_profile(
        node_s, effective_speed_limits, static, dynamic, terminal_stop)
    if (not terminal_stop and len(speeds)
            and float(speeds[-1]) < _TERMINAL_STOP_SPEED_THRESHOLD):
        terminal_stop = True
        effective_speed_limits[-1] = 0.0
        speeds, segment_speeds, times = _motion_profile(
            node_s, effective_speed_limits, static, dynamic, True)
    arrival = times[nodes]
    point_speeds = speeds[nodes]
    if len(nodes) > 1:
        candidate_durations = np.diff(arrival)
        output_segment_speeds = np.divide(
            np.diff(candidate_s),
            candidate_durations,
            out=np.zeros(len(nodes) - 1, dtype=float),
            where=(
                np.isfinite(candidate_durations)
                & (candidate_durations > 1e-9)
            ),
        )
    else:
        output_segment_speeds = np.empty(0, dtype=float)
    terminal_speed = float(point_speeds[-1]) if len(point_speeds) else 0.0
    if terminal_stop:
        terminal_speed = 0.0
        if len(point_speeds):
            point_speeds[-1] = 0.0
    return WaypointWhole(
        vehicleId=prepared['vehicleId'],
        points=points,
        futurePathS=candidate_s,
        arrivalTimes=arrival,
        segmentSpeeds=output_segment_speeds,
        pointSpeeds=point_speeds,
        terminalSpeed=terminal_speed,
        stopped=terminal_stop,
        conflictPointSegments=tuple(conflict_segments),
        futurePath=prepared['futurePath'],
        motionS=node_s.copy(),
        motionTimes=times,
        motionPointSpeeds=speeds,
        motionSegmentSpeeds=segment_speeds,
        motionSpeedLimits=effective_speed_limits,
        candidateNodeIndices=nodes.copy(),
    )


def _candidate_arrival_time(
        prepared, speed_limits, static, dynamic, candidate_index):
    _, _, times = _motion_profile(
        prepared['nodeS'], speed_limits, static, dynamic, False)
    return float(times[prepared['candidateNodes'][candidate_index]])


def _motion_profile(node_s, speed_limits, static, dynamic, stopped):
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
        speeds[np.asarray(speed_limits, dtype=float) <= 1e-9] = 0.0
        segment_speeds = 0.5 * (speeds[:-1] + speeds[1:])
        durations = np.divide(
            distances,
            segment_speeds,
            out=np.full(len(distances), np.inf, dtype=float),
            where=segment_speeds > 1e-9,
        )
        times = np.r_[0.0, np.cumsum(durations)]
    return speeds, segment_speeds, times


def _last_conflicting_positions(
        low_geometry, high_geometry, high_stopped, longitudinal_gap):
    collision = _overlap_matrix(
        low_geometry, high_geometry, longitudinal_gap)
    high_count = len(high_geometry['sampleS'])
    high_indices = np.arange(high_count, dtype=int)
    last = np.max(np.where(collision, high_indices[None, :], -1), axis=1)
    valid = last >= 0
    conflict_s = np.full(len(last), np.nan, dtype=float)
    conflict_s[valid] = high_geometry['sampleS'][last[valid]]
    blocked = valid & (last == high_count - 1) & bool(high_stopped)

    refining = np.flatnonzero(valid & (last < high_count - 1))
    if len(refining):
        lower = high_geometry['sampleS'][last[refining]].copy()
        upper = high_geometry['sampleS'][last[refining] + 1].copy()
        low_centers = low_geometry['centers'][refining]
        low_tangents = low_geometry['tangents'][refining]
        low_normals = low_geometry['normals'][refining]
        for _ in range(16):
            middle = 0.5 * (lower + upper)
            high_centers, high_tangents = _path_states_at_s(
                high_geometry['path'], high_geometry['pathS'], middle)
            high_normals = np.column_stack((
                -high_tangents[:, 1], high_tangents[:, 0]))
            overlaps = _overlap_pairs(
                low_centers, low_tangents, low_normals,
                high_centers, high_tangents, high_normals,
                low_geometry['halfLength'] + longitudinal_gap,
                low_geometry['halfWidth'],
                high_geometry['halfLength'],
                high_geometry['halfWidth'],
            )
            lower = np.where(overlaps, middle, lower)
            upper = np.where(overlaps, upper, middle)
        conflict_s[refining] = lower
    return conflict_s, blocked


def _overlap_matrix(low, high, longitudinal_gap):
    delta = low['centers'][:, None, :] - high['centers'][None, :, :]
    low_tangent = low['tangents']
    low_normal = low['normals']
    high_tangent = high['tangents']
    high_normal = high['normals']

    tangent_tangent = np.abs(low_tangent @ high_tangent.T)
    normal_tangent = np.abs(low_normal @ high_tangent.T)
    tangent_normal = np.abs(low_tangent @ high_normal.T)
    normal_normal = np.abs(low_normal @ high_normal.T)

    low_length = low['halfLength'] + longitudinal_gap
    low_width = low['halfWidth']
    high_length = high['halfLength']
    high_width = high['halfWidth']
    tolerance = 1e-9

    return (
        np.abs(np.einsum('lhk,hk->lh', delta, high_tangent))
        <= high_length + low_length * tangent_tangent
        + low_width * normal_tangent + tolerance
    ) & (
        np.abs(np.einsum('lhk,hk->lh', delta, high_normal))
        <= high_width + low_length * tangent_normal
        + low_width * normal_normal + tolerance
    ) & (
        np.abs(np.einsum('lhk,lk->lh', delta, low_tangent))
        <= low_length + high_length * tangent_tangent
        + high_width * tangent_normal + tolerance
    ) & (
        np.abs(np.einsum('lhk,lk->lh', delta, low_normal))
        <= low_width + high_length * normal_tangent
        + high_width * normal_normal + tolerance
    )


def _overlap_pairs(
        low_centers, low_tangents, low_normals,
        high_centers, high_tangents, high_normals,
        low_length, low_width, high_length, high_width):
    delta = low_centers - high_centers
    tangent_tangent = np.abs(np.sum(low_tangents * high_tangents, axis=1))
    normal_tangent = np.abs(np.sum(low_normals * high_tangents, axis=1))
    tangent_normal = np.abs(np.sum(low_tangents * high_normals, axis=1))
    normal_normal = np.abs(np.sum(low_normals * high_normals, axis=1))
    tolerance = 1e-9
    return (
        np.abs(np.sum(delta * high_tangents, axis=1))
        <= high_length + low_length * tangent_tangent
        + low_width * normal_tangent + tolerance
    ) & (
        np.abs(np.sum(delta * high_normals, axis=1))
        <= high_width + low_length * tangent_normal
        + low_width * normal_normal + tolerance
    ) & (
        np.abs(np.sum(delta * low_tangents, axis=1))
        <= low_length + high_length * tangent_tangent
        + high_width * tangent_normal + tolerance
    ) & (
        np.abs(np.sum(delta * low_normals, axis=1))
        <= low_width + high_length * normal_tangent
        + high_width * normal_normal + tolerance
    )


def _overlap_at_s(low_geometry, low_s, high_geometry, high_s, gap):
    low_center, low_tangent = _path_states_at_s(
        low_geometry['path'], low_geometry['pathS'], [low_s])
    high_center, high_tangent = _path_states_at_s(
        high_geometry['path'], high_geometry['pathS'], [high_s])
    low_normal = np.column_stack((-low_tangent[:, 1], low_tangent[:, 0]))
    high_normal = np.column_stack((-high_tangent[:, 1], high_tangent[:, 0]))
    return bool(_overlap_pairs(
        low_center, low_tangent, low_normal,
        high_center, high_tangent, high_normal,
        low_geometry['halfLength'] + gap,
        low_geometry['halfWidth'],
        high_geometry['halfLength'],
        high_geometry['halfWidth'],
    )[0])
