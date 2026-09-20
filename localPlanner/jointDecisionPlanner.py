"""第九阶段：逐协同块完成优先级、时间窗速度和waypoint联合决策。

走廊面域只负责提取成对冲突锚点。优先级搜索和延误估值均在path弧长
坐标中完成，不进行逐帧轨迹仿真；优先级确定后才生成AGV速度包络与waypoint。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from tools.default_params import DEFAULT_PLANNING_PARAMS
from tools.path_geometry import (
    arc_lengths, as_xy, extend_polyline_ends, heading_at_s, headings_at_s,
    points_at_s, project_to_polyline)


@dataclass(frozen=True)
class ConflictRelation:
    vehicleA: str
    vehicleB: str
    rawSA: float
    rawSB: float
    anchorSA: float
    anchorSB: float
    threshold: float
    geometricDistance: float

    def values_for(self, vehicle_id: str):
        if vehicle_id == self.vehicleA:
            return self.rawSA, self.rawSB, self.anchorSA, self.anchorSB, self.vehicleB
        if vehicle_id == self.vehicleB:
            return self.rawSB, self.rawSA, self.anchorSB, self.anchorSA, self.vehicleA
        raise KeyError(vehicle_id)


@dataclass
class TimingProfile:
    pathS: np.ndarray
    speedLimit: np.ndarray
    speed: np.ndarray
    cumulativeTime: np.ndarray

    def speed_at_s(self, value: float) -> float:
        return float(self.speeds_at_s([value])[0])

    def speeds_at_s(self, values) -> np.ndarray:
        targets = np.atleast_1d(np.asarray(values, dtype=float))
        if len(self.pathS) == 1:
            return np.full(len(targets), float(self.speed[0]))
        targets = np.clip(targets, self.pathS[0], self.pathS[-1])
        indices = np.clip(
            np.searchsorted(self.pathS, targets, side="right")-1,
            0, len(self.pathS)-2)
        segments = self.pathS[indices+1]-self.pathS[indices]
        acceleration = np.divide(
            self.speed[indices+1]**2-self.speed[indices]**2,
            2.0*segments, out=np.zeros(len(targets)), where=segments > 1e-9)
        return np.sqrt(np.maximum(
            0.0, self.speed[indices]**2
            + 2.0*acceleration*(targets-self.pathS[indices])))

    def time_at_s(self, value: float) -> float:
        if len(self.pathS) == 1:
            return 0.0
        target = min(max(float(value), float(self.pathS[0])),
                     float(self.pathS[-1]))
        index = int(np.searchsorted(self.pathS, target, side="right")-1)
        index = max(0, min(index, len(self.pathS)-2))
        base = float(self.cumulativeTime[index])
        ds = target-self.pathS[index]
        if ds <= 1e-9:
            return base
        speed_sum = float(self.speed[index])+self.speed_at_s(target)
        return float("inf") if speed_sum <= 1e-9 else base+2.0*ds/speed_sum

    def state_at_time(self, value: float):
        if len(self.pathS) == 1:
            return float(self.pathS[0]), float(self.speed[0])
        target = max(0.0, float(value))
        finite = np.flatnonzero(np.isfinite(self.cumulativeTime))
        last = int(finite[-1]) if len(finite) else 0
        if target >= self.cumulativeTime[last]:
            return float(self.pathS[last]), float(self.speed[last])
        upper = int(np.searchsorted(
            self.cumulativeTime, target, side="right"))
        upper = max(1, min(upper, len(self.pathS)-1))
        lower = upper-1
        duration = self.cumulativeTime[upper]-self.cumulativeTime[lower]
        if not np.isfinite(duration) or duration <= 1e-9:
            return float(self.pathS[lower]), float(self.speed[lower])
        local_time = target-self.cumulativeTime[lower]
        acceleration = (self.speed[upper]-self.speed[lower])/duration
        path_s = (self.pathS[lower]+self.speed[lower]*local_time
                  + 0.5*acceleration*local_time**2)
        speed = max(0.0, self.speed[lower]+acceleration*local_time)
        return float(path_s), float(speed)


@dataclass
class PriorityDecision:
    members: Tuple[str, ...]
    priorityOrder: Tuple[str, ...]
    priorityEdges: Tuple[Tuple[str, str], ...]
    physicalEdges: Tuple[Tuple[str, str], ...]
    conflicts: Tuple[ConflictRelation, ...]
    totalDelay: float
    searchDiagnostics: Dict[str, Any]


@dataclass
class WaypointDecision:
    waypointResults: Dict[str, np.ndarray]
    plannedPaths: Dict[str, np.ndarray]


@dataclass
class _PrioritySearchNode:
    graph: Dict[str, set]
    parent: Any = None
    prior: float = 1.0
    depth: int = 0
    visitCount: int = 0
    totalValue: float = 0.0
    bestValue: float = -float("inf")

    @property
    def meanValue(self):
        return self.totalValue/self.visitCount if self.visitCount else 0.0


def calculate_block_priority(
    cooperation_block: Sequence[str],
    selected_paths: Mapping[str, np.ndarray],
    sequence_priorities: Sequence[Tuple[str, str]],
    system_priorities: Mapping[str, int],
    vehicle_static_info: Mapping[str, Mapping[str, Any]],
    vehicle_dynamic_info: Mapping[str, Mapping[str, Any]],
    historical_priority_edges: Iterable[Tuple[str, str]] = (),
) -> PriorityDecision:
    """计算冲突和优先级DAG；不生成控制速度或waypoint。"""
    params = DEFAULT_PLANNING_PARAMS
    priority_deadline = perf_counter()+max(
        0.0, float(params["prioritySearchTimeBudgetMs"]))/1000.0
    members = tuple(str(value) for value in cooperation_block
                    if str(value) in selected_paths)
    paths = {vehicle_id: np.asarray(selected_paths[vehicle_id], dtype=float)
             for vehicle_id in members}
    free_profiles = {
        vehicle_id: _build_priority_profile(
            paths[vehicle_id], vehicle_dynamic_info.get(vehicle_id, {}), params)
        for vehicle_id in members
    }
    # FIFO仅用于排序：v+0.5避免静止车辆ETA无穷大，不修改实际速度规划。
    eta_profiles = {
        vehicle_id: _build_priority_profile(
            paths[vehicle_id], vehicle_dynamic_info.get(vehicle_id, {}), params,
            fifo=True)
        for vehicle_id in members
    }
    conflicts = _detect_conflicts(
        members, paths, vehicle_static_info, vehicle_dynamic_info, params)
    conflict_pairs = {frozenset((item.vehicleA, item.vehicleB))
                      for item in conflicts}
    physical_edges = {
        (str(high), str(low)) for high, low in sequence_priorities
        if str(high) in members and str(low) in members
    }
    order, edges, delay, diagnostics = _choose_priority_graph(
        members, conflicts, sequence_priorities, system_priorities,
        vehicle_static_info, vehicle_dynamic_info, eta_profiles, free_profiles,
        paths, historical_priority_edges, params, priority_deadline)
    return PriorityDecision(
        members, tuple(order), tuple(sorted(edges)),
        tuple(sorted(physical_edges)), tuple(conflicts), float(delay),
        diagnostics)


def plan_block_waypoints(
    priority_result: PriorityDecision,
    selected_paths: Mapping[str, np.ndarray],
    vehicle_static_info: Mapping[str, Mapping[str, Any]],
    vehicle_dynamic_info: Mapping[str, Mapping[str, Any]],
    controllable_vehicle_ids: Iterable[str],
    previous_waypoints: Mapping[str, np.ndarray] | None = None,
) -> WaypointDecision:
    """约束指定车辆的速度，并为块内全部request车辆生成waypoint。"""
    params = DEFAULT_PLANNING_PARAMS
    controllable_ids = {str(value) for value in controllable_vehicle_ids}
    paths = {
        vehicle_id: np.asarray(selected_paths[vehicle_id], dtype=float)
        for vehicle_id in priority_result.members
    }
    free_profiles = {
        vehicle_id: _build_timing_profile(
            paths[vehicle_id], vehicle_dynamic_info.get(vehicle_id, {}), params)
        for vehicle_id in priority_result.members
    }
    waypoints, planned_paths = _plan_block_motion(
        priority_result.priorityOrder, set(priority_result.physicalEdges),
        priority_result.conflicts, paths, free_profiles,
        controllable_ids,
        vehicle_static_info, vehicle_dynamic_info, params)
    previous_waypoints = previous_waypoints or {}
    stable_waypoints = {
        vehicle_id: (
            _reuse_waypoints(
                previous_waypoints.get(vehicle_id), waypoints[vehicle_id],
                planned_paths[vehicle_id], params)
            if vehicle_id in controllable_ids
            else waypoints[vehicle_id])
        for vehicle_id in waypoints
    }
    return WaypointDecision(stable_waypoints, planned_paths)


def _detect_conflicts(members, paths, static_info, dynamic_info, params):
    """Detect overlap of complete vehicle corridors and return original-path anchors."""
    output = []
    spacing = max(0.2, float(params["conflictSampleSpacing"]))
    lateral_margin = max(0.0, float(params["cooperationSafetyMargin"]))
    longitudinal_margin = max(0.0, float(params["cooperationStopBuffer"]))
    upstream = float(params["conflictUpstreamOffset"])
    corridor_paths = {}
    backward_offsets = {}
    original_lengths = {}
    lateral_radii = {}
    for vehicle_id in members:
        dynamic = dynamic_info.get(vehicle_id, {})
        longitudinal_radius = (
            0.5*_vehicle_length(vehicle_id, static_info, params)
            + _longitudinal_uncertainty(dynamic)
            + 0.5*longitudinal_margin)
        lateral_radius = (
            0.5*_vehicle_width(vehicle_id, static_info, params)
            + _lateral_uncertainty(dynamic)
            + 0.5*lateral_margin)
        original = as_xy(paths[vehicle_id])
        corridor_paths[vehicle_id] = extend_polyline_ends(
            original, longitudinal_radius, longitudinal_radius,
            _number(dynamic.get("heading"), 0.0))
        backward_offsets[vehicle_id] = longitudinal_radius
        original_lengths[vehicle_id] = (
            float(arc_lengths(original)[-1]) if len(original) else 0.0)
        lateral_radii[vehicle_id] = lateral_radius

    sampled = {vehicle_id: _sample_path(corridor_paths[vehicle_id], spacing)
               for vehicle_id in members}
    sample_boxes = {
        vehicle_id: (
            np.min(points, axis=0), np.max(points, axis=0))
        if len(points) else None
        for vehicle_id, (points, _) in sampled.items()
    }
    for vehicle_a, vehicle_b in combinations(members, 2):
        points_a, _ = sampled[vehicle_a]
        points_b, _ = sampled[vehicle_b]
        if len(points_a) == 0 or len(points_b) == 0:
            continue
        threshold = lateral_radii[vehicle_a]+lateral_radii[vehicle_b]
        lower_a, upper_a = sample_boxes[vehicle_a]
        lower_b, upper_b = sample_boxes[vehicle_b]
        axis_gap = np.maximum(0.0, np.maximum(lower_a-upper_b, lower_b-upper_a))
        if float(np.linalg.norm(axis_gap)) > threshold:
            continue
        entry_a = _first_corridor_entry(
            corridor_paths[vehicle_a], corridor_paths[vehicle_b],
            threshold, spacing)
        entry_b = _first_corridor_entry(
            corridor_paths[vehicle_b], corridor_paths[vehicle_a],
            threshold, spacing)
        if entry_a is None or entry_b is None:
            continue

        # Extended-path negative s belongs to the vehicle body behind its current
        # centre. Map it to s=0 so ETA and waypoint logic stay in the original frame.
        raw_a = float(np.clip(
            entry_a[0]-backward_offsets[vehicle_a],
            0.0, original_lengths[vehicle_a]))
        raw_b = float(np.clip(
            entry_b[0]-backward_offsets[vehicle_b],
            0.0, original_lengths[vehicle_b]))
        distance_a, distance_b = entry_a[1], entry_b[1]
        anchor_a = max(
            0.0, raw_a-upstream
            - _longitudinal_uncertainty(dynamic_info.get(vehicle_a, {})))
        anchor_b = max(
            0.0, raw_b-upstream
            - _longitudinal_uncertainty(dynamic_info.get(vehicle_b, {})))
        output.append(ConflictRelation(
            vehicle_a, vehicle_b, raw_a, raw_b, anchor_a, anchor_b,
            threshold, float(max(distance_a, distance_b))))
    return output

def _first_corridor_entry(path, other_path, threshold, spacing):
    """Return the first own-path s entering the tube around the full peer path."""
    xy = as_xy(path)
    other = as_xy(other_path)
    if len(xy) == 0 or len(other) == 0:
        return None
    path_s = arc_lengths(xy)
    total = float(path_s[-1]) if len(path_s) else 0.0
    if total <= 1e-9:
        distance = float(_point_to_path_distances(xy[:1], other)[0])
        return (0.0, distance) if distance <= threshold+1e-9 else None

    sample_s = np.r_[np.arange(0.0, total, max(0.05, float(spacing))), total]
    sample_points = points_at_s(xy, sample_s)
    distances = _point_to_path_distances(sample_points, other)
    entries = np.flatnonzero(distances <= threshold+1e-9)
    if len(entries) == 0:
        return None
    index = int(entries[0])
    if index == 0:
        return 0.0, float(distances[0])

    lower, upper = float(sample_s[index-1]), float(sample_s[index])
    for _ in range(12):
        middle = 0.5*(lower+upper)
        point = points_at_s(xy, [middle])
        if _point_to_path_distances(point, other)[0] <= threshold:
            upper = middle
        else:
            lower = middle
    distance = float(_point_to_path_distances(
        points_at_s(xy, [upper]), other)[0])
    return float(upper), distance


def _point_to_path_distances(points, path):
    """Vectorized point-to-finite-segment distances, processed in blocks."""
    query = np.asarray(points, dtype=float)[:, :2]
    line = as_xy(path)
    if len(line) == 0:
        return np.full(len(query), float("inf"))
    if len(line) == 1:
        return np.linalg.norm(query-line[0], axis=1)
    starts, vectors = line[:-1], np.diff(line, axis=0)
    squared = np.einsum("si,si->s", vectors, vectors)
    output = np.empty(len(query), dtype=float)
    for begin in range(0, len(query), 256):
        block = query[begin:begin+256]
        offsets = block[:, None, :]-starts[None, :, :]
        ratios = np.divide(
            np.einsum("bsi,si->bs", offsets, vectors), squared[None, :],
            out=np.zeros((len(block), len(starts))),
            where=squared[None, :] > 1e-12)
        projections = starts[None, :, :]+np.clip(
            ratios, 0.0, 1.0)[:, :, None]*vectors[None, :, :]
        output[begin:begin+len(block)] = np.min(
            np.linalg.norm(block[:, None, :]-projections, axis=2), axis=1)
    return output


def _build_priority_profile(path, dynamic, params, fifo=False):
    """仅用于排序估值的加速—匀速—减速时间表，不作为可执行速度。

    0.5m/s是估值用的正则速度：停车点不被解释为无限期占用。
    真实停车约束仍由plan_block_waypoints读取原始path并独立处理。
    """
    estimate_path = np.asarray(path, dtype=float).copy()
    if estimate_path.ndim == 2 and estimate_path.shape[1] >= 3:
        estimate_path[:, 2] = np.maximum(0.5, estimate_path[:, 2])
    estimate_dynamic = dict(dynamic)
    if fifo:
        estimate_dynamic["speed"] = max(0.0, _number(dynamic.get("speed"), 0.0))+0.5
    profile = _build_timing_profile(
        estimate_path, estimate_dynamic, params, cruise_at_current_speed=fifo)
    acceleration = max(0.1, float(params["vehicleMaxAcceleration"]))
    braking = max(0.1, float(params["vehicleMaxDeceleration"]))
    distances = np.diff(profile.pathS)
    starts, ends = profile.speed[:-1], profile.speed[1:]
    ceilings = np.maximum.reduce((
        starts, ends,
        np.minimum(profile.speedLimit[:-1], profile.speedLimit[1:])))
    peak_squared = (
        2.0*distances+starts**2/acceleration+ends**2/braking
    )/(1.0/acceleration+1.0/braking)
    peaks = np.minimum(
        ceilings,
        np.sqrt(np.maximum.reduce((starts**2, ends**2, peak_squared))))
    accelerate_distance = np.maximum(
        0.0, (peaks**2-starts**2)/(2.0*acceleration))
    brake_distance = np.maximum(
        0.0, (peaks**2-ends**2)/(2.0*braking))
    cruise_distance = np.maximum(
        0.0, distances-accelerate_distance-brake_distance)
    durations = ((peaks-starts)/acceleration+(peaks-ends)/braking
                 + cruise_distance/np.maximum(peaks, 0.5))
    profile.cumulativeTime[1:] = np.cumsum(durations)
    return profile


def _priority_arrival(profile, path_s):
    """从优先级时间表查ETA；末端外清空距离按0.5m/s兜底估值，不外推真实轨迹。"""
    target = max(0.0, float(path_s))
    arrival = float(np.interp(target, profile.pathS, profile.cumulativeTime))
    extra = max(0.0, target-float(profile.pathS[-1]))
    return arrival+extra/max(0.5, float(profile.speed[-1]))


def _build_timing_profile(path, dynamic, params, end_s=None,
                          speed_constraints=(), terminal_speed=None,
                          cruise_at_current_speed=False, inherited_cap=None,
                          emergency_braking=False):
    """建立O(N)空间速度包络和累计时间表，不做逐帧运动仿真。"""
    array = np.asarray(path, dtype=float)
    xy = as_xy(array)
    source_s = arc_lengths(xy)
    if len(xy) < 2 or len(source_s) == 0 or source_s[-1] <= 1e-9:
        return TimingProfile(*(np.asarray([0.0]) for _ in range(4)))
    total = float(source_s[-1])
    end = total if end_s is None else float(np.clip(end_s, 0.0, total))
    spacing = max(0.25, float(params["priorityTimingSpacing"]))
    profile_s = np.r_[np.arange(0.0, end, spacing), end]
    for position, _ in speed_constraints:
        profile_s = np.r_[profile_s, np.clip(float(position), 0.0, end)]
    profile_s = np.unique(profile_s)
    if array.ndim == 2 and array.shape[1] >= 3 and len(array) == len(source_s):
        source_limits = np.maximum(0.0, array[:, 2])
    else:
        source_limits = np.full(len(source_s), float(params["vehicleMaxSpeed"]))
    speed_limit = np.interp(profile_s, source_s, source_limits)
    if inherited_cap is not None:
        speed_limit = np.minimum(
            speed_limit, inherited_cap.speeds_at_s(profile_s))
    current_speed = max(0.0, _number(dynamic.get("speed"), 0.0))
    if cruise_at_current_speed:
        speed_limit = np.minimum(speed_limit, current_speed)
    for position, value in speed_constraints:
        index = int(np.argmin(np.abs(profile_s-float(position))))
        speed_limit[index] = min(speed_limit[index], max(0.0, float(value)))
    if terminal_speed is not None:
        speed_limit[-1] = min(speed_limit[-1], max(0.0, float(terminal_speed)))
    speed = speed_limit.copy()
    speed[0] = min(current_speed, speed_limit[0])
    acceleration = max(0.1, float(params["vehicleMaxAcceleration"]))
    braking = max(0.1, float(params[
        "vehicleEmergencyDeceleration" if emergency_braking else "vehicleMaxDeceleration"]))
    speed = _forward_speed_envelope(speed, profile_s, acceleration)
    speed = _backward_speed_envelope(speed, profile_s, braking)
    speed = _forward_speed_envelope(speed, profile_s, acceleration)
    distances = np.diff(profile_s)
    speed_sums = speed[:-1]+speed[1:]
    durations = np.divide(
        2.0*distances, speed_sums,
        out=np.full_like(distances, np.inf), where=speed_sums > 1e-9)
    durations[distances <= 1e-9] = 0.0
    cumulative = np.r_[0.0, np.cumsum(durations)]
    return TimingProfile(profile_s, speed_limit, speed, cumulative)


def _forward_speed_envelope(speed_limit, path_s, acceleration):
    squared = np.maximum(0.0, np.asarray(speed_limit, dtype=float))**2
    transformed = squared-2.0*float(acceleration)*path_s
    return np.sqrt(np.maximum(
        0.0,
        2.0*float(acceleration)*path_s+np.minimum.accumulate(transformed)))


def _backward_speed_envelope(speed_limit, path_s, braking):
    squared = np.maximum(0.0, np.asarray(speed_limit, dtype=float))**2
    transformed = squared+2.0*float(braking)*path_s
    suffix_minimum = np.minimum.accumulate(transformed[::-1])[::-1]
    return np.sqrt(np.maximum(
        0.0, suffix_minimum-2.0*float(braking)*path_s))


def _choose_priority_graph(
    members, conflicts, sequence_priorities, system_priorities,
    static_info, dynamic_info, eta_profiles, free_profiles, paths,
    historical_priority_edges, params, deadline,
):
    """以限时Beam-PUCT搜索优先级DAG，并用双策略rollout补全。"""
    historical_priority_edges = tuple(historical_priority_edges)
    hard_graph = {vehicle_id: set() for vehicle_id in members}
    conflict_pairs = {frozenset((item.vehicleA, item.vehicleB))
                      for item in conflicts}
    for high, low in sequence_priorities:
        if high in hard_graph and low in hard_graph:
            if not _try_add_edge(hard_graph, high, low):
                raise ValueError("物理前后关系存在环，拒绝删除或反转物理优先级")
    # 第二层硬约束：若一辆车当前已经进入另一辆车的未来走廊，
    # 占用者先行。物理前后关系已确定的车对不会被本层覆盖。
    occupancy_edges = set()
    for relation in conflicts:
        first, second = relation.vehicleA, relation.vehicleB
        if _comparable(hard_graph, first, second):
            continue
        occupancy_margin = 0.5*max(0.0, float(params["cooperationStopBuffer"]))
        first_occupancy_reach = (
            0.5*_vehicle_length(first, static_info, params)
            + _longitudinal_uncertainty(dynamic_info.get(first, {}))
            + occupancy_margin)
        second_occupancy_reach = (
            0.5*_vehicle_length(second, static_info, params)
            + _longitudinal_uncertainty(dynamic_info.get(second, {}))
            + occupancy_margin)
        first_inside = relation.rawSA <= first_occupancy_reach+1e-6
        second_inside = relation.rawSB <= second_occupancy_reach+1e-6
        preferred = None
        if first_inside != second_inside:
            preferred = ((first, second) if first_inside else (second, first))
        elif first_inside and second_inside:
            # 双方都已进入彼此走廊时几何关系无法单独定向；只复用仍合法的历史方向。
            for edge in historical_priority_edges:
                if (len(edge) == 2 and frozenset(edge) == frozenset((first, second))):
                    preferred = (str(edge[0]), str(edge[1]))
                    break
        if preferred is not None and _try_add_edge(hard_graph, *preferred):
            occupancy_edges.add(preferred)

    # 第三层硬约束：调箱门区域优先，然后才是系统优先级。
    for relation in conflicts:
        first, second = relation.vehicleA, relation.vehicleB
        if _comparable(hard_graph, first, second):
            continue
        priority_first = (0 if dynamic_info.get(first, {}).get("inDxmArea") else 1,
                          int(system_priorities.get(first, 1)))
        priority_second = (0 if dynamic_info.get(second, {}).get("inDxmArea") else 1,
                           int(system_priorities.get(second, 1)))
        if priority_first != priority_second:
            high, low = ((first, second) if priority_first < priority_second
                         else (second, first))
            _try_add_edge(hard_graph, high, low)

    # relation的ETA排序和FIFO方向与搜索分支无关，只计算一次。
    relation_keys = {}
    fifo_directions = {}
    for relation in conflicts:
        pair = frozenset((relation.vehicleA, relation.vehicleB))
        relation_keys[pair] = _relation_search_key(
            relation, system_priorities, eta_profiles, static_info, params)
        eta_a = _front_eta(
            relation.vehicleA, relation.anchorSA,
            eta_profiles, static_info, params)
        eta_b = _front_eta(
            relation.vehicleB, relation.anchorSB,
            eta_profiles, static_info, params)
        fifo_directions[pair] = (
            (relation.vehicleA, relation.vehicleB)
            if (eta_a, relation.vehicleA) <= (eta_b, relation.vehicleB)
            else (relation.vehicleB, relation.vehicleA))

    current_pairs = {frozenset((item.vehicleA, item.vehicleB))
                     for item in conflicts}
    history_edges = {
        tuple(edge) for edge in historical_priority_edges
        if len(edge) == 2 and edge[0] in hard_graph and edge[1] in hard_graph
        and frozenset(edge) in current_pairs
    }

    root = _PrioritySearchNode(_copy_graph(hard_graph))
    frontier = [root]
    candidates = []
    evaluated_signatures = set()
    evaluation_log = []
    beam_width = max(1, int(params["priorityBeamWidth"]))
    max_iterations = max(1, int(params["prioritySearchMaxIterations"]))
    exploration = max(0.0, float(params["priorityPuctExploration"]))

    # 先生成完整基准解，保证任何时间预算下都有可行输出。
    root_results = _evaluate_priority_rollouts(
        root.graph, conflicts, relation_keys, fifo_directions, history_edges,
        free_profiles, paths, static_info, dynamic_info, system_priorities, params)
    history_candidate = None
    for delay, completed, policy in root_results:
        evaluation_log.append(_priority_evaluation_record(
            policy, delay, completed, system_priorities, eta_profiles,
            static_info, params))
        signature = _edge_signature(completed)
        if signature not in evaluated_signatures:
            candidates.append((delay, completed))
            evaluated_signatures.add(signature)
        if policy == "CACHE_FIFO":
            history_candidate = (delay, completed)
    best_root_delay = min(value[0] for value in root_results)
    if np.isfinite(best_root_delay):
        _backpropagate_priority_value(root, -best_root_delay)
    else:
        # 完整排序已经生成。所有估值非有限时直接保留基准解，不向PUCT传播inf。
        frontier.clear()

    # frontier保留所有尚未展开的节点；每轮只取PUCT最高的beamWidth个。
    # 因此不同分支深度可以不同，搜索深度由时间预算自适应决定。
    for _ in range(max_iterations):
        if perf_counter() >= deadline or not frontier:
            break
        frontier.sort(key=lambda node: (
            -_priority_puct_score(node, exploration),
            node.depth,
            _edge_signature(node.graph),
        ))
        beam = frontier[:beam_width]
        selected_ids = {id(node) for node in beam}
        frontier = [node for node in frontier if id(node) not in selected_ids]

        for node in beam:
            if perf_counter() >= deadline:
                frontier.append(node)
                continue
            pending = _unresolved_relations(conflicts, node.graph)
            if not pending:
                continue
            relation = min(
                pending,
                key=lambda item: relation_keys[
                    frozenset((item.vehicleA, item.vehicleB))])
            child_graphs = []
            for high, low in ((relation.vehicleA, relation.vehicleB),
                              (relation.vehicleB, relation.vehicleA)):
                child_graph = _copy_graph(node.graph)
                if _try_add_edge(child_graph, high, low):
                    child_graphs.append(child_graph)
            child_prior = 1.0/max(1, len(child_graphs))
            for child_graph in child_graphs:
                child = _PrioritySearchNode(
                    child_graph, node, child_prior, node.depth+1)
                rollout_results = _evaluate_priority_rollouts(
                    child.graph, conflicts, relation_keys, fifo_directions,
                    history_edges, free_profiles, paths, static_info, dynamic_info,
                    system_priorities, params)
                best_child_delay = min(value[0] for value in rollout_results)
                if np.isfinite(best_child_delay):
                    _backpropagate_priority_value(child, -best_child_delay)
                for delay, completed, policy in rollout_results:
                    evaluation_log.append(_priority_evaluation_record(
                        f"SEARCH_D{child.depth}_{policy}", delay, completed,
                        system_priorities, eta_profiles, static_info, params))
                    signature = _edge_signature(completed)
                    if signature not in evaluated_signatures:
                        candidates.append((delay, completed))
                        evaluated_signatures.add(signature)
                if np.isfinite(best_child_delay) and _unresolved_relations(conflicts, child.graph):
                    frontier.append(child)

    if not candidates:
        candidates.append((0.0, hard_graph))
    candidates.sort(key=lambda value: (value[0], _edge_signature(value[1])))
    chosen_delay, chosen = candidates[0]
    chosen_policy = "GLOBAL_BEST"
    tolerance = max(0.0, float(params["priorityHistoryDelayTolerance"]))
    if (history_candidate is not None
            and np.isfinite(history_candidate[0])
            and history_candidate[0] <= chosen_delay+tolerance):
        chosen_delay, chosen = history_candidate
        chosen_policy = "CACHE_STABLE"
    order = _topological_order(
        chosen, system_priorities, eta_profiles, static_info, params)
    chosen_edges = _materialize_conflict_edges(chosen, conflicts)
    diagnostics = {
        "historicalEdges": tuple(sorted(history_edges)),
        "evaluations": tuple(evaluation_log),
        "chosenOrder": tuple(order),
        "chosenEdges": tuple(sorted(chosen_edges)),
        "occupancyEdges": tuple(sorted(occupancy_edges)),
        "chosenCompletionTime": float(chosen_delay),
        "chosenDelay": float(chosen_delay),  # 兼容现有仿真诊断字段
        "chosenPolicy": chosen_policy,
    }
    return order, chosen_edges, chosen_delay, diagnostics


def _fifo_rollout(graph, conflicts, relation_keys, fifo_directions):
    """用ETA补全搜索尚未展开的冲突边；FIFO本身不计算cost。"""
    output = _copy_graph(graph)
    while True:
        pending = _unresolved_relations(conflicts, output)
        if not pending:
            return output
        relation = min(
            pending,
            key=lambda item: relation_keys[
                frozenset((item.vehicleA, item.vehicleB))])
        preferred = fifo_directions[
            frozenset((relation.vehicleA, relation.vehicleB))]
        if not _try_add_edge(output, *preferred):
            _try_add_edge(output, preferred[1], preferred[0])


def _cache_fifo_rollout(
    graph, conflicts, relation_keys, fifo_directions, history_edges,
):
    """先加入当前仍合法的历史边，再用FIFO补全剩余冲突。"""
    output = _copy_graph(graph)
    current_pairs = {
        frozenset((item.vehicleA, item.vehicleB)) for item in conflicts}
    ordered_history = sorted(
        history_edges,
        key=lambda edge: (
            relation_keys.get(frozenset(edge), (float("inf"),)), edge))
    for high, low in ordered_history:
        if (frozenset((high, low)) not in current_pairs
                or _comparable(output, high, low)):
            continue
        _try_add_edge(output, high, low)
    return _fifo_rollout(output, conflicts, relation_keys, fifo_directions)


def _evaluate_priority_rollouts(
    graph, conflicts, relation_keys, fifo_directions, history_edges,
    free_profiles, paths, static_info, dynamic_info, system_priorities, params,
):
    """用整段path完成时间评价FIFO与Cache-FIFO两个完整方案。"""
    completed_graphs = [("FIFO", _fifo_rollout(
        graph, conflicts, relation_keys, fifo_directions))]
    if history_edges:
        completed_graphs.append(("CACHE_FIFO", _cache_fifo_rollout(
            graph, conflicts, relation_keys, fifo_directions, history_edges)))
    results = []
    for policy, completed in completed_graphs:
        try:
            delay = _estimate_total_delay(
                completed, conflicts, free_profiles, paths, static_info,
                dynamic_info, system_priorities, params)
        except ArithmeticError:
            delay = float("inf")
        # 数值估值失败不等于排序失败；完整DAG仍可返回。
        results.append((
            float(delay) if np.isfinite(delay) else float("inf"),
            completed, policy))
    return results


def _priority_evaluation_record(policy, delay, graph, system_priorities,
                                eta_profiles, static_info, params):
    return {
        "policy": str(policy),
        "order": tuple(_topological_order(
            graph, system_priorities, eta_profiles, static_info, params)),
        "edges": tuple(_edge_signature(graph)),
        "completionTime": float(delay),
        "delay": float(delay),  # compatibility alias for existing diagnostics
    }


def _priority_puct_score(node, exploration):
    if node.parent is None:
        return node.meanValue
    parent_visits = max(1, node.parent.visitCount)
    return (node.meanValue
            + exploration*node.prior*np.sqrt(parent_visits)
            /(1+node.visitCount))


def _backpropagate_priority_value(node, value):
    while node is not None:
        node.visitCount += 1
        node.totalValue += float(value)
        node.bestValue = max(node.bestValue, float(value))
        node = node.parent


def _estimate_total_delay(graph, conflicts, free_profiles, paths, static_info,
                          dynamic_info, system_priorities, params):
    """按DAG顺序构造简化ST时间表，并返回所有车辆整段path完成时间之和。

    该估值只在弧长上的冲突事件处更新速度包络或等待时间，不逐帧仿真。
    后车读取所有已规划高优先级车辆的清空时间，因此让行代价会沿DAG传播。
    """
    order = _topological_order(
        graph, system_priorities, free_profiles, static_info, params)
    schedules = {}
    total_completion_time = 0.0
    clearance = max(0.0, float(params["priorityClearanceTime"]))
    relation_by_pair = {
        frozenset((item.vehicleA, item.vehicleB)): item for item in conflicts}
    effective_edges = _materialize_conflict_edges(graph, conflicts)

    for low in order:
        profile = free_profiles[low]
        events = []
        estimate_path = np.asarray(paths[low], dtype=float).copy()
        if estimate_path.ndim == 2 and estimate_path.shape[1] >= 3:
            # 排序估值不把规划时域末端的0限速解释为永久占用。
            estimate_path[:, 2] = np.maximum(0.5, estimate_path[:, 2])

        incoming = [high for high, follower in effective_edges if follower == low]
        constraints = []
        for high in incoming:
            relation = relation_by_pair.get(frozenset((high, low)))
            if relation is None or high not in schedules:
                continue
            high_raw, _, _, _, _ = relation.values_for(high)
            _, _, low_anchor, _, _ = relation.values_for(low)
            high_clear_s = (
                high_raw+0.5*_vehicle_length(high, static_info, params)
                + _longitudinal_uncertainty(dynamic_info.get(high, {})))
            low_entry_s = max(
                0.0, low_anchor-0.5*_vehicle_length(low, static_info, params))
            high_schedule = schedules[high]
            high_clear_time = (
                _priority_arrival(high_schedule["profile"], high_clear_s)
                + _delay_before(high_schedule["events"], high_clear_s)
                + clearance)
            constraints.append((low_entry_s, high_clear_time))

        for conflict_s, required_time in sorted(constraints):
            current_arrival = (
                _priority_arrival(profile, conflict_s)
                + _delay_before(events, conflict_s))
            if not np.isfinite(required_time) or not np.isfinite(current_arrival):
                return float("inf")
            if current_arrival >= required_time-1e-9:
                continue

            # 优先用减速包络满足窗口；只有减到最低估值速度仍不够时才加入停车等待。
            required_without_wait = required_time-_delay_before(events, conflict_s)
            adjusted = _profile_for_required_arrival(
                estimate_path, dynamic_info.get(low, {}), conflict_s,
                required_without_wait, params, inherited_cap=profile)
            if adjusted is not None:
                profile = adjusted
                current_arrival = (
                    _priority_arrival(profile, conflict_s)
                    + _delay_before(events, conflict_s))
            if current_arrival < required_time-1e-9:
                previous_s = max((position for position, _ in events
                                  if position <= conflict_s), default=0.0)
                event_delay = _estimate_yield_delay(
                    required_time-current_arrival,
                    profile.speed_at_s(conflict_s),
                    max(0.0, conflict_s-previous_s),
                    max(0.0, profile.pathS[-1]-conflict_s), params)
                events.append((conflict_s, event_delay))

        completion_time = (
            _priority_arrival(profile, float(profile.pathS[-1]))
            + _delay_before(events, float("inf")))
        if not np.isfinite(completion_time):
            return float("inf")
        schedules[low] = {
            "profile": profile,
            "events": tuple(events),
            "completionTime": float(completion_time),
        }
        total_completion_time += completion_time
    return float(total_completion_time)

def _estimate_yield_delay(required_delay, speed, approach_distance, remaining_distance, params):
    """时间窗下的减速/等待/再加速近似损失；使用解析运动学，不逐帧仿真。"""
    speed = max(0.5, float(speed))
    acceleration = max(0.1, float(params["vehicleMaxAcceleration"]))
    braking = max(0.1, float(params["vehicleMaxDeceleration"]))
    # 先估计让行速度，再限制为当前可用距离内能够减到的速度。
    target = speed*approach_distance/max(1e-9, approach_distance+speed*required_delay)
    target = max(target, np.sqrt(max(0.0, speed**2-2.0*braking*approach_distance)))
    brake_time = (speed-target)/braking
    brake_distance = (speed**2-target**2)/(2.0*braking)
    brake_loss = max(0.0, brake_time-brake_distance/speed)
    # path剩余距离不足时，不计入path外的完整加速过程。
    recovery_distance = min(remaining_distance, (speed**2-target**2)/(2.0*acceleration))
    recovered_speed = np.sqrt(target**2+2.0*acceleration*recovery_distance)
    recovery_time = (recovered_speed-target)/acceleration
    recovery_loss = max(0.0, recovery_time-recovery_distance/speed)
    # required_delay已包括减速导致的迟到，不与brake_loss重复累加。
    return max(required_delay, brake_loss)+recovery_loss


def _relation_search_key(relation, system_priorities, eta_profiles,
                         static_info, params):
    eta_a = _front_eta(
        relation.vehicleA, relation.anchorSA, eta_profiles, static_info, params)
    eta_b = _front_eta(
        relation.vehicleB, relation.anchorSB, eta_profiles, static_info, params)
    return (
        min(int(system_priorities.get(relation.vehicleA, 1)),
            int(system_priorities.get(relation.vehicleB, 1))),
        min(eta_a, eta_b), max(eta_a, eta_b),
        min(relation.vehicleA, relation.vehicleB))


def _front_eta(vehicle_id, anchor_s, profiles, static_info, params):
    front_s = max(
        0.0, anchor_s-0.5*_vehicle_length(vehicle_id, static_info, params))
    return _priority_arrival(profiles[vehicle_id], front_s)


def _plan_block_motion(order, physical_edges, conflicts, paths, free_profiles,
                       controllable, static_info, dynamic_info, params):
    """按DAG从高到低规划；低优先级车只处理最近的高优先级冲突。"""
    rank = {vehicle_id: index for index, vehicle_id in enumerate(order)}
    accepted_profiles = {}
    waypoints, planned_paths = {}, {}
    for vehicle_id in order:
        path = paths[vehicle_id]
        final_profile = free_profiles[vehicle_id]
        control_s = ()
        if vehicle_id in controllable:
            approaches = []
            for relation in conflicts:
                if vehicle_id not in (relation.vehicleA, relation.vehicleB):
                    continue
                _, _, low_anchor, _, high = relation.values_for(vehicle_id)
                if rank.get(high, len(order)) >= rank[vehicle_id] or high not in accepted_profiles:
                    continue
                stop_line = max(0.0, low_anchor
                                -0.5*_vehicle_length(vehicle_id, static_info, params)
                                -float(params["cooperationStopBuffer"]))
                approaches.append((stop_line, high, relation))
            if approaches:
                foremost, high, relation = min(approaches, key=lambda value: (value[0], value[1]))
                final_profile, control_s = _coordinate_nearest_conflict(
                    path, vehicle_id, high, foremost, relation, final_profile,
                    accepted_profiles[high], (high, vehicle_id) in physical_edges,
                    static_info, dynamic_info, params)
        if control_s:
            # 四点很稀疏：协同事件的减速参考值传给后方第一个控制点，
            # 不移动格点，也不要求四点拟合速度曲线。同步收紧内部包络，
            # 使低优先级车辆读取的结果与本车发布的参考速度一致。
            positions = _event_waypoint_s(
                path, float(final_profile.pathS[-1]), params, control_s)
            reference_constraints = []
            for event_s in control_s:
                downstream = np.flatnonzero(positions[1:] >= event_s-1e-9)
                if len(downstream):
                    target_s = float(positions[1+downstream[0]])
                    event_speed = final_profile.speed_at_s(event_s)
                    if event_speed < final_profile.speed_at_s(target_s)-1e-6:
                        reference_constraints.append((target_s, event_speed))
            if reference_constraints:
                final_profile = _build_timing_profile(
                    path, dynamic_info.get(vehicle_id, {}), params,
                    end_s=float(final_profile.pathS[-1]),
                    speed_constraints=reference_constraints,
                    inherited_cap=final_profile)
        accepted_profiles[vehicle_id] = final_profile
        planned_paths[vehicle_id] = _profile_path(path, final_profile)
        # 所有request车辆都输出waypoint；PlanningEngine当前把AV/HV都放入
        # controllable集合，使低优先级车辆读取高优先级车辆已调整后的速度曲线。
        waypoints[vehicle_id] = _select_waypoints(
            path, final_profile, params, control_s,
            dynamic_info.get(vehicle_id, {}))
    return waypoints, planned_paths


def _coordinate_nearest_conflict(path, low, high, stop_s, relation, base,
                                 high_profile, is_following, static, dynamic, params):
    """参考2.0：虚拟前车决定减速/AEB，停车位置始终固定在冲突锚点上游。"""
    profile = base
    low_raw, high_raw, low_anchor, _, _ = relation.values_for(low)
    stop_s = min(stop_s, float(base.pathS[-1]))
    half_lengths = 0.5*(_vehicle_length(low, static, params)
                        + _vehicle_length(high, static, params))
    uncertainty = (_longitudinal_uncertainty(dynamic.get(low, {}))
                   + _longitudinal_uncertainty(dynamic.get(high, {})))
    clear_s = (high_raw+0.5*_vehicle_length(high, static, params)
               + _longitudinal_uncertainty(dynamic.get(high, {})))
    emergency = False
    must_stop = False

    # 交叉/汇入额外检查车尾清空时间；跟驰沿用虚拟投影间距。
    if not is_following:
        required = (float("inf") if clear_s > high_profile.pathS[-1]+1e-9
                    and high_profile.speed[-1] <= 1e-6
                    else high_profile.time_at_s(clear_s)+float(params["priorityClearanceTime"]))
        entry_s = max(0.0, low_anchor-0.5*_vehicle_length(low, static, params))
        if profile.time_at_s(entry_s) < required:
            adjusted = _profile_for_required_arrival(
                path, dynamic.get(low, {}), entry_s, required, params, inherited_cap=profile)
            if adjusted is None:
                must_stop = True
            else:
                profile = adjusted

    for _ in range(max(1, int(params["cooperationIterations"]))):
        if must_stop:
            break
        arrival = profile.time_at_s(stop_s)
        if not np.isfinite(arrival):
            return profile, ()  # 已停在检查点之前，不会进入冲突区域
        high_s, high_speed = high_profile.state_at_time(arrival)
        if not is_following and high_s >= clear_s-1e-6:
            return profile, (_deceleration_start(base, profile, stop_s), stop_s)
        low_speed = profile.speed_at_s(stop_s)
        bumper_gap = (low_raw-stop_s)-(high_raw-high_s)-half_lengths-uncertainty
        delta_v = max(0.0, low_speed-high_speed)
        desired = float(params["cooperationFollowingBaseGap"])+delta_v*float(params["cooperationFollowingTimeGap"])
        aeb_gap = delta_v**2/(2.0*float(params["vehicleEmergencyDeceleration"]))+float(params["cooperationStopBuffer"])
        if bumper_gap <= aeb_gap+1e-6:
            emergency = True
            break
        if bumper_gap >= desired-1e-6:
            return profile, (_deceleration_start(base, profile, stop_s), stop_s)
        target = max(float(params["cooperationMinimumCrawlSpeed"]),
                     high_speed+max(0.0, (bumper_gap-float(params["cooperationFollowingBaseGap"]))
                                    /float(params["cooperationFollowingTimeGap"])))
        profile = _build_timing_profile(
            path, dynamic.get(low, {}), params,
            speed_constraints=((stop_s, min(target, low_speed)),), inherited_cap=profile)
    else:
        # 最后一轮减速后再复核一次，避免只因为迭代次数结束就停车。
        arrival = profile.time_at_s(stop_s)
        if not np.isfinite(arrival):
            return profile, ()
        high_s, high_speed = high_profile.state_at_time(arrival)
        low_speed = profile.speed_at_s(stop_s)
        gap = (low_raw-stop_s)-(high_raw-high_s)-half_lengths-uncertainty
        delta_v = max(0.0, low_speed-high_speed)
        desired = float(params["cooperationFollowingBaseGap"])+delta_v*float(params["cooperationFollowingTimeGap"])
        aeb_gap = delta_v**2/(2.0*float(params["vehicleEmergencyDeceleration"]))+float(params["cooperationStopBuffer"])
        if not is_following and high_s >= clear_s-1e-6:
            return profile, (_deceleration_start(base, profile, stop_s), stop_s)
        emergency = gap <= aeb_gap+1e-6
        if not emergency and gap >= desired-1e-6:
            return profile, (_deceleration_start(base, profile, stop_s), stop_s)
    stopped = _build_timing_profile(
        path, dynamic.get(low, {}), params, end_s=stop_s, terminal_speed=0.0,
        inherited_cap=profile, emergency_braking=emergency)
    return stopped, (_deceleration_start(base, stopped, stop_s), stop_s)


def _deceleration_start(base, adjusted, checkpoint_s):
    candidates = adjusted.pathS[adjusted.pathS <= checkpoint_s+1e-9]
    for value in candidates:
        if adjusted.speed_at_s(value) < base.speed_at_s(value)-0.05:
            return float(value)
    return float(checkpoint_s)


def _profile_for_required_arrival(path, dynamic, checkpoint, required, params,
                                  inherited_cap=None):
    """二分冲突点速度约束；不能靠减速满足时间窗时返回None触发停车。"""
    if not np.isfinite(required):
        return None
    base = _build_timing_profile(path, dynamic, params, inherited_cap=inherited_cap)
    base_speed = base.speed_at_s(checkpoint)
    crawl = min(base_speed, float(params["cooperationMinimumCrawlSpeed"]))
    slowest = _build_timing_profile(
        path, dynamic, params, speed_constraints=((checkpoint, crawl),),
        inherited_cap=base)
    if slowest.time_at_s(checkpoint) < required-1e-6:
        return None
    lower, upper, selected = crawl, base_speed, slowest
    for _ in range(12):
        middle = 0.5*(lower+upper)
        candidate = _build_timing_profile(
            path, dynamic, params, speed_constraints=((checkpoint, middle),),
            inherited_cap=base)
        if candidate.time_at_s(checkpoint) >= required:
            selected, lower = candidate, middle
        else:
            upper = middle
    return selected


def _profile_path(path, profile):
    xy = as_xy(path)
    points = points_at_s(xy, profile.pathS)
    return np.column_stack((points, profile.speed))


def _select_waypoints(path, profile, params, control_s=(), dynamic=None):
    """复刻2.0：真实path采样组合搜索，控制事件参与内部点选择。"""
    end_s = float(profile.pathS[-1])
    positions = _event_waypoint_s(path, end_s, params, control_s)
    xy = as_xy(path)
    points = points_at_s(xy, positions)
    headings = headings_at_s(xy, positions)
    speeds = profile.speeds_at_s(positions)
    dynamic = dynamic or {}
    headings[0] = _number(dynamic.get("heading"), headings[0])
    # 输出是期望上限，不用实测速度覆盖本轮限速，尤其是当前位置停车。
    if profile.speed[-1] <= 1e-6:
        speeds[positions >= end_s-1e-9] = 0.0
    return np.column_stack((points, headings, speeds))


def _select_waypoint_s(path, end_s, params):
    minimum = float(params["waypointMinSpacing"])
    maximum = float(params["waypointMaxSpacing"])
    if end_s < 3.0*minimum-1e-9:
        return np.asarray([0.0, end_s, end_s, end_s], dtype=float)
    path_s = arc_lengths(as_xy(path))
    pool = path_s[(path_s > 1e-8) & (path_s < end_s-1e-8)]
    if len(pool) < 2:
        return np.asarray([0.0, end_s, end_s, end_s], dtype=float)
    best = _best_waypoint_pair(
        pool, end_s, minimum, maximum, controls=())
    return (np.asarray([0.0, end_s, end_s, end_s], dtype=float)
            if best is None else
            np.asarray([0.0, best[1], best[2], end_s], dtype=float))


def _event_waypoint_s(path, end_s, params, control_s):
    minimum = float(params["waypointMinSpacing"])
    maximum = float(params["waypointMaxSpacing"])
    if end_s < 3.0*minimum-1e-9:
        return np.asarray([0.0, end_s, end_s, end_s], dtype=float)
    controls = sorted(
        float(np.clip(value, 0.0, end_s)) for value in control_s
        if value is not None and np.isfinite(value)
        and 1e-6 < value < end_s-1e-6)
    if not controls:
        return _select_waypoint_s(path, end_s, params)
    path_s = arc_lengths(as_xy(path))
    pool = path_s[(path_s > 1e-8) & (path_s < end_s-1e-8)]
    pool = np.unique(np.r_[pool, controls])
    best = _best_waypoint_pair(
        pool, end_s, minimum, maximum, controls=controls)
    return (_select_waypoint_s(path, end_s, params) if best is None else
            np.asarray([0.0, best[1], best[2], end_s], dtype=float))


def _best_waypoint_pair(pool, end_s, minimum, maximum, controls):
    """向量化2.0的两内部点穷举；评分和行优先tie-break保持不变。"""
    pool = np.asarray(pool, dtype=float)
    if len(pool) < 2:
        return None
    first = pool[:, None]
    second = pool[None, :]
    valid = np.arange(len(pool))[:, None] < np.arange(len(pool))[None, :]
    gap_first = np.broadcast_to(first, valid.shape)
    gap_second = np.broadcast_to(second-first, valid.shape)
    gap_third = np.broadcast_to(end_s-second, valid.shape)
    target = end_s/3.0
    violation = sum(
        np.maximum(0.0, minimum-gap)+np.maximum(0.0, gap-maximum)
        for gap in (gap_first, gap_second, gap_third))
    score = 1e6*violation + sum(
        (gap-target)**2 for gap in (gap_first, gap_second, gap_third))
    if controls:
        event_error = np.zeros_like(score)
        for event in controls:
            event_error += np.minimum(
                (event-first)**2, (event-second)**2)
        score += 10.0*event_error
    score = np.where(valid, score, np.inf)
    flat_index = int(np.argmin(score))
    first_index, second_index = np.unravel_index(flat_index, score.shape)
    if not np.isfinite(score[first_index, second_index]):
        return None
    return (float(score[first_index, second_index]),
            float(pool[first_index]), float(pool[second_index]))


def _reuse_waypoints(previous, newly_planned, planned_path, params):
    """首尾始终使用新点；最多复用两个能近似表达新path与速度的旧中间点。"""
    new = np.asarray(newly_planned, dtype=float)
    old = np.asarray(previous, dtype=float) if previous is not None else None
    path = np.asarray(planned_path, dtype=float)
    if (old is None or old.ndim != 2 or old.shape[1] < 4
            or len(new) != 4 or len(old) != 4
            or path.ndim != 2 or path.shape[1] < 3 or len(path) < 2):
        return new.copy()

    # planned_path的坐标和速度必须同步去重。as_xy会单独删除重复坐标，
    # 若继续拿原path的速度列插值会发生长度错位；旧实现因此直接放弃了整帧缓存。
    finite = np.all(np.isfinite(path[:, :3]), axis=1)
    path = path[finite]
    if len(path) < 2:
        return new.copy()
    keep = np.concatenate((
        [True], np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 1e-8))
    path = path[keep]
    if len(path) < 2:
        return new.copy()
    xy = path[:, :2]
    path_s = arc_lengths(xy)
    end_s = float(path_s[-1])
    minimum, maximum = float(params["waypointMinSpacing"]), float(params["waypointMaxSpacing"])
    new_s = np.asarray([project_to_polyline(row[:2], xy)["s"] for row in new])
    candidates = [(new[i].copy(), float(new_s[i]), False) for i in (1, 2)]
    for row in old[1:-1]:
        if not np.all(np.isfinite(row[:4])):
            continue
        projection = project_to_polyline(row[:2], xy)
        s = float(projection["s"])
        if not float(params["waypointReuseReachedTolerance"]) < s < end_s-1e-6:
            continue
        if np.linalg.norm(row[:2]-projection["point"]) > float(params["waypointReuseLateralTolerance"]):
            continue
        heading_error = abs(np.arctan2(
            np.sin(row[2]-heading_at_s(xy, s)), np.cos(row[2]-heading_at_s(xy, s))))
        if heading_error > np.deg2rad(float(params["waypointReuseHeadingToleranceDeg"])):
            continue
        speed = float(np.interp(s, path_s, path[:, 2]))
        if (abs(row[3]-speed) > float(params["waypointReuseSpeedTolerance"])
                or (row[3] <= 0.1) != (speed <= 0.1)):
            continue
        candidates.append((row[:4].copy(), s, True))
    candidates.sort(key=lambda value: value[1])
    best, best_score = new.copy(), (0, float("inf"), float("inf"))
    for first, second in combinations(candidates, 2):
        if second[1] <= first[1]+1e-6:
            continue
        gaps = np.diff([0.0, first[1], second[1], end_s])
        violation = np.sum(np.maximum(0.0, minimum-gaps)+np.maximum(0.0, gaps-maximum))
        # 缓存的目标是抑制发布点抖动，而不是每帧重新获得严格的间距最优解。
        # 旧点已通过新path、新速度和停车状态检查，因此先最大化复用数量；
        # 仅在复用数量相同时，才用点间距和相对新点的位置误差选择更好的组合。
        score = (-(int(first[2])+int(second[2])),
                 float(violation),
                 abs(first[1]-new_s[1])+abs(second[1]-new_s[2]))
        if score < best_score:
            best_score = score
            best = np.asarray([new[0], first[0], second[0], new[-1]])
    return best


def _sample_path(path, spacing):
    xy = as_xy(path)
    if len(xy) == 0:
        return xy, np.empty(0)
    cumulative = arc_lengths(xy)
    total = float(cumulative[-1])
    if total <= 1e-9:
        return xy[:1], np.asarray([0.0])
    values = np.r_[np.arange(0.0, total, spacing), total]
    points = np.column_stack((
        np.interp(values, cumulative, xy[:, 0]),
        np.interp(values, cumulative, xy[:, 1])))
    return points, values


def _unresolved_relations(conflicts, graph):
    return [item for item in conflicts
            if not _comparable(graph, item.vehicleA, item.vehicleB)]


def _comparable(graph, first, second):
    return _reachable(graph, first, second) or _reachable(graph, second, first)


def _try_add_edge(graph, high, low):
    if high == low or _reachable(graph, low, high):
        return False
    graph[high].add(low)
    return True


def _reachable(graph, source, target):
    if source == target:
        return True
    stack, visited = list(graph.get(source, ())), set()
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        stack.extend(graph.get(current, ()))
    return False


def _topological_order(graph, system_priorities, profiles, static_info, params):
    indegree = {vehicle_id: 0 for vehicle_id in graph}
    for followers in graph.values():
        for follower in followers:
            indegree[follower] += 1
    remaining, output = set(graph), []
    while remaining:
        available = [value for value in remaining if indegree[value] == 0]
        if not available:
            raise ValueError("优先级硬约束形成环")
        chosen = min(available, key=lambda vehicle_id: (
            int(system_priorities.get(vehicle_id, 1)),
            float(profiles[vehicle_id].cumulativeTime[-1]),
            vehicle_id))
        output.append(chosen)
        remaining.remove(chosen)
        for follower in graph[chosen]:
            indegree[follower] -= 1
    return output


def _copy_graph(graph):
    return {vehicle_id: set(followers)
            for vehicle_id, followers in graph.items()}


def _edge_set(graph):
    return {(high, low) for high, followers in graph.items() for low in followers}


def _edge_signature(graph):
    return tuple(sorted(_edge_set(graph)))


def _materialize_conflict_edges(graph, conflicts):
    """把传递可达关系也显式保存为当前每一对冲突车辆的高->低边。"""
    output = set()
    for relation in conflicts:
        first, second = relation.vehicleA, relation.vehicleB
        if _reachable(graph, first, second):
            output.add((first, second))
        elif _reachable(graph, second, first):
            output.add((second, first))
    return output


def _delay_before(events, path_s):
    return sum(delay for position, delay in events if position <= path_s+1e-9)


def _vehicle_length(vehicle_id, static_info, params):
    return max(0.0, _number(
        static_info.get(vehicle_id, {}).get("length"), params["vehicleLength"]))


def _vehicle_width(vehicle_id, static_info, params):
    return max(0.0, _number(
        static_info.get(vehicle_id, {}).get("width"), params["vehicleWidth"]))


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
    return result if np.isfinite(result) else float(default)
