"""第十阶段：为每个协同块生成完整、无环的车辆优先级。"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from localPlanner.pathGeometry import clean_xy
from parameters import (
    COOPERATION_LATERAL_MARGIN,
    PRIORITY_CONFLICT_SAMPLE_SPACING,
    PRIORITY_CONFLICT_UPSTREAM_DISTANCE,
)


@dataclass(frozen=True)
class ConflictRelation:
    vehicleA: str
    vehicleB: str
    rawSA: float
    rawSB: float
    anchorSA: float
    anchorSB: float
    threshold: float


@dataclass(frozen=True)
class PriorityDecision:
    members: tuple
    priorityOrder: tuple
    priorityEdges: tuple
    physicalEdges: tuple
    conflicts: tuple


def calculate_block_priority(
        cooperation_block, selected_paths, physical_priorities,
        system_priorities, dxm_vehicle_ids, vehicle_static_info,
        vehicle_dynamic_info, historical_priority_edges=()):
    """依次使用物理前后、占用、系统、调箱门和群体FIFO生成优先级。"""
    members = tuple(str(value) for value in cooperation_block
                    if str(value) in selected_paths)
    graph = {vehicle_id: set() for vehicle_id in members}
    paths = {vehicle_id: _prepare_path(selected_paths[vehicle_id])
             for vehicle_id in members}

    physical_edges = add_physical_priorities(graph, physical_priorities)
    conflicts = detect_conflicts(
        members, paths, physical_edges,
        vehicle_static_info, vehicle_dynamic_info)
    history = {(str(high), str(low)) for high, low in historical_priority_edges
               if str(high) in graph and str(low) in graph}

    unresolved = []
    for relation in conflicts:
        first, second = relation.vehicleA, relation.vehicleB
        if _comparable(graph, first, second):
            continue
        edge = calculate_occupancy_priority(
            relation, vehicle_static_info, vehicle_dynamic_info, history)
        if edge is None:
            edge = calculate_system_priority(
                first, second, system_priorities, vehicle_static_info)
        if edge is None:
            edge = calculate_dxm_priority(first, second, dxm_vehicle_ids)
        if edge is None:
            unresolved.append(relation)
        else:
            _try_add_edge(graph, *edge)

    calculate_optimized_priorities(
        graph, unresolved, paths, vehicle_static_info,
        vehicle_dynamic_info, history)
    retained_history = _retain_history_edges(graph, history)
    order = _topological_order(graph, system_priorities, vehicle_static_info)
    priority_edges = set(physical_edges)
    priority_edges.update(_materialize_conflict_edges(graph, conflicts))
    priority_edges.update(retained_history)
    return PriorityDecision(
        members, tuple(order), tuple(sorted(priority_edges)),
        tuple(sorted(physical_edges)), tuple(conflicts))


def _retain_history_edges(graph, historical_priority_edges):
    """同一协同块内继续保留不与本帧约束冲突、且不会成环的历史边。"""
    retained = set()
    for high, low in sorted(historical_priority_edges):
        if _reachable(graph, low, high):
            continue
        if not _reachable(graph, high, low):
            _try_add_edge(graph, high, low)
        retained.add((high, low))
    return retained


def add_physical_priorities(graph, physical_priorities):
    """加入第九阶段的物理前车->后车硬约束。"""
    output = set()
    for high, low in physical_priorities:
        high, low = str(high), str(low)
        if high not in graph or low not in graph:
            continue
        if not _try_add_edge(graph, high, low):
            raise ValueError("物理前后优先级形成环")
        output.add((high, low))
    return output


def calculate_occupancy_priority(
        relation, vehicle_static_info, vehicle_dynamic_info,
        historical_priority_edges):
    """计算一对车辆的占用优先级；无法定向时返回None。"""
    first, second = relation.vehicleA, relation.vehicleB
    first_reach = (0.5 * float(vehicle_static_info[first]['length'])
                   + _longitudinal_uncertainty(vehicle_dynamic_info[first]))
    second_reach = (0.5 * float(vehicle_static_info[second]['length'])
                    + _longitudinal_uncertainty(vehicle_dynamic_info[second]))
    first_inside = relation.rawSA <= first_reach
    second_inside = relation.rawSB <= second_reach
    if first_inside != second_inside:
        return (first, second) if first_inside else (second, first)
    if first_inside and second_inside:
        if (first, second) in historical_priority_edges:
            return first, second
        if (second, first) in historical_priority_edges:
            return second, first
    return None


def calculate_system_priority(
        first, second, system_priorities, vehicle_static_info):
    """比较内部整数系统优先级；数值越小优先级越高。"""
    first_value = _system_priority(
        first, system_priorities, vehicle_static_info)
    second_value = _system_priority(
        second, system_priorities, vehicle_static_info)
    if first_value == second_value:
        return None
    return (first, second) if first_value < second_value else (second, first)


def calculate_dxm_priority(first, second, dxm_vehicle_ids):
    """系统优先级相同时，正在执行调箱门任务的AGV优先。"""
    first_dxm = first in dxm_vehicle_ids
    second_dxm = second in dxm_vehicle_ids
    if first_dxm == second_dxm:
        return None
    return (first, second) if first_dxm else (second, first)


def calculate_optimized_priorities(
        graph, unresolved_relations, paths, vehicle_static_info,
        vehicle_dynamic_info, historical_priority_edges):
    """可替换的群体优化接口：先复用合法历史边，再用FIFO补全。"""
    relation_by_pair = {
        frozenset((item.vehicleA, item.vehicleB)): item
        for item in unresolved_relations}
    for high, low in sorted(historical_priority_edges):
        if (frozenset((high, low)) in relation_by_pair
                and not _comparable(graph, high, low)):
            _try_add_edge(graph, high, low)

    eta_tables = {
        vehicle_id: _build_eta_table(
            paths[vehicle_id], vehicle_dynamic_info[vehicle_id])
        for vehicle_id in graph}
    pending = [item for item in unresolved_relations
               if not _comparable(graph, item.vehicleA, item.vehicleB)]
    pending.sort(key=lambda item: _fifo_sort_key(
        item, eta_tables, vehicle_static_info))
    for relation in pending:
        first, second = relation.vehicleA, relation.vehicleB
        if _comparable(graph, first, second):
            continue
        first_eta, second_eta = _relation_eta(
            relation, eta_tables, vehicle_static_info)
        preferred = ((first, second) if (first_eta, first) <= (second_eta, second)
                     else (second, first))
        if not _try_add_edge(graph, *preferred):
            _try_add_edge(graph, preferred[1], preferred[0])


def detect_conflicts(
        members, paths, physical_edges,
        vehicle_static_info, vehicle_dynamic_info):
    """只为尚未由物理关系定向的车辆对计算占用走廊冲突。"""
    corridor_paths, backward_offsets = {}, {}
    original_lengths, lateral_radii, boxes = {}, {}, {}
    for vehicle_id in members:
        dynamic = vehicle_dynamic_info[vehicle_id]
        longitudinal_radius = (
            0.5 * float(vehicle_static_info[vehicle_id]['length'])
            + _longitudinal_uncertainty(dynamic))
        lateral_radii[vehicle_id] = (
            0.5 * float(vehicle_static_info[vehicle_id]['width'])
            + _lateral_uncertainty(dynamic) + COOPERATION_LATERAL_MARGIN)
        original = paths[vehicle_id]['points']
        corridor = _extend_path(
            original, longitudinal_radius, float(dynamic['heading']))
        corridor_paths[vehicle_id] = _prepare_path(corridor)
        backward_offsets[vehicle_id] = longitudinal_radius
        original_lengths[vehicle_id] = paths[vehicle_id]['total']
        boxes[vehicle_id] = np.min(corridor, axis=0), np.max(corridor, axis=0)

    physical_pairs = {frozenset(edge) for edge in physical_edges}
    output = []
    for first, second in combinations(members, 2):
        if frozenset((first, second)) in physical_pairs:
            continue
        threshold = lateral_radii[first] + lateral_radii[second]
        first_lower, first_upper = boxes[first]
        second_lower, second_upper = boxes[second]
        axis_gap = np.maximum(
            0.0, np.maximum(first_lower - second_upper,
                            second_lower - first_upper))
        if float(axis_gap @ axis_gap) > threshold * threshold:
            continue
        first_entry = _first_corridor_entry(
            corridor_paths[first], corridor_paths[second], threshold)
        second_entry = _first_corridor_entry(
            corridor_paths[second], corridor_paths[first], threshold)
        if first_entry is None or second_entry is None:
            continue
        raw_first = float(np.clip(
            first_entry - backward_offsets[first], 0.0,
            original_lengths[first]))
        raw_second = float(np.clip(
            second_entry - backward_offsets[second], 0.0,
            original_lengths[second]))
        output.append(ConflictRelation(
            first, second, raw_first, raw_second,
            max(0.0, raw_first - PRIORITY_CONFLICT_UPSTREAM_DISTANCE
                - _longitudinal_uncertainty(vehicle_dynamic_info[first])),
            max(0.0, raw_second - PRIORITY_CONFLICT_UPSTREAM_DISTANCE
                - _longitudinal_uncertainty(vehicle_dynamic_info[second])),
            threshold))
    return output


def _prepare_path(path):
    array = np.asarray(path, dtype=float)
    points = clean_xy(array)
    if len(points) == 0:
        points = np.zeros((1, 2), dtype=float)
    speed_limits = None
    if array.ndim == 2 and array.shape[1] >= 3 and len(array) == len(points):
        speed_limits = np.maximum(0.0, array[:, 2])
    if len(points) == 1:
        return {'points': points, 'arc': np.array([0.0]), 'total': 0.0,
                'segments': np.empty((0, 2)), 'lengthSquared': np.empty(0),
                'speedLimits': speed_limits}
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    return {'points': points, 'arc': arc, 'total': float(arc[-1]),
            'segments': segments,
            'lengthSquared': np.sum(segments * segments, axis=1),
            'speedLimits': speed_limits}


def _extend_path(points, distance, heading):
    if len(points) == 1:
        start_tangent = end_tangent = np.array(
            [np.cos(heading), np.sin(heading)])
    else:
        start_tangent = _unit(points[1] - points[0])
        end_tangent = _unit(points[-1] - points[-2])
    return np.vstack((points[0] - distance * start_tangent, points,
                      points[-1] + distance * end_tangent))


def _first_corridor_entry(path, other, threshold):
    if path['total'] <= 1e-9:
        return (0.0 if _point_to_path_distance(path['points'][0], other)
                <= threshold else None)
    spacing = max(0.25, float(PRIORITY_CONFLICT_SAMPLE_SPACING))
    sample_s = np.r_[np.arange(0.0, path['total'], spacing), path['total']]
    distances = _points_to_path_distances(_points_at_s(path, sample_s), other)
    entries = np.flatnonzero(distances <= threshold)
    if len(entries) == 0:
        return None
    index = int(entries[0])
    if index == 0:
        return 0.0
    lower, upper = float(sample_s[index - 1]), float(sample_s[index])
    for _ in range(8):
        middle = 0.5 * (lower + upper)
        point = _points_at_s(path, np.array([middle]))[0]
        if _point_to_path_distance(point, other) <= threshold:
            upper = middle
        else:
            lower = middle
    return upper


def _points_at_s(path, values):
    if len(path['points']) == 1:
        return np.repeat(path['points'], len(values), axis=0)
    values = np.clip(values, 0.0, path['total'])
    indices = np.clip(np.searchsorted(path['arc'], values, side='right') - 1,
                      0, len(path['segments']) - 1)
    lengths = path['arc'][indices + 1] - path['arc'][indices]
    ratios = np.divide(values - path['arc'][indices], lengths,
                       out=np.zeros(len(values)), where=lengths > 1e-12)
    return path['points'][indices] + ratios[:, None] * path['segments'][indices]


def _points_to_path_distances(points, path):
    if len(path['segments']) == 0:
        return np.linalg.norm(points - path['points'][0], axis=1)
    starts, output = path['points'][:-1], np.empty(len(points))
    for begin in range(0, len(points), 128):
        block = points[begin:begin + 128]
        offsets = block[:, None, :] - starts[None, :, :]
        ratios = np.divide(
            np.einsum('bsi,si->bs', offsets, path['segments']),
            path['lengthSquared'][None, :],
            out=np.zeros((len(block), len(starts))),
            where=path['lengthSquared'][None, :] > 1e-12)
        residuals = offsets - np.clip(
            ratios, 0.0, 1.0)[:, :, None] * path['segments'][None, :, :]
        output[begin:begin + len(block)] = np.sqrt(
            np.min(np.sum(residuals * residuals, axis=2), axis=1))
    return output


def _point_to_path_distance(point, path):
    return float(_points_to_path_distances(point[None, :], path)[0])


def _build_eta_table(path, dynamic_info):
    arc = path['arc']
    if len(arc) == 1:
        return arc, np.array([0.0])
    current_speed = max(0.5, float(dynamic_info['speed']))
    segment_speed = np.full(len(arc) - 1, current_speed)
    limits = path['speedLimits']
    if limits is not None:
        segment_caps = np.maximum(
            0.5, 0.5 * (limits[:-1] + limits[1:]))
        segment_speed = np.minimum(segment_speed, segment_caps)
    return arc, np.r_[0.0, np.cumsum(np.diff(arc) / segment_speed)]


def _relation_eta(relation, eta_tables, vehicle_static_info):
    first_target = max(0.0, relation.anchorSA
                       - 0.5 * float(vehicle_static_info[relation.vehicleA]['length']))
    second_target = max(0.0, relation.anchorSB
                        - 0.5 * float(vehicle_static_info[relation.vehicleB]['length']))
    return (_eta_at(eta_tables[relation.vehicleA], first_target),
            _eta_at(eta_tables[relation.vehicleB], second_target))


def _fifo_sort_key(relation, eta_tables, vehicle_static_info):
    first_eta, second_eta = _relation_eta(
        relation, eta_tables, vehicle_static_info)
    return min(first_eta, second_eta), max(first_eta, second_eta), (
        relation.vehicleA, relation.vehicleB)


def _eta_at(table, target):
    arc, cumulative = table
    if target <= arc[-1]:
        return float(np.interp(target, arc, cumulative))
    speed = max(0.5, arc[-1] / max(cumulative[-1], 1e-9))
    return float(cumulative[-1] + (target - arc[-1]) / speed)


def _system_priority(vehicle_id, system_priorities, vehicle_static_info):
    if vehicle_id in system_priorities:
        return int(system_priorities[vehicle_id])
    return 1 if vehicle_static_info[vehicle_id].get('vehicleType') == 'AGV' else 0


def _topological_order(graph, system_priorities, vehicle_static_info):
    indegree = {vehicle_id: 0 for vehicle_id in graph}
    for followers in graph.values():
        for follower in followers:
            indegree[follower] += 1
    remaining, output = set(graph), []
    while remaining:
        available = [vehicle_id for vehicle_id in remaining
                     if indegree[vehicle_id] == 0]
        if not available:
            raise ValueError("优先级图形成环")
        chosen = min(available, key=lambda vehicle_id: (
            _system_priority(vehicle_id, system_priorities,
                             vehicle_static_info), vehicle_id))
        output.append(chosen)
        remaining.remove(chosen)
        for follower in graph[chosen]:
            indegree[follower] -= 1
    return output


def _try_add_edge(graph, high, low):
    if high == low or _reachable(graph, low, high):
        return False
    graph[high].add(low)
    return True


def _comparable(graph, first, second):
    return _reachable(graph, first, second) or _reachable(graph, second, first)


def _reachable(graph, source, target):
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


def _materialize_conflict_edges(graph, conflicts):
    output = set()
    for relation in conflicts:
        first, second = relation.vehicleA, relation.vehicleB
        if _reachable(graph, first, second):
            output.add((first, second))
        elif _reachable(graph, second, first):
            output.add((second, first))
    return output


def _longitudinal_uncertainty(dynamic_info):
    return max(0.0, float(dynamic_info.get('longitudinalUncertainty', 0.0)))


def _lateral_uncertainty(dynamic_info):
    return max(0.0, float(dynamic_info.get('lateralUncertainty', 0.0)))


def _unit(vector):
    return vector / float(np.linalg.norm(vector))
