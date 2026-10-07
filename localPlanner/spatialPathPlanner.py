"""单车PTGR空间图、懒曲线验证与候选path生成。"""

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from localPlanner.pathCollision import (
    first_polygon_stop_s,
    interval_contains,
    interval_crossed,
    line_blocked_intervals,
    path_vehicle_intersects,
)
from localPlanner.pathGeometry import (
    append_polyline,
    clean_xy,
    frenet_hermite_connector,
    frenet_connector,
    hermite_connector,
    path_curvatures,
    project_points_to_polyline,
    resample_polyline,
    truncate_polyline,
)
from parameters import (
    PATH_CURVATURE_TOLERANCE,
    PATH_CURVATURE_WINDOW,
    PATH_COLLISION_SAMPLE_SPACING,
    PATH_CURVE_SAMPLE_SPACING,
    PATH_G0_LATERAL_TOLERANCE,
    PATH_GRAPH_START_OFFSET,
    PATH_LANE_CHANGE_LONGITUDINAL_RATIO,
    PATH_LANE_CHANGE_PENALTY,
    PATH_LATERAL_SAFETY_MARGIN,
    PATH_MAX_CANDIDATES,
    PATH_MAX_GRAPH_ITERATIONS,
    PATH_MAX_LANE_CHANGE_LONGITUDINAL_DISTANCE,
    PATH_MAX_LATERAL_CONNECTION_DISTANCE,
    PATH_MAX_SIGNATURES_PER_NODE,
    PATH_NODE_SPACING,
    PATH_OBSTACLE_REAR_CLEARANCE,
    PATH_OUTPUT_POINT_SPACING,
    PATH_REFERENCE_WEIGHT,
    PATH_SEARCH_VALIDATION_BATCH_SIZE,
    PATH_STATIC_STOP_DISTANCE,
    REVERSE_POINT_LOCATE_TOLERANCE,
)


@dataclass
class PathCandidate:
    xy: np.ndarray
    score: float
    lineSequence: tuple
    maxCurvature: float
    terminalStop: bool = False
    terminalReason: str | None = None
    degraded: bool = False
    degradationReason: str | None = None


@dataclass
class PathPlanResult:
    success: bool
    candidates: list[PathCandidate] = field(default_factory=list)
    failureReason: str | None = None
    searchNodePaths: tuple[np.ndarray, ...] = ()


@dataclass
class _Node:
    id: str
    point: np.ndarray
    progress: float
    lateral: float
    referenceS: float
    referenceLateral: float
    roles: set[str] = field(default_factory=set)
    lineS: dict[int, float] = field(default_factory=dict)
    headings: dict[int, float] = field(default_factory=dict)


@dataclass
class _Edge:
    id: int
    source: str
    target: str
    sourceLine: int | None
    targetLine: int
    kind: str
    geometry: np.ndarray | None = None
    state: str = "UNKNOWN"
    maxCurvature: float = 0.0
    cost: float = 0.0


@dataclass
class _Graph:
    lines: list[np.ndarray]
    lineMetrics: list
    referenceLine: np.ndarray
    nodes: dict[str, _Node]
    edges: dict[int, _Edge]
    adjacency: dict[str, list[int]]
    goals: list[str]
    stops: list[str]
    primaryLine: int | None
    currentLine: int


@dataclass
class _LineMetric:
    """一次规划内复用的折线弧长、方向和投影数据。"""
    points: np.ndarray
    arc: np.ndarray
    segments: np.ndarray
    segmentLengths: np.ndarray
    headings: np.ndarray

    @classmethod
    def from_line(cls, line):
        points = clean_xy(line)
        segments = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        return cls(points, arc, segments, segment_lengths, headings)

    @property
    def total(self):
        return float(self.arc[-1])

    def values_at(self, values):
        targets = np.clip(
            np.asarray(values, dtype=float), 0.0, self.arc[-1])
        indices = np.minimum(
            np.searchsorted(self.arc, targets, side="right") - 1,
            len(self.points) - 2,
        )
        indices = np.maximum(indices, 0)
        lengths = self.segmentLengths[indices]
        ratios = np.divide(
            targets - self.arc[indices], lengths,
            out=np.zeros(len(targets), dtype=float),
            where=lengths > 1e-12,
        )
        points = (
            self.points[indices]
            + ratios[:, None] * self.segments[indices]
        )
        return points, self.headings[indices]

    def project_many(self, query_points):
        queries = np.asarray(query_points, dtype=float).reshape(-1, 2)
        lengths_squared = self.segmentLengths * self.segmentLengths
        residuals = queries[:, None, :] - self.points[:-1][None, :, :]
        ratios = np.divide(
            np.sum(residuals * self.segments[None, :, :], axis=2),
            lengths_squared[None, :],
            out=np.zeros((len(queries), len(self.segments)), dtype=float),
            where=lengths_squared[None, :] > 1e-12,
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = (
            self.points[:-1][None, :, :]
            + ratios[:, :, None] * self.segments[None, :, :]
        )
        differences = queries[:, None, :] - projections
        distances_squared = np.sum(differences * differences, axis=2)
        indices = np.argmin(distances_squared, axis=1)
        rows = np.arange(len(queries))
        lengths = self.segmentLengths[indices]
        tangents = np.divide(
            self.segments[indices], lengths[:, None],
            out=np.tile(np.asarray([[1.0, 0.0]]), (len(queries), 1)),
            where=lengths[:, None] > 1e-12,
        )
        residual = differences[rows, indices]
        return {
            "s": self.arc[indices] + ratios[rows, indices] * lengths,
            "d": tangents[:, 0] * residual[:, 1]
                 - tangents[:, 1] * residual[:, 0],
            "distance": np.sqrt(distances_squared[rows, indices]),
            "point": projections[rows, indices],
            "tangent": tangents,
            "segmentIndex": indices,
        }

    def project(self, point):
        values = self.project_many(np.asarray(point, dtype=float)[:2])
        return {
            key: (value[0].copy() if isinstance(value[0], np.ndarray)
                  else float(value[0]))
            for key, value in values.items()
        }

    def slice(self, start_s, end_s):
        start = float(np.clip(start_s, 0.0, self.arc[-1]))
        end = float(np.clip(end_s, start, self.arc[-1]))
        boundaries, _ = self.values_at([start, end])
        mask = (self.arc > start + 1e-9) & (self.arc < end - 1e-9)
        middle = self.points[mask]
        if end <= start + 1e-9:
            return boundaries[:1]
        return clean_xy(np.vstack((boundaries[0], middle, boundaries[1])))


def plan_spatial_path(
        drivable_lines, current_position, current_heading, reference_line,
        polygons, vehicle_static_info, vehicle_dynamic_info,
        reverse_point=None, stop_at_route_end=False):
    current = np.asarray(current_position, dtype=float)[:2]
    reference = clean_xy(reference_line)
    if (stop_at_route_end and len(reference) == 1
            and np.linalg.norm(reference[0] - current) <= 1e-6):
        return PathPlanResult(True, [PathCandidate(
            current.reshape(1, 2), 0.0, (), 0.0,
            True, "GLOBAL_ROUTE_END",
        )])
    if len(reference) < 2:
        return PathPlanResult(False, failureReason="INVALID_REFERENCE_LINE")

    lines = []
    line_lane_segments = []
    for raw_line in drivable_lines:
        line = clean_xy(getattr(raw_line, 'points', raw_line))
        if len(line) >= 2:
            lines.append(line)
            line_lane_segments.append(tuple(
                getattr(raw_line, 'laneSegments', ())
            ))
    if not lines:
        return PathPlanResult(False, failureReason="NO_DRIVABLE_LINE")
    graph = _build_graph(
        lines, line_lane_segments,
        current, float(current_heading), reference, polygons,
        vehicle_static_info, vehicle_dynamic_info, reverse_point,
    )
    searchNodePaths = []
    searchGoals = graph.goals
    terminalLine = graph.primaryLine
    if stop_at_route_end:
        terminalLine = min(
            range(len(graph.lineMetrics)),
            key=lambda index: (
                graph.lineMetrics[index].project(reference[-1])['distance'],
                index,
            ),
        )
        searchGoals = [
            node_id for node_id in graph.goals
            if terminalLine in graph.nodes[node_id].lineS
        ]

    immediate = []
    immediate.extend((node_id, "REVERSE_POINT")
                     for node_id in searchGoals
                     if "R" in graph.nodes[node_id].roles)
    if stop_at_route_end:
        immediate.extend((node_id, "GLOBAL_ROUTE_END")
                         for node_id in searchGoals
                         if "G" in graph.nodes[node_id].roles)
    immediate.extend((node_id, "STATIC_OBSTACLE_STOP")
                     for node_id in graph.stops)
    for node_id, reason in immediate:
        if np.linalg.norm(graph.nodes[node_id].point - current) <= 1e-6:
            return PathPlanResult(True, [PathCandidate(
                current.reshape(1, 2), 0.0,
                (graph.currentLine,), 0.0, True, reason,
            )])

    for goal_id in searchGoals:
        candidates, safeNodePaths, _, _ = _strict_candidates(
            graph, goal_id, reference, polygons,
            vehicle_static_info, vehicle_dynamic_info,
        )
        searchNodePaths.extend(safeNodePaths)
        if candidates:
            terminal = graph.nodes[goal_id]
            stop = "R" in terminal.roles or (
                stop_at_route_end and "G" in terminal.roles)
            reason = (
                "REVERSE_POINT" if "R" in terminal.roles
                else "GLOBAL_ROUTE_END" if stop else None
            )
            candidates = [
                PathCandidate(
                    item.xy, item.score, item.lineSequence,
                    item.maxCurvature, stop, reason,
                )
                for item in candidates
            ]
            return PathPlanResult(
                True, select_optimal_paths(candidates, reference),
                searchNodePaths=tuple(searchNodePaths))

        if safeNodePaths:
            terminal = graph.nodes[goal_id]
            terminal_reason = (
                "REVERSE_POINT" if "R" in terminal.roles
                else "GLOBAL_ROUTE_END"
                if stop_at_route_end and "G" in terminal.roles
                else None
            )
            fallback = _frenet_fallback(
                current, graph.nodes[goal_id].point, reference, polygons,
                vehicle_static_info, vehicle_dynamic_info,
                terminal_reason=terminal_reason,
            )
            return PathPlanResult(
                True, [fallback],
                searchNodePaths=tuple(searchNodePaths))

    ordered_stops = sorted(
        graph.stops,
        key=lambda node_id: (
            0 if (
                terminalLine is not None
                and terminalLine in graph.nodes[node_id].lineS
            ) else 1,
            graph.nodes[node_id].lateral,
            graph.nodes[node_id].progress,
        ),
    )
    for stop_id in ordered_stops:
        candidates, safeNodePaths, _, _ = _strict_candidates(
            graph, stop_id, reference, polygons,
            vehicle_static_info, vehicle_dynamic_info,
        )
        searchNodePaths.extend(safeNodePaths)
        if candidates:
            candidates = [
                PathCandidate(
                    item.xy, item.score, item.lineSequence,
                    item.maxCurvature, True, "STATIC_OBSTACLE_STOP",
                )
                for item in candidates
            ]
            return PathPlanResult(
                True, select_optimal_paths(candidates, reference),
                searchNodePaths=tuple(searchNodePaths))
    return PathPlanResult(
        False, failureReason="NO_SPATIAL_PATH",
        searchNodePaths=tuple(searchNodePaths))

def select_optimal_paths(candidates, reference_line):
    """???????????path?????????????????"""
    best_by_signature = {}
    for candidate in candidates:
        path = clean_xy(candidate.xy)
        lateral = float(np.mean(np.abs(
            project_points_to_polyline(path, reference_line)["d"]
        )))
        length = float(np.sum(np.linalg.norm(
            np.diff(path, axis=0), axis=1)))
        curvature = float(candidate.maxCurvature)
        laneChanges = sum(
            sourceLine != targetLine
            for sourceLine, targetLine in zip(
                candidate.lineSequence[:-1],
                candidate.lineSequence[1:],
            )
        )
        score = (
            length
            + PATH_REFERENCE_WEIGHT * lateral
            + PATH_LANE_CHANGE_PENALTY * laneChanges
        )
        value = PathCandidate(
            path, score, candidate.lineSequence, curvature,
            candidate.terminalStop, candidate.terminalReason,
            candidate.degraded, candidate.degradationReason,
        )
        previous = best_by_signature.get(value.lineSequence)
        if previous is None or value.score < previous.score:
            best_by_signature[value.lineSequence] = value
    ranked = sorted(
        best_by_signature.values(),
        key=lambda item: (item.score, len(item.lineSequence)),
    )
    selected = []
    geometry_keys = set()
    for candidate in ranked:
        geometry_key = tuple(map(tuple, np.round(candidate.xy, 3)))
        if geometry_key in geometry_keys:
            continue
        geometry_keys.add(geometry_key)
        selected.append(candidate)
        if len(selected) >= PATH_MAX_CANDIDATES:
            break
    return selected


def _build_graph(
        lines, line_lane_segments,
        current, current_heading, reference, polygons,
        static_info, dynamic_info, reverse_point):
    line_metrics = [_LineMetric.from_line(line) for line in lines]
    reference_metric = _LineMetric.from_line(reference)
    current_projections = [metric.project(current) for metric in line_metrics]
    current_line = min(
        range(len(lines)),
        key=lambda index: current_projections[index]["distance"],
    )
    reference_start_projection = reference_metric.project(current)
    reference_start_s = reference_start_projection["s"]
    nodes = {
        "S": _Node(
            "S", current.copy(), 0.0,
            abs(float(reference_start_projection["d"])),
            float(reference_start_s),
            float(reference_start_projection["d"]), {"S"},
            {current_line: float(current_projections[current_line]["s"])},
            {current_line: float(current_heading)},
        )
    }
    node_keys = {}
    line_nodes = {index: [] for index in range(len(lines))}
    blocked = {}

    length = float(static_info["length"])
    width = float(static_info['width'])
    longitudinal_uncertainty = float(
        dynamic_info.get('longitudinalUncertainty', 0.0))
    lateral_uncertainty = float(
        dynamic_info.get('lateralUncertainty', 0.0))
    for line_index, line in enumerate(lines):
        metric = line_metrics[line_index]
        blocked[line_index] = line_blocked_intervals(
            line, polygons, length, width,
            longitudinal_uncertainty, lateral_uncertainty,
            PATH_LATERAL_SAFETY_MARGIN,
            PATH_STATIC_STOP_DISTANCE,
            PATH_OBSTACLE_REAR_CLEARANCE,
            PATH_CURVE_SAMPLE_SPACING,
        )

        total = metric.total
        line_start_s = float(current_projections[line_index]['s'])
        sample_s = np.arange(
            line_start_s + PATH_GRAPH_START_OFFSET,
            total,
            PATH_NODE_SPACING,
        )
        lane_segments = line_lane_segments[line_index]
        if lane_segments:
            # 普通车道仍按固定间距放置P点；交叉口内虚拟车道不切分，
            # 在搜索图中只保留该车道的头、尾节点。
            sample_s = [
                float(value) for value in sample_s
                if any(
                    not segment.isJunction
                    and segment.startS - 1e-6 <= value
                    <= segment.endS + 1e-6
                    for segment in lane_segments
                )
            ]
        specifications = [
            (float(value), {"P"}, "P")
            for value in sample_s
            if not interval_contains(blocked[line_index], value)
        ]

        for segment_index, segment in enumerate(lane_segments):
            if not segment.isJunction:
                continue
            for boundary_index, value in enumerate(
                    (segment.startS, segment.endS)):
                if (value <= line_start_s + 1e-6
                        or value > total + 1e-6
                        or interval_contains(blocked[line_index], value)):
                    continue
                specifications.append((
                    float(value), {"P"},
                    f"J{segment_index}_{boundary_index}",
                ))

        for obstacle_index, interval in enumerate(blocked[line_index]):
            stop_s = float(interval.startS)
            junction_segment = next((
                segment for segment in lane_segments
                if segment.isJunction
                and segment.startS + 1e-6 < interval.startS
                < segment.endS - 1e-6
            ), None)
            if junction_segment is not None:
                if line_start_s < junction_segment.startS - 1e-6:
                    stop_s = float(junction_segment.startS)
                elif line_start_s < junction_segment.endS - 1e-6:
                    stop_s = float(line_start_s)
                else:
                    continue
            if (line_index == current_line
                    and stop_s <= line_start_s + 1e-6):
                nodes["S"].roles.add("T")
                continue
            specifications.append((
                stop_s, {"T"}, f"T{obstacle_index}"))

        if not interval_contains(blocked[line_index], total):
            specifications.append((total, {"G"}, "G"))
        values = [item[0] for item in specifications]
        points, headings = metric.values_at(values)
        reference_projections = reference_metric.project_many(points)
        for item_index, (value, roles, suffix) in enumerate(specifications):
            _add_node(
                nodes, node_keys, line_nodes, line_index, value,
                points[item_index], headings[item_index],
                float(reference_projections["s"][item_index]),
                float(reference_projections["d"][item_index]),
                float(reference_projections["s"][item_index]
                      - reference_start_s),
                abs(float(reference_projections["d"][item_index])),
                line_metrics, roles, suffix=suffix,
            )

    reverse_id = None
    if reverse_point is not None:
        reverse = np.asarray(reverse_point, dtype=float)[:2]
        ranked = []
        for line_index, metric in enumerate(line_metrics):
            projection = metric.project(reverse)
            residual = reverse - projection["point"]
            along = float(np.dot(residual, projection["tangent"]))
            if ((projection["s"] >= metric.total - 1e-6
                 and along > 1e-6)
                    or (projection["s"] <= 1e-6 and along < -1e-6)):
                continue
            if projection["distance"] > REVERSE_POINT_LOCATE_TOLERANCE:
                continue
            if projection["s"] <= current_projections[line_index]["s"] + 1e-6:
                continue
            if interval_contains(blocked[line_index], projection["s"]):
                continue
            ranked.append((projection["distance"], line_index, projection))
        if ranked:
            _, line_index, projection = min(ranked)
            reference_projection = reference_metric.project(reverse)
            reverse_id = _add_node(
                nodes, node_keys, line_nodes, line_index,
                projection["s"], reverse,
                line_metrics[line_index].values_at(
                    [projection["s"]])[1][0],
                float(reference_projection["s"]),
                float(reference_projection["d"]),
                float(reference_projection["s"] - reference_start_s),
                abs(float(reference_projection["d"])),
                line_metrics, {"R"}, suffix="R",
            )

    for line_index in line_nodes:
        unique = sorted(
            set(line_nodes[line_index]),
            key=lambda node_id: nodes[node_id].lineS[line_index],
        )
        line_nodes[line_index] = unique

    goals, primary_line = _ordered_goals(
        nodes, line_metrics, current, current_line, reverse_id)
    stops = [
        node.id for node in nodes.values() if "T" in node.roles
    ]
    edges = {}
    adjacency = {node_id: [] for node_id in nodes}
    edge_keys = set()

    # 起点先接入当前所在可行域线。这样正常直行沿零换线代价向前传播，
    # 不会为了抵达第一个P点而枚举全部跨线Hermite边。
    current_start_s = current_projections[current_line]["s"]
    current_junction_end_s = None
    for segment in line_lane_segments[current_line]:
        if (segment.isJunction
                and segment.startS - 1e-6 <= current_start_s
                < segment.endS - 1e-6):
            current_junction_end_s = float(segment.endS)
            break

    for target_id in line_nodes[current_line]:
        target = nodes[target_id]
        target_s = target.lineS[current_line]
        if target_s <= current_start_s + 1e-6:
            continue
        is_current_junction_exit = (
            current_junction_end_s is not None
            and abs(target_s - current_junction_end_s) <= 1e-6
        )
        if (target.progress > PATH_MAX_LANE_CHANGE_LONGITUDINAL_DISTANCE
                and not is_current_junction_exit):
            break
        if interval_crossed(
                blocked[current_line], current_start_s, target_s):
            continue
        projected_start = current_projections[current_line]["point"]
        heading_error = abs(math.atan2(
            math.sin(current_heading - line_metrics[current_line].values_at(
                [current_start_s])[1][0]),
            math.cos(current_heading - line_metrics[current_line].values_at(
                [current_start_s])[1][0]),
        ))
        if (np.linalg.norm(projected_start - current) <= 1e-6
                and heading_error <= math.radians(1.0)):
            geometry = line_metrics[current_line].slice(
                current_start_s, target_s)
            geometry[0] = current
        else:
            geometry = hermite_connector(
                current, current_heading,
                target.point, target.headings[current_line],
                PATH_CURVE_SAMPLE_SPACING,
            )
        _add_edge(
            edges, adjacency, edge_keys, "S", target_id,
            current_line, current_line, "SAME", geometry, nodes,
        )

    for line_index, group in line_nodes.items():
        for source_id, target_id in zip(group[:-1], group[1:]):
            source, target = nodes[source_id], nodes[target_id]
            if source.roles.intersection({"G", "T", "R"}):
                continue
            source_s = source.lineS[line_index]
            target_s = target.lineS[line_index]
            if interval_crossed(blocked[line_index], source_s, target_s):
                continue
            _add_edge(
                edges, adjacency, edge_keys, source_id, target_id,
                line_index, line_index, "SAME", None, nodes,
            )

    sources = ["S"] + [
        node.id for node in nodes.values()
        if "P" in node.roles and node.id != "S"
    ]
    source_points = np.asarray([nodes[node_id].point for node_id in sources])
    lateral_to_line = {}
    for target_line, metric in enumerate(line_metrics):
        distances = metric.project_many(source_points)["distance"]
        for source_id, distance in zip(sources, distances):
            lateral_to_line[(source_id, target_line)] = float(distance)
    for source_id in sources:
        source = nodes[source_id]
        source_lines = set(source.lineS)
        for target_line, group in line_nodes.items():
            if target_line in source_lines:
                continue
            lateral = lateral_to_line[(source_id, target_line)]
            if lateral > PATH_MAX_LATERAL_CONNECTION_DISTANCE:
                continue
            for target_id in group:
                target = nodes[target_id]
                delta = target.progress - source.progress
                if delta <= 1e-6:
                    continue
                if delta < lateral * PATH_LANE_CHANGE_LONGITUDINAL_RATIO:
                    continue
                if delta > PATH_MAX_LANE_CHANGE_LONGITUDINAL_DISTANCE:
                    continue
                source_line = (
                    min(source_lines) if source_lines else None
                )
                _add_edge(
                    edges, adjacency, edge_keys, source_id, target_id,
                    source_line, target_line, "CROSS", None, nodes,
                )

    return _Graph(
        lines, line_metrics, reference, nodes, edges, adjacency, goals, stops,
        primary_line, current_line)


def _add_node(
        nodes, node_keys, line_nodes, line_index, line_s,
        point, heading, reference_s, reference_d, progress, lateral,
        line_metrics,
        roles, suffix="P"):
    key = (
        round(float(point[0]), 6),
        round(float(point[1]), 6),
        round(float(reference_s), 3),
    )
    node_id = None
    for candidate_id in node_keys.get(key, ()):
        candidate = nodes[candidate_id]
        if any(_continuous_shared_point(
                line_metrics[line_index], line_s,
                line_metrics[other_line], other_s)
                for other_line, other_s in candidate.lineS.items()):
            node_id = candidate_id
            break
    if node_id is None:
        node_id = f"N{len(nodes)}_{suffix}"
        node_keys.setdefault(key, []).append(node_id)
        nodes[node_id] = _Node(
            node_id,
            np.asarray(point, dtype=float)[:2],
            float(progress), float(lateral), float(reference_s),
            float(reference_d),
        )
    node = nodes[node_id]
    node.roles.update(roles)
    node.lineS[line_index] = float(line_s)
    node.headings[line_index] = float(heading)
    line_nodes[line_index].append(node_id)
    return node_id


def _continuous_shared_point(first, first_s, second, second_s):
    """仅合并连续重合区段；孤立几何交叉点保持为两个节点。"""
    first_heading = first.values_at([first_s])[1][0]
    second_heading = second.values_at([second_s])[1][0]
    if math.cos(first_heading - second_heading) < 1.0 - 1e-6:
        return False
    first_total = first.total
    second_total = second.total
    for offset in (-0.5, 0.5):
        first_test = float(first_s) + offset
        second_test = float(second_s) + offset
        if (0.0 <= first_test <= first_total
                and 0.0 <= second_test <= second_total
                and np.linalg.norm(
                    first.values_at([first_test])[0][0]
                    - second.values_at([second_test])[0][0]
                ) <= 1e-5):
            return True
    return False


def _ordered_goals(nodes, line_metrics, current, current_line, reverse_id):
    if reverse_id is not None:
        line_index = min(nodes[reverse_id].lineS)
        return [reverse_id], line_index
    goals = [
        node for node in nodes.values() if "G" in node.roles
    ]
    if not goals:
        return [], None
    primary = min(goals, key=lambda item: (
        0 if item.lateral < PATH_G0_LATERAL_TOLERANCE else 1,
        item.lateral, -item.progress, item.id,
    ))
    primary_line = min(primary.lineS)

    current_on_primary = (
        line_metrics[primary_line].project(current)["distance"] <= 1e-6)
    current_side = 0.0
    if not current_on_primary and current_line != primary_line:
        current_projection = line_metrics[primary_line].project(
            line_metrics[current_line].points[-1])
        current_side = math.copysign(
            1.0, current_projection["d"]
        ) if abs(current_projection["d"]) > 1e-6 else 0.0

    def order(node):
        if node.id == primary.id:
            return (-1, 0.0, node.id)
        signed = line_metrics[primary_line].project(node.point)["d"]
        same_side = (
            current_side == 0.0
            or abs(signed) <= 1e-6
            or math.copysign(1.0, signed) == current_side
        )
        return (0 if same_side else 1, abs(signed), node.id)

    ordered = sorted(goals, key=order)
    return [item.id for item in ordered], primary_line


def _add_edge(
        edges, adjacency, edge_keys, source_id, target_id,
        source_line, target_line, kind, geometry, nodes):
    key = (source_id, target_id, source_line, target_line)
    if key in edge_keys:
        return
    edge_keys.add(key)
    edge_id = len(edges)
    source, target = nodes[source_id], nodes[target_id]
    if geometry is not None:
        cost = float(np.sum(np.linalg.norm(
            np.diff(geometry, axis=0), axis=1)))
    elif kind == "SAME":
        cost = abs(
            target.lineS[target_line] - source.lineS[target_line])
    else:
        cost = math.hypot(
            target.referenceS - source.referenceS,
            target.referenceLateral - source.referenceLateral,
        )
    cost += PATH_REFERENCE_WEIGHT * 0.5 * (
        source.lateral + target.lateral)
    if kind == "CROSS":
        cost += PATH_LANE_CHANGE_PENALTY
    edges[edge_id] = _Edge(
        edge_id, source_id, target_id, source_line, target_line,
        kind, geometry, "UNKNOWN", 0.0, cost,
    )
    adjacency[source_id].append(edge_id)


def _strict_candidates(
        graph, target_id, reference, polygons, static_info, dynamic_info):
    collision_pairs = set()
    curvature_pairs = set()
    accepted = {}
    safe_node_paths = {}

    current_line = graph.currentLine
    terminal = graph.nodes[target_id]
    if current_line in terminal.lineS:
        start_s = graph.nodes["S"].lineS[current_line]
        end_s = terminal.lineS[current_line]
        if end_s > start_s + 1e-9:
            path = graph.lineMetrics[current_line].slice(start_s, end_s)
            collision = path_vehicle_intersects(
                path, polygons,
                static_info["length"], static_info["width"],
                dynamic_info.get("longitudinalUncertainty", 0.0),
                dynamic_info.get("lateralUncertainty", 0.0),
                PATH_LATERAL_SAFETY_MARGIN,
                PATH_COLLISION_SAMPLE_SPACING,
            )
            if not collision:
                signature = (current_line,)
                ordered_nodes = sorted(
                    (
                        (node.lineS[current_line], node.point)
                        for node in graph.nodes.values()
                        if current_line in node.lineS
                        and start_s - 1e-9 <= node.lineS[current_line]
                        <= end_s + 1e-9
                    ),
                    key=lambda item: item[0],
                )
                safe_node_paths[signature] = clean_xy(np.asarray([
                    point for _, point in ordered_nodes
                ], dtype=float))
                maximum = float(np.max(path_curvatures(
                    path, PATH_CURVATURE_WINDOW
                ))) if len(path) > 2 else 0.0
                limit = (
                    PATH_CURVATURE_TOLERANCE
                    / max(float(static_info["minTurningRadius"]), 1e-6)
                )
                if maximum <= limit + 1e-9:
                    sampled = resample_polyline(
                        path, PATH_OUTPUT_POINT_SPACING)
                    scored_curvature = float(np.max(path_curvatures(
                        sampled, PATH_CURVATURE_WINDOW
                    ))) if len(sampled) > 2 else 0.0
                    accepted[signature] = PathCandidate(
                        sampled, 0.0, signature, scored_curvature,
                    )

    baseline_available = bool(accepted)
    for _ in range(PATH_MAX_GRAPH_ITERATIONS):
        paths = _search(
            graph, target_id, collision_pairs, curvature_pairs)
        if not paths:
            return (
                list(accepted.values()),
                tuple(safe_node_paths.values()),
                collision_pairs,
                curvature_pairs,
            )
        valid = []
        changed = False
        for edge_ids, signature in paths:
            edge_invalid = False
            for edge_id in edge_ids:
                edge = graph.edges[edge_id]
                previous_state = edge.state
                _validate_edge(
                    graph, edge, polygons, static_info, dynamic_info)
                if edge.state in ("COLLISION_INVALID", "CURVATURE_INVALID"):
                    edge_invalid = True
                    if previous_state == "UNKNOWN":
                        changed = True
                    break
            if edge_invalid:
                continue

            path = _join_edges(graph, edge_ids)
            collision = path_vehicle_intersects(
                path, polygons,
                static_info["length"], static_info["width"],
                dynamic_info.get("longitudinalUncertainty", 0.0),
                dynamic_info.get("lateralUncertainty", 0.0),
                PATH_LATERAL_SAFETY_MARGIN,
                PATH_COLLISION_SAMPLE_SPACING,
            )
            if collision:
                pair = _invalid_pair(
                    graph, edge_ids, polygons, static_info, dynamic_info,
                    collision=True)
                if pair is not None and pair not in collision_pairs:
                    collision_pairs.add(pair)
                    changed = True
                continue

            if (signature not in safe_node_paths
                    and len(safe_node_paths) < PATH_MAX_CANDIDATES):
                safe_node_paths[signature] = _edge_node_path(graph, edge_ids)

            maximum = float(np.max(path_curvatures(
                path, PATH_CURVATURE_WINDOW))) if len(path) > 2 else 0.0
            limit = (
                PATH_CURVATURE_TOLERANCE
                / max(float(static_info["minTurningRadius"]), 1e-6)
            )
            if maximum > limit + 1e-9:
                pair = _invalid_pair(
                    graph, edge_ids, polygons, static_info, dynamic_info,
                    collision=False)
                if pair is not None and pair not in curvature_pairs:
                    curvature_pairs.add(pair)
                    changed = True
                continue

            sampled = resample_polyline(path, PATH_OUTPUT_POINT_SPACING)
            scored_curvature = float(np.max(path_curvatures(
                sampled, PATH_CURVATURE_WINDOW
            ))) if len(sampled) > 2 else 0.0
            valid.append(PathCandidate(
                sampled, 0.0, signature, scored_curvature,
            ))

        for candidate in valid:
            if len(accepted) >= PATH_MAX_CANDIDATES:
                break
            accepted.setdefault(candidate.lineSequence, candidate)
        if len(accepted) >= PATH_MAX_CANDIDATES:
            break
        if baseline_available:
            break
        if not changed:
            break

    return (
        list(accepted.values()),
        tuple(safe_node_paths.values()),
        collision_pairs,
        curvature_pairs,
    )

def _search(graph, target_id, collision_pairs, curvature_pairs):
    queue = [(
        0.0, 0.0, 0, "S", None,
        (graph.currentLine,), tuple(),
    )]
    serial = 1
    best = {}
    results = {}
    result_limit = PATH_SEARCH_VALIDATION_BATCH_SIZE
    settled_signatures_per_node = {}
    pair_sensitive = bool(collision_pairs or curvature_pairs)
    while queue:
        cost, _, _, node_id, previous_edge, signature, edge_ids = heapq.heappop(queue)
        state_key = (
            node_id,
            previous_edge if pair_sensitive else None,
            signature,
        )
        if cost > best.get(state_key, float("inf")) + 1e-9:
            continue
        settled_signatures = settled_signatures_per_node.setdefault(node_id, set())
        if signature not in settled_signatures:
            if len(settled_signatures) >= PATH_MAX_SIGNATURES_PER_NODE:
                continue
            settled_signatures.add(signature)
        if node_id == target_id:
            results.setdefault(signature, (cost, edge_ids))
            if len(results) >= result_limit:
                break
            continue
        for edge_id in graph.adjacency.get(node_id, ()):
            edge = graph.edges[edge_id]
            if edge.state in ("COLLISION_INVALID", "CURVATURE_INVALID"):
                continue
            pair = (previous_edge, edge_id)
            if pair in collision_pairs or pair in curvature_pairs:
                continue
            new_signature = signature
            if not new_signature or new_signature[-1] != edge.targetLine:
                new_signature = signature + (edge.targetLine,)
            new_cost = cost + edge.cost
            new_key = (
                edge.target,
                edge_id if pair_sensitive else None,
                new_signature,
            )
            if new_cost >= best.get(new_key, float("inf")) - 1e-9:
                continue
            best[new_key] = new_cost
            heapq.heappush(queue, (
                new_cost, -graph.nodes[edge.target].progress,
                serial, edge.target, edge_id,
                new_signature, edge_ids + (edge_id,),
            ))
            serial += 1
    return [
        (edge_ids, signature)
        for signature, (_, edge_ids) in sorted(
            results.items(), key=lambda item: item[1][0])
    ]

def _edge_node_path(graph, edge_ids):
    if not edge_ids:
        return np.empty((0, 2), dtype=float)
    nodeIds = [graph.edges[edge_ids[0]].source]
    nodeIds.extend(graph.edges[edgeId].target for edgeId in edge_ids)
    return np.asarray([
        graph.nodes[nodeId].point for nodeId in nodeIds
    ], dtype=float)




def _validate_edge(graph, edge, polygons, static_info, dynamic_info):
    if edge.state != "UNKNOWN":
        return
    if edge.kind == "SAME" and not polygons:
        edge.state = "VALID"
        return
    _materialize_edge(graph, edge, dynamic_info)
    if path_vehicle_intersects(
            edge.geometry, polygons,
            static_info["length"], static_info["width"],
            dynamic_info.get("longitudinalUncertainty", 0.0),
            dynamic_info.get("lateralUncertainty", 0.0),
            PATH_LATERAL_SAFETY_MARGIN,
            PATH_COLLISION_SAMPLE_SPACING):
        edge.state = "COLLISION_INVALID"
        return
    if edge.kind == "SAME":
        edge.state = "VALID"
        return
    maximum = float(np.max(path_curvatures(
        edge.geometry, PATH_CURVATURE_WINDOW
    ))) if len(edge.geometry) > 2 else 0.0
    edge.maxCurvature = maximum
    limit = (
        PATH_CURVATURE_TOLERANCE
        / max(float(static_info["minTurningRadius"]), 1e-6)
    )
    edge.state = (
        "VALID" if maximum <= limit + 1e-9
        else "CURVATURE_INVALID"
    )


def _materialize_edge(graph, edge, dynamic_info=None):
    if edge.geometry is not None:
        return edge.geometry
    source, target = graph.nodes[edge.source], graph.nodes[edge.target]
    if edge.kind == "SAME":
        edge.geometry = graph.lineMetrics[edge.targetLine].slice(
            source.lineS[edge.targetLine],
            target.lineS[edge.targetLine],
        )
    else:
        source_heading = (
            float((dynamic_info or {}).get("heading", 0.0))
            if edge.source == "S"
            else source.headings[edge.sourceLine]
        )
        edge.geometry = frenet_hermite_connector(
            source.point, source_heading,
            source.referenceS,
            target.point, target.headings[edge.targetLine],
            target.referenceS,
            graph.referenceLine,
            PATH_CURVE_SAMPLE_SPACING,
        )
    return edge.geometry


def _join_edges(graph, edge_ids):
    path = graph.nodes["S"].point.reshape(1, 2)
    for edge_id in edge_ids:
        edge = graph.edges[edge_id]
        path = append_polyline(path, _materialize_edge(graph, edge))
    return clean_xy(path)


def _invalid_pair(
        graph, edge_ids, polygons, static_info, dynamic_info, collision):
    for previous, current in zip(edge_ids[:-1], edge_ids[1:]):
        joined = append_polyline(
            graph.edges[previous].geometry,
            graph.edges[current].geometry,
        )
        if collision:
            invalid = path_vehicle_intersects(
                joined, polygons,
                static_info["length"], static_info["width"],
                dynamic_info.get("longitudinalUncertainty", 0.0),
                dynamic_info.get("lateralUncertainty", 0.0),
                PATH_LATERAL_SAFETY_MARGIN,
                PATH_COLLISION_SAMPLE_SPACING,
            )
        else:
            maximum = float(np.max(path_curvatures(
                joined, PATH_CURVATURE_WINDOW)))
            invalid = maximum > (
                PATH_CURVATURE_TOLERANCE
                / max(float(static_info["minTurningRadius"]), 1e-6)
            ) + 1e-9
        if invalid:
            return previous, current
    return None


def _frenet_fallback(
        current, target, reference, polygons, static_info, dynamic_info,
        terminal_reason):
    path = frenet_connector(
        current, target, reference, PATH_OUTPUT_POINT_SPACING)
    stop_s = first_polygon_stop_s(
        path, polygons,
        static_info["length"],
        0.0,
        PATH_STATIC_STOP_DISTANCE,
        PATH_CURVE_SAMPLE_SPACING,
    )
    stop = terminal_reason == "REVERSE_POINT"
    reason = terminal_reason
    if stop_s is not None:
        path = truncate_polyline(path, stop_s)
        stop = True
        reason = "STATIC_OBSTACLE_STOP"
    maximum = float(np.max(path_curvatures(
        path, PATH_CURVATURE_WINDOW))) if len(path) > 2 else 0.0
    return PathCandidate(
        resample_polyline(path, PATH_OUTPUT_POINT_SPACING),
        0.0, ("FRENET",), maximum, stop, reason,
        True, "FRENET_FALLBACK",
    )
