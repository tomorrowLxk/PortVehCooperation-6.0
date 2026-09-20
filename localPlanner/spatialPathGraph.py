"""Old waypoint-planner spatial graph, separated from waypoint/speed logic."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.path_corridor_collision import (
    BlockedPathInterval,
    blocked_path_intervals,
    path_corridor_intersects,
)
from tools.path_geometry import (
    arc_lengths,
    as_xy,
    heading_at_s,
    headings_at_s,
    point_at_s,
    points_at_s,
    polyline_length,
    project_points_to_polyline,
    project_to_polyline,
    resample_polyline,
    slice_from_s,
)


def build_spatial_graph(lines: Sequence[np.ndarray], current: np.ndarray,
                        current_heading: float, reference: np.ndarray,
                        obstacles: Iterable[Any], params: Dict[str, Any],
                        vehicle_info: Dict[str, float],
                        longitudinal_uncertainty: float,
                        lateral_uncertainty: float,
                        reverse_point: Optional[np.ndarray] = None):
    node_spacing = float(params["pathNodeSpacing"])
    start_offset = float(params["pathGraphStartOffset"])
    max_lateral = float(params["pathMaxLateralConnection"])
    max_start = float(params["pathMaxStartConnectionLength"])
    min_change = float(params["pathMinLaneChangeLength"])
    max_change = float(params["pathMaxLaneChangeLength"])
    lane_change_penalty = float(params["pathLaneChangePenalty"])
    curvature_weight = float(params["pathCurvatureWeight"])
    width = float(vehicle_info["width"])
    length = float(vehicle_info["length"])
    min_radius = max(0.1, float(vehicle_info["minTurningRadius"]))
    corridor_half_width = (0.5 * width + lateral_uncertainty
                           + float(params["pathObstacleSafetyMargin"]))
    stop_clearance = (0.5 * length + longitudinal_uncertainty
                      + float(params["pathStopClearance"]))
    curvature_limit = float(params["pathCurvatureTolerance"]) / min_radius
    curvature_window = float(params["pathCurvatureWindow"])

    projections = [project_to_polyline(current, line) for line in lines]
    current_line = int(np.argmin([abs(item["d"]) for item in projections]))
    reference_start_s = float(project_to_polyline(current, reference)["s"])
    endpoint_projections = project_points_to_polyline(
        np.asarray([line[-1] for line in lines]), reference)
    goal_line = min(range(len(lines)), key=lambda index: (
        abs(float(endpoint_projections["d"][index])),
        -float(endpoint_projections["s"][index]), index))

    line_intervals = {
        index: blocked_path_intervals(line, obstacles, corridor_half_width)
        for index, line in enumerate(lines)
    }
    line_nodes: Dict[int, List[dict]] = {index: [] for index in range(len(lines))}
    line_arcs = [arc_lengths(line) for line in lines]
    line_totals = [float(values[-1]) for values in line_arcs]

    for line_index, line in enumerate(lines):
        total = line_totals[line_index]
        start_s = float(projections[line_index]["s"])
        samples = np.arange(start_s + start_offset, total, node_spacing)
        samples = samples[samples < total - 0.5]
        sample_points = points_at_s(line, samples)
        sample_headings = headings_at_s(line, samples)
        sample_reference = project_points_to_polyline(sample_points, reference)
        for sample_index, (line_s, point, heading) in enumerate(zip(
                samples, sample_points, sample_headings)):
            if _inside_blocked(line_s, line_intervals[line_index]):
                continue
            line_nodes[line_index].append(_node(
                f"P_{line_index}_{sample_index}", line_index, line_s,
                point, heading,
                float(sample_reference["s"][sample_index]-reference_start_s)
                + (1.0 if goal_line == line_index else 0.0),
                abs(float(sample_reference["d"][sample_index])),
                "P"))

    # Polygon fences create explicit upstream T nodes.  Same-line edges are
    # later rejected by interval overlap, so they do not need repeated polygon
    # tests.
    stop_nodes = []
    for line_index, intervals in line_intervals.items():
        line = lines[line_index]
        start_s = float(projections[line_index]["s"])
        for interval_index, interval in enumerate(intervals):
            stop_s = max(start_s, float(interval.startS - stop_clearance))
            if stop_s >= line_totals[line_index] - 1e-9:
                continue
            point = point_at_s(line, stop_s)
            terminal = _node(
                f"T_{line_index}_{interval_index}", line_index, stop_s,
                point, heading_at_s(line, stop_s),
                _progress(point, reference, reference_start_s,
                          goal_line == line_index),
                _lateral(point, reference), "T", True,
                "STATIC_OBSTACLE_STOP")
            line_nodes[line_index].append(terminal)
            stop_nodes.append(terminal)

    goal_nodes = []
    for line_index, line in enumerate(lines):
        total = line_totals[line_index]
        point = line[-1].copy()
        terminal_progress = (
            float(endpoint_projections["s"][line_index]-reference_start_s)
            + (1.0 if goal_line == line_index else 0.0))
        if line_nodes[line_index]:
            # Projection onto a finite futurePath saturates at its endpoint.
            # Keep G topologically downstream of the final P node so an early
            # lane change can follow the target line all the way to its goal.
            terminal_progress = max(
                terminal_progress,
                max(item["progress"] for item in line_nodes[line_index]) + 1e-3)
        terminal = _node(
            f"G_{line_index}", line_index, total, point,
            heading_at_s(line, total), terminal_progress,
            abs(float(endpoint_projections["d"][line_index])), "G", False, None,
            goal_line == line_index)
        line_nodes[line_index].append(terminal)
        goal_nodes.append(terminal)

    reverse_node = _make_reverse_node(
        reverse_point, lines, projections, reference, reference_start_s,
        goal_line, float(params["reversePointLocateTolerance"]))
    if reverse_node is not None:
        line_nodes[reverse_node["lineIndex"]].append(reverse_node)

    for line_index, group in line_nodes.items():
        if reverse_node is not None:
            group[:] = [item for item in group
                        if item["progress"] <= reverse_node["progress"]+1e-6]
        group.sort(key=lambda item: item["lineS"])
        # R/T可能与普通P/G重合，合并为停车节点，避免零长度边断开图。
        unique = []
        for item in group:
            if unique and abs(item["lineS"]-unique[-1]["lineS"]) <= 1e-6:
                if item["kind"] == "R" or (item["terminalStop"] and unique[-1]["kind"] != "R"):
                    unique[-1] = item
            else:
                unique.append(item)
        line_nodes[line_index] = unique

    start = _node("S", current_line, float(projections[current_line]["s"]),
                  current.copy(), current_heading, 0.0, 0.0, "S")
    nodes = {"S": start}
    for group in line_nodes.values():
        nodes.update({item["id"]: item for item in group})
    adjacency = {node_id: [] for node_id in nodes}

    # Centreline edges use precomputed blocked s intervals.
    for line_index, group in line_nodes.items():
        line = lines[line_index]
        for first, second in zip(group[:-1], group[1:]):
            if first["terminalStop"] or second["progress"] <= first["progress"] + 1e-6:
                continue
            if _range_blocked(first["lineS"], second["lineS"],
                              line_intervals[line_index]):
                continue
            sample_count = max(
                2, int(np.ceil((second["lineS"]-first["lineS"])/1.0))+1)
            curve = points_at_s(line, np.linspace(
                first["lineS"], second["lineS"], sample_count))
            _add_edge(adjacency, first, second, curve, "same", False,
                      reference, lane_change_penalty, curvature_weight,
                      curvature_limit, curvature_window)

    # S can connect to the current line or a nearby line exactly as in the old
    # waypoint planner, but the complete connector corridor must be fence-free.
    for line_index, group in line_nodes.items():
        for target in group:
            if target["progress"] <= 0.0:
                continue
            if line_index == current_line:
                line_start = float(projections[line_index]["s"])
                projection_point = np.asarray(
                    projections[line_index]["point"], dtype=float)
                line_heading = heading_at_s(lines[line_index], line_start)
                heading_error = abs(math.atan2(
                    math.sin(float(current_heading)-line_heading),
                    math.cos(float(current_heading)-line_heading)))
                requires_smooth_start = (
                    np.linalg.norm(current-projection_point) > 1e-3
                    or heading_error > math.radians(1.0))
                if (requires_smooth_start
                        and target["lineS"] > line_start + 1e-6):
                    # During a rolling lane change the current position is
                    # normally between two centrelines. Prepending it to the
                    # newly-nearest centreline creates a polyline corner.
                    curve = hermite_connector(
                        current, current_heading,
                        target["point"], target["heading"])
                    lane_change = True
                    connection_length = polyline_length(curve)
                elif target["lineS"] > line_start + 1e-6:
                    sample_count = max(
                        2, int(np.ceil((target["lineS"]-line_start)/1.0))+1)
                    curve = points_at_s(lines[line_index], np.linspace(
                        line_start, target["lineS"], sample_count))
                    lane_change = False
                    connection_length = target["lineS"]-line_start
                else:
                    curve = np.asarray([current, target["point"]], dtype=float)
                    lane_change = requires_smooth_start
                    connection_length = polyline_length(curve)
                if (not requires_smooth_start and len(curve)
                        and np.linalg.norm(curve[0]-current) > 1e-6):
                    curve = np.vstack((current, curve))
                connection_length = max(connection_length, 1e-6)
            else:
                curve = hermite_connector(
                    current, current_heading, target["point"], target["heading"])
                lane_change = True
                connection_length = polyline_length(curve)
            if len(curve) < 2 or connection_length > max_start:
                continue
            if path_corridor_intersects(curve, obstacles, corridor_half_width):
                continue
            _add_edge(adjacency, start, target, curve, "start", lane_change,
                      reference, lane_change_penalty, curvature_weight,
                      curvature_limit, curvature_window)

    # Preserve the old exhaustive 20--40 m inter-line lattice connections.
    for source_index, source_group in line_nodes.items():
        for target_index, target_group in line_nodes.items():
            if source_index == target_index:
                continue
            for source in source_group:
                if source["terminalStop"] or source["progress"] < -0.5:
                    continue
                for target in target_group:
                    delta = target["progress"] - source["progress"]
                    if delta < min_change or delta > max_change:
                        continue
                    if abs(project_to_polyline(source["point"], lines[target_index])["d"]) > max_lateral:
                        continue
                    if abs(project_to_polyline(target["point"], lines[source_index])["d"]) > max_lateral:
                        continue
                    curve = hermite_connector(
                        source["point"], source["heading"],
                        target["point"], target["heading"])
                    if path_corridor_intersects(curve, obstacles,
                                                corridor_half_width):
                        continue
                    _add_edge(adjacency, source, target, curve, "lane_change", True,
                              reference, lane_change_penalty, curvature_weight,
                              curvature_limit, curvature_window)

    if reverse_node is not None:
        terminals = [reverse_node] + sorted(
            [item for item in stop_nodes
             if item["id"] in nodes and item["progress"] < reverse_node["progress"]],
            key=lambda item: item["progress"])
    else:
        terminals = sorted([item for item in goal_nodes if item["id"] in nodes], key=lambda item: (
            item["lateral"], -item["progress"], item["lineIndex"]))
        terminals += sorted([item for item in stop_nodes if item["id"] in nodes],
                            key=lambda item: item["progress"])
    return {
        "lines": lines, "nodes": nodes, "adjacency": adjacency,
        "terminals": terminals, "currentLine": current_line,
        "curvatureLimit": curvature_limit,
    }


def hermite_connector(start, start_heading, end, end_heading, samples=41):
    '''旧版 2.0 跨可行线连接：在全局 XY 中满足两端位置与方向。'''
    start = np.asarray(start, dtype=float)[:2]
    end = np.asarray(end, dtype=float)[:2]
    chord = float(np.linalg.norm(end-start))
    if chord <= 1e-9:
        return start.reshape(1, 2)
    first_derivative = chord*np.asarray(
        [math.cos(float(start_heading)), math.sin(float(start_heading))])
    last_derivative = chord*np.asarray(
        [math.cos(float(end_heading)), math.sin(float(end_heading))])
    u = np.linspace(0.0, 1.0, max(2, int(samples)))
    h00 = 2*u**3-3*u**2+1
    h10 = u**3-2*u**2+u
    h01 = -2*u**3+3*u**2
    h11 = u**3-u**2
    return (h00[:, None]*start+h10[:, None]*first_derivative
            + h01[:, None]*end+h11[:, None]*last_derivative)


def path_curvatures(path, window=5.0):
    """Estimate curvature with the old planner's arc-length window method."""
    points = as_xy(path)
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)
    cumulative = arc_lengths(points)
    total = float(cumulative[-1])
    half_window = max(0.1, float(window)) / 2.0
    segments = np.diff(points, axis=0)
    segment_heading = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))

    lower = np.maximum(0.0, cumulative-half_window)
    upper = np.minimum(total, cumulative+half_window)
    if total > 2.0*half_window:
        near_start = cumulative-half_window < 0.0
        near_end = cumulative+half_window > total
        lower[near_start] = 0.0
        upper[near_start] = 2.0*half_window
        lower[near_end] = total-2.0*half_window
        upper[near_end] = total

    lower_index = np.clip(
        np.searchsorted(cumulative, lower, side="right")-1,
        0, len(segment_heading)-1)
    upper_index = np.clip(
        np.searchsorted(cumulative, upper, side="right")-1,
        0, len(segment_heading)-1)
    delta = np.arctan2(
        np.sin(segment_heading[upper_index]-segment_heading[lower_index]),
        np.cos(segment_heading[upper_index]-segment_heading[lower_index]))
    return np.abs(delta) / np.maximum(upper-lower, 1e-6)


def _add_edge(adjacency, source, target, curve, kind, lane_change,
              reference, lane_change_penalty, curvature_weight,
              curvature_limit, curvature_window):
    curve = as_xy(curve)
    if len(curve) < 2 or target["progress"] <= source["progress"] + 1e-6:
        return
    maximum_curvature = float(np.max(path_curvatures(curve, curvature_window)))
    cost = (target["lateral"] if kind == "start"
            else 0.5*(source["lateral"]+target["lateral"]))
    if lane_change:
        cost += lane_change_penalty + curvature_weight * maximum_curvature
    adjacency[source["id"]].append({
        "from": source["id"], "to": target["id"], "kind": kind,
        "curve": curve, "cost": float(cost),
        "maxCurvature": maximum_curvature,
        # 与2.0一致：地图中心线及同车道起步边不在短边层面淘汰。
        # 只有人工生成的换道曲线先做局部曲率检查，完整path仍会复检。
        "curvatureOK": (not lane_change
                        or maximum_curvature <= curvature_limit + 1e-9),
        "targetLine": target["lineIndex"],
    })


def _make_reverse_node(reverse_point, lines, projections, reference,
                       reference_start_s, goal_line, tolerance):
    if reverse_point is None:
        return None
    point = np.asarray(reverse_point, dtype=float).reshape(-1)[:2]
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    candidates = []
    for index, line in enumerate(lines):
        projection = project_to_polyline(point, line)
        residual = point-projection["point"]
        along = float(np.dot(residual, projection["tangent"]))
        if ((projection["s"] >= polyline_length(line)-1e-6 and along > 1e-6)
                or (projection["s"] <= 1e-6 and along < -1e-6)):
            continue
        if (projection["s"] > projections[index]["s"] + 1e-6
                and projection["s"] <= polyline_length(line) + 1e-6):
            # d只是法向距离；R在线段延长线上时d也可为0，必须看实际投影距离。
            candidates.append((float(np.linalg.norm(point-projection["point"])),
                               index, projection))
    if not candidates:
        return None
    distance, line_index, projection = min(candidates)
    if distance > tolerance:
        return None
    line_s = float(projection["s"])
    return _node(
        "R", line_index, line_s, point, heading_at_s(lines[line_index], line_s),
        _progress(point, reference, reference_start_s,
                  goal_line == line_index),
        _lateral(point, reference), "R", True, "REVERSE_POINT")


def _node(node_id, line_index, line_s, point, heading, progress, lateral,
          kind, terminal_stop=False, terminal_reason=None, is_global=False):
    return {
        "id": node_id, "lineIndex": int(line_index), "lineS": float(line_s),
        "point": np.asarray(point, dtype=float)[:2], "heading": float(heading),
        "progress": float(progress), "lateral": float(lateral), "kind": kind,
        "terminalStop": bool(terminal_stop), "terminalReason": terminal_reason,
        "isGlobal": bool(is_global),
    }


def _progress(point, reference, reference_start_s, preferred):
    value = float(project_to_polyline(point, reference)["s"] - reference_start_s)
    return value + (1.0 if preferred else 0.0)


def _lateral(point, reference):
    return abs(float(project_to_polyline(point, reference)["d"]))


def _inside_blocked(value, intervals):
    return any(item.startS - 1e-9 <= value <= item.endS + 1e-9
               for item in intervals)


def _range_blocked(start, end, intervals):
    return any(start < item.endS - 1e-9 and end > item.startS + 1e-9
               for item in intervals)
