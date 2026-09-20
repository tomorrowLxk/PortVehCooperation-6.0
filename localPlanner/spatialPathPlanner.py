"""Unified single-vehicle spatial path planning for PortVehCooperation 4.0.

This module deliberately stops before speed and waypoint generation.  It keeps
S/P/G/T/R nodes, XY Hermite cross-line connectors, forward dynamic
programming and one optimum per non-homotopic line sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from localPlanner.spatialPathGraph import build_spatial_graph, path_curvatures
from tools.path_corridor_collision import blocked_path_intervals
from tools.default_params import DEFAULT_PLANNING_PARAMS
from tools.path_geometry import (
    arc_lengths, as_xy, project_points_to_polyline, project_to_polyline, resample_polyline,
    truncate_polyline)


@dataclass(frozen=True)
class PathCandidate:
    xy: np.ndarray
    score: float
    lineSequence: tuple
    maxCurvature: float
    terminalStop: bool = False
    terminalReason: Optional[str] = None
    degraded: bool = False
    degradationReason: Optional[str] = None


@dataclass
class PathPlanResult:
    success: bool
    failureReason: Optional[str] = None
    candidates: List[PathCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class AGVPathPlanningJob:
    vehicleId: str
    drivableAreaPlanner: Any
    position: np.ndarray
    heading: float
    speed: float
    globalReferenceLine: np.ndarray
    obstacles: Sequence[Any]
    vehicleStaticInfo: Dict[str, float]
    longitudinalUncertainty: float = 0.0
    lateralUncertainty: float = 0.0
    reversePoint: Optional[np.ndarray] = None
    stopAtRouteEnd: bool = False
    drivableLines: Optional[Sequence[np.ndarray]] = None


def plan_spatial_path(
    drivable_lines,
    position,
    heading,
    global_reference_line,
    obstacles,
    vehicle_static_info,
    longitudinal_uncertainty=0.0,
    lateral_uncertainty=0.0,
    reverse_point=None,
    stop_at_route_end=False,
    params=None,
) -> PathPlanResult:
    """Plan one vehicle independently; no vehicle state is retained."""
    planning_params = dict(DEFAULT_PLANNING_PARAMS)
    if params:
        planning_params.update(params)
    current = np.asarray(position, dtype=float).reshape(-1)[:2]
    if current.shape != (2,) or not np.all(np.isfinite(current)):
        return PathPlanResult(False, "INVALID_VEHICLE_POSITION")
    reference = as_xy(global_reference_line)
    if reverse_point is not None and np.linalg.norm(
            current-np.asarray(reverse_point, dtype=float)[:2]) <= 1e-6:
        return PathPlanResult(True, candidates=[PathCandidate(
            current.reshape(1, 2), 0.0, (), 0.0, True, "REVERSE_POINT")])
    # 到达全局终点时futurePath退化成一个位置；仍返回成功的停车path，
    # 不能将重复坐标去重后的单点误判为缺少可行线/参考线。
    destination_tolerance = max(
        1e-6, float(planning_params["responseDestinationTolerance"]))
    if (stop_at_route_end and len(reference) == 1
            and np.linalg.norm(current-reference[0]) <= destination_tolerance):
        return PathPlanResult(True, candidates=[PathCandidate(
            current.reshape(1, 2), 0.0, (), 0.0, True, "GLOBAL_ROUTE_END")])
    lines = []
    centerline_spacing = float(planning_params["pathCenterlineSpacing"])
    for raw_line in drivable_lines or []:
        line = as_xy(raw_line)
        if len(line) < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
        if (len(segment_lengths)
                and np.max(segment_lengths) > centerline_spacing + 1e-9):
            line = resample_polyline(line, centerline_spacing)
        lines.append(line)
    if not lines:
        # The map centreline endpoint and the long-route endpoint may differ by
        # a few millimetres.  Only suppress NO_DRIVABLE_LINE when the remaining
        # route is already inside the declared destination tolerance; an empty
        # drivable area anywhere else is still a planning failure.
        if (stop_at_route_end and len(reference) >= 1
                and np.linalg.norm(
                    current-reference[-1]) <= destination_tolerance):
            return PathPlanResult(True, candidates=[PathCandidate(
                current.reshape(1, 2), 0.0, (), 0.0,
                True, "GLOBAL_ROUTE_END")])
        return PathPlanResult(False, "NO_DRIVABLE_LINE")
    if len(reference) < 2:
        return PathPlanResult(False, "INVALID_GLOBAL_REFERENCE_LINE")

    static_info = dict(vehicle_static_info or {})
    vehicle_info = {
        "length": float(static_info.get(
            "length", planning_params["vehicleLength"])),
        "width": float(static_info.get(
            "width", planning_params["vehicleWidth"])),
        "minTurningRadius": float(static_info.get(
            "minTurningRadius", planning_params["minTurningRadius"])),
    }
    graph = build_spatial_graph(
        lines, current, float(heading), reference, obstacles or [],
        planning_params, vehicle_info,
        max(0.0, float(longitudinal_uncertainty)),
        max(0.0, float(lateral_uncertainty)),
        None if reverse_point is None else np.asarray(reverse_point, dtype=float),
    )
    candidates = _search_first_reachable_terminal(
        graph, reference, planning_params, bool(stop_at_route_end))
    if candidates:
        return PathPlanResult(True, candidates=candidates)

    # A fallback is not a curvature-relaxed lattice search. The upstream route
    # is trusted as the only feasible line; static fences can only truncate it.
    fallback = _reference_route_fallback(
        current, reference, obstacles or [], vehicle_info,
        max(0.0, float(longitudinal_uncertainty)),
        max(0.0, float(lateral_uncertainty)),
        bool(stop_at_route_end), reverse_point, planning_params)
    if fallback is not None:
        return PathPlanResult(True, candidates=[fallback])

    stationary = PathCandidate(
        current.reshape(1, 2), 0.0, (graph["currentLine"],), 0.0,
        True, "NO_COLLISION_FREE_PATH")
    return PathPlanResult(False, "NO_COLLISION_FREE_PATH", [stationary])


def plan_spatial_paths_batch(
    jobs: Sequence[AGVPathPlanningJob],
    params=None,
) -> Dict[str, PathPlanResult]:
    """逐车循环计算，Python阶段不使用线程池。"""
    if not jobs:
        return {}
    planning_params = dict(DEFAULT_PLANNING_PARAMS)
    if params:
        planning_params.update(params)

    def execute(job):
        try:
            lines = job.drivableLines
            if lines is None:
                lines = job.drivableAreaPlanner.plan(
                    job.position, currentHeading=job.heading,
                    currentSpeed=job.speed, reversePoint=job.reversePoint)
            result = plan_spatial_path(
                lines, job.position, job.heading, job.globalReferenceLine,
                job.obstacles, job.vehicleStaticInfo,
                job.longitudinalUncertainty, job.lateralUncertainty,
                job.reversePoint, job.stopAtRouteEnd, planning_params)
        except Exception as error:
            result = PathPlanResult(
                False, f"PATH_PLANNING_EXCEPTION: {error}")
        return job.vehicleId, result

    return dict(execute(job) for job in jobs)

def _reference_route_fallback(
    current, reference, obstacles, vehicle_info,
    longitudinal_uncertainty, lateral_uncertainty,
    stop_at_route_end, reverse_point, params,
):
    """Use only the trusted route; fences may shorten it to a safe stop."""
    route = as_xy(reference)
    if len(route) < 2:
        return None
    if np.linalg.norm(route[0]-current) > 1e-6:
        route = np.vstack((current, route))

    half_width = (
        0.5*float(vehicle_info["width"])
        + float(lateral_uncertainty)
        + float(params["pathObstacleSafetyMargin"]))
    intervals = blocked_path_intervals(route, obstacles, half_width)
    route_end_s = float(arc_lengths(route)[-1])
    terminal_s = route_end_s
    terminal_stop = bool(stop_at_route_end)
    terminal_reason = "GLOBAL_ROUTE_END" if terminal_stop else None
    reverse_xy = None
    if reverse_point is not None:
        candidate = np.asarray(reverse_point, dtype=float).reshape(-1)[:2]
        if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
            projection = project_to_polyline(candidate, route)
            if projection["distance"] <= float(params["reversePointLocateTolerance"]):
                terminal_s = float(projection["s"])
                terminal_stop = True
                terminal_reason = "REVERSE_POINT"
                reverse_xy = candidate
    if intervals:
        stop_clearance = (
            0.5*float(vehicle_info["length"])
            + float(longitudinal_uncertainty)
            + float(params["pathStopClearance"]))
        obstacle_stop_s = max(
            0.0, float(intervals[0].startS)-stop_clearance)
        if obstacle_stop_s < terminal_s-1e-9:
            terminal_s = obstacle_stop_s
            terminal_stop = True
            terminal_reason = "STATIC_OBSTACLE_STOP"
            reverse_xy = None
    route = truncate_polyline(route, terminal_s)
    if (terminal_reason == "REVERSE_POINT" and reverse_xy is not None
            and len(route)):
        route[-1] = reverse_xy

    if len(route) == 0:
        route = current.reshape(1, 2)
    maximum_curvature = float(np.max(path_curvatures(
        route, float(params["pathCurvatureWindow"])))) if len(route) > 1 else 0.0
    return PathCandidate(
        xy=route[:, :2].copy(), score=0.0, lineSequence=("ROUTE",),
        maxCurvature=maximum_curvature,
        terminalStop=terminal_stop, terminalReason=terminal_reason,
        degraded=True, degradationReason="REFERENCE_ROUTE_FALLBACK")


def _search_first_reachable_terminal(graph, reference, params,
                                     stop_at_route_end):
    """Search only curvature-feasible graph edges; fallback is handled outside."""
    states, parent = _dynamic_programming(graph, max_signatures=32)
    for terminal in graph["terminals"]:
        normal = _first_terminal_candidates(
            graph, states, parent, reference, params, stop_at_route_end,
            terminals=(terminal,))
        if normal:
            return normal
    return []

def _first_terminal_candidates(graph, states, parent, reference, params,
                               stop_at_route_end, terminals=None):
    maximum = max(1, int(params["pathMaxCandidates"]))
    curvature_window = float(params["pathCurvatureWindow"])
    terminal_sequence = graph["terminals"] if terminals is None else terminals
    for terminal in terminal_sequence:
        if (terminal["terminalStop"] and np.linalg.norm(
                terminal["point"]-graph["nodes"]["S"]["point"]) <= 1e-6):
            return [PathCandidate(
                terminal["point"].reshape(1, 2), 0.0,
                (graph["currentLine"],), 0.0, True,
                terminal["terminalReason"])]
        entries = states.get(terminal["id"], {})
        if not entries:
            continue
        reconstructed = []
        for signature, cost in sorted(entries.items(), key=lambda item: item[1]):
            edges = _reconstruct(terminal["id"], signature, parent)
            path = _join_edges(graph["nodes"]["S"]["point"], edges)
            if len(path) < 2:
                continue
            curvatures = path_curvatures(path, curvature_window)
            maximum_curvature = (
                float(np.max(curvatures)) if len(curvatures) else 0.0)
            mean_lateral = float(np.mean(np.abs(
                project_points_to_polyline(path, reference)["d"])))
            lane_changes = max(0, len(signature)-1)
            score = (
                mean_lateral
                + float(params["pathLaneChangePenalty"])*lane_changes
                + float(params["pathCurvatureWeight"])*maximum_curvature)
            reconstructed.append((
                True, mean_lateral, score, signature, path,
                maximum_curvature))
            if len(reconstructed) >= maximum:
                break
        if not reconstructed:
            continue
        terminal_stop = bool(terminal["terminalStop"] or (
            stop_at_route_end and terminal["kind"] == "G"))
        terminal_reason = terminal["terminalReason"]
        if terminal_reason is None and stop_at_route_end and terminal["kind"] == "G":
            terminal_reason = "GLOBAL_ROUTE_END"
        chosen = sorted(
            reconstructed, key=lambda item: (item[2], item[1]))[:maximum]
        return [_candidate(item, terminal_stop, terminal_reason)
                for item in chosen]
    return []

def _candidate(item, terminal_stop, terminal_reason):
    _, _, score, signature, path, maximum_curvature = item
    return PathCandidate(
        xy=np.asarray(path, dtype=float)[:, :2].copy(),
        score=float(score), lineSequence=tuple(signature),
        maxCurvature=float(maximum_curvature),
        terminalStop=terminal_stop, terminalReason=terminal_reason)


def _dynamic_programming(graph, max_signatures=32):
    nodes, adjacency = graph["nodes"], graph["adjacency"]
    start_signature = (int(graph["currentLine"]),)
    states = {"S": {start_signature: 0.0}}
    parent = {}
    order = sorted(nodes.values(), key=lambda item: (item["progress"], item["id"]))
    for node in order:
        entries = states.get(node["id"])
        if not entries:
            continue
        if len(entries) > max_signatures:
            entries = dict(sorted(
                entries.items(), key=lambda item: item[1])[:max_signatures])
            states[node["id"]] = entries
        for signature, cost in list(entries.items()):
            for edge in adjacency.get(node["id"], []):
                if not edge["curvatureOK"]:
                    continue
                target = nodes[edge["to"]]
                if target["progress"] <= node["progress"] + 1e-6:
                    continue
                new_signature = signature
                if signature[-1] != edge["targetLine"]:
                    new_signature = signature + (int(edge["targetLine"]),)
                new_cost = cost + float(edge["cost"])
                destination = states.setdefault(edge["to"], {})
                if new_cost < destination.get(new_signature, float("inf"))-1e-9:
                    destination[new_signature] = new_cost
                    parent[(edge["to"], new_signature)] = (
                        node["id"], signature, edge)
    return states, parent

def _reconstruct(terminal_id, state_key, parent):
    edges = []
    cursor = (terminal_id, state_key)
    while cursor in parent:
        previous_id, previous_key, edge = parent[cursor]
        edges.append(edge)
        cursor = (previous_id, previous_key)
    edges.reverse()
    return edges


def _join_edges(start, edges):
    points = [np.asarray(start, dtype=float)[:2]]
    # 图节点投影可能生成亚毫米级伪边；保留这种边会让它不稳定的航向
    # 污染5m曲率窗口，并在当前位置制造虚假的极低限速。
    minimum_segment_length = 1e-3
    for edge in edges:
        for point in as_xy(edge["curve"])[1:]:
            if np.linalg.norm(point - points[-1]) > minimum_segment_length:
                points.append(point)
    return np.asarray(points, dtype=float)
