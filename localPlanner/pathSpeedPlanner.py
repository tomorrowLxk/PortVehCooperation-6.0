"""Attach per-point speed limits to a continuous spatial path."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from localPlanner.spatialPathGraph import path_curvatures
from tools.default_params import DEFAULT_PLANNING_PARAMS
from tools.path_geometry import as_xy


def build_speed_limit_path(
    path_xy: Any,
    vehicle_static_info: Mapping[str, Any] | None,
    terminal_stop: bool = False,
    stationary: bool = False,
    params: Mapping[str, Any] = DEFAULT_PLANNING_PARAMS,
) -> np.ndarray:
    """Return ``N x 3 [x, y, vMax]`` and always keep at least 2 rows.

    For AGVs, vMax is the static/curvature speed limit. For HVs it is
    still a geometric speed limit used by later ETA calculations, not a command.
    A required stop is represented by a zero speed at the final path point;
    longitudinal smoothing belongs to the later speed-planning stage.
    """
    points = as_xy(path_xy)
    if len(points) == 0:
        raise ValueError("path缺少有效坐标")
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)

    static_info = vehicle_static_info or {}
    maximum_speed = _positive(
        static_info.get("maxSpeed"), float(params["vehicleMaxSpeed"]))
    lateral_acceleration = _positive(
        params.get("vehicleMaxLateralAcceleration"), 1.5)

    speeds = np.full(len(points), maximum_speed, dtype=float)
    if not stationary:
        curvature = path_curvatures(
            points, float(params["pathCurvatureWindow"]))
        curved = curvature > 1e-6
        if np.any(curved):
            # a_y = v^2 * |kappa|, so v <= sqrt(a_y,max / |kappa|).
            curve_limit = np.sqrt(
                lateral_acceleration/np.maximum(curvature[curved], 1e-6))
            speeds[curved] = np.minimum(maximum_speed, curve_limit)
    else:
        speeds[:] = 0.0

    if terminal_stop:
        speeds[-1] = 0.0
    return np.column_stack((points, speeds))


def stationary_path(position: Any) -> np.ndarray:
    """Return a safe two-row stationary ``[x, y, 0]`` path."""
    point = np.asarray(position, dtype=float).reshape(-1)[:2]
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("无法为无效当前位置生成停车path")
    return np.asarray([
        [point[0], point[1], 0.0],
        [point[0], point[1], 0.0],
    ], dtype=float)


def _positive(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) and result > 0.0 else float(default)