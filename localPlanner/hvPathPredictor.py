"""HV沿全局route的短时空间占用路径预测。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from tools.default_params import DEFAULT_PLANNING_PARAMS
from tools.path_corridor_collision import first_path_corridor_contact_s
from tools.path_geometry import as_xy, truncate_polyline


def predict_hv_path(
    future_path: Any,
    speed: float,
    prediction_horizon: float,
    obstacles: Iterable[Any],
    vehicle_length: float,
    vehicle_width: float,
    longitudinal_uncertainty: float = 0.0,
    lateral_uncertainty: float = 0.0,
    safety_margin: float = 0.0,
    collision_check_spacing: float = 2.0,
) -> np.ndarray:
    """按当前速度匀速预测HV中心路径，并在该车可见围栏前截断。"""
    path = as_xy(future_path)
    if len(path) < 2:
        return path
    prediction_distance = max(0.0, float(speed)) * max(0.0, float(prediction_horizon))
    if prediction_distance <= 1e-8:
        return path[:1].copy()
    predicted = truncate_polyline(path, prediction_distance)
    contact = first_path_corridor_contact_s(
        predicted, obstacles, 0.5*vehicle_width+lateral_uncertainty+safety_margin)
    if contact is None:
        return predicted
    stop_s = max(0.0, contact-0.5*vehicle_length-longitudinal_uncertainty
                 -float(DEFAULT_PLANNING_PARAMS["pathStopClearance"]))
    return truncate_polyline(predicted, stop_s)
