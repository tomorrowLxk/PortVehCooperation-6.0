"""车辆动静态信息缓存更新。"""

import numpy as np

from localPlanner.pathGeometry import navigation_heading_to_cartesian
from parameters import (
    DEFAULT_AGV_STATIC_INFO,
    DEFAULT_NON_AGV_STATIC_INFO,
    VEHICLE_CACHE_MAX_MISSING_FRAMES,
    VEHICLE_LATERAL_UNCERTAINTY_SCALE,
    VEHICLE_LONGITUDINAL_UNCERTAINTY_SCALE,
)


def update_vehicle_dynamic_cache(all_vehicle_states, previous_cache):
    """用本帧BSM更新动态缓存，连续缺失指定帧数后删除车辆。"""
    updated = {}
    for vehicle_id, state in (all_vehicle_states or {}).items():
        position = _field(state, "position")
        x = _field(position, "longitude")
        y = _field(position, "latitude")
        raw_speed = _field(state, "speed", 0)
        accuracy_level = max(0.0, float(_field(state, "posAccuracy", 0) or 0))
        updated[str(vehicle_id)] = {
            "valid": x is not None and y is not None,
            "position": np.asarray([
                float("nan") if x is None else float(x),
                float("nan") if y is None else float(y),
            ]),
            "posAccuracy": _field(state, "posAccuracy"),
            "lateralUncertainty": (
                VEHICLE_LATERAL_UNCERTAINTY_SCALE * accuracy_level
            ),
            "longitudinalUncertainty": (
                VEHICLE_LONGITUDINAL_UNCERTAINTY_SCALE * accuracy_level
            ),
            "speed": 0.0 if raw_speed == 8191 else max(0.0, float(raw_speed) * 0.02),
            "heading": float(_field(state, "heading", 0.0)),
            "direction": int(_field(state, "direction", 1)),
            "_missingFrames": 0,
        }

        updated[str(vehicle_id)]['heading'] = (
            navigation_heading_to_cartesian(
                updated[str(vehicle_id)]['heading'])
        )

    for vehicle_id, old_info in (previous_cache or {}).items():
        vehicle_id = str(vehicle_id)
        if vehicle_id in updated:
            continue
        missing_frames = int(old_info.get("_missingFrames", 0)) + 1
        if missing_frames < VEHICLE_CACHE_MAX_MISSING_FRAMES:
            updated[vehicle_id] = dict(old_info, _missingFrames=missing_frames)
    return updated


def update_vehicle_static_cache(all_vehicle_static_info, previous_cache):
    """用本帧VehicleStaticVO更新静态缓存，缺失字段采用车型默认值。"""
    updated = {}
    for vehicle_id, static_info in (all_vehicle_static_info or {}).items():
        vehicle_type = _field(static_info, "vehicleType", "AGV")
        defaults = (DEFAULT_AGV_STATIC_INFO if vehicle_type == "AGV"
                    else DEFAULT_NON_AGV_STATIC_INFO)
        bounding_box = _field(static_info, "boundingBox")
        kinematics = _field(static_info, "kinematics")
        length = _field(bounding_box, "length")
        width = _field(bounding_box, "width")
        record = {
            "vehicleType": vehicle_type,
            "length": float(length) / 100.0 if length else defaults["length"],
            "width": float(width) / 100.0 if width else defaults["width"],
            "_missingFrames": 0,
        }
        for name in (
            "maxSpeed", "maxAcceleration", "maxDeceleration", "minTurningRadius",
        ):
            value = _field(kinematics, name)
            record[name] = float(value) if value else defaults[name]
        updated[str(vehicle_id)] = record

    for vehicle_id, old_info in (previous_cache or {}).items():
        vehicle_id = str(vehicle_id)
        if vehicle_id in updated:
            continue
        missing_frames = int(old_info.get("_missingFrames", 0)) + 1
        if missing_frames < VEHICLE_CACHE_MAX_MISSING_FRAMES:
            updated[vehicle_id] = dict(old_info, _missingFrames=missing_frames)
    return updated


def _field(value, name, default=None):
    """读取接口字典字段。"""
    if value is None:
        return default
    return value.get(name, default)
