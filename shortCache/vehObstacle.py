"根据上一帧短路径和本帧车辆状态维护静态障碍物判定缓存。"

from typing import Any, Dict, Iterable, Mapping

import numpy as np

from parameters import (
    VEHICLE_DESTINATION_DISTANCE_THRESHOLD,
    VEHICLE_OBSTACLE_CONFIRM_FRAMES,
    VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD,
    VEHICLE_STOP_SPEED_THRESHOLD,
)


def update_vehicle_obstacle_cache(
        request_vehicle_ids: Iterable[str],
        vehicle_path_cache: Mapping[str, Any],
        vehicle_trajectory: Mapping[str, Any],
        vehicle_dynamic_info_cache: Mapping[str, Mapping[str, Any]],
        previous_cache: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    "返回与车辆动态信息缓存具有完全相同车辆ID集合的新缓存。"
    requested = {str(vehicle_id) for vehicle_id in request_vehicle_ids}
    updated: Dict[str, Dict[str, Any]] = {}

    for vehicle_id, dynamic_info in vehicle_dynamic_info_cache.items():
        old = previous_cache.get(vehicle_id, {})
        is_static_obstacle = bool(old.get('isStaticObstacle', False))
        unexecuted_frames = int(old.get('unexecutedFrames', 0))

        bsm_is_current = (
            bool(dynamic_info.get('valid', False))
            and int(dynamic_info.get('_missingFrames', 0)) == 0
        )
        if bsm_is_current:
            if _remaining_distance_to_destination(
                    vehicle_trajectory.get(vehicle_id),
                    dynamic_info.get('position'),
            ) <= VEHICLE_DESTINATION_DISTANCE_THRESHOLD:
                is_static_obstacle = True
                unexecuted_frames = 0
            else:
                is_suspected_stopped = (
                    float(dynamic_info.get('speed', 0.0)) < VEHICLE_STOP_SPEED_THRESHOLD
                )
                if not is_suspected_stopped:
                    is_static_obstacle = False
                    unexecuted_frames = 0
                elif not is_static_obstacle:
                    has_executable_path = (
                        vehicle_id in requested
                        and _remaining_path_length(
                            vehicle_path_cache.get(vehicle_id),
                            dynamic_info.get('position'),
                        ) > VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD
                    )
                    unexecuted_frames = unexecuted_frames + 1 if has_executable_path else 0
                    is_static_obstacle = (
                        unexecuted_frames >= VEHICLE_OBSTACLE_CONFIRM_FRAMES
                    )

        updated[vehicle_id] = {
            'isStaticObstacle': is_static_obstacle,
            'unexecutedFrames': unexecuted_frames,
        }

    return updated


def _remaining_distance_to_destination(trajectory_data: Any, current_position: Any) -> float:
    '''计算当前位置投影到全局参考线后，沿线到终点的剩余弧长。'''
    try:
        trajectory = np.asarray(trajectory_data, dtype=float)
        position = np.asarray(current_position, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float('inf')
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2 or position.size < 2:
        return float('inf')
    if trajectory.shape[0] == 1:
        return float(np.linalg.norm(position[:2] - trajectory[0, :2]))
    return _remaining_path_length(trajectory, position)


def _remaining_path_length(path_data: Any, current_position: Any) -> float:
    "计算当前位置投影到path后，从投影点到path末端的弧长。"
    try:
        path = np.asarray(path_data, dtype=float)
        position = np.asarray(current_position, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2 or position.size < 2:
        return 0.0

    points = path[:, :2]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid_indices = np.flatnonzero(lengths > 1e-9)
    if valid_indices.size == 0:
        return 0.0

    valid_segments = segments[valid_indices]
    starts = points[valid_indices]
    ratios = np.clip(
        np.einsum('ij,ij->i', position[:2] - starts, valid_segments)
        / np.square(lengths[valid_indices]),
        0.0,
        1.0,
    )
    projections = starts + ratios[:, None] * valid_segments
    nearest = int(np.argmin(np.linalg.norm(projections - position[:2], axis=1)))
    segment_index = int(valid_indices[nearest])
    return float(
        (1.0 - ratios[nearest]) * lengths[segment_index]
        + lengths[segment_index + 1:].sum()
    )
