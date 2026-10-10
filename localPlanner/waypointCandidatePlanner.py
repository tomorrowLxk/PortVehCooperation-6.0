"""生成与basePath绑定的稳定waypoint候选，并随futurePath同步裁剪。"""

import math

import numpy as np

from localPlanner.pathGeometry import (
    clean_xy,
    point_at_s,
    polyline_length,
    project_points_to_polyline,
    project_to_polyline,
    tangent_at_s,
)
from parameters import (
    VEHICLE_LANE_LOCATE_MAX_DISTANCE,
    VEHICLE_LANE_LOCATE_HEADING_WEIGHT,
    WAYPOINT_CANDIDATE_SPACING,
)


def build_base_waypoint_candidates(
        base_path, centerline_dict, junction_lane_ids,
        final_destination=None):
    """普通区域每隔固定距离取点；虚拟车道内部只保留头尾边界。"""
    path = np.asarray(base_path, dtype=float)
    path_xy = clean_xy(path)
    if len(path_xy) < 2:
        point = path_xy[0] if len(path_xy) else np.zeros(2)
        return np.asarray([[point[0], point[1], 0.0]], dtype=float)

    total = polyline_length(path_xy)
    sample_s = list(np.arange(0.0, total, WAYPOINT_CANDIDATE_SPACING))
    sample_s.append(total)
    junction_boundary_s = _junction_boundary_s(
        path_xy, centerline_dict, junction_lane_ids)
    sample_s.extend(junction_boundary_s)
    sample_s = np.asarray(sorted(set(
        round(float(value), 6) for value in sample_s
    )), dtype=float)

    points = np.asarray([
        point_at_s(path_xy, value) for value in sample_s
    ], dtype=float)
    inside = _points_inside_junction_lanes(
        points, centerline_dict, junction_lane_ids)

    # basePath起点属于固定基础候选；本帧当前位置只在future候选中加入。
    boundary = np.zeros(len(sample_s), dtype=bool)
    if junction_boundary_s:
        boundary = np.any(
            np.abs(sample_s[:, None]
                   - np.asarray(junction_boundary_s)[None, :]) <= 1e-6,
            axis=1)
    keep = ~inside | boundary
    keep[0] = True
    if (final_destination is not None
            and np.linalg.norm(
                points[-1] - np.asarray(final_destination, dtype=float)[:2]
            ) <= 1e-3):
        keep[-1] = True
    return np.column_stack((points[keep], sample_s[keep]))


def build_future_waypoint_candidates(
        base_path, base_candidates, future_path, current_position,
        centerline_dict, junction_lane_ids):
    """裁剪已驶过候选，并以本帧当前位置作为新的第一个候选点。"""
    base = np.asarray(base_path, dtype=float)
    base_xy = clean_xy(base)
    future = np.asarray(future_path, dtype=float)
    future_xy = clean_xy(future)
    position = np.asarray(current_position, dtype=float)[:2]

    if len(base_xy) < 2 or len(future_xy) < 2:
        return np.asarray([[position[0], position[1], 0.0]], dtype=float)

    progress = float(project_to_polyline(position, base_xy)['s'])
    future_end = float(project_to_polyline(future_xy[-1], base_xy)['s'])
    future_end = max(progress, future_end)

    candidates = np.asarray(base_candidates, dtype=float)
    selected = [[position[0], position[1], 0.0]]
    for candidate in candidates:
        base_s = float(candidate[2])
        if (base_s <= progress + 1e-6
                or base_s > future_end + 1e-6):
            continue
        future_s = float(project_to_polyline(
            candidate[:2], future_xy)['s'])
        if future_s <= 1e-6:
            continue
        selected.append([candidate[0], candidate[1], future_s])

    end_point = future_xy[-1]
    end_inside = bool(_points_inside_junction_lanes(
        end_point.reshape(1, 2), centerline_dict, junction_lane_ids)[0])
    if not end_inside:
        selected.append([
            end_point[0], end_point[1], polyline_length(future_xy)
        ])

    if not selected:
        return np.empty((0, 3), dtype=float)

    selected.sort(key=lambda item: item[2])
    output = []
    for item in selected:
        if (output
                and abs(item[2] - output[-1][2]) <= 1e-6
                and np.linalg.norm(
                    np.asarray(item[:2]) - np.asarray(output[-1][:2])
                ) <= 1e-6):
            continue
        output.append(item)
    return np.asarray(output, dtype=float)


def match_current_junction_lane(
        current_position, current_heading,
        centerline_dict, junction_lane_ids):
    """用有向虚拟车道中心线匹配当前位置，返回虚拟车道ID或None。"""
    position = np.asarray(current_position, dtype=float)[:2]
    heading = float(current_heading)
    candidates = []

    for lane_id in junction_lane_ids:
        if lane_id not in centerline_dict:
            continue
        lane = clean_xy(centerline_dict[lane_id])
        lane_length = polyline_length(lane)
        projection = project_to_polyline(position, lane)
        if (projection['distance'] > VEHICLE_LANE_LOCATE_MAX_DISTANCE
                or projection['s'] <= 1e-6
                or projection['s'] >= lane_length - 1e-6):
            continue

        tangent = tangent_at_s(lane, projection['s'])
        lane_heading = math.atan2(tangent[1], tangent[0])
        heading_error = abs(math.atan2(
            math.sin(heading - lane_heading),
            math.cos(heading - lane_heading),
        ))
        score = (
            projection['distance']
            + VEHICLE_LANE_LOCATE_HEADING_WEIGHT * heading_error
        )
        candidates.append((score, projection['distance'], str(lane_id)))

    if not candidates:
        return None
    return min(candidates)[2]


def _junction_boundary_s(path, centerline_dict, junction_lane_ids):
    values = []
    path_min = np.min(path, axis=0) - VEHICLE_LANE_LOCATE_MAX_DISTANCE
    path_max = np.max(path, axis=0) + VEHICLE_LANE_LOCATE_MAX_DISTANCE
    for lane_id in junction_lane_ids:
        if lane_id not in centerline_dict:
            continue
        lane = clean_xy(centerline_dict[lane_id])
        if len(lane) < 2:
            continue
        if (np.any(np.max(lane, axis=0) < path_min)
                or np.any(np.min(lane, axis=0) > path_max)):
            continue
        for boundary in (lane[0], lane[-1]):
            projection = project_to_polyline(boundary, path)
            if projection['distance'] <= VEHICLE_LANE_LOCATE_MAX_DISTANCE:
                values.append(float(projection['s']))
    return values


def _points_inside_junction_lanes(
        points, centerline_dict, junction_lane_ids):
    points = np.asarray(points, dtype=float)[:, :2]
    inside = np.zeros(len(points), dtype=bool)
    points_min = (
        np.min(points, axis=0) - VEHICLE_LANE_LOCATE_MAX_DISTANCE)
    points_max = (
        np.max(points, axis=0) + VEHICLE_LANE_LOCATE_MAX_DISTANCE)
    for lane_id in junction_lane_ids:
        if lane_id not in centerline_dict:
            continue
        lane = clean_xy(centerline_dict[lane_id])
        if len(lane) < 2:
            continue
        if (np.any(np.max(lane, axis=0) < points_min)
                or np.any(np.min(lane, axis=0) > points_max)):
            continue
        lane_length = polyline_length(lane)
        projection = project_points_to_polyline(points, lane)
        inside |= (
            (projection['distance'] <= VEHICLE_LANE_LOCATE_MAX_DISTANCE)
            & (projection['s'] > 1e-6)
            & (projection['s'] < lane_length - 1e-6)
        )
    return inside
