"""生成与basePath绑定的稳定waypoint候选，并随futurePath同步裁剪。"""

import numpy as np

from localPlanner.pathGeometry import (
    clean_xy,
    point_at_s,
    polyline_length,
    project_points_to_polyline,
    project_to_polyline,
)
from parameters import (
    JUNCTION_EXIT_PATH_EXTENSION,
    VEHICLE_LANE_LOCATE_MAX_DISTANCE,
    WAYPOINT_CANDIDATE_SPACING,
    WAYPOINT_SPARSE_MIN_SPACING,
)


WAYPOINT_CANDIDATE_REGULAR = 0
WAYPOINT_CANDIDATE_PATH_END = 1
WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE = 2
WAYPOINT_CANDIDATE_JUNCTION_EXIT = 4
WAYPOINT_CANDIDATE_POST_JUNCTION = 8


def build_base_waypoint_candidates(
        base_path, centerline_dict, junction_lane_ids,
        final_destination=None, base_path_lane_ids=None):
    """普通区域每隔固定距离取点；虚拟车道内部只保留头尾边界。"""
    path = np.asarray(base_path, dtype=float)
    path_xy = clean_xy(path)
    if len(path_xy) < 2:
        point = path_xy[0] if len(path_xy) else np.zeros(2)
        return np.asarray([[
            point[0], point[1], 0.0,
            WAYPOINT_CANDIDATE_PATH_END,
        ]], dtype=float)

    total = polyline_length(path_xy)
    candidate_types = {
        round(float(value), 6): WAYPOINT_CANDIDATE_REGULAR
        for value in np.arange(0.0, total, WAYPOINT_CANDIDATE_SPACING)
    }
    _merge_candidate_type(
        candidate_types, total, WAYPOINT_CANDIDATE_PATH_END)
    junction_intervals = _junction_intervals(
        path_xy, centerline_dict, junction_lane_ids,
        base_path_lane_ids)
    for entrance_s, exit_s in junction_intervals:
        _merge_candidate_type(
            candidate_types, entrance_s,
            WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE)
        _merge_candidate_type(
            candidate_types, exit_s,
            WAYPOINT_CANDIDATE_JUNCTION_EXIT)
        post_exit_s = _post_junction_s(
            exit_s, total, junction_intervals)
        if post_exit_s > exit_s + 1e-6:
            _merge_candidate_type(
                candidate_types, post_exit_s,
                WAYPOINT_CANDIDATE_POST_JUNCTION)
    sample_s = np.asarray(sorted(candidate_types), dtype=float)

    points = np.asarray([
        point_at_s(path_xy, value) for value in sample_s
    ], dtype=float)
    relevant_junction_lane_ids = _relevant_junction_lane_ids(
        junction_lane_ids, base_path_lane_ids)
    inside = _points_inside_junction_lanes(
        points, centerline_dict, relevant_junction_lane_ids)

    # basePath起点属于固定基础候选；本帧当前位置只在future候选中加入。
    candidate_type = np.asarray([
        candidate_types[round(float(value), 6)]
        for value in sample_s
    ], dtype=float)
    keep = ~inside | (candidate_type != WAYPOINT_CANDIDATE_REGULAR)
    keep[0] = True
    if (final_destination is not None
            and np.linalg.norm(
                points[-1] - np.asarray(final_destination, dtype=float)[:2]
            ) <= 1e-3):
        keep[-1] = True
    return np.column_stack((
        points[keep], sample_s[keep], candidate_type[keep]
    ))


def build_future_waypoint_candidates(
        base_path, base_candidates, future_path, current_position):
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
    selected = [[
        position[0], position[1], 0.0,
        WAYPOINT_CANDIDATE_REGULAR,
    ]]
    for candidate in candidates:
        base_s = float(candidate[2])
        if (base_s <= progress + 1e-6
                or base_s > future_end + 1e-6):
            continue
        future_s = float(project_to_polyline(
            candidate[:2], future_xy)['s'])
        if future_s <= 1e-6:
            continue
        selected.append([
            candidate[0], candidate[1], future_s,
            candidate[3] if len(candidate) >= 4
            else WAYPOINT_CANDIDATE_REGULAR,
        ])

    end_point = future_xy[-1]
    end_inside = _s_inside_candidate_junction(
        future_end, candidates)
    if not end_inside:
        selected.append([
            end_point[0], end_point[1], polyline_length(future_xy),
            WAYPOINT_CANDIDATE_REGULAR,
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
            output[-1][3] = float(
                int(round(output[-1][3])) | int(round(item[3])))
            continue
        output.append(item)
    return np.asarray(output, dtype=float)


def _s_inside_candidate_junction(value, candidates):
    entrances = []
    for candidate in candidates:
        if len(candidate) < 4:
            continue
        candidate_s = float(candidate[2])
        candidate_type = int(round(candidate[3]))
        if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_ENTRANCE:
            entrances.append(candidate_s)
        if candidate_type & WAYPOINT_CANDIDATE_JUNCTION_EXIT:
            for index, entrance_s in enumerate(entrances):
                if candidate_s > entrance_s + 1e-6:
                    if entrance_s + 1e-6 < value < candidate_s - 1e-6:
                        return True
                    entrances.pop(index)
                    break
    return False


def _junction_intervals(
        path, centerline_dict, junction_lane_ids,
        base_path_lane_ids=None):
    intervals = []
    path_min = np.min(path, axis=0) - VEHICLE_LANE_LOCATE_MAX_DISTANCE
    path_max = np.max(path, axis=0) + VEHICLE_LANE_LOCATE_MAX_DISTANCE
    for lane_id in _relevant_junction_lane_ids(
            junction_lane_ids, base_path_lane_ids):
        if lane_id not in centerline_dict:
            continue
        lane = clean_xy(centerline_dict[lane_id])
        if len(lane) < 2:
            continue
        if (np.any(np.max(lane, axis=0) < path_min)
                or np.any(np.min(lane, axis=0) > path_max)):
            continue
        entrance = project_to_polyline(lane[0], path)
        exit_point = project_to_polyline(lane[-1], path)
        if (entrance['distance'] <= VEHICLE_LANE_LOCATE_MAX_DISTANCE
                and exit_point['distance'] <= VEHICLE_LANE_LOCATE_MAX_DISTANCE
                and exit_point['s'] > entrance['s'] + 1e-6):
            intervals.append((
                float(entrance['s']), float(exit_point['s'])))
    return sorted(set(
        (round(entrance_s, 6), round(exit_s, 6))
        for entrance_s, exit_s in intervals
    ))


def _relevant_junction_lane_ids(
        junction_lane_ids, base_path_lane_ids=None):
    if base_path_lane_ids is None:
        return list(junction_lane_ids)
    allowed = {str(lane_id) for lane_id in junction_lane_ids}
    output = []
    for lane_id in np.asarray(base_path_lane_ids, dtype=object):
        if lane_id is None:
            continue
        lane_id = str(lane_id)
        if lane_id in allowed and lane_id not in output:
            output.append(lane_id)
    return output


def _post_junction_s(exit_s, total, junction_intervals):
    remaining = total - exit_s
    if (remaining >= JUNCTION_EXIT_PATH_EXTENSION
            and remaining
            < JUNCTION_EXIT_PATH_EXTENSION + WAYPOINT_SPARSE_MIN_SPACING):
        value = total
    else:
        value = min(exit_s + JUNCTION_EXIT_PATH_EXTENSION, total)
    for entrance_s, following_exit_s in junction_intervals:
        if entrance_s + 1e-6 < value < following_exit_s - 1e-6:
            return following_exit_s
    return value


def _merge_candidate_type(candidate_types, value, candidate_type):
    key = round(float(value), 6)
    candidate_types[key] = (
        int(candidate_types.get(key, WAYPOINT_CANDIDATE_REGULAR))
        | int(candidate_type)
    )


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
