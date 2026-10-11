"""第十三阶段：将waypointSparse转换为接口要求的PathVO字典。"""

import math

import numpy as np

from localPlanner.pathGeometry import clean_xy
from parameters import (
    RESPONSE_DESTINATION_TOLERANCE,
    RESPONSE_MERGE_ANGLE_TOLERANCE_RAD,
    RESPONSE_MERGE_LATERAL_TOLERANCE,
    RESPONSE_TURN_THRESHOLD_RAD,
    RESPONSE_UTURN_THRESHOLD_RAD,
)


def build_response_path_lists(
        waypoints, planned_path, centerline_dict, connector_dict,
        lane_ids, junction_lane_ids, full_path_list, request):
    """返回单车的(shortPathList, mergePathList)，不改变response状态。"""
    waypoint_array = np.asarray(waypoints, dtype=float)
    path = clean_xy(planned_path)
    path_geometry = _prepare_polyline(path)
    metadata = {
        str(item.get('laneId')): item
        for item in full_path_list
        if item.get('laneId') is not None
    }
    candidate_lane_ids = _candidate_lane_ids(
        lane_ids, metadata, request, centerline_dict)
    goal = _request_goal(request)
    movable = [
        index for index in range(len(waypoint_array) - 1)
        if np.linalg.norm(
            waypoint_array[index + 1, :2] - waypoint_array[index, :2]
        ) > 1e-9
    ]

    if not movable:
        lane_id = _nearest_lanes(
            waypoint_array[:1, :2], candidate_lane_ids, centerline_dict)[0]
        lane_metadata = metadata.get(lane_id, {})
        stop = _path_vo(
            final_destination=_is_final(waypoint_array[0, :2], goal),
            lane_id=lane_id,
            points=[
                _position(waypoint_array[0]),
                _position(waypoint_array[0]),
            ],
            speed=waypoint_array[0, 3],
            action=_declared_action(
                lane_metadata, lane_id, junction_lane_ids),
            direction='STRAIGHT',
            length=0.0,
            junction_id=lane_metadata.get('junctionId'),
        )
        return [stop], [_copy_path(stop)]

    if len(path) >= 2:
        waypoint_s = _project_many(
            waypoint_array[:, :2], path_geometry)[0]
        intervals = [
            (
                float(waypoint_s[index]),
                max(float(waypoint_s[index]), float(waypoint_s[index + 1])),
            )
            for index in movable
        ]
        middle_points = np.asarray([
            _point_at_prepared(path_geometry, 0.5 * (start_s + end_s))
            for start_s, end_s in intervals
        ])
    else:
        intervals = [
            (0.0, float(np.linalg.norm(
                waypoint_array[index + 1, :2]
                - waypoint_array[index, :2]
            )))
            for index in movable
        ]
        middle_points = np.asarray([
            0.5 * (
                waypoint_array[index, :2]
                + waypoint_array[index + 1, :2]
            )
            for index in movable
        ])

    lane_matches = _nearest_lanes(
        np.vstack((waypoint_array[:, :2], middle_points)),
        candidate_lane_ids,
        centerline_dict,
    )
    middle_offset = len(waypoint_array)
    short_paths = []
    last_movable = movable[-1]
    for movable_index, index in enumerate(movable):
        start = waypoint_array[index]
        end = waypoint_array[index + 1]
        start_s, end_s = intervals[movable_index]
        length = max(0.0, end_s - start_s)
        duration = float(end[4] - start[4])
        segment_speed = (
            length / duration
            if np.isfinite(duration) and duration > 1e-9
            else 0.0
        )
        start_lane = lane_matches[index]
        end_lane = lane_matches[index + 1]
        middle_lane = lane_matches[middle_offset + movable_index]
        lane_metadata = metadata.get(middle_lane, {})
        junction_id = lane_metadata.get('junctionId')
        turn = _normalize_angle(float(end[2] - start[2]))
        action = _segment_action(
            start_lane, middle_lane, end_lane,
            lane_metadata, connector_dict, junction_lane_ids, turn)
        direction = _turn_direction(turn)
        final_destination = (
            index == last_movable and _is_final(end[:2], goal)
        )
        segment = _path_vo(
            final_destination=final_destination,
            lane_id=middle_lane,
            points=[_position(start), _position(end)],
            speed=segment_speed,
            action=action,
            direction=direction,
            length=length,
            junction_id=junction_id,
        )
        short_paths.append(segment)

    return short_paths, _merge_paths(short_paths)


def _candidate_lane_ids(lane_ids, metadata, request, centerline_dict):
    ordered = []
    for lane_id in lane_ids or ():
        lane_id = str(lane_id)
        if lane_id in centerline_dict and lane_id not in ordered:
            ordered.append(lane_id)
    for lane_id in metadata:
        if lane_id in centerline_dict and lane_id not in ordered:
            ordered.append(lane_id)
    for key in ('startPosition', 'endPosition'):
        lane_id = (request.get(key) or {}).get('laneId')
        if lane_id in centerline_dict and lane_id not in ordered:
            ordered.append(lane_id)
    return tuple(ordered)


def _nearest_lanes(points, lane_ids, centerline_dict):
    queries = np.asarray(points, dtype=float)[:, :2]
    best_lanes = [None] * len(queries)
    best_distances = np.full(len(queries), float('inf'))
    for lane_id in lane_ids:
        geometry = _prepare_polyline(centerline_dict[lane_id])
        distances = _project_many(queries, geometry)[1]
        improved = distances < best_distances
        best_distances[improved] = distances[improved]
        for index in np.flatnonzero(improved):
            best_lanes[int(index)] = lane_id
    return best_lanes


def _prepare_polyline(line):
    points = clean_xy(line)
    if len(points) < 2:
        return points, np.empty((0, 2)), np.empty(0), np.asarray([0.0])
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    return points, segments, lengths, arc


def _project_many(query_points, geometry):
    points, segments, lengths, arc = geometry
    queries = np.asarray(query_points, dtype=float)[:, :2]
    if len(segments) == 0:
        distances = np.linalg.norm(queries - points[0], axis=1)
        return np.zeros(len(queries)), distances
    length_squared = lengths * lengths
    residuals = queries[:, None, :] - points[:-1][None, :, :]
    ratios = np.divide(
        np.sum(residuals * segments[None, :, :], axis=2),
        length_squared[None, :],
        out=np.zeros((len(queries), len(segments))),
        where=length_squared[None, :] > 1e-12,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = (
        points[:-1][None, :, :]
        + ratios[:, :, None] * segments[None, :, :]
    )
    distance_squared = np.sum(
        (queries[:, None, :] - projections) ** 2, axis=2)
    indices = np.argmin(distance_squared, axis=1)
    rows = np.arange(len(queries))
    return (
        arc[indices] + ratios[rows, indices] * lengths[indices],
        np.sqrt(distance_squared[rows, indices]),
    )


def _point_at_prepared(geometry, target_s):
    points, segments, lengths, arc = geometry
    target = float(np.clip(target_s, 0.0, arc[-1]))
    index = min(
        int(np.searchsorted(arc, target, side='right') - 1),
        len(segments) - 1,
    )
    ratio = (
        (target - arc[index]) / lengths[index]
        if lengths[index] > 1e-12 else 0.0
    )
    return points[index] + ratio * segments[index]


def _segment_action(
        start_lane, middle_lane, end_lane, metadata,
        connector_dict, junction_lane_ids, turn):
    declared = metadata.get('action')
    if (metadata.get('junctionId') is not None
            or middle_lane in junction_lane_ids
            or declared == 'TURN'):
        return 'TURN'
    if (start_lane is not None and end_lane is not None
            and start_lane != end_lane
            and not _forward_reachable(
                start_lane, end_lane, connector_dict)):
        return 'LANECHANGE'
    if declared in ('LANECHANGE', 'ROUND'):
        return declared
    return 'TURN' if abs(turn) >= RESPONSE_TURN_THRESHOLD_RAD else 'STRAIGHT'


def _forward_reachable(start_lane, end_lane, connector_dict):
    visited = {start_lane}
    pending = [start_lane]
    while pending:
        lane_id = pending.pop()
        for successor in connector_dict[lane_id]['successor']:
            if successor == end_lane:
                return True
            if successor not in visited:
                visited.add(successor)
                pending.append(successor)
    return False


def _declared_action(metadata, lane_id, junction_lane_ids):
    if (metadata.get('junctionId') is not None
            or lane_id in junction_lane_ids
            or metadata.get('action') == 'TURN'):
        return 'TURN'
    return metadata.get('action') or 'STRAIGHT'


def _turn_direction(turn):
    magnitude = abs(turn)
    if magnitude < RESPONSE_TURN_THRESHOLD_RAD:
        return 'STRAIGHT'
    if magnitude >= RESPONSE_UTURN_THRESHOLD_RAD:
        return 'U_TURN'
    return 'LEFT' if turn > 0.0 else 'RIGHT'


def _request_goal(request):
    position = (request.get('endPosition') or {}).get('position')
    if position is None:
        return None
    return np.asarray([position['x'], position['y']], dtype=float)


def _is_final(point, goal):
    return bool(
        goal is not None
        and np.linalg.norm(np.asarray(point)[:2] - goal)
        <= RESPONSE_DESTINATION_TOLERANCE
    )


def _position(row):
    return {
        'x': float(row[0]),
        'y': float(row[1]),
        'z': 0.0,
        'hdg': _normalize_angle(float(row[2])),
    }


def _path_vo(
        final_destination, lane_id, points, speed,
        action, direction, length, junction_id):
    return {
        'isFinalDestination': bool(final_destination),
        'laneId': lane_id,
        'points': points,
        'speedLimit': max(0.0, float(speed)),
        'action': action,
        'direction': direction,
        'length': max(0.0, float(length)),
        'junctionId': junction_id,
    }


def _merge_paths(short_paths):
    merged = []
    for segment in short_paths:
        if not merged or not _can_merge(merged[-1], segment):
            merged.append(_copy_path(segment))
            continue
        previous = merged[-1]
        first_length = previous['length']
        second_length = segment['length']
        total_length = first_length + second_length
        if total_length > 1e-9:
            first_speed = previous['speedLimit']
            second_speed = segment['speedLimit']
            if first_speed > 1e-9 and second_speed > 1e-9:
                total_time = (
                    first_length / first_speed
                    + second_length / second_speed
                )
                previous['speedLimit'] = total_length / total_time
            else:
                previous['speedLimit'] = 0.0
        previous['points'] = [
            previous['points'][0],
            dict(segment['points'][-1]),
        ]
        previous['length'] = total_length
        previous['isFinalDestination'] = (
            previous['isFinalDestination']
            or segment['isFinalDestination']
        )
    return merged


def _can_merge(first, second):
    if any(first[key] != second[key] for key in (
            'laneId', 'action', 'direction', 'junctionId')):
        return False
    first_stopped = first['speedLimit'] <= 1e-9
    second_stopped = second['speedLimit'] <= 1e-9
    if first_stopped != second_stopped:
        return False
    if len(first['points']) < 2 or len(second['points']) < 2:
        return False
    a0 = first['points'][0]
    a1 = first['points'][-1]
    b0 = second['points'][0]
    b1 = second['points'][-1]
    if math.hypot(a1['x'] - b0['x'], a1['y'] - b0['y']) > 1e-6:
        return False
    first_vector = np.asarray([a1['x'] - a0['x'], a1['y'] - a0['y']])
    second_vector = np.asarray([b1['x'] - b0['x'], b1['y'] - b0['y']])
    first_length = float(np.linalg.norm(first_vector))
    second_length = float(np.linalg.norm(second_vector))
    if first_length <= 1e-9 or second_length <= 1e-9:
        return False
    angle = abs(_normalize_angle(
        math.atan2(second_vector[1], second_vector[0])
        - math.atan2(first_vector[1], first_vector[0])
    ))
    end_offset = np.asarray([b1['x'] - a0['x'], b1['y'] - a0['y']])
    lateral = abs(
        first_vector[0] * end_offset[1]
        - first_vector[1] * end_offset[0]
    ) / first_length
    heading = abs(_normalize_angle(b1['hdg'] - a1['hdg']))
    return (
        angle <= RESPONSE_MERGE_ANGLE_TOLERANCE_RAD
        and lateral <= RESPONSE_MERGE_LATERAL_TOLERANCE
        and heading <= RESPONSE_MERGE_ANGLE_TOLERANCE_RAD
    )


def _copy_path(path):
    copied = dict(path)
    copied['points'] = [dict(point) for point in path['points']]
    return copied


def _normalize_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi
