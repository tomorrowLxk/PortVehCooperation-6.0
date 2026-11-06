"车辆任务级完整可行域及本帧局部可行域规划。"

from collections import deque
from dataclasses import dataclass

import numpy as np

from parameters import (
    DRIVABLE_LINE_FORWARD_LENGTH,
    DRIVABLE_LINE_JOIN_TOLERANCE,
    DRIVABLE_LINE_MAX_COUNT,
    DRIVABLE_LINE_POINT_SPACING,
    FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE,
    FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE,
    JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE,
    JUNCTION_EXIT_PATH_EXTENSION,
    FEASIBLE_DOMAIN_SEARCH_DEPTH,
    VEHICLE_LANE_LOCATE_HEADING_MIN_SPEED,
    VEHICLE_LANE_LOCATE_HEADING_WEIGHT,
    VEHICLE_LANE_LOCATE_MAX_DISTANCE,
    VEHICLE_LANE_LOCATE_ROUTE_PREFERENCE,
    VEHICLE_LANE_LOCATE_TOPOLOGY_DEPTH,
)


@dataclass(frozen=True)
class DrivableLineSegment:
    "一条局部可行域线中的有序车道段及其在线上的累计弧长范围。"
    laneId: str
    startS: float
    endS: float
    isJunction: bool


@dataclass(frozen=True)
class DrivableLine:
    "兼容坐标数组的局部可行域线，同时保留每一段的真实车道ID。"
    points: np.ndarray
    laneSegments: tuple[DrivableLineSegment, ...]

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.points, dtype=dtype)
        return array.copy() if copy else array


@dataclass(frozen=True)
class _PolylineMetric:
    """任务周期内保持不变的折线投影数据。"""
    points: np.ndarray
    segments: np.ndarray
    segmentLengths: np.ndarray
    segmentLengthsSquared: np.ndarray
    cumulativeS: np.ndarray

    @classmethod
    def from_line(cls, line):
        points = np.asarray(line, dtype=float)[:, :2]
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return cls(
            points=points,
            segments=segments,
            segmentLengths=lengths,
            segmentLengthsSquared=lengths * lengths,
            cumulativeS=np.concatenate(([0.0], np.cumsum(lengths))),
        )

    def project_many(self, query_points):
        queries = np.asarray(query_points, dtype=float).reshape(-1, 2)
        relative = queries[:, None, :] - self.points[:-1][None, :, :]
        ratios = np.divide(
            np.sum(relative * self.segments[None, :, :], axis=2),
            self.segmentLengthsSquared[None, :],
            out=np.zeros((len(queries), len(self.segments)), dtype=float),
            where=self.segmentLengthsSquared[None, :] > 1e-12,
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = (
            self.points[:-1][None, :, :]
            + ratios[:, :, None] * self.segments[None, :, :]
        )
        differences = projections - queries[:, None, :]
        distances_squared = np.sum(differences * differences, axis=2)
        indices = np.argmin(distances_squared, axis=1)
        rows = np.arange(len(queries))
        return (
            self.cumulativeS[indices]
            + ratios[rows, indices] * self.segmentLengths[indices],
            np.sqrt(distances_squared[rows, indices]),
        )

    def mean_distance(self, query_points):
        return float(np.mean(self.project_many(query_points)[1]))

class DrivableAreaPlanner:
    "保存单车当前任务的完整可行域和跨帧位置匹配进度。"

    def __init__(self, connector_dict, centerline_dict, referLaneIds, referLine,
                 dynamicInfo, reversePoint=None,
                 forbiddenLaneIds=None, mapVersion=None,
                 junctionLaneIds=None):
        self.reset_planning_context(
            connector_dict, centerline_dict, referLaneIds, referLine,
            dynamicInfo, reversePoint, forbiddenLaneIds, mapVersion,
            junctionLaneIds,
        )

    def reset_planning_context(self, connector_dict, centerline_dict,
                               referLaneIds, referLine,
                               dynamicInfo, reversePoint=None,
                               forbiddenLaneIds=None, mapVersion=None,
                               junctionLaneIds=None, resume_route_s=None):
        """使用新的地图和长路径重建任务级完整可行域。"""
        lane_ids = list(referLaneIds)
        route = np.asarray(referLine, dtype=float)[:, :2].copy()
        forbidden = set(forbiddenLaneIds or [])

        # 新任务可能恰好从首条车道末端开始。此时该车道在route起点之后
        # 已无有效长度，应让其直接下游route车道成为可行域搜索起点。
        while len(lane_ids) > 1:
            first_line = np.asarray(
                centerline_dict[lane_ids[0]], dtype=float
            )[:, :2]
            first_s = self._project_to_line(route[0], first_line)[0]
            first_length = float(np.linalg.norm(
                np.diff(first_line, axis=0), axis=1
            ).sum())
            if first_length - first_s > 1e-6:
                break
            lane_ids.pop(0)

        # 长路径车道序列可能包含route终点尚未实际进入的下游车道。
        # 这类车道在终点阶段没有有效纵向区间，不应参与完整可行域构建。
        while len(lane_ids) > 1:
            last_line = np.asarray(
                centerline_dict[lane_ids[-1]], dtype=float
            )[:, :2]
            last_s = self._project_to_line(route[-1], last_line)[0]
            if last_s > 1e-6:
                break
            print("#"*100)
            lane_ids.pop()

        normalized_connectors = {}
        for lane_id, source in connector_dict.items():
            info = dict(source)
            info['predecessor'] = list(info['predecessor'])
            info['successor'] = list(info['successor'])
            info['left_lane'] = info.get(
                'left_lane', info.get('leftLane')
            )
            info['right_lane'] = info.get(
                'right_lane', info.get('rightLane')
            )
            normalized_connectors[lane_id] = info

        centerlines, connectors, route_stages, connection_bounds = (
            self._build_complete_feasible_domain(
                normalized_connectors,
                centerline_dict,
                lane_ids,
                route,
                forbidden,
            )
        )

        self.mapVersion = mapVersion
        self.referLaneIds = lane_ids
        self.referLine = route
        self.centerline_dict = centerlines
        self.connector_dict = connectors
        self.routeStages = route_stages
        self.connectionBounds = connection_bounds
        self.junctionLaneIds = frozenset(junctionLaneIds or ())
        self.feasibleLaneIds = frozenset(
            lane for stage in self.routeStages for lane in stage
        )
        lane_stage_indices = {}
        for stage_index, stage in enumerate(self.routeStages):
            for lane in stage:
                lane_stage_indices.setdefault(lane, []).append(stage_index)
        self.laneStageIndices = {
            lane: tuple(indices)
            for lane, indices in lane_stage_indices.items()
        }
        self.referenceMetric = _PolylineMetric.from_line(self.referLine)
        self.routeStageLineSimilarity = {}
        self.laneCumulativeS = {
            lane: np.concatenate((
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1)),
            ))
            for lane, line in self.centerline_dict.items()
        }
        (
            self.routeStageBounds,
            self.routeStageReferenceLaneIds,
        ) = self._build_route_stage_context()

        temporary_forward_lanes = {
            lane_id for lane_id in self.connector_dict
            if lane_id.endswith('__DXM_FORWARD')
        }
        temporary_reverse_lanes = {
            lane_id for lane_id in self.connector_dict
            if lane_id.endswith('__DXM_REVERSE')
        }
        dxm_protected_lanes = (
                temporary_forward_lanes | temporary_reverse_lanes
        )
        for lane_id in temporary_forward_lanes:
            dxm_protected_lanes.update(
                self.connector_dict[lane_id]['predecessor']
            )
        self.dxmProtectedLaneIds = frozenset(dxm_protected_lanes)
        self.is_dxm_update_protected = False
        self.lastLaneId = None
        self.lastLaneS = None
        self.lastRouteS = resume_route_s
        self.lastRouteStageIndex = None
        self.reverseStageIndex = self._find_reverse_stage(reversePoint)
        self.passedReversePoint = False
        self._update_position_match(
            dynamicInfo, reversePoint, initial=resume_route_s is None
        )

    @staticmethod
    def _expand_lateral_lanes(seed_lanes, connector_dict, forbidden):
        """从种子车道出发，只沿左右相互确认的关系进行横向扩展。"""
        result = set()
        queue = deque(lane for lane in seed_lanes if lane not in forbidden)
        while queue:
            lane = queue.popleft()
            if lane in result:
                continue
            result.add(lane)

            left_lane = connector_dict[lane]['left_lane']
            if (left_lane is not None
                    and left_lane not in forbidden
                    and connector_dict[left_lane]['right_lane'] == lane
                    and left_lane not in result):
                queue.append(left_lane)

            right_lane = connector_dict[lane]['right_lane']
            if (right_lane is not None
                    and right_lane not in forbidden
                    and connector_dict[right_lane]['left_lane'] == lane
                    and right_lane not in result):
                queue.append(right_lane)
        return result

    @staticmethod
    def _find_bridge_lanes(route_stages, connector_dict, forbidden):
        """搜索连接较早和较晚长路径阶段的有限深度纵向车道。"""
        if FEASIBLE_DOMAIN_SEARCH_DEPTH <= 0:
            return set(), {}

        base_lanes = set().union(*route_stages)
        stage_index = {}
        for index, stage in enumerate(route_stages):
            for lane in stage:
                stage_index.setdefault(lane, set()).add(index)

        bridge_lanes = set()
        bridge_stages = {}
        for start_index, stage in enumerate(route_stages):
            for start_lane in stage:
                stack = [(start_lane, tuple(), {start_lane})]
                while stack:
                    lane, middle_lanes, visited = stack.pop()
                    for next_lane in connector_dict[lane]['successor']:
                        if next_lane in forbidden or next_lane in visited:
                            continue

                        if next_lane in base_lanes:
                            target_indexes = [
                                index for index in stage_index[next_lane]
                                if index > start_index
                            ]
                            if middle_lanes and target_indexes:
                                bridge_lanes.update(middle_lanes)
                                target_index = min(target_indexes)
                                available_stage_count = (
                                        target_index - start_index - 1
                                )
                                for offset, middle_lane in enumerate(
                                        middle_lanes, start=1
                                ):
                                    if available_stage_count > 0:
                                        assigned_index = start_index + min(
                                            offset, available_stage_count
                                        )
                                    else:
                                        assigned_index = start_index
                                    bridge_stages.setdefault(
                                        assigned_index, set()
                                    ).add(middle_lane)
                            continue

                        if len(middle_lanes) >= FEASIBLE_DOMAIN_SEARCH_DEPTH:
                            continue

                        stack.append((
                            next_lane,
                            middle_lanes + (next_lane,),
                            visited | {next_lane},
                        ))
        return bridge_lanes, bridge_stages

    @staticmethod
    def _prune_unreachable_lanes(candidate_lanes, connector_dict,
                                 start_lane, goal_lane):
        "保留从长路径起点可达并且能够继续到达终点的候选车道。"
        forward = {lane: [] for lane in candidate_lanes}
        reverse = {lane: [] for lane in candidate_lanes}

        for lane in candidate_lanes:
            info = connector_dict[lane]
            neighbours = [
                next_lane for next_lane in info['successor']
                if next_lane in candidate_lanes
            ]

            left_lane = info['left_lane']
            if (left_lane in candidate_lanes
                    and connector_dict[left_lane]['right_lane'] == lane):
                neighbours.append(left_lane)

            right_lane = info['right_lane']
            if (right_lane in candidate_lanes
                    and connector_dict[right_lane]['left_lane'] == lane):
                neighbours.append(right_lane)

            for neighbour in neighbours:
                if neighbour not in forward[lane]:
                    forward[lane].append(neighbour)
                    reverse[neighbour].append(lane)

        def reachable(seed, topology):
            result = set()
            queue = deque([seed])
            while queue:
                lane = queue.popleft()
                if lane in result:
                    continue
                result.add(lane)
                queue.extend(topology[lane])
            return result

        return reachable(start_lane, forward) & reachable(goal_lane, reverse)

    @staticmethod
    def _project_to_line(point, line):
        "把点投影到折线，返回累计弧长和横向距离。"
        segments = np.diff(line, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        relative = point - line[:-1]
        ratio = np.divide(
            np.sum(relative * segments, axis=1),
            lengths * lengths,
            out=np.zeros_like(lengths),
            where=lengths > 1e-12,
        )
        ratio = np.clip(ratio, 0.0, 1.0)
        projections = line[:-1] + ratio[:, None] * segments
        distances = np.linalg.norm(projections - point, axis=1)
        index = int(np.argmin(distances))
        arc = np.concatenate(([0.0], np.cumsum(lengths)))
        return arc[index] + ratio[index] * lengths[index], distances[index]

    @staticmethod
    def _slice_line(line, start_s, end_s):
        "按累计弧长截取中心线，保留精确的首尾插值点。"
        lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(lengths)))
        start_s = float(np.clip(start_s, 0.0, arc[-1]))
        end_s = float(np.clip(end_s, start_s, arc[-1]))

        def point_at(target_s):
            index = min(int(np.searchsorted(arc, target_s, side='right') - 1),
                        len(lengths) - 1)
            ratio = (0.0 if lengths[index] <= 1e-12 else
                     (target_s - arc[index]) / lengths[index])
            return line[index] + ratio * (line[index + 1] - line[index])

        points = [point_at(start_s)]
        points.extend(
            line[index] for index in range(1, len(line) - 1)
            if start_s < arc[index] < end_s
        )
        points.append(point_at(end_s))
        return np.asarray(points, dtype=float)

    def _build_route_stage_context(self):
        route_length = float(np.linalg.norm(
            np.diff(self.referLine, axis=0), axis=1
        ).sum())
        bounds = []
        reference_lanes = []
        route_lane_index = 0
        cursor = 0.0

        for stage in self.routeStages:
            stage_reference_lanes = []
            while (route_lane_index < len(self.referLaneIds)
                   and self.referLaneIds[route_lane_index] in stage):
                stage_reference_lanes.append(
                    self.referLaneIds[route_lane_index]
                )
                route_lane_index += 1
            last_reference_lane = stage_reference_lanes[-1]
            end_point = self.centerline_dict[last_reference_lane][-1]
            if route_lane_index < len(self.referLaneIds):
                next_reference_lane = self.referLaneIds[route_lane_index]
                connection = self.connectionBounds.get(
                    (last_reference_lane, next_reference_lane)
                )
                if connection is not None:
                    end_point = self._slice_line(
                        self.centerline_dict[last_reference_lane],
                        connection['fromEndS'],
                        connection['fromEndS'],
                    )[0]
            route_suffix = self._slice_line(
                self.referLine, cursor, route_length
            )
            stage_end = cursor + self._project_to_line(
                end_point, route_suffix
            )[0]
            stage_end = min(max(stage_end, cursor), route_length)
            bounds.append((cursor, stage_end))
            reference_lanes.append(tuple(stage_reference_lanes))
            cursor = stage_end
        return bounds, reference_lanes

    @staticmethod
    def _angle_error(first, second):
        return abs(float(np.arctan2(
            np.sin(first - second),
            np.cos(first - second),
        )))

    def _update_dxm_protection(self):
        "根据最近一次成功匹配的车道统一维护调箱门更新保护状态。"
        self.is_dxm_update_protected = (
            self.lastLaneId in self.dxmProtectedLaneIds
        )

    def _position_match_candidates(self):
        if self.lastLaneId is None:
            if self.lastRouteS is not None:
                return {
                    lane
                    for index, stage in enumerate(self.routeStages)
                    if self.routeStageBounds[index][1] >= self.lastRouteS - 1e-6
                    for lane in stage
                }
            return {self.referLaneIds[0]}

        candidates = {self.lastLaneId}
        frontier = {self.lastLaneId}
        for _ in range(VEHICLE_LANE_LOCATE_TOPOLOGY_DEPTH):
            next_frontier = set()
            for lane in frontier:
                info = self.connector_dict[lane]
                if info['left_lane'] is not None:
                    next_frontier.add(info['left_lane'])
                if info['right_lane'] is not None:
                    next_frontier.add(info['right_lane'])
                next_frontier.update(info['successor'])
            next_frontier -= candidates
            candidates.update(next_frontier)
            frontier = next_frontier

        for lane in tuple(candidates):
            info = self.connector_dict[lane]
            if info['left_lane'] is not None:
                candidates.add(info['left_lane'])
            if info['right_lane'] is not None:
                candidates.add(info['right_lane'])

        return candidates & self.feasibleLaneIds

    def _find_reverse_stage(self, reverse_point):
        if reverse_point is None:
            return None
        point = np.asarray(reverse_point, dtype=float)[:2]
        distance, stage_index = min(
            (
                float(np.linalg.norm(
                    self.centerline_dict[lane][-1] - point
                )),
                index,
            )
            for index, lanes in enumerate(
                self.routeStageReferenceLaneIds
            )
            for lane in lanes
        )
        if distance <= VEHICLE_LANE_LOCATE_MAX_DISTANCE:
            return stage_index
        return None

    def _update_position_match(self, dynamic_info, reverse_point,
                               initial=False):
        if not dynamic_info or not dynamic_info.get('valid', True):
            return None

        position = np.asarray(dynamic_info['position'], dtype=float)[:2]
        heading = float(dynamic_info.get('heading', 0.0))
        speed = float(dynamic_info.get('speed', 0.0))
        if reverse_point is not None and self.reverseStageIndex is None:
            self.reverseStageIndex = self._find_reverse_stage(reverse_point)

        ranked = []
        if initial:
            first_lane = self.referLaneIds[0]
            initial_lanes = [first_lane]
            initial_lanes.extend(
                lane
                for lane in self.connector_dict[first_lane]['successor']
                if lane in self.feasibleLaneIds
            )

            for candidate_order, lane in enumerate(initial_lanes):
                lane_s, distance = self._project_to_line(
                    position, self.centerline_dict[lane]
                )
                stage_index = self.laneStageIndices[lane][0]
                ranked.append((
                    distance,
                    candidate_order,
                    distance,
                    stage_index,
                    lane,
                    lane_s,
                ))
            match_candidates = ()
        else:
            match_candidates = self._position_match_candidates()

        for lane in match_candidates:
            line = self.centerline_dict[lane]
            lane_s, distance = self._project_to_line(position, line)
            if distance > VEHICLE_LANE_LOCATE_MAX_DISTANCE:
                continue

            arc = self.laneCumulativeS[lane]
            lengths = np.diff(arc)
            segment_index = min(
                int(np.searchsorted(arc, lane_s, side='right') - 1),
                len(lengths) - 1,
            )
            tangent = line[segment_index + 1] - line[segment_index]
            lane_heading = float(np.arctan2(tangent[1], tangent[0]))
            angle_error = (
                0.0
                if abs(speed) < VEHICLE_LANE_LOCATE_HEADING_MIN_SPEED
                else self._angle_error(heading, lane_heading)
            )
            if angle_error > np.pi / 2.0:
                continue

            occurrences = (
                index for index in self.laneStageIndices[lane]
                if (self.lastRouteStageIndex is not None
                    and index >= self.lastRouteStageIndex)
                or (self.lastRouteStageIndex is None
                    and self.lastRouteS is not None
                    and self.routeStageBounds[index][1] >= self.lastRouteS - 1e-6)
            )
            for stage_index in occurrences:
                if (self.reverseStageIndex is not None
                        and not self.passedReversePoint
                        and reverse_point is not None
                        and stage_index > self.reverseStageIndex):
                    continue
                if (self.reverseStageIndex is not None
                        and self.passedReversePoint
                        and stage_index <= self.reverseStageIndex):
                    continue

                lane_range = self.routeStages[stage_index][lane]
                effective_start_s = lane_range['startS']
                effective_end_s = lane_range['endS']
                if stage_index > 0:
                    predecessor_starts = [
                        self.connectionBounds[(predecessor, lane)]['toStartS']
                        for predecessor in self.connector_dict[lane]['predecessor']
                        if predecessor in self.routeStages[stage_index - 1]
                           and (predecessor, lane) in self.connectionBounds
                    ]
                    if predecessor_starts:
                        effective_start_s = max(
                            effective_start_s,
                            min(predecessor_starts),
                        )
                if stage_index + 1 < len(self.routeStages):
                    successor_ends = [
                        self.connectionBounds[(lane, successor)]['fromEndS']
                        for successor in self.connector_dict[lane]['successor']
                        if successor in self.routeStages[stage_index + 1]
                           and (lane, successor) in self.connectionBounds
                    ]
                    if successor_ends:
                        effective_end_s = min(
                            effective_end_s,
                            max(successor_ends),
                        )
                if not (
                        effective_start_s - 1e-6
                        <= lane_s
                        <= effective_end_s + 1e-6
                ):
                    continue
                geometric_score = (
                        distance
                        + VEHICLE_LANE_LOCATE_HEADING_WEIGHT * angle_error
                )
                route_penalty = (
                    0.0
                    if lane in self.routeStageReferenceLaneIds[stage_index]
                    else VEHICLE_LANE_LOCATE_ROUTE_PREFERENCE
                )
                ranked.append((
                    geometric_score + route_penalty,
                    geometric_score,
                    distance,
                    stage_index,
                    lane,
                    lane_s,
                ))

        if not ranked:
            return None

        _, _, _, stage_index, lane, lane_s = min(ranked)
        minimum_s, maximum_s = self.routeStageBounds[stage_index]
        route_window = self._slice_line(
            self.referLine, minimum_s, maximum_s
        )
        route_s = minimum_s + self._project_to_line(
            position, route_window
        )[0]
        route_s = min(max(route_s, minimum_s), maximum_s)

        self.lastLaneId = lane
        self.lastLaneS = lane_s
        self.lastRouteStageIndex = stage_index
        self.lastRouteS = (
            route_s if self.lastRouteS is None
            else max(self.lastRouteS, route_s)
        )

        self._update_dxm_protection()

        return {
            'laneId': lane,
            'laneS': lane_s,
            'stageIndex': stage_index,
            'routeS': self.lastRouteS,
        }

    def update_position_match(self, dynamicInfo, reversePoint=None):
        """只更新车辆在完整可行域和全局参考线上的匹配进度。"""
        return self._update_position_match(
            dynamicInfo,
            reversePoint,
            initial=False,
        )

    def _build_stage_ranges(self, route_stages, connector_dict,
                            centerline_dict, route_lane_ids, route):
        "计算每个route阶段中各车道真正可使用的纵向区间。"
        removed = set()
        all_lanes = set(connector_dict)

        def lateral_distance(lane, targets, stage):
            queue = deque([(lane, 0)])
            visited = {lane}
            while queue:
                current, distance = queue.popleft()
                if current in targets:
                    return distance
                info = connector_dict[current]
                for neighbour, reverse_key in (
                        (info['left_lane'], 'right_lane'),
                        (info['right_lane'], 'left_lane')):
                    if (neighbour in stage and neighbour not in visited
                            and connector_dict[neighbour][reverse_key] == current):
                        visited.add(neighbour)
                        queue.append((neighbour, distance + 1))
            return None

        while True:
            active = [
                {lane for lane in stage if (index, lane) not in removed}
                for index, stage in enumerate(route_stages)
            ]

            def lateral_closure(lanes, stage):
                return self._expand_lateral_lanes(
                    lanes,
                    connector_dict,
                    all_lanes - stage,
                ) & stage

            backward = [set() for _ in active]
            backward[-1] = {
                min(
                    active[-1],
                    key=lambda lane: self._project_to_line(
                        route[-1], centerline_dict[lane]
                    )[1],
                )
            }
            for index in range(len(active) - 2, -1, -1):
                next_reachable = lateral_closure(
                    backward[index + 1], active[index + 1]
                )
                backward[index] = {
                    lane for lane in active[index]
                    if any(
                        successor in next_reachable
                        for successor in connector_dict[lane]['successor']
                    )
                }

            forward = [set() for _ in active]
            if route_lane_ids[0] in active[0]:
                forward[0].add(route_lane_ids[0])
            for index in range(1, len(active)):
                previous_reachable = lateral_closure(
                    forward[index - 1], active[index - 1]
                )
                previous_core = (
                    previous_reachable & backward[index - 1]
                )
                forward[index] = {
                    lane for lane in active[index]
                    if any(
                        lane in connector_dict[previous]['successor']
                        for previous in previous_core
                    )
                }

            ranges = []
            newly_removed = set()
            for index, stage in enumerate(active):
                stage_ranges = {}
                for lane in stage:
                    line = centerline_dict[lane]
                    total = float(np.linalg.norm(
                        np.diff(line, axis=0), axis=1
                    ).sum())

                    if index == 0:
                        start_s = self._project_to_line(route[0], line)[0]
                        distance = lateral_distance(
                            lane, forward[index], stage
                        )
                        if distance is None:
                            newly_removed.add((index, lane))
                            continue
                        start_s += (
                                distance * FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE
                        )
                    elif lane in forward[index]:
                        start_s = 0.0
                    else:
                        previous_core = (
                                forward[index - 1] & backward[index - 1]
                        )
                        previous_expanded = self._expand_lateral_lanes(
                            previous_core,
                            connector_dict,
                            set(connector_dict) - active[index - 1],
                        ) & active[index - 1]
                        enters_directly = any(
                            lane in connector_dict[previous]['successor']
                            for previous in previous_expanded
                        )
                        distance = lateral_distance(
                            lane, forward[index], stage
                        )
                        if not enters_directly and distance is None:
                            newly_removed.add((index, lane))
                            continue
                        start_s = (
                            0.0 if enters_directly else
                            distance * FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE
                        )

                    if index == len(active) - 1:
                        end_s = self._project_to_line(route[-1], line)[0]
                        distance = lateral_distance(
                            lane, backward[index], stage
                        )
                        if distance is None:
                            newly_removed.add((index, lane))
                            continue
                        end_s -= (
                                distance * FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE
                        )
                    elif lane in backward[index]:
                        end_s = total
                    else:
                        next_core = (
                                backward[index + 1] & forward[index + 1]
                        )
                        next_expanded = self._expand_lateral_lanes(
                            next_core,
                            connector_dict,
                            set(connector_dict) - active[index + 1],
                        ) & active[index + 1]
                        exits_directly = any(
                            successor in next_expanded
                            for successor in connector_dict[lane]['successor']
                        )
                        distance = lateral_distance(
                            lane, backward[index], stage
                        )
                        if not exits_directly and distance is None:
                            newly_removed.add((index, lane))
                            continue
                        end_s = (
                            total if exits_directly else
                            total - distance
                            * FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE
                        )

                    if end_s - start_s <= 1e-6:
                        newly_removed.add((index, lane))
                        continue

                    stage_ranges[lane] = {
                        'startS': start_s,
                        'endS': end_s,
                        'centerline': self._slice_line(
                            line, start_s, end_s
                        ),
                    }
                ranges.append(stage_ranges)

            if not newly_removed:
                return ranges
            removed.update(newly_removed)

    def _build_complete_feasible_domain(self, connector_dict, centerline_dict,
                                        route_lane_ids, route, forbidden):
        "根据长路径、完整地图和禁行车道生成任务级完整可行域。"
        # 第一次扩展：对长路径的每条车道执行双向确认的横向扩展。
        route_stages = []
        for route_lane in route_lane_ids:
            stage = self._expand_lateral_lanes(
                [route_lane], connector_dict, forbidden
            )
            if route_stages and route_stages[-1] & stage:
                route_stages[-1].update(stage)
            else:
                route_stages.append(stage)

        base_lanes = set().union(*route_stages)

        # 第二次扩展：搜索能够重新接回较晚长路径阶段的连接车道。
        bridge_lanes, bridge_stages = self._find_bridge_lanes(
            route_stages, connector_dict, forbidden
        )

        # 第三次扩展：把新加入连接车道的可换入、可换出车道一并加入。
        bridge_lateral_lanes = set()
        for stage_index, stage_bridge_lanes in bridge_stages.items():
            expanded = self._expand_lateral_lanes(
                stage_bridge_lanes, connector_dict, forbidden
            )
            route_stages[stage_index].update(expanded)
            bridge_lateral_lanes.update(expanded - bridge_lanes)
        candidate_lanes = base_lanes | bridge_lanes | bridge_lateral_lanes

        selected_lanes = self._prune_unreachable_lanes(
            candidate_lanes,
            connector_dict,
            route_lane_ids[0],
            route_lane_ids[-1],
        )

        # 完整可行域就是该车辆任务长期保存的局部地图。
        local_centerlines = {
            lane: np.asarray(centerline_dict[lane], dtype=float)[:, :2].copy()
            for lane in selected_lanes
        }
        local_connectors = {}
        for lane in selected_lanes:
            source = connector_dict[lane]
            left_lane = source['left_lane']
            right_lane = source['right_lane']

            if (left_lane not in selected_lanes
                    or connector_dict[left_lane]['right_lane'] != lane):
                left_lane = None
            if (right_lane not in selected_lanes
                    or connector_dict[right_lane]['left_lane'] != lane):
                right_lane = None

            local_connectors[lane] = {
                **source,
                'predecessor': [
                    value for value in source['predecessor']
                    if value in selected_lanes
                ],
                'successor': [
                    value for value in source['successor']
                    if value in selected_lanes
                ],
                'left_lane': left_lane,
                'right_lane': right_lane,
            }

        route_stages = [stage & selected_lanes for stage in route_stages]
        route_stages = self._build_stage_ranges(
            route_stages,
            local_connectors,
            local_centerlines,
            route_lane_ids,
            route,
        )

        # 前后车道存在几何重叠时，记录每条有向连接的无重叠拼接边界。
        connection_bounds = {}
        lane_lengths = {
            lane: float(np.linalg.norm(
                np.diff(line, axis=0), axis=1
            ).sum())
            for lane, line in local_centerlines.items()
        }
        for lane, info in local_connectors.items():
            upstream = local_centerlines[lane]
            upstream_length = lane_lengths[lane]
            for successor in info['successor']:
                downstream = local_centerlines[successor]
                downstream_length = lane_lengths[successor]
                from_end_s = upstream_length
                to_start_s = 0.0

                # 优先完整保留上游车道，并删除下游车道与其重叠的开头。
                # 这样车辆已经驶到上游末端时，拼线不会退回下游起点。
                projected_s, distance = self._project_to_line(
                    upstream[-1], downstream
                )
                if (distance <= FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE
                        and projected_s
                        > FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE):
                    to_start_s = min(projected_s, downstream_length)
                else:
                    projected_s, distance = self._project_to_line(
                        downstream[0], upstream
                    )
                    if (distance
                            <= FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE
                            and projected_s
                            < upstream_length
                            - FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE):
                        from_end_s = projected_s

                connection_bounds[(lane, successor)] = {
                    'fromEndS': from_end_s,
                    'toStartS': to_start_s,
                }

        return (
            local_centerlines,
            local_connectors,
            route_stages,
            connection_bounds,
        )

    @staticmethod
    def _line_length(line):
        return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())

    @staticmethod
    def _resample_line(line, spacing):
        segment_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total = float(cumulative[-1])
        sample_s = np.arange(0.0, total, spacing)
        if len(sample_s) == 0 or total - sample_s[-1] > 1e-8:
            sample_s = np.append(sample_s, total)
        else:
            sample_s[-1] = total

        segment_indexes = np.searchsorted(
            cumulative, sample_s, side='right'
        ) - 1
        segment_indexes = np.clip(
            segment_indexes, 0, len(segment_lengths) - 1
        )
        ratios = np.divide(
            sample_s - cumulative[segment_indexes],
            segment_lengths[segment_indexes],
            out=np.zeros_like(sample_s),
            where=segment_lengths[segment_indexes] > 1e-12,
        )
        return (
            line[segment_indexes]
            + ratios[:, None]
            * (line[segment_indexes + 1] - line[segment_indexes])
        )

    def _lane_distance(self, first_lane, second_lane, stage_index):
        if first_lane == second_lane:
            return 0
        stage_lanes = set(self.routeStages[stage_index])
        queue = deque([(first_lane, 0)])
        visited = {first_lane}
        while queue:
            lane, distance = queue.popleft()
            info = self.connector_dict[lane]
            for neighbour, reverse_key in (
                    (info['left_lane'], 'right_lane'),
                    (info['right_lane'], 'left_lane')):
                if (neighbour not in stage_lanes
                        or neighbour in visited
                        or self.connector_dict[neighbour][reverse_key] != lane):
                    continue
                if neighbour == second_lane:
                    return distance + 1
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
        if (first_lane in self.junctionLaneIds
                and second_lane in self.junctionLaneIds
                and self._parallel_junction_lanes(
                    first_lane, second_lane, stage_index)):
            return 1
        return None

    def _parallel_junction_lanes(self, first_lane, second_lane, stage_index):
        """Infer first-stage junction adjacency from feasible neighbouring stages."""
        for adjacent_stage, connection_key in (
                (stage_index - 1, 'predecessor'),
                (stage_index + 1, 'successor')):
            if not 0 <= adjacent_stage < len(self.routeStages):
                continue
            stage_lanes = self.routeStages[adjacent_stage]
            first_connected = [
                lane for lane in self.connector_dict[first_lane][connection_key]
                if lane in stage_lanes
            ]
            second_connected = {
                lane for lane in self.connector_dict[second_lane][connection_key]
                if lane in stage_lanes
            }
            for lane in first_connected:
                info = self.connector_dict[lane]
                for neighbour, reverse_key in (
                        (info['left_lane'], 'right_lane'),
                        (info['right_lane'], 'left_lane')):
                    if (neighbour in second_connected
                            and self.connector_dict[neighbour][reverse_key]
                            == lane):
                        return True
        return False

    def _line_similarity(self, line):
        return self.referenceMetric.mean_distance(line)

    def _stage_line_similarity(self, stage_index, lane):
        key = (stage_index, lane)
        value = self.routeStageLineSimilarity.get(key)
        if value is None:
            value = self._line_similarity(
                self.routeStages[stage_index][lane]['centerline'])
            self.routeStageLineSimilarity[key] = value
        return value

    def _planning_target(self, current_stage_index, reverse_point):
        route_length = self._line_length(self.referLine)
        phase_end_s = route_length
        if (reverse_point is not None
                and not self.passedReversePoint
                and self.reverseStageIndex is not None):
            phase_end_s = min(
                phase_end_s,
                self.routeStageBounds[self.reverseStageIndex][1],
            )

        target_route_s = min(
            self.lastRouteS + DRIVABLE_LINE_FORWARD_LENGTH,
            phase_end_s,
        )
        target_stage_index = len(self.routeStageBounds) - 1
        for index in range(current_stage_index, len(self.routeStageBounds)):
            if target_route_s <= self.routeStageBounds[index][1] + 1e-6:
                target_stage_index = index
                break

        # If extending beyond one junction reaches another junction, continue
        # through the latter as well. This keeps the final junction exit and
        # its short downstream section in the same planning horizon.
        for _ in range(len(self.routeStageBounds)):
            terminal_junction_stage = None
            for index in range(current_stage_index,
                               target_stage_index + 1):
                if not any(
                        lane in self.junctionLaneIds
                        for lane in self.routeStageReferenceLaneIds[index]):
                    continue
                junction_exit_route_s = self.routeStageBounds[index][1]
                if (target_route_s
                        <= junction_exit_route_s
                        + JUNCTION_EXIT_PATH_EXTENSION + 1e-6):
                    terminal_junction_stage = index
            if terminal_junction_stage is None:
                break

            junction_exit_route_s = self.routeStageBounds[
                terminal_junction_stage
            ][1]
            remaining_after_exit = phase_end_s - junction_exit_route_s
            if remaining_after_exit <= 1e-6:
                break
            if (remaining_after_exit
                    <= JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE):
                next_target_route_s = phase_end_s
            else:
                next_target_route_s = min(
                    junction_exit_route_s + JUNCTION_EXIT_PATH_EXTENSION,
                    phase_end_s,
                )
            if next_target_route_s <= target_route_s + 1e-6:
                break

            target_route_s = next_target_route_s
            target_stage_index = len(self.routeStageBounds) - 1
            for index in range(terminal_junction_stage,
                               len(self.routeStageBounds)):
                if (target_route_s
                        <= self.routeStageBounds[index][1] + 1e-6):
                    target_stage_index = index
                    break
        if (reverse_point is not None
                and not self.passedReversePoint
                and abs(target_route_s - phase_end_s) <= 1e-6):
            target_point = np.asarray(reverse_point, dtype=float)[:2]
        else:
            target_point = self._slice_line(
                self.referLine, target_route_s, target_route_s
            )[0]
        return target_stage_index, target_point, target_route_s

    def _target_lane_ends(self, target_stage_index, target_point):
        target_ends = {}
        for lane, lane_range in self.routeStages[target_stage_index].items():
            # 按候选终点车道ID判断；若它是交叉口内虚拟车道，
            # 终点放到该车道末端，不继续额外拼接出口道路。
            if lane in self.junctionLaneIds:
                target_ends[lane] = lane_range['endS']
                continue

            lane_s, _ = self._project_to_line(
                target_point, self.centerline_dict[lane]
            )
            if (lane_range['startS'] - 1e-6
                    <= lane_s
                    <= lane_range['endS'] + 1e-6):
                target_ends[lane] = min(
                    max(lane_s, lane_range['startS']),
                    lane_range['endS'],
                )
        return target_ends

    def _backward_viable_lanes(self, current_stage_index,
                               target_stage_index, target_lanes):
        viable = {target_stage_index: set(target_lanes)}
        for index in range(target_stage_index - 1,
                           current_stage_index - 1, -1):
            next_lanes = viable[index + 1]
            viable[index] = {
                lane for lane in self.routeStages[index]
                if any(
                    successor in next_lanes
                    for successor in self.connector_dict[lane]['successor']
                )
            }
        return viable

    def _build_drivable_line(self, first_lane, current_position,
                             lateral_distance, current_stage_index,
                             target_stage_index, target_ends, viable):
        lane = first_lane
        previous_lane = None
        parts = []
        lane_segments = []
        accumulated_s = 0.0

        for stage_index in range(current_stage_index,
                                 target_stage_index + 1):
            lane_range = self.routeStages[stage_index][lane]
            line = self.centerline_dict[lane]

            if stage_index == current_stage_index:
                projected_s, _ = self._project_to_line(
                    current_position, line
                )
                start_s = max(
                    lane_range['startS'],
                    projected_s
                    + lateral_distance
                    * FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE,
                )
            else:
                start_s = lane_range['startS']
                connection = self.connectionBounds.get(
                    (previous_lane, lane)
                )
                if connection is not None:
                    start_s = max(start_s, connection['toStartS'])

            next_lane = None
            if stage_index == target_stage_index:
                end_s = target_ends[lane]
            else:
                successors = [
                    successor
                    for successor in self.connector_dict[lane]['successor']
                    if successor in viable[stage_index + 1]
                ]
                if not successors:
                    return None
                next_lane = min(
                    successors,
                    key=lambda successor: self._stage_line_similarity(
                        stage_index + 1, successor),
                )
                end_s = lane_range['endS']
                connection = self.connectionBounds.get((lane, next_lane))
                if connection is not None:
                    end_s = min(end_s, connection['fromEndS'])

            if end_s > start_s + 1e-6:
                part = self._resample_line(
                    self._slice_line(line, start_s, end_s),
                    DRIVABLE_LINE_POINT_SPACING,
                )
                segment_start_s = accumulated_s
                if parts:
                    gap = float(np.linalg.norm(parts[-1][-1] - part[0]))
                    if gap <= DRIVABLE_LINE_JOIN_TOLERANCE:
                        part = part[1:]
                if len(part):
                    if parts:
                        accumulated_s += self._line_length(
                            np.vstack((parts[-1][-1], part))
                        )
                    else:
                        accumulated_s += self._line_length(part)
                    parts.append(part)
                    lane_segments.append(DrivableLineSegment(
                        laneId=lane,
                        startS=segment_start_s,
                        endS=accumulated_s,
                        isJunction=lane in self.junctionLaneIds,
                    ))

            previous_lane = lane
            lane = next_lane

        if not parts:
            return None
        result = np.concatenate(parts)
        if len(result) < 2 or self._line_length(result) <= 1e-6:
            return None
        return DrivableLine(result, tuple(lane_segments))

    def plan(self, currentPosition, currentHeading=0.0,
             currentSpeed=0.0, reversePoint=None):
        "更新本帧位置匹配；局部可行线将在下一阶段实现。"
        location = self._update_position_match(
            {
                'valid': True,
                'position': currentPosition,
                'heading': currentHeading,
                'speed': currentSpeed,
            },
            reversePoint,
            initial=False,
        )
        if location is None:
            return [], None

        current_position = np.asarray(currentPosition, dtype=float)[:2]
        current_stage_index = location['stageIndex']
        target_stage_index, target_point, target_route_s = self._planning_target(
            current_stage_index, reversePoint
        )
        target_ends = self._target_lane_ends(
            target_stage_index, target_point
        )
        if not target_ends:
            return [], target_route_s
        viable = self._backward_viable_lanes(
            current_stage_index,
            target_stage_index,
            target_ends,
        )

        candidates = []
        for lane in viable[current_stage_index]:
            lateral_distance = self._lane_distance(
                self.lastLaneId, lane, current_stage_index
            )
            if lateral_distance is None:
                continue
            line = self._build_drivable_line(
                lane,
                current_position,
                lateral_distance,
                current_stage_index,
                target_stage_index,
                target_ends,
                viable,
            )
            if line is None:
                continue
            candidates.append((
                lane,
                line,
                self._line_similarity(line.points),
            ))

        candidates.sort(key=lambda item: (
            0 if item[0] == self.lastLaneId else 1,
            item[2],
            item[0],
        ))
        return ([
            line for _, line, _ in candidates[:DRIVABLE_LINE_MAX_COUNT]
        ], target_route_s)

    def set_progress_at_point(self, point):
        "车辆到达调箱门倒车点后，将匹配进度切换到下游反向stage。"
        if self.reverseStageIndex is None:
            self.reverseStageIndex = self._find_reverse_stage(point)
        reverse_stage = self.reverseStageIndex
        if reverse_stage is None:
            return

        reverse_route_s = self.routeStageBounds[reverse_stage][1]
        self.lastRouteS = (
            reverse_route_s if self.lastRouteS is None
            else max(self.lastRouteS, reverse_route_s)
        )
        self.passedReversePoint = True

        next_stage = reverse_stage + 1
        if next_stage >= len(self.routeStages):
            return

        source_lane = self.lastLaneId
        if source_lane not in self.routeStages[reverse_stage]:
            source_lane = self.routeStageReferenceLaneIds[
                reverse_stage
            ][-1]
        connected = [
            lane for lane in self.connector_dict[source_lane]['successor']
            if lane in self.routeStages[next_stage]
        ]
        if not connected:
            return

        anchor = np.asarray(point, dtype=float)[:2]
        lane = min(
            connected,
            key=lambda lane_id: self._project_to_line(
                anchor, self.centerline_dict[lane_id]
            )[1],
        )
        lane_s, distance = self._project_to_line(
            anchor, self.centerline_dict[lane]
        )
        if distance > VEHICLE_LANE_LOCATE_MAX_DISTANCE:
            return

        self.lastLaneId = lane
        self.lastLaneS = lane_s
        self.lastRouteStageIndex = next_stage
        self._update_dxm_protection()
