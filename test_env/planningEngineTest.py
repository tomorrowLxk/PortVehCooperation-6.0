import time
import numpy as np
from typing import Any, Dict, List

from shortCache.vehInfo import update_vehicle_dynamic_cache, update_vehicle_static_cache
from shortCache.requestInfo import build_request_temporary_info

from parameters import (
    AGV_DESTINATION_REMAINING_LENGTH_THRESHOLD,
    HV_DESTINATION_DISTANCE_THRESHOLD,
    VEHICLE_OBSTACLE_CONFIRM_FRAMES,
    VEHICLE_OBSTACLE_MAX_FRAMES,
    VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD,
    VEHICLE_STOP_SPEED_THRESHOLD,
    AGV_BASE_PATH_DEVIATION_THRESHOLD,
    AGV_BASE_PATH_ENDPOINT_SAME_THRESHOLD,
    AGV_BASE_PATH_REPLAN_WAYPOINT_DISTANCE,
    PATH_COLLISION_SAMPLE_SPACING,
    PATH_LATERAL_SAFETY_MARGIN,
)

from localPlanner.cooperationBlocker import group_by_future_paths
from localPlanner.physicalRelationPlanner import determine_physical_front_rear_edges

from localPlanner.drivableAreaPlanner import DrivableAreaPlanner
from localPlanner.dxmPlanner import fixed_dxm
from localPlanner.priorityPlanner import calculate_block_priority

from localPlanner.spatialPlanningStage import (
    build_obstacle_polygons,
    exclude_vehi_polygons,
    mark_response_failure,
    plan_agv_spatial,
    plan_hv_spatial,
)
from localPlanner.pathCollision import path_vehicle_intersects
from localPlanner.responsePlanner import build_response_path_lists
from localPlanner.pathGeometry import (
    clean_xy,
    polyline_length,
    project_to_polyline,
    slice_polyline,
)
from localPlanner.trafficSignalPathProcessor import (
    build_future_path,
    classify_path_lane_ids,
)
from localPlanner.waypointCandidatePlanner import (
    build_base_waypoint_candidates,
    build_future_waypoint_candidates,
)
from localPlanner.waypointWholePlanner import (
    align_old_waypoint_whole,
    plan_block_waypoint_whole,
    should_replace_waypoint_whole,
)
from localPlanner.waypointSparsePlanner import (
    build_hv_waypoint_sparse,
    build_waypoint_sparse,
)


class PlanningEngine:
    def __init__(self, centerlineDict, connectorDict, mapVersion) -> None:
        # 测试版全部地图数据由场景JSON加载后从外部传入。
        self.centerline_dict = centerlineDict
        self.connector_dict = connectorDict

        # 长路径计算结果
        self.responses = None
        self.shortPathFailedVehicleIds = []

        # 必要的，储存地图版本
        self.map_version = str(mapVersion)

        # 仅AGV需要DrivableAreaPlanner；Planner内部只持有该任务涉及的局部小地图。
        self.drivableAreaPlannerCaches: Dict[str, Any] = {}

        # 所有车辆修复前的原始laneId序列、全局参考线和禁行车道集合。
        self.vehicleLaneIDCaches: Dict[str, List[str]] = {}
        self.vehicleRouteCaches: Dict[str, Any] = {}
        self.vehicleForbiddenLaneIDCaches: Dict[str, frozenset[str]] = {}

        # 各车完整basePath；AGV允许跨帧复用，HV每帧重新预测并覆盖。
        self.vehicleBasePathCaches: Dict[str, Any] = {}
        # 与basePath同时更新和清除的固定空间候选：[x, y, basePathS]。
        self.vehicleBaseWaypointCandidateCaches: Dict[str, Any] = {}
        # 与basePath逐点对应的laneId，仅AGV在生成新basePath时计算一次。
        self.vehicleBasePathLaneIDCaches: Dict[str, Any] = {}
        # AGV上一轮实际采用的带时间完整waypoint序列。
        self.vehicleWaypointWholeCaches: Dict[str, Any] = {}
        # AGV上一轮最终向车端发布的稀疏waypoint序列。
        self.vehicleWaypointSparseCaches: Dict[str, Any] = {}
        # 本帧必须重新向车端下发waypoint的车辆ID；每帧开始时清空。
        self.vehicleIdsNeedWaypointUpdate: set[str] = set()
        # 本帧空间重规划已经能够确定必须下发或必须不下发的车辆ID。
        self.vehicleIdsMustDispatch: set[str] = set()
        self.vehicleIdsMustNotDispatch: set[str] = set()
        # 已经向车端下发最终目的地的车辆；同一任务后续不再重复规划和下发。
        self.finalDestinationVehicleIds: set[str] = set()

        # AV/HV统一内部静态参数：长宽、限速、最小转弯半径等。
        self.vehicleStaticInfoCaches: Dict[str, Any] = {}
        # AV/HV统一内部动态信息：坐标、朝向、速度、纵横向定位不确定性
        self.vehicleDynamicInfoCaches: Dict[str, Any] = {}
        # 每辆车是否应作为静态障碍物处理，以及“应走但未走”的连续帧数。
        self.vehicleObstacleCaches: Dict[str, Any] = {}

        # reversePointCaches以车辆ID为key，只保存当前调箱门任务的倒车切换点xy。
        self.reversePointCaches: Dict[str, Any] = {}
        # 完整调箱门任务车辆集合；到达R点后仍保留，直到任务变化或车辆驶出。
        self.dxmTaskVehicleIds: set[str] = set()
        # AV/HV任务身份与R点是否已通过无关；只有新commandId才开始新任务。
        self.vehicleTaskCaches: Dict[str, Dict[str, Any]] = {}
        # HV只沿route定位，单独维护弧长进度，不实例化可行域规划器。
        self.vehicleRouteProgressCaches: Dict[str, Dict[str, Any]] = {}

        # 全局优先级有向图与协同分块无关。edges只保留上一帧仍有冲突的，高->低关系；AGV节点长期保留，消失于BSM的HV节点在每帧末删除。
        self.priorityGraphCache: Dict[str, Any] = {
            "nodes": {},
            "edges": set(),
        }

    def prepareVehicle(self, request, response, laneIds, route, forbiddenLaneIds,
                       dynamicInfo, staticInfo, centerlineDict, connectorDict,
                       mapVersion, junctionLaneIds):
        """根据本帧长路径结果，原地准备或更新单车规划上下文。"""
        vehicleId = str(request.get('deviceId'))
        vehicleType = request.get('deviceType')
        isAgv = vehicleType == 'AGV'
        commandId = request.get('commandId')
        allowReverse = bool(request.get('allowReverse')) if isAgv else False

        planner = self.drivableAreaPlannerCaches.get(vehicleId)
        previousTask = self.vehicleTaskCaches.get(vehicleId)
        reversePoint = self.reversePointCaches.get(vehicleId)
        taskChanged = (
                previousTask is None
                or previousTask.get('commandId') != commandId
        )
        if taskChanged:
            self.dxmTaskVehicleIds.discard(vehicleId)
            self.finalDestinationVehicleIds.discard(vehicleId)

        # 同一个调箱门任务仍处于保护区时，冻结短路径内部上下文。
        if (isAgv and planner is not None and not taskChanged
                and previousTask.get('allowReverse')
                and planner.is_dxm_update_protected):
            return

        # 本帧长路径失败：这里只决定保留或清除历史上下文；
        # 实际采用的降级方式由后续空间规划执行并写入response。
        if not response.get('success', True):
            if not isAgv:
                return

            hasHistoricalContext = (
                    not taskChanged
                    and vehicleId in self.vehicleLaneIDCaches
                    and vehicleId in self.vehicleRouteCaches
                    and vehicleId in self.vehicleForbiddenLaneIDCaches
                    and planner is not None
            )
            if hasHistoricalContext:
                return

            self.vehicleLaneIDCaches.pop(vehicleId, None)
            self.vehicleRouteCaches.pop(vehicleId, None)
            self.vehicleForbiddenLaneIDCaches.pop(vehicleId, None)
            self.drivableAreaPlannerCaches.pop(vehicleId, None)
            self.reversePointCaches.pop(vehicleId, None)
            self.vehicleBasePathCaches.pop(vehicleId, None)
            self.vehicleBaseWaypointCandidateCaches.pop(vehicleId, None)
            self.vehicleBasePathLaneIDCaches.pop(vehicleId, None)
            self.vehicleWaypointWholeCaches.pop(vehicleId, None)
            self.vehicleWaypointSparseCaches.pop(vehicleId, None)
            self.vehicleRouteProgressCaches.pop(vehicleId, None)
            self.vehicleTaskCaches.pop(vehicleId, None)
            return

        sourceLaneIds = [str(value) for value in laneIds if value is not None]
        forbiddenLaneIds = frozenset(str(value) for value in forbiddenLaneIds if value is not None)

        # 非AGV只维护修复前长路径；任务变化时旧短路径不能继续参与停车判断。
        if not isAgv:
            sourceRoute = np.array(route, dtype=float, copy=True)[:, :2]
            if taskChanged:
                self.vehicleBasePathCaches.pop(vehicleId, None)
                self.vehicleBaseWaypointCandidateCaches.pop(vehicleId, None)
                self.vehicleBasePathLaneIDCaches.pop(vehicleId, None)
                self.vehicleWaypointWholeCaches.pop(vehicleId, None)
                self.vehicleWaypointSparseCaches.pop(vehicleId, None)
                self.vehicleRouteProgressCaches.pop(vehicleId, None)

            self.vehicleTaskCaches[vehicleId] = {
                'commandId': commandId,
                'allowReverse': False,
            }
            self.vehicleLaneIDCaches[vehicleId] = sourceLaneIds
            self.vehicleRouteCaches[vehicleId] = sourceRoute
            self.vehicleForbiddenLaneIDCaches[vehicleId] = forbiddenLaneIds
            return

        # 仅比较修复前数据；route只比较点数，避免坐标微小变化频繁重置Planner。
        contextChanged = True
        if planner is not None:
            cachedLaneIds = self.vehicleLaneIDCaches.get(vehicleId)
            cachedRoute = self.vehicleRouteCaches.get(vehicleId)
            mapChanged = getattr(planner, 'mapVersion', None) != mapVersion
            forbiddenChanged = self.vehicleForbiddenLaneIDCaches.get(vehicleId) != forbiddenLaneIds
            laneIdsChanged = cachedLaneIds != sourceLaneIds
            routeChanged = (
                    cachedRoute is None
                    or len(cachedRoute) != len(route)
            )
            contextChanged = (
                    mapChanged
                    or forbiddenChanged
                    or laneIdsChanged
                    or routeChanged
            )

        if not taskChanged and planner is not None and not contextChanged:
            return

        # route只比较点数；确认需要重建上下文后再转换完整坐标，避免正常帧重复复制。
        sourceRoute = np.array(route, dtype=float, copy=True)[:, :2]

        # 任务/地图/长路径变化后，所有依赖旧任务的规划结果立即失效。
        if taskChanged:
            self.reversePointCaches.pop(vehicleId, None)
        self.vehicleBasePathCaches.pop(vehicleId, None)
        self.vehicleBaseWaypointCandidateCaches.pop(vehicleId, None)
        self.vehicleBasePathLaneIDCaches.pop(vehicleId, None)
        self.vehicleWaypointWholeCaches.pop(vehicleId, None)
        self.vehicleWaypointSparseCaches.pop(vehicleId, None)
        self.vehicleRouteProgressCaches.pop(vehicleId, None)

        # 调箱门修复属于局部规划上下文构建：新任务必修复，同任务仅在R点尚存时修复。
        effectiveLaneIds = sourceLaneIds
        effectiveRoute = sourceRoute
        localConnector = connectorDict
        localCenterlines = centerlineDict

        applyDxm = allowReverse and (taskChanged or reversePoint is not None)
        nextReversePoint = None
        if applyDxm:
            (effectiveLaneIds, effectiveRoute, localConnector, localCenterlines, nextReversePoint) = fixed_dxm(
                sourceLaneIds,
                sourceRoute,
                connectorDict,
                centerlineDict,
                vehicleStaticInfo=staticInfo,
            )
        effectiveReversePoint = nextReversePoint if applyDxm else None

        if planner is None:
            try:
                planner = DrivableAreaPlanner(
                    localConnector, localCenterlines,  # 修复后的地图
                    effectiveLaneIds, effectiveRoute,  # 修复后的长路径
                    dynamicInfo=dynamicInfo,
                    reversePoint=effectiveReversePoint,
                    forbiddenLaneIds=forbiddenLaneIds,
                    mapVersion=mapVersion,
                    junctionLaneIds=junctionLaneIds,
                )
            except Exception as error:
                mark_response_failure(
                    response,
                    'DRIVABLE_AREA_INITIALIZATION_FAILED',
                    f'可行域规划器初始化失败：{error}',
                )
                return
            self.drivableAreaPlannerCaches[vehicleId] = planner
        else:
            try:
                planner.reset_planning_context(
                    connector_dict=localConnector,
                    centerline_dict=localCenterlines,
                    referLaneIds=effectiveLaneIds,
                    referLine=effectiveRoute,
                    dynamicInfo=dynamicInfo,
                    reversePoint=effectiveReversePoint,
                    forbiddenLaneIds=forbiddenLaneIds,
                    mapVersion=mapVersion,
                    junctionLaneIds=junctionLaneIds,
                )
            except Exception as error:
                self.drivableAreaPlannerCaches.pop(vehicleId, None)
                mark_response_failure(
                    response,
                    'DRIVABLE_AREA_UPDATE_FAILED',
                    f'可行域规划器更新失败：{error}。',
                )
                return

        if applyDxm:
            self.reversePointCaches[vehicleId] = nextReversePoint

        self.vehicleLaneIDCaches[vehicleId] = sourceLaneIds
        self.vehicleRouteCaches[vehicleId] = sourceRoute
        self.vehicleForbiddenLaneIDCaches[vehicleId] = forbiddenLaneIds
        self.vehicleTaskCaches[vehicleId] = {
            'commandId': commandId,
            'allowReverse': allowReverse,
        }
        if allowReverse:
            self.dxmTaskVehicleIds.add(vehicleId)
        else:
            self.dxmTaskVehicleIds.discard(vehicleId)

    def removeExitedVehicleCaches(self):
        """统一删除已连续缺失并从动静态缓存中移除的车辆全部历史状态。"""
        retainedVehicleIds = (
            set(self.vehicleStaticInfoCaches)
            & set(self.vehicleDynamicInfoCaches)
        )
        vehicleCaches = (
            self.drivableAreaPlannerCaches,
            self.vehicleLaneIDCaches,
            self.vehicleRouteCaches,
            self.vehicleForbiddenLaneIDCaches,
            self.vehicleBasePathCaches,
            self.vehicleBaseWaypointCandidateCaches,
            self.vehicleBasePathLaneIDCaches,
            self.vehicleWaypointWholeCaches,
            self.vehicleWaypointSparseCaches,
            self.vehicleObstacleCaches,
            self.reversePointCaches,
            self.vehicleTaskCaches,
            self.vehicleRouteProgressCaches,
        )
        for cache in vehicleCaches:
            for vehicleId in tuple(cache):
                if vehicleId not in retainedVehicleIds:
                    cache.pop(vehicleId, None)

        self.dxmTaskVehicleIds.intersection_update(retainedVehicleIds)
        self.finalDestinationVehicleIds.intersection_update(retainedVehicleIds)
        priorityNodes = {
            vehicleId: vehicleType
            for vehicleId, vehicleType in self.priorityGraphCache.get(
                'nodes', {}).items()
            if vehicleId in retainedVehicleIds
        }
        self.priorityGraphCache = {
            'nodes': priorityNodes,
            'edges': {
                edge for edge in self.priorityGraphCache.get('edges', ())
                if edge[0] in priorityNodes and edge[1] in priorityNodes
            },
        }

    def shouldReplanAgvBasePath(
            self, vehicleId, dynamicInfo, obstaclePolygons):
        """判断AGV的完整basePath缓存是否需要重新进行空间规划。"""
        basePath = self.vehicleBasePathCaches.get(vehicleId)
        if basePath is not None:
            basePath = clean_xy(basePath)
        noHistory = basePath is None or len(basePath) == 0
        historyFinished = False
        vehicleDeviated = False
        historyBlocked = False
        position = np.asarray(dynamicInfo['position'], dtype=float)[:2]

        if not noHistory:
            if len(basePath) >= 2:
                positionMatch = project_to_polyline(position, basePath)
                basePathProgress = positionMatch['s']
                basePathLength = polyline_length(basePath)
                remainingBasePath = slice_polyline(
                    basePath,
                    basePathProgress,
                    basePathLength,
                )
                historyFinished = (
                        basePathLength - basePathProgress
                        <= AGV_BASE_PATH_REPLAN_WAYPOINT_DISTANCE
                )
                vehicleDeviated = (
                        positionMatch['distance']
                        > AGV_BASE_PATH_DEVIATION_THRESHOLD
                )
            else:
                remainingBasePath = basePath
                remainingDistance = np.linalg.norm(
                    position - basePath[0, :2])
                historyFinished = (
                        remainingDistance
                        <= AGV_BASE_PATH_REPLAN_WAYPOINT_DISTANCE
                )
                vehicleDeviated = (
                        remainingDistance
                        > AGV_BASE_PATH_DEVIATION_THRESHOLD
                )

            if len(remainingBasePath):
                staticInfo = self.vehicleStaticInfoCaches[vehicleId]
                historyBlocked = path_vehicle_intersects(
                    remainingBasePath,
                    obstaclePolygons,
                    staticInfo['length'],
                    staticInfo['width'],
                    dynamicInfo.get('longitudinalUncertainty', 0.0),
                    dynamicInfo.get('lateralUncertainty', 0.0),
                    PATH_LATERAL_SAFETY_MARGIN,
                    PATH_COLLISION_SAMPLE_SPACING,
                )

        return noHistory, historyFinished, vehicleDeviated, historyBlocked

    def updateVehicleObstacleCaches(
            self, requests, effectiveWaypointWholeResults,
            agvExecutableShortPathVehicleIds):
        """在速度规划完成后，按实际采用的waypointWhole维护全部车辆死车状态。"""
        requestMap = {
            str(request.get('deviceId')): request for request in requests
        }
        responseMap = {
            str(request.get('deviceId')): self.responses[index]
            for index, request in enumerate(requests)
        }
        updated = {}

        for vehicleId, dynamicInfo in self.vehicleDynamicInfoCaches.items():
            vehicleId = str(vehicleId)
            old = self.vehicleObstacleCaches.get(vehicleId, {})

            # 没有本帧有效BSM时无法重新判断，缓存随车辆信息缓存暂时保留。
            if (not dynamicInfo.get('valid', False)
                    or int(dynamicInfo.get('_missingFrames', 0)) > 0):
                updated[vehicleId] = {
                    'dead': bool(old.get('dead', False)),
                    'frames': min(
                        int(old.get('frames', 0)),
                        VEHICLE_OBSTACLE_MAX_FRAMES,
                    ),
                    'reason': old.get('reason'),
                }
                continue

            request = requestMap.get(vehicleId)
            response = responseMap.get(vehicleId)
            staticInfo = self.vehicleStaticInfoCaches[vehicleId]
            vehicleType = (
                request.get('deviceType')
                if request is not None
                else staticInfo.get('vehicleType')
            )
            isAgv = vehicleType == 'AGV'
            position = np.asarray(dynamicInfo['position'], dtype=float)[:2]
            reason = None

            # 1. 每帧根据本帧BSM重新判断终点死车。
            if isAgv:
                planner = self.drivableAreaPlannerCaches.get(vehicleId)
                if planner is not None and planner.lastRouteS is not None:
                    routeLength = float(planner.routeStageBounds[-1][1])
                    if (routeLength - float(planner.lastRouteS)
                            <= AGV_DESTINATION_REMAINING_LENGTH_THRESHOLD):
                        reason = '终点'
            else:
                route = self.vehicleRouteCaches.get(vehicleId)
                if route is not None:
                    route = np.asarray(route, dtype=float)
                    if (len(route)
                            and np.linalg.norm(position - route[-1, :2])
                            <= HV_DESTINATION_DISTANCE_THRESHOLD):
                        reason = '终点'

            # 2. 有response时实时判断规划失败；无response时只继承上一帧规划失败。
            if reason is None:
                if response is not None and not response.get('success', False):
                    # 长路径失败但历史Planner仍成功产生可执行短路径时，
                    # response继续保持失败；死车状态则由实际执行情况判断。
                    if not (
                            isAgv
                            and vehicleId
                            in agvExecutableShortPathVehicleIds):
                        failureReason = str(
                            response.get('message')
                            or response.get('code')
                            or '规划失败'
                        )
                        reason = (
                            failureReason
                            if failureReason.startswith('失败：')
                            else f'失败：{failureReason}'
                        )
                elif response is None:
                    oldReason = str(old.get('reason') or '')
                    if oldReason.startswith('失败：'):
                        reason = oldReason

            # 3. 前面均未命中时，仅AGV根据实际采用的速度规划结果判断其他死车。
            if (reason is None and isAgv
                    and float(dynamicInfo.get('speed', 0.0))
                    < VEHICLE_STOP_SPEED_THRESHOLD):
                waypointWhole = effectiveWaypointWholeResults.get(vehicleId)
                shouldMove = (
                    waypointWhole is not None
                    and len(waypointWhole.motionS) > 0
                    and float(waypointWhole.motionS[-1])
                    > VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD
                )
                if shouldMove:
                    reason = '其他'

            if reason is None:
                updated[vehicleId] = {
                    'dead': False,
                    'frames': 0,
                    'reason': None,
                }
                continue

            sameReason = old.get('reason') == reason
            if reason == '其他':
                frames = int(old.get('frames', 0)) + 1 if sameReason else 1
            else:
                frames = (
                    int(old.get('frames', 0)) + 1
                    if sameReason and int(old.get('frames', 0)) >= 3
                    else 3
                )
            frames = min(frames, VEHICLE_OBSTACLE_MAX_FRAMES)
            updated[vehicleId] = {
                'dead': frames >= VEHICLE_OBSTACLE_CONFIRM_FRAMES,
                'frames': frames,
                'reason': reason,
            }

        self.vehicleObstacleCaches = updated

    def batchRoutePlan(
            self, requests: List[Any], *, allVehicleStates,
            allVehicleStaticInfo, centerlineDict, connectorDict, mapVersion,
            junctionLaneIds, obstacleMap, laneSignal_zy, longRouteResult
    ) -> List[Any]:
        self.vehicleIdsNeedWaypointUpdate.clear()
        self.vehicleIdsMustDispatch.clear()
        self.vehicleIdsMustNotDispatch.clear()

        # todo. 一、短路径所有输入，包括车辆动静态信息、地图、障碍物、长路径、信号灯，不能改动
        CenterlineDict = centerlineDict
        ConnectorDict = connectorDict
        result = longRouteResult
        # List<PathPlanningResponseVO> batchRoutePlan最终返回的结果，长路径进行了填充
        self.responses = result.responses
        # Map<String, List<String>> vehicleLaneMap是每辆车的laneId序列；
        vehicleLaneMap = result.vehicleLaneMap
        # Map<String, double[][]> 每辆车的全局参考线
        vehicleTrajectory = result.vehicleTrajectoryMap
        # Map<String, List<String>> 每辆车最终禁行的laneId集合
        vehicleForbiddenLane = result.vehicleForbiddenLaneMap

        # todo. 二、更新车辆缓存
        self.vehicleStaticInfoCaches = update_vehicle_static_cache(allVehicleStaticInfo, self.vehicleStaticInfoCaches)
        self.vehicleDynamicInfoCaches = update_vehicle_dynamic_cache(allVehicleStates, self.vehicleDynamicInfoCaches)
        self.removeExitedVehicleCaches()

        # todo. 三、根据request更新临时信息
        systemPriorities, vehiclePlanningMaxSpeeds = build_request_temporary_info(requests, self.vehicleStaticInfoCaches)

        # todo. 四、准备全部车辆的规划上下文
        for request_index, request in enumerate(requests):
            vehicleId = str(request.get('deviceId'))
            self.prepareVehicle(
                request=request,
                response=self.responses[request_index],
                laneIds=vehicleLaneMap.get(vehicleId),
                route=vehicleTrajectory.get(vehicleId),
                forbiddenLaneIds=vehicleForbiddenLane.get(vehicleId),
                dynamicInfo=self.vehicleDynamicInfoCaches.get(vehicleId, {}),
                staticInfo=self.vehicleStaticInfoCaches.get(vehicleId, {}),
                centerlineDict=CenterlineDict,
                connectorDict=ConnectorDict,
                mapVersion=mapVersion,
                junctionLaneIds=junctionLaneIds,
            )

        # 空间规划前，统一更新本帧全部BSM中已有Planner的AGV位置匹配。
        for rawVehicleId in allVehicleStates:
            vehicleId = str(rawVehicleId)
            planner = self.drivableAreaPlannerCaches.get(vehicleId)
            if planner is not None:
                planner.update_position_match(
                    self.vehicleDynamicInfoCaches[vehicleId],
                    self.reversePointCaches.get(vehicleId),
                )

        # todo. 五、为必要车辆生成basePath，并更新vehicleBasePathCaches。
        # AGV按request规划候选；HV按本帧全部BSM生成固定长度basePath。
        AGV_basePathCandidates = {}  # 仅用于调试，vehicleId -> List[Nx3]
        spatialDrivableLines = {}  # 仅用于调试/可视化
        spatialSearchNodePaths = {}  # 仅用于调试/可视化
        agvExecutableShortPathVehicleIds = set()

        all_obstacle_polygons = build_obstacle_polygons(
            obstacle_map=obstacleMap,
            vehicle_obstacle_caches=self.vehicleObstacleCaches,
            vehicle_dynamic_caches=self.vehicleDynamicInfoCaches,
            vehicle_static_caches=self.vehicleStaticInfoCaches,
        )

        # 只为空间规划请求中的AGV；缓存仍有效时直接复用完整basePath。
        for request_index, request in enumerate(requests):
            if request.get('deviceType') != 'AGV':
                continue
            vehicleId = str(request.get('deviceId'))
            if vehicleId in self.finalDestinationVehicleIds:
                continue
            dynamicInfo = self.vehicleDynamicInfoCaches[vehicleId]
            staticInfo = self.vehicleStaticInfoCaches[vehicleId]
            vehi_obstacle_polygons = exclude_vehi_polygons(all_obstacle_polygons, dynamicInfo)
            oldBasePath = self.vehicleBasePathCaches.get(vehicleId)
            if oldBasePath is not None:
                oldBasePath = clean_xy(oldBasePath).copy()
            replanFlags = self.shouldReplanAgvBasePath(
                vehicleId, dynamicInfo, vehi_obstacle_polygons)
            (noHistory, historyFinished,
             vehicleDeviated, historyBlocked) = replanFlags
            if not any(replanFlags):
                AGV_basePathCandidates[vehicleId] = [self.vehicleBasePathCaches[vehicleId].copy()]
                agvExecutableShortPathVehicleIds.add(vehicleId)
                continue

            agv_spatial_result = plan_agv_spatial(
                dynamic_info=dynamicInfo,
                static_info=staticInfo,
                planning_max_speed=vehiclePlanningMaxSpeeds[vehicleId],
                drivable_area_planner=self.drivableAreaPlannerCaches.get(vehicleId),
                reverse_point=self.reversePointCaches.get(vehicleId),
                polygons=vehi_obstacle_polygons,
                previous_base_path=oldBasePath,
                prefer_previous_path=(
                    historyFinished
                    and not vehicleDeviated
                    and not historyBlocked
                ),
            )
            if not agv_spatial_result.success:
                failureReason = (
                    agv_spatial_result.failureReason
                    or 'SPATIAL_PLANNING_FAILED'
                )
                mark_response_failure(
                    self.responses[request_index],
                    failureReason,
                    f'短路径空间规划失败：{failureReason}',
                )
            else:
                agvExecutableShortPathVehicleIds.add(vehicleId)
            AGV_basePathCandidates[vehicleId] = agv_spatial_result.candidatePaths
            spatialDrivableLines[vehicleId] = agv_spatial_result.drivableLines
            spatialSearchNodePaths[vehicleId] = (
                agv_spatial_result.searchNodePaths
            )
            if agv_spatial_result.candidatePaths:
                selectedBasePath = agv_spatial_result.candidatePaths[0].copy()
                if noHistory:
                    self.vehicleIdsMustNotDispatch.discard(vehicleId)
                    self.vehicleIdsMustDispatch.add(vehicleId)
                elif historyFinished:
                    endpointDistance = np.linalg.norm(
                        selectedBasePath[-1, :2]
                        - oldBasePath[-1, :2]
                    )
                    if (endpointDistance
                            > AGV_BASE_PATH_ENDPOINT_SAME_THRESHOLD):
                        self.vehicleIdsMustNotDispatch.discard(vehicleId)
                        self.vehicleIdsMustDispatch.add(vehicleId)
                    else:
                        self.vehicleIdsMustDispatch.discard(vehicleId)
                        self.vehicleIdsMustNotDispatch.add(vehicleId)
                elif vehicleDeviated or historyBlocked:
                    self.vehicleIdsMustNotDispatch.discard(vehicleId)
                    self.vehicleIdsMustDispatch.add(vehicleId)
                self.vehicleBasePathCaches[vehicleId] = selectedBasePath
                planner = self.drivableAreaPlannerCaches.get(vehicleId)
                candidateCenterlines = (
                    planner.centerline_dict if planner is not None
                    else CenterlineDict
                )
                candidateConnectors = (
                    planner.connector_dict if planner is not None
                    else ConnectorDict
                )
                self.vehicleBasePathLaneIDCaches[vehicleId] = (
                    classify_path_lane_ids(
                        selectedBasePath,
                        candidateCenterlines,
                        candidateConnectors,
                    )
                )
                self.vehicleBaseWaypointCandidateCaches[vehicleId] = (
                    build_base_waypoint_candidates(
                        selectedBasePath,
                        candidateCenterlines,
                        junctionLaneIds,
                        final_destination=(
                            self.vehicleRouteCaches[vehicleId][-1]
                            if vehicleId in self.vehicleRouteCaches
                            and len(self.vehicleRouteCaches[vehicleId])
                            else None
                        ),
                        base_path_lane_ids=(
                            self.vehicleBasePathLaneIDCaches[vehicleId]
                        ),
                        lane_change_boundaries=(
                            agv_spatial_result
                            .candidateLaneChangeBoundaries[0]
                            if agv_spatial_result
                            .candidateLaneChangeBoundaries
                            else ()
                        ),
                    )
                )
            if agv_spatial_result.reversePointReached:
                self.reversePointCaches.pop(vehicleId, None)

        # 本帧BSM中的全部非AGV每帧更新。有历史route缓存时保持横向d并沿纵向s延伸固定空间长度；没有route缓存时按当前速度预测固定时长。
        for rawVehicleId in allVehicleStates:
            vehicleId = str(rawVehicleId)
            staticInfo = self.vehicleStaticInfoCaches[vehicleId]
            if staticInfo.get('vehicleType') == 'AGV':
                continue
            dynamicInfo = self.vehicleDynamicInfoCaches[vehicleId]
            vehi_obstacle_polygons = exclude_vehi_polygons(all_obstacle_polygons, dynamicInfo)
            hv_spatial_result = plan_hv_spatial(
                dynamic_info=dynamicInfo,
                static_info=staticInfo,
                planning_max_speed=vehiclePlanningMaxSpeeds.get(vehicleId, staticInfo['maxSpeed']),
                route=self.vehicleRouteCaches.get(vehicleId),
                polygons=vehi_obstacle_polygons,
                route_progress_cache=self.vehicleRouteProgressCaches.setdefault(vehicleId, {}),
            )
            if hv_spatial_result.candidatePaths:
                selectedBasePath = hv_spatial_result.candidatePaths[0].copy()
                self.vehicleBasePathCaches[vehicleId] = selectedBasePath
                candidateLaneIds = classify_path_lane_ids(
                    selectedBasePath, CenterlineDict, ConnectorDict)
                self.vehicleBaseWaypointCandidateCaches[vehicleId] = (
                    build_base_waypoint_candidates(
                        selectedBasePath,
                        CenterlineDict,
                        junctionLaneIds,
                        final_destination=(
                            self.vehicleRouteCaches[vehicleId][-1]
                            if vehicleId in self.vehicleRouteCaches
                            else None
                        ),
                        base_path_lane_ids=candidateLaneIds,
                    )
                )

        # todo. 六、根据vehicleBasePathCaches生成全部BSM车辆的futurePath
        future_path = {}
        futureWaypointCandidates = {}
        for rawVehicleId in allVehicleStates:
            vehicleId = str(rawVehicleId)
            dynamicInfo = self.vehicleDynamicInfoCaches[vehicleId]
            staticInfo = self.vehicleStaticInfoCaches[vehicleId]
            basePath = self.vehicleBasePathCaches.get(vehicleId)
            if basePath is None:
                position = np.asarray(dynamicInfo['position'], dtype=float)[:2]
                basePath = np.asarray([[position[0], position[1], 0.0]])
                self.vehicleBasePathCaches[vehicleId] = basePath.copy()
                self.vehicleBaseWaypointCandidateCaches[vehicleId] = np.asarray([[
                    position[0], position[1], 0.0, 1.0
                ]])
                self.vehicleBasePathLaneIDCaches.pop(vehicleId, None)

            if staticInfo.get('vehicleType') == 'AGV':
                planner = self.drivableAreaPlannerCaches.get(vehicleId)
                signalCenterlines = (
                    planner.centerline_dict if planner is not None
                    else CenterlineDict
                )
                signalConnectors = (
                    planner.connector_dict if planner is not None
                    else ConnectorDict
                )
                basePathLaneIds = self.vehicleBasePathLaneIDCaches.get(
                    vehicleId)
            else:
                signalCenterlines = CenterlineDict
                signalConnectors = ConnectorDict
                basePathLaneIds = None

            future_path[vehicleId] = build_future_path(
                base_path=basePath,
                current_position=dynamicInfo['position'],
                current_speed=dynamicInfo['speed'],
                max_acceleration=staticInfo['maxAcceleration'],
                max_deceleration=staticInfo['maxDeceleration'],
                centerline_dict=signalCenterlines,
                connector_dict=signalConnectors,
                lane_signals=laneSignal_zy,
                base_path_lane_ids=basePathLaneIds,
            )
            baseWaypointCandidates = self.vehicleBaseWaypointCandidateCaches[vehicleId]
            futureWaypointCandidates[vehicleId] = (
                build_future_waypoint_candidates(
                    base_path=basePath,
                    base_candidates=baseWaypointCandidates,
                    future_path=future_path[vehicleId],
                    current_position=dynamicInfo['position'],
                )
            )

        # todo 八、仅根据futurePath的空间接近关系生成协同连通块，保留单车块。
        cooperationBlocks = group_by_future_paths(
            future_path,
            self.vehicleStaticInfoCaches,
            self.vehicleDynamicInfoCaches,
        )

        # todo.九、基于最终选定path确定物理前后关系。
        # 每一项为(物理前车ID, 物理后车ID)，暂不与其他优先级合并。
        physicalFrontRearEdges = determine_physical_front_rear_edges(
            cooperationBlocks,
            future_path,
            self.vehicleStaticInfoCaches,
            self.vehicleDynamicInfoCaches,
        )

        # todo. 十、生成完整优先级。
        # 每个协同块返回一个PriorityDecision；跨帧缓存只保存最终高->低优先级边。
        priorityDecisions = []
        historicalPriorityEdges = set(
            self.priorityGraphCache.get('edges', ()))
        newPriorityEdges = set()
        dxmVehicleIds = set(self.dxmTaskVehicleIds)

        for cooperationBlock in cooperationBlocks:
            priorityDecision = calculate_block_priority(
                cooperation_block=cooperationBlock,
                selected_paths=future_path,
                physical_priorities=physicalFrontRearEdges,
                system_priorities=systemPriorities,
                dxm_vehicle_ids=dxmVehicleIds,
                vehicle_static_info=self.vehicleStaticInfoCaches,
                vehicle_dynamic_info=self.vehicleDynamicInfoCaches,
                historical_priority_edges=historicalPriorityEdges,
            )
            priorityDecisions.append(priorityDecision)
            newPriorityEdges.update(priorityDecision.priorityEdges)

        priorityNodes = dict(self.priorityGraphCache.get('nodes', {}))
        for vehicleId in future_path:
            priorityNodes[vehicleId] = self.vehicleStaticInfoCaches[
                vehicleId].get('vehicleType')
        currentBsmIds = {str(vehicleId) for vehicleId in allVehicleStates}
        priorityNodes = {
            vehicleId: vehicleType
            for vehicleId, vehicleType in priorityNodes.items()
            if vehicleType == 'AGV' or vehicleId in currentBsmIds
        }
        self.priorityGraphCache = {
            'nodes': priorityNodes,
            'edges': {
                edge for edge in newPriorityEdges
                if edge[0] in priorityNodes and edge[1] in priorityNodes
            },
        }

        self.lastPlanningDiagnostics = {
            'cooperationBlocks': tuple(
                tuple(str(vehicleId) for vehicleId in block)
                for block in cooperationBlocks
            ),
            'priorityDecisions': tuple(priorityDecisions),
            'historicalPriorityEdges': tuple(sorted(historicalPriorityEdges)),
            'currentPriorityEdges': tuple(sorted(newPriorityEdges)),
            'drivableLines': spatialDrivableLines,
            'spatialSearchNodePaths': spatialSearchNodePaths,
            'basePathCandidates': AGV_basePathCandidates,
            'basePaths': dict(self.vehicleBasePathCaches),
            'futurePaths': future_path,
            'futureWaypointCandidates': futureWaypointCandidates,
        }

        # todo. 十一、根据完整优先级进行速度规划。输出本帧全量newWaypointWhole
        newWaypointWholeResults = {}
        for priorityDecision in priorityDecisions:
            blockWaypointWhole = plan_block_waypoint_whole(
                priority_decision=priorityDecision,
                future_paths=future_path,
                waypoint_candidates=futureWaypointCandidates,
                vehicle_static_info=self.vehicleStaticInfoCaches,
                vehicle_dynamic_info=self.vehicleDynamicInfoCaches,
            )
            newWaypointWholeResults.update(blockWaypointWhole)
        self.lastPlanningDiagnostics['waypointWhole'] = {
            vehicleId: result.as_dict()
            for vehicleId, result in newWaypointWholeResults.items()
        }

        # todo. 十二、比较新旧waypointWhole，仅为确实变化的AGV更新waypointSparse。
        waypointSparseResults = {}
        effectiveWaypointWholeResults = {}
        for vehicleId, newWaypointWhole in newWaypointWholeResults.items():
            if self.vehicleStaticInfoCaches[vehicleId].get('vehicleType') != 'AGV':
                waypointSparseResults[vehicleId] = build_hv_waypoint_sparse(
                    newWaypointWhole,
                    self.vehicleDynamicInfoCaches[vehicleId],
                    self.vehicleBasePathCaches[vehicleId],
                )
                effectiveWaypointWholeResults[vehicleId] = newWaypointWhole
                self.vehicleIdsNeedWaypointUpdate.add(vehicleId)
                continue

            oldWaypointWhole = self.vehicleWaypointWholeCaches.get(vehicleId)
            hasHistoricalOutput = (
                oldWaypointWhole is not None
                and vehicleId in self.vehicleWaypointSparseCaches
            )
            if (vehicleId in self.vehicleIdsMustNotDispatch
                    and hasHistoricalOutput):
                alignedOldWaypointWhole = align_old_waypoint_whole(
                    oldWaypointWhole,
                    self.vehicleDynamicInfoCaches[vehicleId]['position'],
                )
                waypointSparseResults[vehicleId] = (
                    self.vehicleWaypointSparseCaches[vehicleId].copy()
                )
                effectiveWaypointWholeResults[vehicleId] = alignedOldWaypointWhole
                continue
            useNewWaypointWhole = (
                    vehicleId in self.vehicleIdsMustDispatch
                    or not hasHistoricalOutput
            )
            if not useNewWaypointWhole:
                alignedOldWaypointWhole = align_old_waypoint_whole(
                    oldWaypointWhole,
                    self.vehicleDynamicInfoCaches[vehicleId]['position'],
                )
                useNewWaypointWhole = should_replace_waypoint_whole(
                    alignedOldWaypointWhole,
                    newWaypointWhole,
                )
            if not useNewWaypointWhole:
                waypointSparseResults[vehicleId] = (
                    self.vehicleWaypointSparseCaches[vehicleId].copy()
                )
                effectiveWaypointWholeResults[vehicleId] = alignedOldWaypointWhole
                continue

            waypointSparse = build_waypoint_sparse(
                waypoint_whole=newWaypointWhole,
                dynamic_info=self.vehicleDynamicInfoCaches[vehicleId],
                base_path=self.vehicleBasePathCaches[vehicleId],
                base_candidates=(
                    self.vehicleBaseWaypointCandidateCaches.get(vehicleId)
                ),
            )
            waypointSparseResults[vehicleId] = waypointSparse
            self.vehicleWaypointWholeCaches[vehicleId] = newWaypointWhole
            self.vehicleWaypointSparseCaches[vehicleId] = waypointSparse.copy()
            self.vehicleIdsNeedWaypointUpdate.add(vehicleId)
            effectiveWaypointWholeResults[vehicleId] = newWaypointWhole

        self.lastPlanningDiagnostics['waypointSparse'] = {
            vehicleId: value.copy()
            for vehicleId, value in waypointSparseResults.items()
        }

        # todo. 七（依赖十一、十二阶段结果，延后执行）：根据实际采用的
        # waypointWhole更新死车，协同主动停车不会被误判。
        self.updateVehicleObstacleCaches(
            requests=requests,
            effectiveWaypointWholeResults=effectiveWaypointWholeResults,
            agvExecutableShortPathVehicleIds=(
                agvExecutableShortPathVehicleIds
            ),
        )

        # todo. 十三、仅为本帧确实需要更新的车辆填充并返回短路径结果。
        allResponses = self.responses
        finalResponses = []
        for requestIndex, request in enumerate(requests):
            vehicleId = str(request.get('deviceId'))
            if vehicleId in self.finalDestinationVehicleIds:
                continue
            if vehicleId not in self.vehicleIdsNeedWaypointUpdate:
                continue
            response = allResponses[requestIndex]
            planner = self.drivableAreaPlannerCaches.get(vehicleId)
            if planner is None:
                responseCenterlines = CenterlineDict
                responseConnectors = ConnectorDict
                responseLaneIds = self.vehicleLaneIDCaches.get(vehicleId, ())
                responseJunctionLaneIds = junctionLaneIds
            else:
                responseCenterlines = planner.centerline_dict
                responseConnectors = planner.connector_dict
                responseLaneIds = planner.feasibleLaneIds
                responseJunctionLaneIds = planner.junctionLaneIds

            responsePath = self.vehicleBasePathCaches[vehicleId]
            shortPathList, mergePathList = build_response_path_lists(
                waypoints=waypointSparseResults[vehicleId],
                planned_path=responsePath,
                centerline_dict=responseCenterlines,
                connector_dict=responseConnectors,
                lane_ids=responseLaneIds,
                junction_lane_ids=responseJunctionLaneIds,
                full_path_list=response.get('fullPathList') or (),
                request=request,
                base_candidates=(
                    self.vehicleBaseWaypointCandidateCaches.get(vehicleId)),
            )
            response['shortPathList'] = shortPathList
            response['mergePathList'] = mergePathList
            response['pathId'] = (
                f"{vehicleId}_{request.get('commandId')}_{time.time_ns()}"
            )
            finalResponses.append(response)
            if any(
                    path.get('isFinalDestination', False)
                    for path in shortPathList):
                self.finalDestinationVehicleIds.add(vehicleId)

        # todo. 十四、把规划失败的车辆ID单列出来，供下一帧触发长路径重规划。
        self.shortPathFailedVehicleIds = [
            str(response.get('deviceId'))
            for response in allResponses
            if not response.get('success', False)
        ]

        self.responses = finalResponses
        return self.responses
