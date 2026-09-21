import pickle
from lib2to3.fixes.fix_input import context

import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from shortCache.vehInfo import update_vehicle_dynamic_cache, update_vehicle_static_cache
from shortCache.vehObstacle import remaining_path_length
from shortCache.requestInfo import build_request_temporary_info

from parameters import (
    AGV_DESTINATION_REMAINING_LENGTH_THRESHOLD,
    HV_DESTINATION_DISTANCE_THRESHOLD,
    VEHICLE_OBSTACLE_CONFIRM_FRAMES,
    VEHICLE_OBSTACLE_MAX_FRAMES,
    VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD,
    VEHICLE_STOP_SPEED_THRESHOLD,
)

from localPlanner.cooperationPlanner import (
    determine_front_rear_priorities, group_by_future_paths, group_by_paths,
    select_default_paths)
from localPlanner.drivableAreaPlanner import DrivableAreaPlanner
from localPlanner.dxmPlanner import fixed_dxm
from localPlanner.jointDecisionPlanner import (
    calculate_block_priority, plan_block_waypoints)

from localPlanner.spatialPlanningStage import (
    build_obstacle_polygons,
    exclude_vehi_polygons,
    mark_response_failure,
    plan_agv_spatial,
    plan_hv_spatial,
)

from tools.obstacle_processing import generateStaticObstacles
from tools.response_processing import complete_planning_responses
from tools.planning_failure import (
    mark_vehicle_failure, failure_waypoints, fail_short_path_batch)


class PlanningEngine:
    def __init__(self) -> None:
        # 占位的，模拟真实系统的地图数据
        projectRoot = Path(__file__).resolve().parent
        staticMapDir = projectRoot / "staticMap"
        centerlineFile = staticMapDir / "centerline_dict0827.pkl"
        connectorFile = staticMapDir / "connector_dict0827.pkl"
        with centerlineFile.open("rb") as file:
            self.centerline_dict = pickle.load(file)
        with connectorFile.open("rb") as file:
            self.connector_dict = pickle.load(file)

        # 最终输出
        self.responses = None

        # 必要的，储存地图版本
        self.map_version = "0827"

        # 仅AGV需要DrivableAreaPlanner；Planner内部只持有该任务涉及的局部小地图。
        self.drivableAreaPlannerCaches: Dict[str, Any] = {}

        # 所有车辆修复前的原始laneId序列、全局参考线和禁行车道集合。
        self.vehicleLaneIDCaches: Dict[str, List[str]] = {}
        self.vehicleRouteCaches: Dict[str, Any] = {}
        self.vehicleForbiddenLaneIDCaches: Dict[str, frozenset[str]] = {}

        # AGV上一轮最终采用的完整path和最终发布的waypoint；HV每帧重新预测，不复用。
        self.vehiclePathCaches: Dict[str, Any] = {}
        self.vehicleWaypointCaches: Dict[str, Any] = {}

        # AV/HV统一内部静态参数：长宽、限速、最小转弯半径等。
        self.vehicleStaticInfoCaches: Dict[str, Any] = {}
        # AV/HV统一内部动态信息：坐标、朝向、速度、纵横向定位不确定性
        self.vehicleDynamicInfoCaches: Dict[str, Any] = {}
        # 每辆车是否应作为静态障碍物处理，以及“应走但未走”的连续帧数。
        self.vehicleObstacleCaches: Dict[str, Any] = {}

        # reversePointCaches以车辆ID为key，只保存当前调箱门任务的倒车切换点xy。
        self.reversePointCaches: Dict[str, Any] = {}
        # AV/HV任务身份与R点是否已通过无关；只有新commandId才开始新任务。
        self.vehicleTaskCaches: Dict[str, Dict[str, Any]] = {}
        # HV只沿route定位，单独维护弧长进度，不实例化可行域规划器。
        self.vehicleRouteProgressCaches: Dict[str, Dict[str, Any]] = {}

        # 全局优先级有向图与协同分块无关。edges只保留上一帧仍有冲突的
        # 高->低关系；AGV节点长期保留，消失于BSM的HV节点在每帧末删除。
        self.priorityGraphCache: Dict[str, Any] = {
            "nodes": {},
            "edges": set(),
        }

    def prepareVehicle(self, request, response, laneIds, route, forbiddenLaneIds,
                       dynamicInfo, staticInfo, centerlineDict, connectorDict, mapVersion):
        """根据本帧长路径结果，原地准备或更新单车规划上下文。"""
        vehicleId = str(request.get('deviceId'))
        vehicleType = str(request.get('deviceType')).upper()
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
            self.vehiclePathCaches.pop(vehicleId, None)
            self.vehicleWaypointCaches.pop(vehicleId, None)
            self.vehicleRouteProgressCaches.pop(vehicleId, None)
            self.vehicleTaskCaches.pop(vehicleId, None)
            return

        sourceLaneIds = [str(value) for value in laneIds if value is not None]
        sourceRoute = np.asarray(route, dtype=float)[:, :2]
        forbiddenLaneIds = frozenset(str(value) for value in forbiddenLaneIds if value is not None)

        # 非AGV只维护修复前长路径；任务变化时旧短路径不能继续参与停车判断。
        if not isAgv:
            if taskChanged:
                self.vehiclePathCaches.pop(vehicleId, None)
                self.vehicleWaypointCaches.pop(vehicleId, None)
                self.vehicleRouteProgressCaches.pop(vehicleId, None)

            self.vehicleTaskCaches[vehicleId] = {
                'commandId': commandId,
                'allowReverse': False,
            }
            self.vehicleLaneIDCaches[vehicleId] = list(sourceLaneIds)
            self.vehicleRouteCaches[vehicleId] = sourceRoute.copy()
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
                    or len(cachedRoute) != len(sourceRoute)
            )
            contextChanged = (
                    mapChanged
                    or forbiddenChanged
                    or laneIdsChanged
                    or routeChanged
            )

        if not taskChanged and planner is not None and not contextChanged:
            return

        # 任务/地图/长路径变化后，所有依赖旧任务的规划结果立即失效。
        if taskChanged:
            self.reversePointCaches.pop(vehicleId, None)
        self.vehiclePathCaches.pop(vehicleId, None)
        self.vehicleWaypointCaches.pop(vehicleId, None)
        self.vehicleRouteProgressCaches.pop(vehicleId, None)

        # 调箱门修复属于局部规划上下文构建：新任务必修复，同任务仅在R点尚存时修复。
        effectiveLaneIds = list(sourceLaneIds)
        effectiveRoute = sourceRoute.copy()
        localConnector = connectorDict
        localCenterlines = centerlineDict

        applyDxm = allowReverse and (taskChanged or reversePoint is not None)
        nextReversePoint = None
        if applyDxm:
            (effectiveLaneIds, effectiveRoute, localConnector, localCenterlines, nextReversePoint) = fixed_dxm(
                list(sourceLaneIds),
                sourceRoute.copy(),
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

        self.vehicleLaneIDCaches[vehicleId] = list(sourceLaneIds)
        self.vehicleRouteCaches[vehicleId] = sourceRoute.copy()
        self.vehicleForbiddenLaneIDCaches[vehicleId] = forbiddenLaneIds
        self.vehicleTaskCaches[vehicleId] = {
            'commandId': commandId,
            'allowReverse': allowReverse,
        }

    def updateVehicleObstacleCaches(
            self, requests, responses, longRouteSuccess,
            plannedPathResults):
        '''在整帧规划结束后统一判断、维持或恢复全部车辆的死车状态。'''
        requestMap = {
            str(request.get('deviceId')): request for request in requests
        }
        responseMap = {
            str(request.get('deviceId')): responses[index]
            for index, request in enumerate(requests)
        }
        updated = {}

        for vehicleId, dynamicInfo in self.vehicleDynamicInfoCaches.items():
            vehicleId = str(vehicleId)
            old = self.vehicleObstacleCaches.get(vehicleId, {})
            request = requestMap.get(vehicleId)

            bsmIsCurrent = (
                    bool(dynamicInfo.get('valid', False))
                    and int(dynamicInfo.get('_missingFrames', 0)) == 0
            )
            if request is None or not bsmIsCurrent:
                updated[vehicleId] = {
                    'dead': bool(old.get('dead', False)),
                    'frames': min(
                        int(old.get('frames', 0)),
                        VEHICLE_OBSTACLE_MAX_FRAMES,
                    ),
                    'reason': old.get('reason'),
                }
                continue

            vehicleType = str(request.get('deviceType')).upper()
            position = np.asarray(dynamicInfo.get('position'), dtype=float)[:2]
            planner = self.drivableAreaPlannerCaches.get(vehicleId)
            reason = None

            # 1. 终点死车。
            if vehicleType == 'AGV':
                if planner is not None and planner.lastRouteS is not None:
                    route = np.asarray(planner.referLine, dtype=float)[:, :2]
                    routeLength = float(
                        np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
                    )
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

            # 2-5. 失败类型统一读取response，不在死车模块中重复推断。
            response = responseMap[vehicleId]
            planningFailed = (
                not response.get('success', False)
                if vehicleType == 'AGV'
                else not longRouteSuccess.get(vehicleId, False)
            )
            if reason is None and planningFailed:
                failureReason = (
                        response.get('message')
                        or response.get('code')
                        or '规划失败'
                )
                failureReason = str(failureReason)
                reason = (
                    failureReason
                    if failureReason.startswith('失败：')
                    else f'失败：{failureReason}'
                )

            # 6. 有可执行短路径但车辆没有执行。
            if reason is None:
                path = plannedPathResults.get(vehicleId)
                if (
                        float(dynamicInfo.get('speed', 0.0))
                        < VEHICLE_STOP_SPEED_THRESHOLD
                        and remaining_path_length(path, position)
                        > VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD
                ):
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

    def batchRoutePlan(self, requests: List[Any]) -> List[Any]:
        # todo. 一、短路径所有输入，包括车辆动静态信息、地图、障碍物、长路径、信号灯，不能改动
        # Map<String, DeviceBsmBodyVO>，所有车辆的状态信息，包括内外车辆，静止车辆
        allVehicleStates = vehicleDynamicLocationService.getAllRealTimeBsm(None)
        # Map<String, VehicleStaticVO>，查询所有车辆的静态信息，包括长宽和运动学信息，无视动力学信息，此外本地需要有一个默认值
        allVehicleStaticInfo = vehicleStaticInfoService.getVehicleStaticInfo(None)

        # Map<String, double[][]>，地图信息，每条车道的中心线坐标
        CenterlineDict = context.getCenterlineDict()
        # Map<String, ConnectorInfo>, 地图信息，每条车道的左右前后连接关系
        ConnectorDict = context.getConnectorDict()
        # String，地图信息，版本名称
        mapVersion = context.getMapVersion()

        # Map<String, ObstacleInfo>, 所有的障碍物，ObstacleInfo的格式比较复杂，我们主要是用polygon这个字段的xy属性，还有status判断是否激活
        obstacleMap = {}
        # <String, <String, String>>, 信号灯数据，包括进口道id，方向和红绿灯状态 {"rightTurn":"RED", "leftTurn":"GREEN", "UTurn":"RED", "straight":"RED"}
        laneSignal_zy = {}

        # 模拟长路径计算
        result = self.planLongRoutes(allVehicleStates, allVehicleStaticInfo, requests)  # 暂时模拟长路径规划
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

        # todo. 三、根据request更新临时信息
        systemPriorities, vehiclePlanningMaxSpeeds = build_request_temporary_info(requests,
                                                                                  self.vehicleStaticInfoCaches)

        # todo. 四、准备全部车辆的规划上下文
        longRouteSuccess = {
            str(request.get('deviceId')): bool(self.responses[index].get('success', False))
            for index, request in enumerate(requests)
        }

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
            )

        # todo. 五、生成所有车辆的futurePath和空间path候选。
        # futurePath为Nx2；HV path和AGV候选path均为Nx3[x,y,vMax]。
        # 此处显式逐request循环；单车函数不遍历车辆，并原地追加本阶段response信息。
        """HV不是无视障碍物，而是“只截停、不绕行。"""

        HV_local_path = {}  # HV唯一预测3s traj，vehicleId -> Nx3[x,y,vMax]
        AGV_all_local_path = {}  # AGV最多5条候选，vehicleId -> List[Nx3]
        future_path = {}  # 所有车辆前方约80m的参考path，vehicleId -> Nx2
        spatialDrivableLines = {}  # AGV本帧局部可行域线, 目前仅用于诊断/可视化

        all_obstacle_polygons = build_obstacle_polygons(
            obstacle_map=obstacleMap,
            vehicle_obstacle_caches=self.vehicleObstacleCaches,
            vehicle_dynamic_caches=self.vehicleDynamicInfoCaches,
            vehicle_static_caches=self.vehicleStaticInfoCaches,
        )

        for request_index, request in enumerate(requests):
            vehicleId = str(request.get('deviceId'))
            vehicleType = str(request.get('deviceType')).upper()
            dynamicInfo = self.vehicleDynamicInfoCaches[vehicleId]
            staticInfo = self.vehicleStaticInfoCaches[vehicleId]

            # 排除掉包含本车的障碍物
            vehi_obstacle_polygons = exclude_vehi_polygons(all_obstacle_polygons, dynamicInfo)

            if vehicleType == 'AGV':
                result = plan_agv_spatial(
                    response=self.responses[request_index],
                    dynamic_info=dynamicInfo,
                    static_info=staticInfo,
                    planning_max_speed=vehiclePlanningMaxSpeeds[vehicleId],
                    drivable_area_planner=self.drivableAreaPlannerCaches.get(vehicleId),
                    reverse_point=self.reversePointCaches.get(vehicleId),
                    polygons=vehi_obstacle_polygons,
                    lane_signals=laneSignal_zy,
                )
            else:
                result = plan_hv_spatial(
                    response=self.responses[request_index],
                    dynamic_info=dynamicInfo,
                    static_info=staticInfo,
                    planning_max_speed=vehiclePlanningMaxSpeeds[vehicleId],
                    route=self.vehicleRouteCaches.get(vehicleId),
                    polygons=vehi_obstacle_polygons,
                    route_progress_cache=self.vehicleRouteProgressCaches.setdefault(vehicleId, {}),
                    route_lane_ids=self.vehicleLaneIDCaches.get(vehicleId),
                    lane_signals=laneSignal_zy,
                    centerline_dict=CenterlineDict,
                    connector_dict=ConnectorDict,
                )

            future_path[vehicleId] = result.futurePath

            if vehicleType == 'AGV':
                spatialDrivableLines[vehicleId] = result.drivableLines
                if result.candidatePaths:
                    AGV_all_local_path[vehicleId] = result.candidatePaths
                if result.reversePointReached:
                    self.reversePointCaches.pop(vehicleId, None)
            elif result.candidatePaths:
                HV_local_path[vehicleId] = result.candidatePaths[0]

        # 六、仅根据futurePath的空间接近关系生成协同连通块，保留单车块。
        cooperationBlocks = group_by_future_paths(
            future_path,
            self.vehicleStaticInfoCaches,
            self.vehicleDynamicInfoCaches,
        )

        # 七、path组合选优。当前占位策略：AGV取第一个候选，HV取唯一预测path。
        # 输出只保留{车辆ID: Nx3[x,y,vMax]}。
        selectedPaths = select_default_paths(
            cooperationBlocks,
            AGV_all_local_path,
            HV_local_path,
            self.vehicleDynamicInfoCaches,
        )

        # 根据全部选定path重新分组，可拆开旧块，也可合并换道后新增冲突的块。
        # 不重新选择path；后续前后关系和联合决策均使用这次分组。
        cooperationBlocks = group_by_paths(
            selectedPaths,
            self.vehicleStaticInfoCaches,
            self.vehicleDynamicInfoCaches,
        )

        # 八、基于最终选定path确定物理前后关系。
        # 每一项为(高优先级/前车ID, 低优先级/后车ID)。
        sequencePriorities = determine_front_rear_priorities(
            cooperationBlocks,
            selectedPaths,
            self.vehicleStaticInfoCaches,
            self.vehicleDynamicInfoCaches,
        )

        # 九、逐协同块完成联合决策与waypoint输出
        waypointResults = {}
        plannedPathResults = {}
        historicalPriorityEdges = set(self.priorityGraphCache.get("edges", ()))
        newPriorityEdges = set()
        framePriorityDiagnostics = []
        self.lastPlanningDiagnostics = {
            "cooperationBlocks": tuple(
                tuple(str(vehicleId) for vehicleId in block)
                for block in cooperationBlocks),
            "historicalPriorityEdges": tuple(sorted(historicalPriorityEdges)),
            "priorityBlocks": framePriorityDiagnostics,
            # 仅附加诊断输出，不参与后续分组、优先级、速度或waypoint计算。
            "drivableLines": spatialDrivableLines,
        }
        # 所有request车辆（包括HV）都参与同一套优先级速度推断：高优先级车辆
        # 已经调整后的速度曲线会继续约束低优先级车辆。HV是否遵从输出不属于规划器职责。
        speedPlanningVehicleIds = set(selectedPaths)
        # 历史waypoint只复用到可控AGV；HV每帧根据最新BSM重新推断，避免旧预测滞留。
        waypointCacheVehicleIds = set(AGV_all_local_path)
        request_indices = {
            str(request.get('deviceId')): index
            for index, request in enumerate(requests)
        }
        for cooperationBlock in cooperationBlocks:
            try:
                priorityResult = calculate_block_priority(
                    cooperation_block=cooperationBlock,
                    selected_paths=selectedPaths,
                    sequence_priorities=sequencePriorities,
                    system_priorities=systemPriorities,
                    vehicle_static_info=self.vehicleStaticInfoCaches,
                    vehicle_dynamic_info=self.vehicleDynamicInfoCaches,
                    historical_priority_edges=historicalPriorityEdges,
                )
                framePriorityDiagnostics.append({
                    "members": tuple(priorityResult.members),
                    "conflicts": tuple({
                                           "vehicles": (item.vehicleA, item.vehicleB),
                                           "rawS": (float(item.rawSA), float(item.rawSB)),
                                           "anchorS": (float(item.anchorSA), float(item.anchorSB)),
                                       } for item in priorityResult.conflicts),
                    **priorityResult.searchDiagnostics,
                })
                blockWaypointResult = plan_block_waypoints(
                    priority_result=priorityResult,
                    selected_paths=selectedPaths,
                    vehicle_static_info=self.vehicleStaticInfoCaches,
                    vehicle_dynamic_info=self.vehicleDynamicInfoCaches,
                    controllable_vehicle_ids=speedPlanningVehicleIds,
                    previous_waypoints={
                        vehicleId: self.vehicleWaypointCaches[vehicleId]
                        for vehicleId in cooperationBlock
                        if vehicleId in waypointCacheVehicleIds
                           and vehicleId in self.vehicleWaypointCaches
                    },
                )
                waypointResults.update(blockWaypointResult.waypointResults)
                plannedPathResults.update(blockWaypointResult.plannedPaths)
                newPriorityEdges.update(priorityResult.priorityEdges)
                for vehicleId, path in blockWaypointResult.plannedPaths.items():
                    if vehicleId in waypointCacheVehicleIds:
                        self.vehiclePathCaches[vehicleId] = path
                for vehicleId, waypoints in blockWaypointResult.waypointResults.items():
                    if vehicleId in waypointCacheVehicleIds:
                        self.vehicleWaypointCaches[vehicleId] = waypoints
            except Exception as error:
                # 优先级/速度是块内耦合计算，异常时整块停车，其他块继续。
                for vehicleId in cooperationBlock:
                    index = request_indices[vehicleId]
                    mark_vehicle_failure(self.responses, index, requests[index],
                                         f"协同块规划失败: {error}",
                                         code="COOPERATION_BLOCK_FAILED")
                    stop = failure_waypoints(self.vehicleDynamicInfoCaches.get(vehicleId))
                    if stop is not None:
                        waypointResults[vehicleId] = stop
                        plannedPathResults[vehicleId] = stop[:2, [0, 1, 3]].copy()
                    self.vehicleWaypointCaches.pop(vehicleId, None)
                    self.vehiclePathCaches.pop(vehicleId, None)
                newPriorityEdges = {edge for edge in newPriorityEdges
                                    if not set(edge).intersection(cooperationBlock)}

        priorityNodes = dict(self.priorityGraphCache.get("nodes", {}))
        for request in requests:
            priorityNodes[str(request.get('deviceId'))] = str(request.get('deviceType'))
        currentBsmIds = {str(vehicleId) for vehicleId in allVehicleStates}
        priorityNodes = {
            vehicleId: vehicleType
            for vehicleId, vehicleType in priorityNodes.items()
            if vehicleType == "AGV" or vehicleId in currentBsmIds
        }
        self.priorityGraphCache = {
            "nodes": priorityNodes,
            "edges": {
                edge for edge in newPriorityEdges
                if edge[0] in priorityNodes and edge[1] in priorityNodes
            },
        }

        self.updateVehicleObstacleCaches(
            requests=requests,
            responses=self.responses,
            longRouteSuccess=longRouteSuccess,
            plannedPathResults=plannedPathResults,
        )

        # 十、直接返回与requests同下标的PathPlanningResponseVO列表，补全
        # success、失败原因和三类路径字段，不再增加finalResult包装对象。
        return complete_planning_responses(
            requests=requests,
            long_route_responses=self.responses,
            waypoint_results=waypointResults,
            planned_paths=plannedPathResults,
            drivable_area_planners=self.drivableAreaPlannerCaches,
            vehicle_dynamic_info=self.vehicleDynamicInfoCaches,
        )
