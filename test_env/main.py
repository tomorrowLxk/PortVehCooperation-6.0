"""5.0短路径规划脚本式闭环仿真入口。

直接修改下面的场景路径或JSON内容即可测试。循环中依次完成：
读取本帧BSM -> 调用batchRoutePlan -> 下发response -> 车辆步进 -> 可视化。
"""

import json

from test_env.planningEngineTest import PlanningEngine
from test_env.simulation import (
    SignalController,
    SimulationVisualizer,
    create_vehicles,
    load_scenario,
)


VISUALIZATION = True
VISUALIZE_GRAPH_SEARCH_RESULTS = True
VISUALIZE_ONLY_FIRST_GRAPH_SEARCH_RESULT = True
PRINT_COOPERATION_BLOCKS = True
PRINT_FULL_RESPONSES = False


# 一、读取单一JSON所描述的全部测试输入并初始化长期对象。
scenario = load_scenario(r'G:\项目\港口车辆运动规划\PortVehCooperation 5.0\test_env\demo\scenario1.json')
planningEngine = PlanningEngine(
    scenario.centerlineDict,
    scenario.connectorDict,
    scenario.mapVersion,
)


vehicles = create_vehicles(scenario)
signalController = SignalController(scenario.laneSignals)
visualizer = (
    SimulationVisualizer(
        scenario,
        vehicles,
        visualizeGraphSearchResults=VISUALIZE_GRAPH_SEARCH_RESULTS,
        visualizeFirstGraphSearchResultOnly=(
            VISUALIZE_ONLY_FIRST_GRAPH_SEARCH_RESULT),
    )
    if VISUALIZATION else None
)

planningStepInterval = max(1, int(round(scenario.planningPeriod / scenario.simulationDt)))

maxSteps = scenario.maxSteps
responses = []

print(f'=== {scenario.name} ===')
print(f'scenario={scenario.sourceFile}')
print(f'vehicles={tuple(vehicles)}, dt={scenario.simulationDt:.2f}s, planningPeriod={scenario.planningPeriod:.2f}s')


# 二、显式闭环循环：规划和仿真步进都在main.py中，便于逐帧调试。
for stepIndex in range(maxSteps):
    simulationTime = stepIndex * scenario.simulationDt

    if stepIndex % planningStepInterval == 0:
        # 1. 仿真车辆生成本帧全部BSM；静态信息、地图、障碍物和信控来自JSON。
        allVehicleStates = {
            vehicleId: vehicle.to_bsm()
            for vehicleId, vehicle in vehicles.items()
        }
        allVehicleStaticInfo = scenario.vehicleStaticInfo
        activeRequests = [
            request for request in scenario.requests
            if not vehicles[str(request['deviceId'])].arrived
        ]
        longRouteResult = scenario.build_long_route_result(activeRequests)

        # 2. 调用一次5.0测试版batchRoutePlan，所有上游数据显式可见。
        responses = planningEngine.batchRoutePlan(
            activeRequests,
            allVehicleStates=allVehicleStates,
            allVehicleStaticInfo=allVehicleStaticInfo,
            centerlineDict=scenario.centerlineDict,
            connectorDict=scenario.connectorDict,
            mapVersion=scenario.mapVersion,
            junctionLaneIds=scenario.junctionLaneIds,
            obstacleMap=scenario.obstacleMap,
            laneSignal_zy=signalController.states,
            longRouteResult=longRouteResult,
        )

        # 3. 输出本帧规划摘要，并将response直接下发给对应仿真车辆。
        print(f'[planning t={simulationTime:.1f}s]')
        for response in responses:
            vehicleId = str(response['deviceId'])
            print(
                f'  {vehicleId}: success={response.get("success")}, '
                f'code={response.get("code")}, '
                f'shortPath={len(response.get("shortPathList") or ())}, '
                f'mergePath={len(response.get("mergePathList") or ())}'
            )
            vehicles[vehicleId].apply_response(
                response,
                planningEngine.lastPlanningDiagnostics[
                    'futurePaths'][vehicleId],
            )
        if PRINT_COOPERATION_BLOCKS:
            print('  cooperationBlocks (high -> low priority):')
            diagnostics = planningEngine.lastPlanningDiagnostics
            blocks = diagnostics.get('cooperationBlocks', ())
            decisions = diagnostics.get('priorityDecisions', ())
            for blockIndex, block in enumerate(blocks):
                orderedBlock = (
                    decisions[blockIndex].priorityOrder
                    if blockIndex < len(decisions)
                    else block
                )
                print(f'    block{blockIndex + 1}: {list(orderedBlock)}')
        if PRINT_FULL_RESPONSES:
            printableResponses = [
                {**response, 'fullPathList': []}
                for response in responses
            ]
            print(json.dumps(
                printableResponses, ensure_ascii=False, indent=2))

    # 4. 仿真环境执行一个小步；信号灯倒计时同步推进。
    for vehicle in vehicles.values():
        vehicle.step()
    signalController.step(scenario.simulationDt)

    # 5. 显示地图、长路径、局部可行域、候选path、futurePath和waypoint。
    if visualizer is not None:
        visualizer.update(
            simulationTime + scenario.simulationDt,
            planningEngine,
            responses,
            signalController.states,
        )
        if visualizer.stopRequested:
            print('ESC pressed, simulation stopped.')
            break

    if all(vehicle.reachedEndpoint for vehicle in vehicles.values()):
        print(f'all vehicles arrived at t='
              f'{simulationTime + scenario.simulationDt:.1f}s')
        break


print('finalStates=' + str({
    vehicleId: {
        'x': round(vehicle.x, 3),
        'y': round(vehicle.y, 3),
        'speed': round(vehicle.speed, 3),
        'arrived': vehicle.arrived,
        'reachedEndpoint': vehicle.reachedEndpoint,
    }
    for vehicleId, vehicle in vehicles.items()
}))
if visualizer is not None:
    visualizer.close()
