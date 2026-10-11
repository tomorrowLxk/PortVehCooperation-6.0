"""5.0脚本式闭环测试所需的场景读取、车辆步进和可视化薄工具。"""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths,
    cartesian_heading_to_navigation,
    clean_xy,
    heading_at_s,
    point_at_s,
    project_points_to_polyline,
    project_to_polyline,
)


@dataclass
class Scenario:
    sourceFile: Path
    name: str
    centerlineDict: dict
    connectorDict: dict
    mapVersion: str
    junctionLaneIds: tuple
    requests: list
    routes: dict
    vehicleStaticInfo: dict
    obstacleMap: dict
    laneSignals: dict
    simulationDt: float
    planningPeriod: float
    maxSteps: int
    viewMargin: float

    def build_long_route_result(self, requests=None):
        """按request顺序生成测试用固定长路径结果，每帧返回全新response。"""
        responses = []
        vehicle_lane_map = {}
        trajectory_map = {}
        forbidden_lane_map = {}

        for request in (self.requests if requests is None else requests):
            vehicle_id = str(request['deviceId'])
            route_data = self.routes[vehicle_id]
            lane_ids = list(route_data['laneIds'])
            route = route_data['points']
            route_length = float(np.sum(
                np.linalg.norm(np.diff(route, axis=0), axis=1)))
            full_path = {
                'isFinalDestination': True,
                'laneId': lane_ids[0],
                'points': [
                    _position(point, _path_heading(route, index))
                    for index, point in enumerate(route)
                ],
                'speedLimit': float(request['speedLimit']) / 3.6,
                'action': 'STRAIGHT',
                'direction': 'STRAIGHT',
                'length': route_length,
                'junctionId': None,
            }
            responses.append({
                'deviceId': vehicle_id,
                'success': True,
                'code': '0',
                'message': None,
                'pathId': f'fixed-route-{vehicle_id}',
                'fullPathList': [full_path],
                'shortPathList': [],
                'mergePathList': [],
                'pathPlanningRequest': deepcopy(request),
            })
            vehicle_lane_map[vehicle_id] = lane_ids
            trajectory_map[vehicle_id] = route.copy()
            forbidden_lane_map[vehicle_id] = []

        return SimpleNamespace(
            responses=responses,
            vehicleLaneMap=vehicle_lane_map,
            vehicleTrajectoryMap=trajectory_map,
            vehicleForbiddenLaneMap=forbidden_lane_map,
        )


def load_scenario(path):
    """从一份JSON加载地图引用、固定长路径、车辆、信控和障碍物。"""
    source = Path(path).resolve()
    with source.open('r', encoding='utf-8') as stream:
        raw = json.load(stream)

    map_data = raw['map']
    centerline_dict = _load_pickle(
        source.parent / map_data['centerlineFile'])
    connector_dict = _load_pickle(
        source.parent / map_data['connectorFile'])
    centerline_dict = {
        str(lane_id): np.asarray(points, dtype=float)[:, :2]
        for lane_id, points in centerline_dict.items()
    }
    connector_dict = {
        str(lane_id): value
        for lane_id, value in connector_dict.items()
    }

    routes = {}
    requests = []
    static_info = {}
    for index, vehicle in enumerate(raw['vehicles']):
        vehicle_id = str(vehicle['vehicleId'])
        vehicle_type = vehicle.get('deviceType', 'AGV')
        lane_ids = [str(value) for value in vehicle['route']['laneIds']]
        route = _route_points(
            vehicle['route'], lane_ids, centerline_dict, source.parent)
        routes[vehicle_id] = {
            'laneIds': lane_ids,
            'points': route,
        }

        initial = vehicle.get('initialState') or {}
        initial_position = np.asarray([
            initial.get('x', route[0, 0]),
            initial.get('y', route[0, 1]),
        ], dtype=float)
        initial_heading = float(initial.get(
            'headingRad', _path_heading(route, 0)))
        request_data = vehicle.get('request') or {}
        requests.append({
            'deviceId': vehicle_id,
            'deviceType': vehicle_type,
            'startPosition': {
                'position': _position(initial_position, initial_heading),
                'laneId': lane_ids[0],
            },
            'endPosition': {
                'position': _position(
                    route[-1], _path_heading(route, len(route) - 2)),
                'laneId': lane_ids[-1],
            },
            'forbidLaneIdList': [
                str(value)
                for value in request_data.get('forbidLaneIdList', ())
            ],
            'laneWhiteList': [
                str(value)
                for value in request_data.get('laneWhiteList', ())
            ],
            'priority': int(request_data.get('priority', 1)),
            'mapVersion': str(map_data['version']),
            'commandId': str(request_data.get(
                'commandId', f'demo-command-{index + 1}')),
            'speedLimit': float(
                request_data.get('speedLimitMps', 4.0)) * 3.6,
            'allowReverse': bool(request_data.get('allowReverse', False)),
            'requestId': str(request_data.get(
                'requestId', f'demo-request-{index + 1}')),
        })

        static = vehicle.get('staticInfo') or {}
        static_info[vehicle_id] = {
            'vehicleType': vehicle_type,
            'boundingBox': {
                'length': float(static.get('lengthMeters', 14.8)) * 100.0,
                'width': float(static.get('widthMeters', 2.8)) * 100.0,
            },
            'kinematics': {
                'maxSpeed': float(static.get('maxSpeedMps', 9.77)),
                'maxAcceleration': float(
                    static.get('maxAccelerationMps2', 2.0)),
                'maxDeceleration': abs(float(
                    static.get('maxDecelerationMps2', -4.0))),
                'minTurningRadius': float(
                    static.get('minTurningRadiusMeters', 10.0)),
            },
            '_initialPosition': initial_position,
            '_initialHeading': initial_heading,
            '_initialSpeed': max(
                0.0, float(initial.get('speedMps', 0.0))),
        }

    environment = raw.get('environment') or {}
    obstacle_map = _id_map(environment.get('obstacles') or (), 'obstacle')
    simulation = raw.get('simulation') or {}
    return Scenario(
        sourceFile=source,
        name=str(raw.get('name', source.stem)),
        centerlineDict=centerline_dict,
        connectorDict=connector_dict,
        mapVersion=str(map_data['version']),
        junctionLaneIds=tuple(
            str(value) for value in map_data.get('junctionLaneIds', ())),
        requests=requests,
        routes=routes,
        vehicleStaticInfo=static_info,
        obstacleMap=obstacle_map,
        laneSignals=deepcopy(environment.get('laneSignals') or {}),
        simulationDt=float(simulation.get('simulationDtSeconds', 0.2)),
        planningPeriod=float(
            simulation.get('planningPeriodSeconds', 1.0)),
        maxSteps=int(simulation.get('maxSteps', 600)),
        viewMargin=(
            90.0 if simulation.get('viewMarginMeters') is None
            else float(simulation['viewMarginMeters'])
        ),
    )


class VehicleSimulator:
    """Execute response waypoints on a route-based Frenet path."""

    def __init__(self, vehicle_id, route, static_info, simulation_dt):
        self.vehicleId = str(vehicle_id)
        self.route = np.asarray(route, dtype=float)[:, :2]
        self.length = float(static_info['boundingBox']['length']) / 100.0
        self.width = float(static_info['boundingBox']['width']) / 100.0
        self.dt = float(simulation_dt)
        initial = np.asarray(static_info['_initialPosition'], dtype=float)
        self.x = float(initial[0])
        self.y = float(initial[1])
        self.heading = float(static_info['_initialHeading'])
        self.speed = float(static_info['_initialSpeed'])
        self.arrived = False
        self.reachedEndpoint = False
        self.executionPath = np.asarray([[self.x, self.y]], dtype=float)
        self.executionArc = np.asarray([0.0], dtype=float)
        self.speedEndS = np.empty(0, dtype=float)
        self.segmentSpeeds = np.empty(0, dtype=float)
        self.displayWaypoints = np.empty((0, 3), dtype=float)
        self.history = [np.asarray([self.x, self.y], dtype=float)]

    def to_bsm(self):
        return {
            'position': {
                'longitude': float(self.x),
                'latitude': float(self.y),
                'elevation': 0.0,
            },
            'posAccuracy': 0,
            'speed': int(np.clip(
                np.rint(max(0.0, self.speed) / 0.02), 0, 8190)),
            'heading': cartesian_heading_to_navigation(self.heading),
            'direction': 1,
        }

    def apply_response(self, response, future_path):
        waypointPoints = []
        intervalSpeeds = []

        for segment in response.get('shortPathList') or ():
            speed = max(0.0, float(segment.get('speedLimit') or 0.0))
            for rawPoint in segment.get('points') or ():
                point = np.asarray(
                    [rawPoint['x'], rawPoint['y']], dtype=float)
                if not waypointPoints:
                    waypointPoints.append(point)
                    continue
                if np.linalg.norm(point - waypointPoints[-1]) <= 1e-8:
                    continue
                waypointPoints.append(point)
                intervalSpeeds.append(speed)

        if not waypointPoints:
            self.displayWaypoints = np.empty((0, 3), dtype=float)
            self._stop_at_current_position()
            return

        waypointPoints = np.asarray(waypointPoints, dtype=float)
        if intervalSpeeds:
            displaySpeeds = np.r_[
                intervalSpeeds[0], np.asarray(intervalSpeeds, dtype=float)]
        else:
            displaySpeeds = np.asarray([0.0])
        self.displayWaypoints = np.column_stack(
            (waypointPoints, displaySpeeds))

        if len(waypointPoints) == 1:
            executionPath = clean_xy(future_path)
            if len(executionPath) < 2:
                self._stop_at_current_position()
                return
            targetProgress = float(project_points_to_polyline(
                waypointPoints, executionPath)['s'][0])
            targetOffset = waypointPoints[0] - np.asarray(
                [self.x, self.y], dtype=float)
            headingVector = np.asarray([
                math.cos(self.heading), math.sin(self.heading)
            ])
            if (targetProgress <= 1e-6
                    or float(targetOffset @ headingVector) <= 1e-6):
                self._stop_at_current_position()
                return
            previousSpeed = (
                float(np.max(self.segmentSpeeds))
                if len(self.segmentSpeeds) else 0.0
            )
            approachSpeed = max(0.0, self.speed, previousSpeed)
            if approachSpeed <= 1e-9:
                self._stop_at_current_position()
                return
            self.executionPath = executionPath
            self.executionArc = arc_lengths(executionPath)
            self.speedEndS = np.asarray([targetProgress], dtype=float)
            self.segmentSpeeds = np.asarray([approachSpeed], dtype=float)
            return

        self.executionPath = clean_xy(future_path)
        if len(self.executionPath) < 2:
            self._stop_at_current_position()
            return

        self.executionArc = arc_lengths(self.executionPath)
        waypointProgress = project_points_to_polyline(
            waypointPoints, self.executionPath)['s']
        waypointProgress = np.maximum.accumulate(waypointProgress)
        self.speedEndS = waypointProgress[1:].copy()
        self.segmentSpeeds = np.asarray(intervalSpeeds, dtype=float)

    def _stop_at_current_position(self):
        self.executionPath = np.asarray([[self.x, self.y]], dtype=float)
        self.executionArc = np.asarray([0.0], dtype=float)
        self.speedEndS = np.empty(0, dtype=float)
        self.segmentSpeeds = np.empty(0, dtype=float)
        self.speed = 0.0

    def step(self):
        if self.reachedEndpoint:
            return

        if len(self.segmentSpeeds) == 0 or len(self.executionPath) < 2:
            self.speed = 0.0
        else:
            currentPosition = np.asarray([self.x, self.y], dtype=float)
            progress = float(project_to_polyline(
                currentPosition, self.executionPath)['s'])
            intervalIndex = int(np.searchsorted(
                self.speedEndS, progress, side='right'))

            if intervalIndex >= len(self.segmentSpeeds):
                self.speed = 0.0
            else:
                speed = max(
                    0.0, float(self.segmentSpeeds[intervalIndex]))
                if speed <= 1e-9:
                    self.speed = 0.0
                else:
                    targetS = min(
                        progress + speed * self.dt,
                        float(self.executionArc[-1]),
                    )
                    position = point_at_s(self.executionPath, targetS)
                    self.x, self.y = map(float, position)
                    self.heading = heading_at_s(
                        self.executionPath, targetS)
                    self.speed = (
                        speed
                        if targetS < self.executionArc[-1] - 1e-9
                        else 0.0
                    )

        distanceToDestination = np.linalg.norm(
            np.asarray([self.x, self.y]) - self.route[-1])
        if distanceToDestination < 1.5:
            self.arrived = True
        if distanceToDestination <= 1e-6:
            self.reachedEndpoint = True
            self.speed = 0.0
        self.history.append(np.asarray([self.x, self.y], dtype=float))

class SignalController:
    """维护JSON给出的信号灯颜色和倒计时。"""

    def __init__(self, lane_signals):
        self.states = deepcopy(lane_signals)

    def step(self, dt):
        for movements in self.states.values():
            for state in movements.values():
                remaining = max(0.0, float(state.get('time', 0.0)) - dt)
                if remaining > 0.0:
                    state['time'] = remaining
                    continue
                color = str(state.get('color', 'GREEN'))
                next_color = 'GREEN' if color == 'RED' else 'RED'
                state['color'] = next_color
                duration_key = (
                    'greenDuration' if next_color == 'GREEN'
                    else 'redDuration'
                )
                state['time'] = float(state.get(duration_key, 30.0))


class SimulationVisualizer:
    """展示地图、长路径、规划中间线、waypoint和车辆状态。"""

    def __init__(
            self, scenario, vehicles, *, visualizeGraphSearchResults=True,
            visualizeFirstGraphSearchResultOnly=False):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.plt = plt
        self.LineCollection = LineCollection
        self.scenario = scenario
        self.vehicles = vehicles
        self.visualizeGraphSearchResults = bool(
            visualizeGraphSearchResults)
        self.visualizeFirstGraphSearchResultOnly = bool(
            visualizeFirstGraphSearchResultOnly)
        self.colors = {
            vehicle_id: plt.get_cmap('tab10')(index % 10)
            for index, vehicle_id in enumerate(vehicles)
        }
        self.centerlineSegments = [
            np.asarray(points, dtype=float)[:, :2]
            for points in scenario.centerlineDict.values()
            if len(points) >= 2
        ]
        routePoints = np.vstack([
            np.asarray(routeData['points'], dtype=float)[:, :2]
            for routeData in scenario.routes.values()
        ])
        self.viewLimits = (
            float(np.min(routePoints[:, 0])) - 50.0,
            float(np.max(routePoints[:, 0])) + 50.0,
            float(np.min(routePoints[:, 1])) - 50.0,
            float(np.max(routePoints[:, 1])) + 50.0,
        )
        self.stopRequested = False
        self.figure = None
        self.axes = None
        self.completeAreaShown = False
        self.drivableLineCaches = {}
        self.spatialSearchNodePathCaches = {}

    def update(self, simulation_time, engine, responses, signals):
        self._show_complete_drivable_areas(engine)
        if self.figure is None:
            self.plt.ion()
            self.figure, self.axes = self.plt.subplots(figsize=(11, 9))
            self.figure.canvas.mpl_connect(
                'key_press_event', self._on_key)

        axes = self.axes
        axes.clear()
        diagnostics = getattr(engine, 'lastPlanningDiagnostics', {})

        axes.add_collection(self.LineCollection(
            self.centerlineSegments, colors='#eeeeee', linewidths=1.2,
            alpha=0.9, zorder=0))

        for polygon in self.scenario.obstacleMap.values():
            points = polygon.get('polygon', polygon)
            array = np.asarray([
                [point['x'], point['y']] if isinstance(point, dict)
                else point[:2]
                for point in points
            ], dtype=float)
            if len(array) >= 3:
                closed = np.vstack((array, array[0]))
                axes.fill(closed[:, 0], closed[:, 1], color='black',
                          alpha=0.25, zorder=3)


        self._draw_diagnostics(axes, diagnostics)

        # Keep every global route complete and above all other line layers.
        for route_data in self.scenario.routes.values():
            route = np.asarray(route_data['points'], dtype=float)
            axes.plot(route[:, 0], route[:, 1], color='black',
                      linewidth=0.45, alpha=0.95, zorder=11)
        for lane_id, movements in signals.items():
            if lane_id not in self.scenario.centerlineDict:
                continue
            point = self.scenario.centerlineDict[lane_id][-1]
            color = ('red' if any(
                state.get('color') == 'RED'
                for state in movements.values()) else 'green')
            axes.scatter(point[0], point[1], s=45, c=color,
                         edgecolors='white', zorder=8)

        status_lines = []
        response_map = {
            str(item.get('deviceId')): item for item in responses
        }
        for vehicle_id, vehicle in self.vehicles.items():
            color = self.colors[vehicle_id]
            history = np.asarray(vehicle.history)
            if len(history) >= 2:
                axes.plot(history[:, 0], history[:, 1], color=color,
                          linewidth=0.9, alpha=0.6, zorder=9)
            self._draw_vehicle(axes, vehicle, color)
            if len(vehicle.displayWaypoints):
                axes.scatter(
                    vehicle.displayWaypoints[:, 0],
                    vehicle.displayWaypoints[:, 1],
                    s=14, color=color, edgecolors='black',
                    linewidths=0.35, zorder=12)
            response = response_map.get(vehicle_id, {})
            status_lines.append(
                f'{vehicle_id}: v={vehicle.speed:.2f}m/s, '
                f'success={response.get("success", "-")}')

        axes.set_xlim(self.viewLimits[0], self.viewLimits[1])
        axes.set_ylim(self.viewLimits[2], self.viewLimits[3])
        axes.set_aspect('equal', adjustable='box')
        axes.text(
            0.01, 0.99, '\n'.join(status_lines),
            transform=axes.transAxes, va='top', fontsize=8,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def _show_complete_drivable_areas(self, engine):
        if (self.completeAreaShown
                or not engine.drivableAreaPlannerCaches):
            return

        self.completeAreaShown = True
        self.plt.ioff()
        figure, axes = self.plt.subplots(figsize=(11, 9))
        axes.add_collection(self.LineCollection(
            self.centerlineSegments, colors='#eeeeee', linewidths=1.2,
            alpha=0.9, zorder=0))

        for vehicle_id, planner in engine.drivableAreaPlannerCaches.items():
            feasibleLines = [
                np.asarray(points, dtype=float)[:, :2]
                for points in planner.centerline_dict.values()
                if len(points) >= 2
            ]
            if feasibleLines:
                axes.add_collection(self.LineCollection(
                    feasibleLines,
                    colors=[self.colors.get(vehicle_id, 'tab:blue')],
                    linewidths=2.4, alpha=0.65, zorder=1))

        axes.set_xlim(self.viewLimits[0], self.viewLimits[1])
        axes.set_ylim(self.viewLimits[2], self.viewLimits[3])
        axes.set_aspect('equal', adjustable='box')
        figure.canvas.draw()
        self.plt.show(block=True)
        self.plt.close(figure)

    def close(self):
        self.plt.ioff()
        self.plt.show()

    def _draw_diagnostics(self, axes, diagnostics):
        for vehicle_id, lines in diagnostics.get('drivableLines', {}).items():
            if lines:
                self.drivableLineCaches[vehicle_id] = [
                    np.asarray(line, dtype=float).copy()
                    for line in lines
                ]

        for vehicle_id, lines in self.drivableLineCaches.items():
            color = self.colors.get(vehicle_id, 'gold')
            for points in lines:
                axes.plot(points[:, 0], points[:, 1], color=color,
                          linewidth=2.0, alpha=0.78, zorder=2)
        if self.visualizeGraphSearchResults:
            for vehicle_id, nodePaths in diagnostics.get(
                    'spatialSearchNodePaths', {}).items():
                paths = tuple(
                    np.asarray(nodePath, dtype=float).copy()
                    for nodePath in nodePaths if len(nodePath)
                )
                if paths:
                    self.spatialSearchNodePathCaches[vehicle_id] = paths

            for nodePaths in self.spatialSearchNodePathCaches.values():
                visiblePaths = (
                    nodePaths[:1]
                    if self.visualizeFirstGraphSearchResultOnly
                    else nodePaths
                )
                for points in visiblePaths:
                    axes.plot(
                        points[:, 0], points[:, 1], color='red',
                        linewidth=0.45, alpha=0.75, zorder=30)
                    axes.scatter(
                        points[:, 0], points[:, 1], s=5, color='red',
                        edgecolors='none', alpha=0.9, zorder=31)


        for vehicle_id, path in diagnostics.get('futurePaths', {}).items():
            points = np.asarray(path, dtype=float)
            color = self.colors.get(vehicle_id, 'tab:blue')
            axes.plot(points[:, 0], points[:, 1], '--', color=color,
                      linewidth=0.75, alpha=0.9, zorder=10)
        for vehicle_id, candidates in diagnostics.get(
                'futureWaypointCandidates', {}).items():
            points = np.asarray(candidates, dtype=float)
            if len(points):
                axes.scatter(points[:, 0], points[:, 1], s=8,
                             color=self.colors.get(vehicle_id, 'tab:blue'),
                             alpha=0.5, zorder=7)
        for vehicle_id, waypoint_whole in diagnostics.get(
                'waypointWhole', {}).items():
            points = np.asarray(waypoint_whole.get('points', ()), dtype=float)
            if len(points):
                axes.scatter(
                    points[:, 0], points[:, 1], marker='o', s=3,
                    color=self.colors.get(vehicle_id, 'tab:blue'),
                    edgecolors='none', alpha=0.9, zorder=8)

    @staticmethod
    def _draw_vehicle(axes, vehicle, color):
        from matplotlib.patches import Polygon

        forward = np.asarray([
            math.cos(vehicle.heading), math.sin(vehicle.heading)])
        lateral = np.asarray([-forward[1], forward[0]])
        center = np.asarray([vehicle.x, vehicle.y])
        corners = np.asarray([
            center + sx * vehicle.length * 0.5 * forward
            + sy * vehicle.width * 0.5 * lateral
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ])
        axes.add_patch(Polygon(
            corners, closed=True, facecolor=color,
            edgecolor='black', alpha=0.8, zorder=8))
        axes.text(vehicle.x, vehicle.y, vehicle.vehicleId,
                  fontsize=7, zorder=9)

    def _on_key(self, event):
        if event.key == 'escape':
            self.stopRequested = True


def create_vehicles(scenario):
    return {
        vehicle_id: VehicleSimulator(
            vehicle_id,
            scenario.routes[vehicle_id]['points'],
            static_info,
            scenario.simulationDt,
        )
        for vehicle_id, static_info in scenario.vehicleStaticInfo.items()
    }


def _load_pickle(path):
    resolved = Path(path).resolve()
    with resolved.open('rb') as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f'{resolved.name}的根对象必须是字典')
    return value


def _route_points(route_data, lane_ids, centerline_dict, base_dir):
    if route_data.get('points') is not None:
        return np.asarray(route_data['points'], dtype=float)[:, :2]
    if route_data.get('npyFile') is not None:
        path = (base_dir / route_data['npyFile']).resolve()
        return np.asarray(np.load(path), dtype=float)[:, :2]
    raise ValueError('每辆车的route必须直接提供points或npyFile')


def _path_heading(path, index):
    first = max(0, min(int(index), len(path) - 2))
    vector = path[first + 1] - path[first]
    return float(math.atan2(vector[1], vector[0]))


def _position(point, heading):
    return {
        'x': float(point[0]),
        'y': float(point[1]),
        'z': 0.0,
        'hdg': cartesian_heading_to_navigation(heading),
    }


def _id_map(items, prefix):
    if isinstance(items, dict):
        return dict(items)
    return {
        str(item.get('id', f'{prefix}-{index + 1}')): item
        for index, item in enumerate(items)
    }
