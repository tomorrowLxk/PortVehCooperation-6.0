"""把PortVehCooperation 4.0/demoTest场景批量升级为5.0测试JSON。"""

from copy import deepcopy
import json
from pathlib import Path
import re


SOURCE_ROOT = Path(
    r'G:\项目\港口车辆运动规划\PortVehCooperation 4.0\demoTest'
)
SOURCE_MAP_ROOT = Path(
    r'G:\项目\港口车辆运动规划\PortVehCooperation 4.0\staticMap'
)
OUTPUT_ROOT = Path(
    r'G:\项目\港口车辆运动规划\PortVehCooperation 5.0\test_env\demo'
)
DEFAULT_MAP_VERSION = '0910'


def convert_all_scenarios():
    converted = []
    for source_file in sorted(SOURCE_ROOT.rglob('*.json')):
        relative_file = source_file.relative_to(SOURCE_ROOT)
        output_file = OUTPUT_ROOT / relative_file
        converted_data = convert_scenario(source_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as stream:
            json.dump(converted_data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        converted.append(output_file)
        print(f'converted: {relative_file}')
    print(f'total={len(converted)}, output={OUTPUT_ROOT}')
    return converted


def convert_scenario(source_file):
    with source_file.open('r', encoding='utf-8') as stream:
        source = json.load(stream)

    result = deepcopy(source)
    result['schemaVersion'] = '5.0'
    map_version = _detect_map_version(source_file, source)
    centerline_file = (
        SOURCE_MAP_ROOT / f'centerline_dict{map_version}.pkl'
    ).resolve()
    connector_file = (
        SOURCE_MAP_ROOT / f'connector_dict{map_version}.pkl'
    ).resolve()
    if not centerline_file.is_file() or not connector_file.is_file():
        raise FileNotFoundError(
            f'{source_file}: 找不到地图版本{map_version}对应的PKL')

    result['map'] = {
        'version': map_version,
        'centerlineFile': str(centerline_file),
        'connectorFile': str(connector_file),
        'junctionLaneIds': _scenario_junction_lane_ids(result['vehicles']),
    }
    result['environment'] = _convert_environment(
        source.get('environment') or {}, source_file)

    for vehicle in result.get('vehicles') or ():
        route = vehicle.get('route') or {}
        points = route.get('points')
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(
                f'{source_file}: {vehicle.get("vehicleId")}缺少原始route.points')
        request = vehicle.setdefault('request', {})
        request['forbidLaneIdList'] = []
        request.setdefault('laneWhiteList', [])
        request.setdefault('priority', 1)
        request.setdefault('allowReverse', False)

    return result


def _detect_map_version(source_file, source):
    for script_name in ('main.py', 'main_gif.py'):
        script_file = source_file.parent / script_name
        if not script_file.is_file():
            continue
        text = script_file.read_text(encoding='utf-8')
        match = re.search(
            r'MAP_VERSION\s*=\s*[^0-9]([0-9]+)', text)
        if match:
            return match.group(1)

    raw_version = str((source.get('map') or {}).get('version', ''))
    return raw_version if raw_version.isdigit() else DEFAULT_MAP_VERSION


def _scenario_junction_lane_ids(vehicles):
    """旧JSON没有junctionMap；保留本场景长路径中出现的虚拟连接车道。"""
    output = []
    for vehicle in vehicles or ():
        for lane_id in (vehicle.get('route') or {}).get('laneIds') or ():
            lane_id = str(lane_id)
            if '_' not in lane_id and lane_id not in output:
                output.append(lane_id)
    return output


def _convert_environment(environment, source_file):
    old_signals = environment.get('trafficSignals') or ()
    if old_signals:
        raise ValueError(
            f'{source_file}: 旧trafficSignals非空，无法自动推断5.0的'
            '进口道ID、方向及倒计时，请人工补充laneSignals')

    obstacles = {}
    for prefix, values in (
        ('geoFence', environment.get('geoFences') or ()),
        ('dynamicObstacle', environment.get('dynamicObstacles') or ()),
    ):
        if isinstance(values, dict):
            obstacles.update(deepcopy(values))
            continue
        for index, value in enumerate(values):
            obstacle_id = str(value.get('id', f'{prefix}-{index + 1}'))
            obstacles[obstacle_id] = deepcopy(value)

    return {
        'obstacles': obstacles,
        'laneSignals': {},
    }


convert_all_scenarios()
