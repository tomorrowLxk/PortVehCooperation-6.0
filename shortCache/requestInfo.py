"""整理仅在本帧规划过程中使用的request信息。"""

from parameters import DEFAULT_AGV_STATIC_INFO, DEFAULT_NON_AGV_STATIC_INFO


def build_request_temporary_info(requests, vehicle_static_info_cache):
    """返回本帧系统优先级和车辆规划速度上限。"""
    system_priorities = {}
    vehicle_planning_max_speeds = {}

    for request in requests:
        vehicle_id = str(request.get('deviceId'))
        vehicle_type = str(request.get('deviceType'))
        is_agv = vehicle_type.upper() == 'AGV'
        defaults = DEFAULT_AGV_STATIC_INFO if is_agv else DEFAULT_NON_AGV_STATIC_INFO
        static_info = vehicle_static_info_cache.get(vehicle_id, {})

        physical_max_speed = float(static_info.get('maxSpeed', defaults['maxSpeed']))
        speed_limit = request.get('speedLimit')
        request_max_speed = (
            None if speed_limit is None
            else float(speed_limit) / 3.6
        )
        planning_max_speed = min(
            physical_max_speed,
            defaults['maxSpeed'] if request_max_speed is None else request_max_speed,
        )

        priority = request.get('priority')
        request_priority = 1 if priority is None else int(priority)
        system_priority = (
            max(request_priority, 1) if is_agv else min(request_priority, 0)
        )
        system_priorities[vehicle_id] = system_priority
        vehicle_planning_max_speeds[vehicle_id] = planning_max_speed

    return system_priorities, vehicle_planning_max_speeds
