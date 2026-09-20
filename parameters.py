"""PortVehCooperation 5.0统一参数。"""


VEHICLE_CACHE_MAX_MISSING_FRAMES = 3  # 车辆连续缺失达到该帧数后，从动静态信息缓存中删除
VEHICLE_OBSTACLE_CONFIRM_FRAMES = 3  # 车辆连续“短路径要求行驶但实际静止”达到该帧数后，视为静态障碍物
VEHICLE_STOP_SPEED_THRESHOLD = 0.5  # 车辆速度低于该值时，开始怀疑车辆处于停车状态，单位m/s
VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD = 2.0  # 短路径剩余长度超过该值时，才认为车辆仍应继续行驶，单位m
VEHICLE_DESTINATION_DISTANCE_THRESHOLD = 2.0  # 沿全局参考线到终点的剩余弧长不超过该值时，直接视为静态障碍物，单位m

REVERSE_POINT_REACH_TOLERANCE = 1.5  # 调箱门车辆到倒车切换点的到达判定距离，单位m
REVERSE_POINT_STOP_SPEED = 0.2  # 调箱门车辆通过倒车切换点前要求达到的停车速度，单位m/s
DXM_TURN_LANE_THRESHOLD_DEG = 75.0  # 车道首尾方向差达到该角度时视为转弯车道，单位deg
DXM_ROUTE_POINT_SPACING = 2.0  # 调箱门新增车道中心线的目标点间距，单位m
DXM_REVERSE_SEARCH_MIN_STEP = 5.0  # 搜索倒车切换点时每次增加的最小距离，单位m
DXM_REVERSE_SEARCH_RADIUS_RATIO = 0.5  # 倒车切换点搜索步长相对最小转弯半径的比例
DXM_REVERSE_SEARCH_COUNT = 16  # 调箱门倒车切换点允许的最大搜索次数
DXM_CURVE_CHECK_SPACING = 0.5  # 调箱门候选曲线曲率检查的采样间距，单位m
DXM_CURVE_CHECK_MIN_POINTS = 30  # 调箱门候选曲线曲率检查的最少采样点数
DXM_CURVATURE_TOLERANCE = 1.02  # 调箱门曲率相对最小转弯半径约束的数值容差

FEASIBLE_DOMAIN_SEARCH_DEPTH = 1  # 完整可行域两段基础横向扩展区域之间允许的最大中间车道数量
FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE = 8.0  # 完整可行域中每横向换一条车道需要预留的纵向距离，单位m
FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE = 0.5  # 前后车道中心线重叠判定的最大横向距离，单位m
VEHICLE_LANE_LOCATE_MAX_DISTANCE = 5.0  # 车辆位置允许匹配到车道中心线的最大横向距离，单位m
VEHICLE_LANE_LOCATE_HEADING_MIN_SPEED = 0.5  # 车辆速度低于该值时，位置匹配不使用航向约束，单位m/s
VEHICLE_LANE_LOCATE_HEADING_WEIGHT = 1.0  # 位置匹配评分中航向误差的权重
VEHICLE_LANE_LOCATE_ROUTE_PREFERENCE = 0.5  # 非长路径参考车道参与位置匹配时增加的评分代价
VEHICLE_LANE_LOCATE_TOPOLOGY_DEPTH = 2  # 后续帧从上一匹配车道向左右及下游搜索的拓扑层数

DEFAULT_AGV_STATIC_INFO = {
    "length": 14.8,             # AGV车辆长度，单位m
    "width": 2.8,               # AGV车辆宽度，单位m
    "maxSpeed": 4.2,            # AGV最大纵向速度，单位m/s
    "maxAcceleration": 2.0,     # AGV最大纵向加速度，单位m/s²
    "maxDeceleration": 4.0,     # AGV最大纵向减速度，单位m/s²
    "minTurningRadius": 10.0,   # AGV最小转弯半径，单位m
}

DEFAULT_NON_AGV_STATIC_INFO = {
    "length": 14.8,             # 非AGV车辆长度，单位m
    "width": 2.8,               # 非AGV车辆宽度，单位m
    "maxSpeed": 10.0,           # 非AGV最大纵向速度，单位m/s
    "maxAcceleration": 2.0,     # 非AGV最大纵向加速度，单位m/s²
    "maxDeceleration": 4.0,     # 非AGV最大纵向减速度，单位m/s²
    "minTurningRadius": 10.0,   # 非AGV最小转弯半径，单位m
}
