"""PortVehCooperation 5.0统一参数。"""


VEHICLE_CACHE_MAX_MISSING_FRAMES = 3  # 车辆连续缺失达到该帧数后，从动静态信息缓存中删除
VEHICLE_OBSTACLE_CONFIRM_FRAMES = 3  # 车辆连续“短路径要求行驶但实际静止”达到该帧数后，视为静态障碍物
VEHICLE_OBSTACLE_MAX_FRAMES = 300  # 死车原因连续帧数的最大缓存值
VEHICLE_STOP_SPEED_THRESHOLD = 0.5  # 车辆速度低于该值时，开始怀疑车辆处于停车状态，单位m/s
VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD = 2.0  # 短路径剩余长度超过该值时，才认为车辆仍应继续行驶，单位m
AGV_DESTINATION_REMAINING_LENGTH_THRESHOLD = 2.0  # AGV沿长路径到终点的剩余弧长不超过该值时，视为终点死车，单位m
HV_DESTINATION_DISTANCE_THRESHOLD = 2.0  # HV当前位置到长路径终点的平面距离不超过该值时，视为终点死车，单位m

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
FEASIBLE_DOMAIN_LANE_CHANGE_DISTANCE = 0.0  # 完整可行域中每横向换一条车道需要预留的纵向距离，单位m
FEASIBLE_DOMAIN_CONNECTION_OVERLAP_TOLERANCE = 0.5  # 前后车道中心线重叠判定的最大横向距离，单位m
DRIVABLE_LINE_FORWARD_LENGTH = 80.0  # 每帧局部可行线相对车辆当前长路径进度的前视距离，单位m
DRIVABLE_LINE_POINT_SPACING = 1.0  # 局部可行线输出点的目标间距，单位m
DRIVABLE_LINE_MAX_COUNT = 3  # 每辆车每帧最多输出的局部可行线数量
VEHICLE_LANE_LOCATE_MAX_DISTANCE = 6.0  # 车辆位置允许匹配到车道中心线的最大横向距离，单位m
VEHICLE_LANE_LOCATE_HEADING_MIN_SPEED = 0.5  # 车辆速度低于该值时，位置匹配不使用航向约束，单位m/s
VEHICLE_LANE_LOCATE_HEADING_WEIGHT = 1.0  # 位置匹配评分中航向误差的权重
VEHICLE_LANE_LOCATE_ROUTE_PREFERENCE = 0.5  # 非长路径参考车道参与位置匹配时增加的评分代价
VEHICLE_LANE_LOCATE_TOPOLOGY_DEPTH = 2  # 后续帧从上一匹配车道向左右及下游搜索的拓扑层数

FUTURE_PATH_LENGTH = 80.0  # 本帧用于分组、HV预测和空间规划的前向参考线长度，单位m
FUTURE_PATH_POINT_SPACING = 1.0  # futurePath重采样点间距，单位m
HV_PREDICTION_HORIZON = 3.0  # HV按当前速度匀速预测的时间范围，单位s
AGV_BASE_PATH_REPLAN_WAYPOINT_DISTANCE = 2.0  # AGV接近历史第二个waypoint并触发basePath重规划的距离，单位m

PATH_NODE_SPACING = 8.0  # 每条可行域线生成普通P节点的纵向间距，单位m
PATH_GRAPH_START_OFFSET = 10.0  # 从车辆当前位置向前首次布置P节点的距离，单位m
PATH_MAX_LATERAL_CONNECTION_DISTANCE = 10.0  # 跨可行域线连边允许的最大横向距离，单位m
PATH_LANE_CHANGE_LONGITUDINAL_RATIO = 2.5  # 跨线所需最小纵向距离相对横向距离的倍率
PATH_MAX_LANE_CHANGE_LONGITUDINAL_DISTANCE = 50.0  # 跨线连边允许的最大纵向距离，单位m
PATH_CURVE_SAMPLE_SPACING = 0.5  # Hermite、Frenet曲线及中心线polygon粗筛的采样间距，单位m
PATH_COLLISION_SAMPLE_SPACING = 0.5  # 候选path连续车辆扫掠检查的纵向采样间距，单位m
PATH_OUTPUT_POINT_SPACING = 1.0  # 空间path最终输出的重采样间距，单位m
PATH_CURVATURE_WINDOW = 5.0  # 评价曲率时使用的前后弧长窗口，单位m
PATH_CURVATURE_TOLERANCE = 1.0  # 人工跨线曲线相对最小转弯半径的曲率容差倍率
PATH_LATERAL_SAFETY_MARGIN = 1.0  # 正常path后验证时叠加到本车横向占用上的固定冗余，单位m
PATH_STATIC_STOP_DISTANCE = 5.0  # polygon上游T点要求的车头静态停车净距，单位m
PATH_OBSTACLE_REAR_CLEARANCE = 12.0  # polygon下游恢复P节点前的额外清理距离，单位m
PATH_G0_LATERAL_TOLERANCE = 1.0  # G末端到全局长路径横向距离小于该值时视为长路径G，单位m
PATH_MAX_CANDIDATES = 5  # ???????5??????????1???path
PATH_MAX_SIGNATURES_PER_NODE = 32  # 每个图节点最多保留的不同分支序列数量
PATH_MAX_GRAPH_ITERATIONS = 64  # 懒验证删除无效边后允许重新搜索当前终点的最大轮数
PATH_REFERENCE_WEIGHT = 1.0  # 4.0占位选优中全局长路径横向偏差的权重
PATH_LANE_CHANGE_PENALTY = 10.0  # 4.0占位选优中每次跨可行域线的代价
PATH_CURVATURE_WEIGHT = 0.5  # 4.0占位选优中最大曲率的权重
PATH_CURVATURE_SPEED_COEFFICIENT = 3.0  # 曲率速度上限公式1/(系数*曲率)中的系数
PATH_CURVATURE_MIN_SPEED = 2.77  # 曲率限速非停车点允许的最低速度上限，单位m/s
TRAFFIC_SIGNAL_MATCH_DISTANCE = 5.0  # path与进口道末端匹配的最大距离，单位m
TRAFFIC_SIGNAL_STOP_DISTANCE = 5.0  # 信控停车位置相对进口道末端的上游距离，单位m
TRAFFIC_SIGNAL_GREEN_ETA_MARGIN = 2.0  # 当前为绿灯时叠加到ETA上的保守时间，单位s
TRAFFIC_SIGNAL_HEADING_WINDOW = 10.0  # 信控方向判断使用的上游path长度，单位m
TRAFFIC_SIGNAL_HEADING_AFTER_WINDOW = 20.0  # 信控方向判断使用的下游path长度，单位m
TRAFFIC_SIGNAL_TURN_THRESHOLD_DEG = 15.0  # 直行与左右转的航向变化分界，单位deg
TRAFFIC_SIGNAL_UTURN_THRESHOLD_DEG = 135.0  # 掉头方向的最小航向变化，单位deg

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
