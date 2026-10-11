"""PortVehCooperation 5.0统一参数。"""


VEHICLE_CACHE_MAX_MISSING_FRAMES = 3  # 车辆连续缺失达到该帧数后，从动静态信息缓存中删除
VEHICLE_LATERAL_UNCERTAINTY_SCALE = 0.0  # 每级posAccuracy对应的横向定位不确定性，单位m，正式映射明确前暂设为0
VEHICLE_LONGITUDINAL_UNCERTAINTY_SCALE = 0.0  # 每级posAccuracy对应的纵向定位不确定性，单位m，正式映射明确前暂设为0
VEHICLE_OBSTACLE_CONFIRM_FRAMES = 3  # 车辆连续“短路径要求行驶但实际静止”达到该帧数后，视为静态障碍物
VEHICLE_OBSTACLE_MAX_FRAMES = 300  # 死车原因连续帧数的最大缓存值
VEHICLE_STOP_SPEED_THRESHOLD = 0.5  # 车辆速度低于该值时，开始怀疑车辆处于停车状态，单位m/s
VEHICLE_PATH_REMAINING_LENGTH_THRESHOLD = 2.0  # 短路径剩余长度超过该值时，才认为车辆仍应继续行驶，单位m
AGV_DESTINATION_REMAINING_LENGTH_THRESHOLD = 2.0  # AGV沿长路径到终点的剩余弧长不超过该值时，视为终点死车，单位m
HV_DESTINATION_DISTANCE_THRESHOLD = 2.0  # HV当前位置到长路径终点的平面距离不超过该值时，视为终点死车，单位m

REVERSE_POINT_REACH_TOLERANCE = 1.5  # 调箱门车辆到倒车切换点的到达判定距离，单位m
REVERSE_POINT_STOP_SPEED = 0.2  # 调箱门车辆通过倒车切换点前要求达到的停车速度，单位m/s
REVERSE_POINT_LOCATE_TOLERANCE = 3.0  # R点匹配当前局部可行域线的最大距离，单位m
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
JUNCTION_EXIT_PATH_EXTENSION = 5.0  # 前视末端落在交叉口内部时，出弯后沿出口道最多补充的路径长度，单位m
JUNCTION_EXIT_DESTINATION_MERGE_DISTANCE = 2.0  # 出口点与全局终点沿路径距离不超过该值时合并，单位m
DRIVABLE_LINE_MAX_COUNT = 3  # 每辆车每帧最多输出的局部可行线数量
VEHICLE_LANE_LOCATE_MAX_DISTANCE = 10.0  # 车辆位置允许匹配到车道中心线的最大横向距离，单位m
VEHICLE_LANE_LOCATE_HEADING_MIN_SPEED = 0.5  # 车辆速度低于该值时，位置匹配不使用航向约束，单位m/s
VEHICLE_LANE_LOCATE_HEADING_WEIGHT = 1.0  # 位置匹配评分中航向误差的权重
VEHICLE_LANE_LOCATE_ROUTE_PREFERENCE = 0.5  # 非长路径参考车道参与位置匹配时增加的评分代价
VEHICLE_LANE_LOCATE_TOPOLOGY_DEPTH = 2  # 后续帧从上一匹配车道向左右及下游搜索的拓扑层数

FUTURE_PATH_LENGTH = 80.0  # 本帧用于分组、HV预测和空间规划的前向参考线长度，单位m
FUTURE_PATH_POINT_SPACING = 1.0  # futurePath重采样点间距，单位m
WAYPOINT_CANDIDATE_SPACING = 2.0  # 普通道路上相邻waypoint候选点的弧长间距，单位m
WAYPOINT_OUTPUT_MIN_FORWARD_DISTANCE = 2.0  # 首个下发点至少位于车辆前方该距离，单位m
WAYPOINT_OUTPUT_FORWARD_TIME = 1.0  # 首个下发点随当前车速前移的最小时间，单位s
WAYPOINT_SPARSE_MIN_SPACING = 10.0  # 非强制waypointSparse与已有下发点的最小沿路径间距，单位m
WAYPOINT_WHOLE_REUSE_MAX_LENGTH_DIFFERENCE = 10.0  # 新旧waypointWhole允许复用的最大剩余长度差，单位m
WAYPOINT_WHOLE_REUSE_MAX_TIME_DIFFERENCE = 2.0  # 新旧waypointWhole允许复用的最大到达时间差，单位s
RESPONSE_DESTINATION_TOLERANCE = 2.0  # waypoint末点进入全局终点该距离内时标记最终到达，单位m
RESPONSE_TURN_THRESHOLD_RAD = 0.08  # 短路径段航向变化达到该值时标记为转弯，单位rad
RESPONSE_UTURN_THRESHOLD_RAD = 2.60  # 短路径段航向变化达到该值时标记为掉头，单位rad
COOPERATION_PATH_POINT_SPACING = 4.0  # 协同分块内部futurePath降采样间距，单位m
COOPERATION_LATERAL_MARGIN = 0.3  # 每辆车参与协同分块时增加的侧向安全冗余，单位m
COOPERATION_PATH_SIMPLIFICATION_MARGIN = 0.5  # 两条4m折线近似产生的合计横向补偿，单位m
COOPERATION_STATIC_GAP = 6.0  # 协同碰撞检测使用的固定纵向跟驰净距，单位m
COOPERATION_TIME_CLEARANCE = 3.0  # 低优先级车辆晚于高优先级车辆清空冲突位置的时间余量，单位s
COOPERATION_CONFLICT_SEGMENT_MERGE_DISTANCE = 6.0  # 两个冲突候选点段允许合并的最大沿futurePath距离，单位m
COOPERATION_STOP_SPEED_THRESHOLD = 0.5  # 协同所需通行速度低于该值时改为停车，单位m/s
COOPERATION_EMERGENCY_DECELERATION = 6.0  # AEB安全距离使用的紧急减速度，单位m/s²
COOPERATION_AEB_STATIC_GAP = 5.0  # AEB安全距离中的固定停车余量，单位m
PHYSICAL_FRONT_REAR_MIN_PROGRESS = 0.5  # 判定物理前后关系所需的最小纵向进度，单位m
PHYSICAL_FRONT_REAR_HEADING_THRESHOLD_DEG = 30.0  # 判定物理前后关系允许的最大航向差，单位°
PRIORITY_CONFLICT_SAMPLE_SPACING = 2.0  # 优先级冲突入口搜索的粗采样间距，单位m
PRIORITY_CONFLICT_UPSTREAM_DISTANCE = 2.0  # 冲突锚点相对冲突区入口的上游偏移，单位m
HV_PREDICTION_HORIZON = 3.0  # HV按当前速度匀速预测的时间范围，单位s
AGV_BASE_PATH_REPLAN_WAYPOINT_DISTANCE = 20.0  # AGV距离历史完整basePath末端不超过该值时触发空间重规划，单位m
AGV_BASE_PATH_DEVIATION_THRESHOLD = 8.0  # AGV当前位置距离历史basePath超过该值时触发空间重规划，单位m
AGV_BASE_PATH_ENDPOINT_SAME_THRESHOLD = 2.0  # 新旧完整basePath末端在该距离内视为相同，单位m
PATH_CONTINUATION_COMPARE_DISTANCE = 20.0  # 历史路径即将执行完时，用于保持换道连续性的前向比较距离，单位m
PATH_CONTINUATION_MAX_DEVIATION = 1.0  # 新候选在比较区间内允许偏离历史未执行路径的最大距离，单位m

PATH_NODE_SPACING = 4.0  # 每条可行域线生成普通P节点的纵向间距，单位m
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
PATH_STATIC_STOP_DISTANCE = 6.0  # polygon上游T点要求的车头静态停车净距，单位m
PATH_OBSTACLE_REAR_CLEARANCE = 12.0  # polygon下游恢复P节点前的额外清理距离，单位m
PATH_G0_LATERAL_TOLERANCE = 1.0  # G末端到全局长路径横向距离小于该值时视为长路径G，单位m
PATH_MAX_CANDIDATES = 3  # 每辆车空间规划最多保留多少条不同同伦类的候选 path
PATH_SEARCH_VALIDATION_BATCH_SIZE = 4  # 每轮懒验证最多检查的内部候选path数量，不改变最终输出数量
PATH_MAX_SIGNATURES_PER_NODE = 8  # 每个图节点最多保留的不同分支序列数量
PATH_MAX_GRAPH_ITERATIONS = 32  # 懒验证删除无效边后允许重新搜索当前终点的最大轮数
PATH_REFERENCE_WEIGHT = 1.0  # 4.0占位选优中全局长路径横向偏差的权重
PATH_LANE_CHANGE_PENALTY = 2.0  # 每次跨可行域线换道叠加的等效距离代价，单位m
VEHICLE_MAX_LATERAL_ACCELERATION = 0.3  # 依据曲率计算速度上限时允许的最大横向加速度，单位m/s²
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
