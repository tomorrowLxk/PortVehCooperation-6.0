# 5.0脚本式闭环仿真

直接运行：

```powershell
python test_env/main.py
```

`main.py`中显式保留了逐帧循环：生成BSM、调用`batchRoutePlan`、打印并下发response、车辆步进、信控倒计时和可视化。按`Esc`结束可视化。

场景入口是单个`scenario.json`。其中：

- `map`指定中心线PKL、连接关系PKL、地图版本和全部交叉口虚拟车道ID；
- `vehicles[].route.laneIds`是固定长路径结果，默认按这些车道中心线自动拼接全局参考线；也可使用`points`直接写坐标，或用`npyFile`引用坐标文件；
- `environment.obstacles`是`id -> {polygon: [{x, y}, ...]}`；
- `environment.laneSignals`是`进口道laneId -> 方向 -> {color, time}`，方向采用`rightTurn/leftTurn/UTurn/straight`，颜色采用`RED/GREEN`；可选的`redDuration/greenDuration`用于闭环倒计时循环；
- `vehicles`同时给出初始状态、静态参数和request字段。

无界面短跑可使用：

```powershell
$env:PORT_SIM_HEADLESS='1'
$env:PORT_SIM_MAX_STEPS='6'
python test_env/main.py
```

可以用`PORT_SIM_SCENARIO`切换JSON，用`PORT_SIM_PRINT_RESPONSE=1`打印完整response。
