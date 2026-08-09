# bc-stark-sdk 1.4.5 Revo3 接口核对

本文基于 [PyPI 1.4.5 发布包](https://pypi.org/project/bc-stark-sdk/1.4.5/)中的
`bc_stark_sdk-1.4.5-cp38-abi3-manylinux_2_31_x86_64.whl`
及其中自动生成的 `bc_stark_sdk/main_mod.pyi`。部署代码只导入
`bc_stark_sdk.main_mod`；所有 `DeviceContext` 通信方法均为异步调用，需要 `await`。

## 连接与版本

当前 Modbus/RS485 链路使用：

```python
from bc_stark_sdk import main_mod as sdk

sdk.init_logging()
version = sdk.get_sdk_version()
_, port, baudrate, slave_id = await sdk.auto_detect_modbus_revo3()
ctx = await sdk.modbus_open(port, sdk.Baudrate.Baud5Mbps)
info = await ctx.get_device_info(slave_id)
# ... await ctx.<method>(slave_id)
sdk.modbus_close(ctx)
```

相关模块级函数包括：

- `init_logging`、`get_sdk_version`
- `auto_detect`、`auto_detect_device`、`auto_detect_modbus_revo1`、
  `auto_detect_modbus_revo2`、`auto_detect_modbus_revo3`
- `available_usb_ports`、`list_available_ports`
- `modbus_open`、`modbus_close`、`protobuf_open`
- `init_device_handler`、`init_from_detected`、`close_device_handler`
- `init_socketcan_can`、`init_socketcan_canfd`、`close_socketcan`、
  `is_socketcan_available`
- `init_zqwl_can`、`init_zqwl_canfd`、`close_zqwl`、`list_zqwl_devices`
- `scan_can_devices`、`scan_canfd_devices`
- `set_can_rx_callback`、`set_can_tx_callback`、`set_modbus_read_holding_callback`、
  `set_modbus_read_input_callback`、`set_modbus_write_callback`

当前部署没有使用 CAN、CANFD、EtherCAT、Protobuf 或 callback 接口。

连接后的通用设备接口还包括 `get_device_info`、`set_hardware_type`、`get_device_sn`、
`get_device_fw_version`、`get_sku_type`、`get_serialport_cfg`、
`get_serialport_baudrate`、`set_serialport_baudrate`、`set_slave_id`、`reboot` 和 `close`。

## 关节状态

部署每帧调用：

```python
status = await ctx.v3_get_motor_status_data(slave_id)
positions_deg = status.positions
```

`V3MotorStatusData` 包含五个长度为 21 的数组：`positions`、`velocities`、
`currents`、`statuses`、`errors`。`v3_get_all_motor_positions` 明确返回角度制；部署将
`status.positions` 按同一 V3 位置单位从 degree 转成 rad。类型 stub 没有再次注明
`status.velocities` 的单位，因此当前闭环只消费位置，不依赖速度单位。

V3 电机状态/基础控制接口：

- `v3_get_all_motor_positions`、`v3_get_all_motor_velocities`、
  `v3_get_all_motor_currents`
- `v3_get_motor_status_data`、`v3_get_all_motor_status`、`v3_get_all_motor_errors`
- `v3_clear_motor_errors`
- `v3_set_motor_position`、`v3_set_motor_current`、`v3_set_motor_mit`
- `v3_set_all_motor_positions`、`v3_set_all_motor_currents`
- `v3_set_calibration_current`、`v3_set_max_continuous_current`

stub 的 `DataCollector.new_v3_basic/full` 文案中有“23 motors”的旧描述，但 V3 状态类型、
读写接口和 Revo3 控制接口均定义为 21 个关节。本部署严格要求返回长度等于 21，不做截断。

## 触觉状态

每个控制周期只进行一次 bulk read：

```python
touch = await ctx.v3_get_all_touch_data(slave_id)
summary = touch.summary   # 16 values
modules = touch.modules   # 11 variable-length arrays
```

`V3TouchData.summary` 是 16 个汇总值，`modules` 是模块 0..10 的 11 个变长压力数组。
与触觉相关的全部 V3 接口为：

- `v3_set_all_touch_modules_enabled`、`v3_get_all_touch_modules_enabled`
- `v3_set_touch_module_enabled`、`v3_get_touch_module_enabled`
- `v3_reset_all_touch_pressure`、`v3_reset_touch_pressure`
- `v3_set_touch_data_type`、`v3_get_touch_data_type`，其中 `0=AD value`、
  `1=calibrated value`
- `v3_get_touch_summary`
- `v3_get_touch_module_data`
- `v3_get_all_touch_data`

当前硬件契约如下，运行时会逐模块检查实际返回长度：

| module | 位置 | 数量 |
|---:|---|---:|
| 0 | palm | 36 |
| 1 | thumb tip | 31 |
| 2 | thumb pad | 57 |
| 3 | index tip | 21 |
| 4 | index pad | 52 |
| 5 | middle tip | 21 |
| 6 | middle pad | 52 |
| 7 | ring tip | 21 |
| 8 | ring pad | 52 |
| 9 | little tip | 21 |
| 10 | little pad | 52 |

需要区分两个编号空间：SDK 数组只能确认 `sdk_channel_id = array_index + 1`；v5 布局中的
`official_sensor_id` 是上位机截图标号，仓库现有资料没有证明二者相等。部署通过
`sdk_channel_ids_by_official_sensor_id` 显式重排，并默认把 identity mapping 标成
`verified: false`。

## Revo3 控制与诊断

策略闭环使用：

```python
await ctx.revo3_multi_mit_set_all(
    slave_id, kp_21, kd_21, positions_deg_21,
    velocities_rpm_21, feedforward_ma_21,
)
```

MIT 参数单位为 position degree、velocity rpm、feed-forward mA，`kp/kd` 合法范围
`[0, 10]`。运行时发送前检查 21 维、有限值和增益范围。

其余 Revo3 可调用接口按用途分组如下：

- 单/多关节：`revo3_single_joint_control`、`revo3_multi_joint_control`、
  `revo3_mit_control`、`revo3_multi_mit_set_joint`、`revo3_multi_mit_set_all`
- 分项 MIT：`revo3_set_all_mit_kp`、`revo3_set_all_mit_kd`、
  `revo3_set_all_mit_positions`、`revo3_set_all_mit_velocities`、
  `revo3_set_all_mit_torques`、`revo3_set_all_mit_batch`
- 单关节轨迹：`revo3_move_joint`、`revo3_move_joint_with_speed`、
  `revo3_move_joint_with_gains`、`revo3_move_joint_with_speed_and_gains`
- 全手轨迹：`revo3_move_hand`、`revo3_move_hand_with_speed`、
  `revo3_move_hand_with_gains`、`revo3_move_hand_with_speed_and_gains`
- 示教回放：`revo3_teach_joint`、`revo3_teach_hand`、`revo3_replay_joint`、
  `revo3_replay_hand`
- 分指控制：`revo3_finger_control`、`revo3_thumb_control`、
  `revo3_finger_mit_control`、`revo3_thumb_mit_control`
- 保护与限制：`revo3_set_global_protect_current`、
  `revo3_get_global_protect_current`、
  `revo3_set_joint_protect_current`、`revo3_get_all_joint_protect_currents`、
  `revo3_get_all_joint_position_limits`、`revo3_set_joint_position_limits`、
  `revo3_get_all_joint_speed_limits`、`revo3_set_joint_speed_limits`
- 模式与安全：`v3_set_teaching_mode`、`revo3_set_teaching_mode`、
  `revo3_set_touch_screen`、`revo3_set_software_e_stop`、
  `revo3_set_use_broadcast_id`、`revo3_reset_finger_defaults`
- 系统：`revo3_get_system_status`、`revo3_get_system_current`、
  `revo3_get_system_voltage`、`revo3_get_system_power`、
  `revo3_get_system_temperature`；完整状态内部采样 5 Hz，不应高于 5 Hz 轮询
- 电机诊断：`revo3_get_motor_online_status`、`revo3_get_all_motor_temperatures`、
  `revo3_get_motor_temperature`、`revo3_get_motor_sn`、`revo3_get_all_motor_sns`、
  `revo3_get_motor_fw_versions`、`revo3_get_hardware_version`
- 保留接口：`revo3_admittance_joint`、`revo3_admittance_hand` 在 1.4.5 stub 中明确
  标为尚未实现触觉集成

`DataCollector.new_v3_full(ctx, V3MotorStatusBuffer, V3TouchDataBuffer, ...)` 也可在 SDK
线程中采集并通过 buffer 读取。当前 20 Hz 策略采用直接 bulk read，便于让关节读、触觉读、
推理和命令处于同一个显式控制周期；两次读取是顺序执行，并非硬件时间同步快照。
