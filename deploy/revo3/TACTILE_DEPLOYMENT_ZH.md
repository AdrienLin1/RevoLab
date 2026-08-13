# Revo3 多任务触觉学生策略真机部署

当前任务注册表支持：

| task | 默认观测手指 | SDK tip module | proprio 单帧 | 触觉单帧 | 动作有效关节 |
| --- | --- | --- | ---: | ---: | ---: |
| `rotate_ball_tactile` | 五指 | 1/3/5/7/9 | 43 | 595 | 21 |
| `rotate_cylinder_tactile` | 五指 | 1/3/5/7/9 | 43 | 595 | 21 |
| `nutbolt_tactile` | thumb/index/middle | 1/3/5 | 42 | 377 | 13 |
| `screwdriver_tactile` | thumb/index/middle/ring | 1/3/5/7 | 42 | 486 | 17 |
| `valvedriver_tactile` | 五指 | 1/3/5/7/9 | 42 | 595 | 21 |
| `valvedriver_tactile25` | 五指 | 1/3/5/7/9 | 42 | 595 | 21 |
| `valvedriver_tactile_40` | 五指 | 1/3/5/7/9 | 42 | 595 | 21 |

历史拼写 `vavledriver_tactile` 以及仓库中对应 Gym ID 会被规范化到同一 canonical task。
其中 `valvedriver_tactile25` 和 `valvedriver_tactile_40` 的数字是阀门名义半径（mm），不是控制频率；七种任务当前均为
20 Hz。

现有可完整导出的实测 checkpoint 示例为：

```text
outputs/hora/revo3_right/run_nutbolt_tactile_20260801_195819/
  stage2_runs/20260802_124312/config_080215.yaml
```

该 checkpoint 的 active fingers 为 `thumb/index/middle`。导出器不会仅凭 task 猜测网络宽度：
task 决定默认观测手指与动作 mask，保存的 `tactile_layout_encoder.finger_names` 和
`student_frame_dim` 决定 checkpoint 的实际输入，二者必须一致。训练时若使用
`--keep_all_tactile_fingers`，允许 nutbolt/screwdriver 读取五指触觉，但动作 mask 仍分别只启用
3 指/4 指。

## 原有 deploy 链路与缺口

原有无触觉链路为：

```text
scripts/export_policy.py --stage stage2
  -> 启动 Isaac Lab，按 task registry 创建 env/runner
  -> load checkpoint
  -> export policy.pt + policy.onnx + policy.yaml
scripts/run_policy.py
  -> sdk_hand_io.read_position_rad
  -> robot_profile: SDK order/offset -> policy order
  -> Stage2InputBuilder
  -> Revo3PolicyRunner/ONNX Runtime
  -> robot_profile: policy target -> SDK order/offset
  -> sdk_hand_io.send_mit_command_rad
```

`Stage2InputBuilder` 的单帧仍是 `[q_norm(21), target_rad(21)]`，保存 30 帧；ONNX 输入为
`obs[B,126]`（最后 3 帧展开）和 `proprio_hist[B,30,42]`，输出 21 维 delta action。

这条 exporter 不能直接导出当前 HORA `TactileStudentPolicy`：学生网络不是 Isaac Lab
RSL-RL policy，checkpoint key 是 `student`，还需要 `proprio_mean_std`，且输入已经变为两个
独立历史张量。新增 `--stage tactile_student`、`tactile_exporter.py`、
`TactileStudentInputBuilder` 和 `Revo3TactilePolicyRunner` 正是补齐这个缺口，没有改变原有
无触觉部署入口。

## 精确输入输出

螺丝/阀门任务的 proprio frame 为 42 维：

```text
q_norm[21] = (2*q - upper - lower) / (upper - lower)
target_rad[21]
```

两类 rotate 任务在末尾追加公开命令，总计 43 维：

```text
q_norm[21], target_rad[21], target_angular_velocity_rad_s[1]
```

rotate 命令的训练随机化范围是 `[0.2, 1.0] rad/s`。部署时该值由
`--target-angvel` 显式给出并固定写入每个 proprio 历史帧；YAML 使用 command v2 schema，
防止把不含命令的 42 维旧模型误用于 rotate task。

这里必须与旧 HORA teacher/adaptation 的 47 维单帧区分。旧单帧为
`21 joint_pos + 21 targets + 5 fingertip contact magnitudes`；触觉蒸馏学生通过
`student_proprio_frame_dim=42/43` 截掉最后 5 维。学生 proprio 中没有原始三维力，原始
`[Fx,Fy,Fz]` 只进入训练期 teacher 触觉分支。真机学生仅使用独立触觉输入中的接触派生量。

学生输入 `student_proprio_hist` 为最近 3 帧，普通任务形状 `[B,3,42]`，rotate 任务形状
`[B,3,43]`。训练时的
`proprio_mean_std` 已封装进 ONNX，真机侧不得再次归一化。

每个物理触觉 node 为 5 维：

```text
[contact, established, released, duration, eta]
```

- `contact` 使用 `threshold_on/threshold_off` 滞回。
- `established/released` 是相邻帧接触边沿。
- `duration` 使用与仿真一致的 log1p 归一化。
- `eta` 是该 taxel 相对接触质心沿平滑质心位移方向的投影。

每个激活手指再追加 4 维：

```text
[shift_x, shift_y, shift_valid, contact_ratio]
```

各任务物理 node 和帧宽为：

```text
nutbolt:     (31 + 21 + 21)           * 5 + 3 * 4 = 377
screwdriver: (31 + 21 + 21 + 21)      * 5 + 4 * 4 = 486
valve:       (31 + 21 + 21 + 21 + 21) * 5 + 5 * 4 = 595
rotate:      (31 + 21 + 21 + 21 + 21) * 5 + 5 * 4 = 595
```

`student_tactile_hist` 形状为 `[B,10,frame_dim]`。reset 时 proprio 和 tactile 均重复首帧，
与当前仿真 `at_reset_env_ids` 逻辑一致。结构触觉训练时不做幅值归一化。

模型始终输出 `action[B,21]`，ONNX 内先 clamp 到 `[-1,1]`，再使用 task 对应的
13/17/21 维有效 action mask。目标更新为：

```text
target = clamp(previous_target + (1/24) * action * action_mask, joint_limits)
```

随后从 policy joint order 变换到 SDK joint order，加 `sim2real_joint_offset`，rad 转 degree，
通过 21 关节 Multi-MIT 写入。

## 控制周期

```text
v3_get_motor_status_data
  -> SDK 21 关节 degree -> rad -> policy order -> 减 offset
v3_get_all_touch_data
  -> 按 task + checkpoint 契约取 tip modules 1/3/5[/7/9]
  -> SDK channel 映射到图纸 sensor ID 顺序
  -> 滞回/边沿/duration/质心 shift/eta
  -> 更新 3 帧 proprio 与 10 帧 tactile history
ONNX Runtime
  -> 21 维 action -> mask -> delta target -> joint limits
  -> policy order -> SDK order -> 加 offset
revo3_multi_mit_set_all
```

关节和触觉是同一软件循环内的顺序读取，不是硬件同步采样。触觉使用一次 bulk read，避免
逐模块读取造成模块间更大的时间偏移。

## 按 task 导出 checkpoint

在包含 PyTorch、ONNX 和本仓库 HORA 代码的 `revolab` 环境中执行：

```bash
cd /home/bingchenzhou/RevoLab
python deploy/revo3/scripts/export_policy.py \
  --stage tactile_student \
  --task nutbolt_tactile \
  --checkpoint outputs/hora/revo3_right/run_nutbolt_tactile_20260801_195819/stage2_runs/20260802_124312/stage2_nn/model_best.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --output-dir deploy/revo3/artifacts/nutbolt_tactile
```

`--config` 可省略：导出器会在 checkpoint 的 stage2 run 目录选择最新的
`config_*.yaml`；新式配置存在 `env_runtime.task` 时会优先筛选与 `--task` 一致的文件。需要
固定配置时仍可显式传入。将 `--task` 换成表中任一任务即可按对应手指、module、帧宽、
20 Hz 控制周期和动作 mask 校验；不能把 nutbolt checkpoint 标成 screwdriver 来导出。

导出器重建 `TactileStudentPolicy`、strict load `checkpoint['student']` 和
`checkpoint['proprio_mean_std']`，替换 opset 17 不支持的 RMSNorm 实现，并在 tracing 时关闭
PyTorch MHA fastpath。输出是 `policy.onnx` 和含完整硬件契约的 `policy.yaml`。
ONNX 首先写入临时文件，并以固定 seed 对照 PyTorch CPU 与 ONNX Runtime CPU 输出；只有
最大绝对误差不超过 `1e-4` 才会原子替换最终模型。`policy.yaml.verification` 保存实测误差，
`artifacts.onnx_sha256` 绑定具体 ONNX 文件，防止把 valve/valve40 或不同训练轮次的 ONNX 与
错误 YAML 混用。

完成通道标定后增加：

```bash
  --sensor-map deploy/revo3/config/tactile_sensor_map_calibrated.yaml
```

映射格式参考五指全集模板 `config/tactile_sensor_map_template.yaml`。列表下标是
`official_sensor_id-1`，列表值是从 1 开始的 SDK channel ID，必须是完整排列。

## 读取真机逐帧数据

先安装硬件依赖：

```bash
cd deploy/revo3
pip install -e ".[hardware]"
```

读取 21 个关节和五个 tip 模块的每个 SDK channel：

```bash
python scripts/inspect_hand.py \
  --profile config/revo3_right.yaml \
  --port /dev/ttyUSB0 \
  --slave-id 126 \
  --touch-data-type 1 \
  --module-ids 1,3,5,7,9 \
  --rate 20 \
  --frames 100 \
  --format jsonl
```

每帧含 `timestamp_ns`、SDK order 的 21 个 `rad/deg` 关节值，以及每个模块的
`sdk_channel_id/value`。不传 `--port` 时按 profile 允许 SDK auto-detect。

不要把 `sdk_channel_id` 直接称为图纸 `official_sensor_id`。应逐个轻压已知图纸位置，记录
唯一显著变化的 SDK channel，完成双射后把 sensor map 的 `verified` 改为 `true`。映射未验证
时 runtime 允许 dry-run，但默认禁止发送电机命令。

## 阈值标定

SDK 返回的是设备触觉数值，不是牛顿制物理力，接触时可达到数千。当前真机默认使用
`touch_data_type=1`、`threshold_on=150`、`threshold_off=100`。每个 taxel 都满足
`threshold_off < threshold_on`；位于 100 到 150 之间时保持上一帧接触状态。仿真的
`0.001/0.0005` 属于 TacSL 缩放力值域，不能用于真机 SDK 数据。

快速测试可用全局阈值：

```bash
python scripts/run_tactile_policy.py \
  --task nutbolt_tactile \
  --onnx artifacts/nutbolt_tactile/policy.onnx \
  --policy artifacts/nutbolt_tactile/policy.yaml \
  --profile config/revo3_right.yaml \
  --touch-data-type 1 \
  --sensor-log-every 1 \
  --dry-run
```

上述命令不传阈值时自动使用 `150/100`。仍可用
`--touch-threshold-on/--touch-threshold-off` 显式覆盖；若选择 `touch_data_type=0`，必须提供
对应 AD 值阈值。

rotate 策略还必须给出训练范围内的目标速度：

```bash
python scripts/run_tactile_policy.py \
  --task rotate_ball_tactile \
  --onnx artifacts/rotate_ball_tactile/policy.onnx \
  --policy artifacts/rotate_ball_tactile/policy.yaml \
  --target-angvel 0.6 \
  --dry-run
```

正式标定使用 `config/tactile_thresholds_template.yaml`，按图纸 sensor ID 顺序填入逐点值：

```bash
  --touch-thresholds config/tactile_thresholds_calibrated.yaml
```

runtime 会先核对传入 task 与 `policy.yaml.task_contract`，再核对 layout version、data type、
module ID、观测手指、每指长度、帧宽、控制频率、action mask、导出数值验证和 ONNX SHA-256，
最后才创建 ONNX session 并连接真机。

旧配置若没有 `estimated_official` 的 `finger_names/student_frame_dim`，说明对应 checkpoint 不是
当前物理 node GNN 输入契约，不能仅靠 task 名安全恢复，应使用当前配置重新训练 Stage2。

## 上真机前的边界

1. 先运行 `inspect_hand.py`，确认 SDK 版本、21 关节顺序、模块 enable mask 和数组长度。
2. 完成 SDK channel 到图纸 sensor ID 的物理激励标定并重新导出。
3. 完成逐 sensor on/off threshold 标定。
4. 使用 `--dry-run --sensor-log-every 1` 检查 action、target 和触觉变化。
5. 核对 profile 中 joint limits、offset、`kp/kd`、slave ID 和 baudrate。
6. 最后移除 `--dry-run`；未验证 sensor map 时程序会拒绝发送命令。

当前代码已验证模型导出与 ONNX Runtime 推理，但本开发环境没有连接目标真手，所以无法替代
步骤 1 至 3 的硬件确认。v5 几何本身仍是低置信度截图估计，不是毫米级实测布局。
