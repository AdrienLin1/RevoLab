# Revo3 HORA Screw / Valve 任务

现有默认配置继续使用 **MLP Stage1 教师 + conv1d/GRU Stage2 学生**。此外可通过独立的
`valvedriver_tactile_frame813.yaml` 显式启用新增的结构化 Stage1 Teacher。触觉
screw/valve 任务默认使用 **`tactile_layout: estimated_official`**（真实物理节点分布）。

## 任务

| CLI `--task` | 说明 |
|---|---|
| `nutbolt_tactile` | 三指螺母 |
| `screwdriver_tactile` | 四指螺丝刀 |
| `valvedriver_tactile` | 五指、名义半径 35 mm 阀门 |
| `valvedriver_tactile25` | 五指、名义半径 25 mm 阀门 |
| `valvedriver_tactile_40` | 五指、名义半径 40 mm 阀门 |
| `rotate_ball_tactile` / `rotate_cylinder_tactile` | 连续旋转触觉任务 |
| `valvedriver_tactile_xy` | 五指 35 mm 阀门 + 二维物理平移台（层级主从策略，23 维动作） |
| `valvedriver_tactile_xyyaw` | 同上 + 末端 yaw 旋转关节（层级主从策略，24 维动作） |
| `valvedriver_tactile_yaw` | 只有末端 yaw、无平移的消融任务（层级主从策略，22 维动作） |

## Stage1：MLP Force Oracle

```bash
python scripts/hora/train.py --task valvedriver_tactile \
  --train_cfg Revo3HandScrewTactile \
  --num_envs 4096 --headless \
  --output_name valvedriver_tactile_mlp
```

```bash
python scripts/hora/train.py \
  --task rotate_cylinder_tactile \
  --train_cfg Revo3HandTactileRotate \
  --num_envs 4096 \
  --headless \
  --output_name cylinder_official115
```

**检查训练阶段的表现**

```bash
python scripts/hora/play.py \
  --task valvedriver_tactile25 \
  --train_cfg Revo3HandScrewTactile \
  --checkpoint /home/miao/RevoLab/outputs/hora/revo3_right/valvedriver_tactile25_08112311/stage1_nn/ep_14000_step_0458M_reward_3434.20.pth \
  --num_envs 16 \
  --tactile_gui_vis \
  --tactile_gui_contact_forces \
  --tactile_gui_env_index 0 \
  --log_every 100
```

- 配置：`Revo3HandScrewTactile.yaml`（含 `tactile_layout: estimated_official`）
- 教师：`tactile_encoder.type: mlp`，将 `priv_info` 展平后过 `priv_mlp`
- `ppo.priv_info_dim` 会由 `train.py` 按 env 自动同步（无需手填物理节点维度）
- 输出：`outputs/hora/revo3_right/<run>/stage1_nn/best.pth`

## Stage1：Frame813 结构化触觉 Teacher（新增）

新增配置：

```text
source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/agents/
valvedriver_tactile_frame813.yaml
```

只有配置中的下列显式选择会启用新结构；已有 YAML 的 `mlp` Teacher 行为不变：

```yaml
network:
  tactile_encoder:
    type: finger_attention_gru
```

### 修改概要

- 每帧每个物理节点从原始 10 通道中选取 `b、d、Fn、Ft1、Ft2`，再加入固定物理坐标
  `u、v`，形成 `[u,v,b,d,Fn,Ft1,Ft2]` 七维节点输入。
- 节点坐标来自 `estimated_official` 布局，按每根手指分别减均值并除以最大节点间距；
  坐标注册为模型 buffer，不在 forward 中读取 JSON 或重新计算距离。
- 不使用逐节点共享网络或节点 pooling。每根活动手指将当帧全部节点按物理顺序展平，
  经过彼此不共享参数的整体 MLP：拇指 `217 → 64 → 32`，其余手指
  `147 → 64 → 32`。
- 每帧活动手指 token 经过一层四头 Self-Attention，然后保持全部 token 并展平，
  输入单层、单向、hidden size 128 的十帧 GRU，最终使用 `h_n[-1]`。
- 新 Teacher 的 Actor 输入为“归一化 141 维公开 observation + 原始基础特权切片
  + 128 维触觉 latent”。基础特权不经过 `env_mlp`、Linear、tanh 或 LayerNorm；
  `priv_info` 中的详细触觉尾部不参与新 Teacher forward。
- PPO rollout、minibatch、`train.py --test`、`play.py` 和 TactileDAgger 冻结 Teacher
  均会传递独立的 `tactile_hist`。
- 新旧 Teacher checkpoint 使用严格架构检查；旧 MLP checkpoint 仍须搭配旧 MLP YAML，
  不能用于恢复 Frame813 Teacher。Stage2 Student 的 Conv1d/GRU 输入和 checkpoint 格式未改。
- rotation 任务不再向 `priv_info[:,8]` 写入 object-size 数据，基础特权严格保持 8 维，
  详细触觉从该位置开始并由环境 observation 更新。

### 主要代码位置

- Teacher 编码器、Actor 融合和 checkpoint 校验：
  [`models.py`](source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/models/models.py)
- PPO 历史存储/更新与 DAgger Teacher 标签链路：
  [`ppo.py`](source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/ppo.py)、
  [`tactile_dagger.py`](source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/padapt/tactile_dagger.py)
- 物理节点坐标与环境运行时元数据：
  [`tactile_layout.py`](source/BrainCo_DexHand/BrainCo_DexHand/tasks/tactile_layout.py)、
  [`revo3_hand_screw_tactile_env_cfg.py`](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/revo3_hand_screw_tactile_env_cfg.py)
- 新配置及 train/play 配置发现：
  [`valvedriver_tactile_frame813.yaml`](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/agents/valvedriver_tactile_frame813.yaml)、
  [`train.py`](scripts/hora/train.py)、[`play.py`](scripts/hora/play.py)
- 核心回归测试：
  [`test_hora_finger_attention_gru_teacher.py`](tests/test_hora_finger_attention_gru_teacher.py)、
  [`test_hora_frame813_config.py`](tests/test_hora_frame813_config.py)

### 任务维度

所有任务的公开 observation 都是 `3 × 47 = 141` 维。

| 任务 | 活动手指/节点 | Teacher 触觉帧 | 完整 `priv_info` | Actor 输入 |
|---|---:|---:|---:|---:|
| `nutbolt_tactile` | 3 指，`31+21+21=73` | `73×10+3×4=742` | `11+742=753` | `141+11+128=280` |
| `screwdriver_tactile` | 4 指，`31+21×3=94` | `94×10+4×4=956` | `11+956=967` | `141+11+128=280` |
| 三个 `valvedriver_tactile*` | 5 指，`31+21×4=115` | `115×10+5×4=1170` | `11+1170=1181` | `141+11+128=280` |
| 两个 `rotate_*_tactile` | 5 指，`31+21×4=115` | `115×10+5×4=1170` | `8+1170=1178` | `141+8+128=277` |

五指任务的触觉处理过程例如：

```text
[B,10,1170]
  → 去掉末尾 5×4 finger context
  → [B,10,115,10]
  → 选择五个动态通道并加入 u/v
  → [B,10,115,7]
  → 每指整体 MLP
  → [B,10,5,32]
  → 同帧 Self-Attention
  → [B,10,160]
  → GRU
  → [B,128]
```

`train.py` 和 `play.py` 都会依次搜索 `hora_rotation/agents` 与
`hora_screw/agents`，因此虽然新 YAML 位于 screw 目录，七个目标任务都能使用同一个
`--train_cfg valvedriver_tactile_frame813`。脚本会在创建环境后将
`ppo.priv_info_dim` 同步为任务真实值，并把活动手指、节点数、触觉帧宽和历史长度写入
运行配置。

### Stage-1 训练

新结构必须从头训练，不要传入已有 MLP Teacher 的 `--checkpoint`。

先选择任务和输出名称：

| `TASK` | 建议 `OUTPUT_NAME` |
|---|---|
| `nutbolt_tactile` | `nutbolt_tactile_frame813` |
| `screwdriver_tactile` | `screwdriver_tactile_frame813` |
| `valvedriver_tactile` | `valvedriver35_tactile_frame813` |
| `valvedriver_tactile25` | `valvedriver25_tactile_frame813` |
| `valvedriver_tactile_40` | `valvedriver40_tactile_frame813` |
| `rotate_ball_tactile` | `rotate_ball_tactile_frame813` |
| `rotate_cylinder_tactile` | `rotate_cylinder_tactile_frame813` |

然后执行：

```bash
TASK=nutbolt_tactile
OUTPUT_NAME=nutbolt_tactile_frame813

python scripts/hora/train.py \
  --task "${TASK}" \
  --algo PPO \
  --train_cfg valvedriver_tactile_frame813 \
  --output_name "${OUTPUT_NAME}" \
  --num_envs 4096 \
  --headless
```

将 `TASK` 和 `OUTPUT_NAME` 替换为上表任意一行即可训练其它六个任务。

### Stage-1 checkpoint 测试

可以用训练入口执行确定性测试：

```bash
TASK=nutbolt_tactile
OUTPUT_NAME=nutbolt_tactile_frame813
CHECKPOINT="outputs/hora/revo3_right/${OUTPUT_NAME}/stage1_nn/best.pth"

python scripts/hora/train.py \
  --task "${TASK}" \
  --algo PPO \
  --train_cfg valvedriver_tactile_frame813 \
  --checkpoint "${CHECKPOINT}" \
  --test \
  --num_envs 16 \
  --headless
```

也可以通过独立回放入口测试，并按需去掉 `--headless` 打开 GUI：

```bash
TASK=nutbolt_tactile
OUTPUT_NAME=nutbolt_tactile_frame813
CHECKPOINT="outputs/hora/revo3_right/${OUTPUT_NAME}/stage1_nn/best.pth"

python scripts/hora/play.py \
  --task "${TASK}" \
  --train_cfg valvedriver_tactile_frame813 \
  --checkpoint "${CHECKPOINT}" \
  --num_envs 16 \
  --steps 2000 \
  --headless
```

### Stage2：使用 Frame813 Teacher 生成 DAgger 标签

```bash
TASK=nutbolt_tactile
OUTPUT_NAME=nutbolt_tactile_frame813
CHECKPOINT="outputs/hora/revo3_right/${OUTPUT_NAME}/stage1_nn/best.pth"

python scripts/hora/train.py \
  --task "${TASK}" \
  --algo ProprioAdapt \
  --train_cfg valvedriver_tactile_frame813 \
  --checkpoint "${CHECKPOINT}" \
  --output_name "${OUTPUT_NAME}_student" \
  --num_envs 1024 \
  --headless
```

### 相关回归测试

```bash
PYTHONPATH=source/BrainCo_DexHand PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q \
  tests/test_hora_finger_attention_gru_teacher.py \
  tests/test_hora_frame813_config.py \
  tests/test_tactile_layout.py \
  tests/test_hora_tactile_rotate.py
```

## Stage2：TactileDAgger 学生

```bash
python scripts/hora/train.py --task valvedriver_tactile --algo ProprioAdapt \
  --train_cfg Revo3HandScrewTactile \
  --checkpoint outputs/.../stage1_nn/best.pth \
  --num_envs 1024 --headless \
  --output_name valvedriver_tactile_student
```

- 学生观测：proprio 历史 + 二进制触觉历史
- 默认学生编码器：`student_tactile_encoder.type: conv1d`
- GRU 变体：`--train_cfg Revo3HandScrewTactileGRU`
- 损失：动作均值 DAgger（`MSE(student μ, teacher μ)`）

## 连续旋转任务

Stage1 / Stage2 分别使用：

- `Revo3HandTactileRotate.yaml`（conv1d 学生）
- `Revo3HandTactileRotateGRU.yaml`（GRU 学生）

两者默认使用 `estimated_official` 五指物理布局：拇指 31 个节点，
其余每指 21 个节点，共 115 个触觉节点。

```bash
python scripts/hora/train.py --task rotate_cylinder_tactile \
  --train_cfg Revo3HandTactileRotate --num_envs 4096 --headless
```

## 层级主从策略：灵巧手 + 二维机械臂平移（新增）

用于验证"末端二维平移是否能提高阀门持续旋转速度上限"。原 `valvedriver_tactile`
任务、`PPO`/`ProprioAdapt` 路径、已有 checkpoint 格式全部保持不变。

### 物理资产

`valvedriver_tactile_xy` 在场景克隆前，把两个**真实 prismatic joint** 写进手的
articulation（`Revo3HandScrewTactileXYEnv._author_robot_stage_overrides`）：

```text
world（被资产自带的全局 fixed root joint 固定）
  -> stage_x_joint  (prismatic, 世界 X, 限位 ±0.05 m)
  -> stage_x_carriage      (0.5 kg 刚体, 无碰撞体, disableGravity)
  -> stage_y_joint  (prismatic, 世界 Y, 限位 ±0.05 m)
  -> right_hand_base_link  (y 滑台 / 手掌安装座)
  -> Revo3 手掌与 21 个手指关节
```

- 原来的 `right_hand_base_joint`（world→手掌 fixed weld）被置为 inactive。
- 关节坐标系用手根四元数的逆做局部旋转，因此轴严格对齐**世界 X / 世界 Y**，
  与掌心向下的抓取姿态无关。
- 全流程**不存在** step 期间的 root teleport：`write_root_*_to_sim` 只在 reset 调用
  （与原任务一致），水平运动完全由有限力矩的 PD 驱动产生。
- 滑台使用独立 actuator 组 `xy_stage`（`stage_.*_joint`），不会被 `right_.*` 手指
  actuator 误匹配。

### 动作与观测

```text
action[:, :21]   -> 手指关节（原路径，未改动）
action[:, 21:23] -> XY 滑台（位置目标增量）
```

- master 观测仍为 141 维（3 帧 × (21 关节角 + 21 目标 + 5 接触)）。
- `student_proprio_frame_dim` 仍为 42（校验改用 `finger_action_space=21`）。
- follower 观测严格 159 维：
  `21 executed_hand_action + 128 tactile latent + 2 pos + 2 vel + 2 target +
  2 prev_action + 2 workspace_margin`。

### 训练

```bash
# 从零开始（Stage 0 先训 master，速度 EMA > 0.8 rad/s 连续 5 个 epoch 后自动激活 follower）
python scripts/hora/train.py \
  --task valvedriver_tactile_xy \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xy \
  --output_name valvedriver_xy_hier \
  --num_envs 4096 --headless
```

```bash
# 用已有 21 维 Stage-1 teacher 热启动 master（只加载权重 + 归一化，不是 resume）
python scripts/hora/train.py \
  --task valvedriver_tactile_xy \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xy \
  --master_checkpoint /ABS/PATH/TO/stage1_nn/best.pth \
  --output_name valvedriver_xy_hier_from_master \
  --num_envs 4096 --headless
```

```bash
# 完整层级恢复（模型 + 优化器 + 课程状态）
python scripts/hora/train.py \
  --task valvedriver_tactile_xy \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xy \
  --checkpoint /ABS/PATH/TO/hier_nn/last.pth \
  --output_name valvedriver_xy_hier_resume \
  --num_envs 4096 --headless
```

```bash
# 冒烟（几分钟内跑完 Stage 0 -> Stage 1 -> Stage 2 全部状态机）
python scripts/hora/train.py \
  --task valvedriver_tactile_xy \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xy_smoke \
  --output_name valvedriver_xy_smoke \
  --num_envs 16 --headless
```

```bash
# 回放 / 评估（确定性主从策略，无统计输出，一直跑到手动中断）
python scripts/hora/train.py \
  --task valvedriver_tactile_xy \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xy \
  --checkpoint /ABS/PATH/TO/hier_nn/best_speed.pth \
  --num_envs 16 --test
```

```bash
# 带统计的评估（推荐）：成功率 / 存活时长 / 阀门转角 / 角速度 + XY 平移台指标
python scripts/hora/play.py \
  --task valvedriver_tactile_xy \
  --checkpoint /ABS/PATH/TO/hier_nn/best_speed.pth \
  --num_envs 16 --steps 2000 --headless
```

控制频率 20 Hz、`episode_length_s = 40`，即一个 episode 共 800 个控制步；
`--steps 2000 --num_envs 16` 约合 40 个完整 episode，足够统计成功率。`--steps 0`
则一直跑到手动中断。

`play.py` 从任务名推断 `--algo HierarchicalPPO`（显式写出来也接受）。它按
checkpoint 里锁存的 stage 与 `agent_steps` 还原课程进度，再把对应的 workspace /
action scale 推给环境，因此回放的物理条件与训练结束时一致。结束时除常规
`[SUMMARY]` 外还打印：

- `[XY SUMMARY]`：平移台平均 / 最大位移（mm）、平均速度（mm/s）、目标跟踪误差、
  workspace margin（1 = 中心，0 = 边界）、平均 `|action|` 与动作饱和比例；
- `[XY DIAGNOSTICS]`：环境自己记录的全部 `xy/*`、`xy_penalty/*`、`curriculum/xy*`
  标量在整个回放上的均值。

`play.py` 的触觉可视化（`--tactile_gui_vis`、`--visualize_tactile`）与鲁棒性扰动
（`--tactile_force_scale`、`--tactile_spatial_dropout`、`--tactile_noise_std`）
对该任务同样可用，作用在 master 读到的 teacher 触觉观测上。

输出目录：`outputs/hora/revo3_right/<run>/hier_nn/{best_reward,best_speed,last}.pth`，
TensorBoard 在 `hier_tb/`。`best_speed` 依据**平滑角速度**（`activation_speed_ema`）
选择，不是 episode reward。

### 课程状态机

| Stage | master | follower | XY 动作 | workspace / action scale |
|---|---|---|---|---|
| 0 `stage0_master` | 正常 PPO 训练 | 不采样、不更新 | 恒为 `[0, 0]` | initial（不生效） |
| 1 `stage1_follower` | 权重与输入归一化全部冻结 | 独立 PPO 训练 | 采样 | `xy_curriculum_ramp_steps` 内 1 cm → 5 cm |
| 2 `stage2_joint_finetune`（可选） | actor trunk/head/critic 解冻，触觉编码器仍冻结，lr = follower_lr × 0.07，并对 Stage-1 起始策略加 KL | 继续训练 | 采样 | 继续 ramp |

- Stage 0 → 1 的门限是**每个 rollout 有符号平均角速度的 EMA 严格大于 0.8 rad/s**，
  且连续满足 `activation_patience`（默认 5）个 epoch。**激活后永久锁存**，速度回落
  不会退回 Stage 0。0.8 rad/s 只用于课程触发，**从不作为奖励门控**。
- Stage 1 → 2 由 `hierarchical.joint_finetune_enable` 控制，默认 `false`
  （即纯 follower 消融实验）。

### 公平对比与速度上限实验

1. **公平对比（A）**：默认 `high_speed_reward_enable: false`，奖励与
   `valvedriver_tactile` 完全一致，只额外扣一组很小的 XY 物理代价
   （速度 / 加速度 / 加加速度 / 力 / 功率 / 边界饱和，均归一化到各自上限后加权，
   默认权重 ≤ 0.05）。手指 torque/work 惩罚仍只统计 21 个手指关节，XY 出力单独统计。
2. **速度上限（B）**：在 env cfg 打开 `high_speed_reward_enable=True`。奖励在
   `angvel_clip_max`（4 rad/s，原 rotate reward 饱和点）之上连续线性上升到
   `high_speed_target`，`high_speed_penalty_threshold` 之上再用
   `rotate_penalty_scale` 抑制超速。全程连续、无 0.8 rad/s 跳变。

### 关键日志

`curriculum/hierarchical_stage`、`curriculum/activation_speed_ema`、
`curriculum/activation_patience_counter`、`curriculum/xy_workspace`、
`curriculum/xy_action_scale`、`hierarchical/master_frozen`、
`hierarchical/joint_finetune_enabled`、`screw/angular_velocity`、
`screw/angular_velocity_positive_mean`、`screw/fraction_above_{0_8,1,2,4}`、
`xy/{position_x,position_y,velocity_norm,acceleration_norm,effort_norm,power,
action_saturation_ratio,boundary_saturation_ratio,workspace_utilization}`、
`losses/{master,follower}_{actor,critic}`、`info/{master,follower}_{kl,lr}`。

`screw/fraction_above_*` 也由基础 screw env 记录，所以 baseline 与 hierarchical
两次 run 可以在 TensorBoard 中直接对齐比较。

### 相关回归测试

```bash
PYTHONPATH=source/BrainCo_DexHand python -m pytest -q tests/test_hora_hierarchical_xy.py
```

---

## 层级主从策略：灵巧手 + 末端 XY + yaw（新增）

在 `valvedriver_tactile_xy` 之上再加一个**真实的世界 Z 轴 revolute 关节**，用于验证
"末端增加一个 yaw 自由度能否进一步提高阀门旋转速度"。原 `valvedriver_tactile` 与
`valvedriver_tactile_xy` 两个任务、它们的 checkpoint 格式与行为**完全不变**。

同时提供**只有 yaw、没有平移**的消融任务 `valvedriver_tactile_yaw`（22 维动作，
1 维 follower），用于区分"yaw 本身的贡献"和"XY 平移的贡献"。

### 物理关节链

`Revo3HandYawStageMixin._author_robot_stage_overrides` 在场景克隆前把整条链写进手的
articulation：

```text
valvedriver_tactile_xyyaw（24 维动作）
world（被资产自带的全局 fixed root joint 固定）
  -> stage_x_joint   (prismatic, 世界 X, 限位 ±0.05 m)
  -> stage_x_carriage       (0.5 kg 刚体, 无碰撞体, disableGravity)
  -> stage_y_joint   (prismatic, 世界 Y, 限位 ±0.05 m)
  -> stage_y_carriage       (0.5 kg 刚体, 无碰撞体, disableGravity)   ← 新增
  -> stage_yaw_joint (revolute, 世界 Z, 限位 ±0.70 rad)               ← 新增
  -> right_hand_base_link   (手腕 / 手掌安装座)
  -> Revo3 手掌与 21 个手指关节

valvedriver_tactile_yaw（22 维动作）
world -> stage_yaw_joint (revolute, 世界 Z, ±0.70 rad) -> right_hand_base_link -> ...
```

- yaw 是**有限位、有限力矩的真实 revolute joint**，不是连续无界关节，也不存在
  root teleport（全仓库 `write_root_*_to_sim` / `set_world_poses` 只在 reset 调用）。
- 旋转轴穿过**手腕安装点**（`LocalPos0 = LocalPos1 = (0,0,0)`，body 为 y 滑台 →
  `right_hand_base_link`），**不是**绕阀门中心公转。
- 关节坐标系同样用手根四元数的逆做局部旋转，因此轴严格对齐**世界 Z**。
- **单位**：`UsdPhysics.RevoluteJoint` 的 `lowerLimit/upperLimit` 与
  `PhysxJointAPI.maxJointVelocity` 按 USD 约定以**度 / 度每秒**写入
  （±0.70 rad → ±40.107°，3.0 rad/s → 171.887 °/s），而 Isaac/PhysX 运行时的
  articulation joint state 是**弧度**。环境启动时用
  `yaw_stage.assert_runtime_yaw_limits` 把读回的硬限位与配置里的 rad 值比对，
  漏转换或重复转换都会立刻报错。
- yaw drive 为 angular force drive：`type=force`、`stiffness=0`、`damping=0`、
  `maxForce = yaw_effort_limit`（N·m，角度单位无关）。环境像 XY 一样施加显式的
  effort-limited PD。
- **actuator 分组严格隔离**：XY 组表达式已从 `stage_.*_joint` 收紧为
  `stage_[xy]_joint`，yaw 用独立组 `yaw_stage` + 精确表达式 `stage_yaw_joint`。
  yaw 绝不会继承 XY 的 120 N 线性力上限。
- 所有 DOF 都按 **joint 名**解析（`yaw_stage.resolve_xyyaw_dof_indices`），
  不依赖 articulation 内部排序。
- yaw 被排除在全部手指专用逻辑之外：`action_mask`、reset 关节噪声、
  pose-diff 惩罚、141 维 master 观测、42 维 student proprio frame、
  手指 torque/work 惩罚。

### 动作与观测

```text
action[:, 0:21]  -> 21 个灵巧手关节（原路径，未改动）
action[:, 21:23] -> 机械臂末端世界 X / Y 平移（位置目标增量，米）
action[:, 23:24] -> 机械臂末端 yaw（位置目标增量，弧度）
```

- master 观测仍为 141 维，teacher 触觉帧、`priv_info`、42 维 student proprio frame 均不变，
  因此已有 21 维 Stage-1 teacher checkpoint 仍可通过 `--master_checkpoint` 严格加载。
- follower 观测按 stage DOF 数参数化（`149 + 5 * D`）：

  | 任务 | D | follower 观测 | 环境动作 |
  |---|---|---|---|
  | `valvedriver_tactile_yaw` | 1 | **154** | 22 |
  | `valvedriver_tactile_xy` | 2 | **159**（保持不变） | 23 |
  | `valvedriver_tactile_xyyaw` | 3 | **164** | 24 |

  164 维布局：`21 executed_hand_action + 128 tactile latent + 3 stage_position +
  3 stage_velocity + 3 stage_target + 3 previous_stage_action +
  3 stage_workspace_margin`，每个 3 宽 block 的通道序固定为 `[x, y, yaw]`。
- 2 DOF 任务继续使用原来的 `xy_*` observation key 与 `FOLLOWER_OBS_SPEC`；
  1/3 DOF 任务使用语义中立的 `stage_*` key（yaw 环境同时也发布 `yaw_*` 与 `xy_*`
  诊断通道，所以已有的 `xy/*` 日志全部保留）。
- actor **不读取**任何 privileged 状态；centralized critic 仍只额外读前 11 维
  base privileged info。

### yaw 默认参数

| 配置项 | 默认值 | 单位 |
|---|---|---|
| `yaw_joint_limit` | 0.70 | rad |
| `yaw_workspace_initial` → `final` | 0.15 → 0.60 | rad |
| `yaw_action_scale_initial` → `final` | 0.015 → 0.040 | rad / control-step |
| `yaw_velocity_limit` | 1.2 | rad/s |
| `yaw_acceleration_limit` | 12.0 | rad/s² |
| `yaw_joint_velocity_limit_sim` | 3.0 | rad/s |
| `yaw_pgain` | 8.0 | N·m/rad |
| `yaw_dgain` | 0.5 | N·m·s/rad |
| `yaw_effort_limit` | 0.30 | N·m |
| `yaw_action_smoothing` | 0.5 | — |
| `yaw_use_action_delay` | true | — |
| `yaw_jerk_reference` | 60.0 | rad/s³ |
| `yaw_boundary_margin` | 0.10 | — |

yaw 动作同样是"累计位置目标增量"：

```text
a_s    = (1 - smoothing) * clamp(a, -1, 1) + smoothing * a_s_prev
delta  = clamp(yaw_action_scale * a_s, prev_delta ± yaw_acceleration_limit * dt²)
delta  = clamp(delta, ± yaw_velocity_limit * dt)
target = clamp(prev_target + delta, ± yaw_workspace)
torque = clamp(yaw_pgain * (target - q) - yaw_dgain * qdot, ± yaw_effort_limit)
```

XY 与 yaw 可以共用同一批 per-env action delay 随机样本，但 target、delta、
smoothed action、归一化尺度与 effort buffer 全部独立；**米和弧度从不进入同一个
scale 或 limit**。reset 时 8 个 yaw controller buffer 全部清零，不跨 episode 泄漏。

> 调参提示：`yaw_acceleration_limit * dt² = 0.03 rad`（20 Hz）小于
> `yaw_action_scale_final = 0.040 rad`，因此满量程指令需要 2 个控制步达到稳态增量、
> 反向需要 3 个控制步——这是有意的平滑，不是 clamp bug。另外
> `yaw_effort_limit / yaw_pgain = 37.5 mrad` 就会饱和力矩，配合 0.60 rad 的终态
> workspace 意味着 yaw 在外围区域几乎总是力矩受限；如果首轮训练发现 yaw 长期贴着
> 硬限位、`yaw/tracking_error` 接近 workspace，请优先上调 `yaw_effort_limit`
> 或下调 `yaw_workspace_final`。

### 训练

```bash
# XY + yaw（推荐主实验）
python scripts/hora/train.py \
  --task valvedriver_tactile_xyyaw \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xyyaw \
  --output_name valvedriver_xyyaw_hier \
  --num_envs 2048 --headless
```

```bash
# 只有 yaw 的消融（22 维动作 / 1 维 follower）
python scripts/hora/train.py \
  --task valvedriver_tactile_yaw \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_yaw \
  --output_name valvedriver_yaw_hier \
  --num_envs 2048 --headless
```

```bash
# 用已有 21 维 Stage-1 teacher 热启动 master（只加载权重 + 归一化，不是 resume）
python scripts/hora/train.py \
  --task valvedriver_tactile_xyyaw \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xyyaw \
  --master_checkpoint /ABS/PATH/TO/stage1_nn/best.pth \
  --output_name valvedriver_xyyaw_from_master \
  --num_envs 2048 --headless
```

```bash
# 冒烟（几分钟内跑完 Stage 0 -> 1 -> 2 全部状态机）
python scripts/hora/train.py --task valvedriver_tactile_xyyaw --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xyyaw_smoke \
  --output_name xyyaw_smoke --num_envs 16 --headless
python scripts/hora/train.py --task valvedriver_tactile_yaw --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_yaw_smoke \
  --output_name yaw_smoke --num_envs 16 --headless
```

`--train_cfg` 可以省略：两个新任务会分别自动选用
`valvedriver_tactile_frame813_xyyaw` 与 `valvedriver_tactile_frame813_yaw`。

### 三阶段课程（与 XY 任务同一状态机）

| Stage | master | follower | stage 动作 | workspace / action scale |
|---|---|---|---|---|
| 0 `stage0_master` | 正常 PPO 训练 | 不采样、不更新 | 严格 `[0, 0, 0]` | initial（不生效） |
| 1 `stage1_follower` | 权重与输入归一化全部冻结 | 3 维 `[x, y, yaw]` 一起训练 | 采样 | 同步 ramp（见下） |
| 2 `stage2_joint_finetune` | actor trunk / 21 维 head / critic 解冻，触觉编码器冻结，lr = follower_lr × 0.07，对 Stage-1 起始策略加 KL | 继续训练 | 采样 | 继续 ramp |

- **同步激活**：Stage 0 → 1 只有**一个**锁存点——每 rollout 有符号平均角速度的 EMA
  严格大于 0.8 rad/s 且连续 `activation_patience`（默认 5）个 epoch。XY 与 yaw 在
  **完全相同的 `agent_step`** 激活，结构上不可能分开。激活后永久锁存。
  0.8 rad/s 只用于课程触发，**从不作为奖励门控**。
- **同步课程**：一个无量纲 progress
  `clamp((agent_steps - activation_agent_step) / xy_curriculum_ramp_steps, 0, 1)`
  同时驱动两者，各自插值到自己的单位（默认 ramp = 20 000 000 agent steps）：

  | progress | XY action scale | XY workspace | yaw action scale | yaw workspace |
  |---|---|---|---|---|
  | 0.00 | 0.002 m | 0.01 m | 0.015 rad | 0.15 rad |
  | 0.50 | 0.0035 m | 0.03 m | 0.0275 rad | 0.375 rad |
  | 1.00 | 0.005 m | 0.05 m | 0.040 rad | 0.60 rad |

  "同步"指 progress 与激活时刻同步，**不是**把米和弧度设成同一个数值。
- Stage 1 → 2 由 `hierarchical.joint_finetune_enable` 控制。两个新 YAML 里
  **默认 `true`**（`follower_only_steps: 50000000`），XY 基线 YAML 仍是 `false`。
- Stage 2 中 master 与 follower 在**同一个 rollout** 内各自执行 optimizer step，
  共享 team reward，但保留各自的 PPO ratio、value、optimizer 与 normalizer；
  follower loss 不会反传进 master。

### yaw 物理代价与诊断

yaw 不是免费能源：与 XY 同风格的一组归一化非正代价（默认权重 ≤ 0.05）——
速度 / 加速度 / 加加速度 / 力矩 / 机械功率 / 边界饱和。yaw 力矩**不计入**
手指 torque/work 惩罚（后者只索引 21 个手指的 `actuated_dof_indices`）。

新增 TensorBoard / env extras：

```text
yaw/{position,velocity,target,tracking_error,effort,power,action_abs,
     action_saturation_ratio,boundary_saturation_ratio,workspace_utilization,
     at_positive_limit_ratio,at_negative_limit_ratio,stage_reward}
yaw_cost/{velocity,acceleration,jerk,effort,power,boundary}
yaw_penalty/{velocity,acceleration,jerk,effort,power,boundary}
curriculum/{yaw_workspace,yaw_action_scale,stage_progress}
hierarchical/follower_action_dim
```

全部已有的 `xy/*`、`xy_cost/*`、`xy_penalty/*`、`curriculum/xy_*` 日志保持不变。

### 回放

```bash
python scripts/hora/play.py \
  --task valvedriver_tactile_xyyaw \
  --checkpoint /ABS/PATH/TO/hier_nn/best_speed.pth \
  --num_envs 16 --steps 2000 --headless
```

- 从任务名推断 `--algo HierarchicalPPO`；按 checkpoint 里锁存的 stage 与
  `agent_steps` 还原同步课程，再把 workspace / action scale 推给环境。
- 确定性回放执行完整的 24 维（或 22 维）动作。Stage 0 的 `[x, y, yaw]` 严格为零。
- 结束时分别打印 `[XY SUMMARY]`（mm / mm·s⁻¹）与 `[YAW SUMMARY]`
  （rad、deg、rad·s⁻¹），**yaw 绝不以毫米输出**；`[STAGE DIAGNOSTICS]` 汇总所有
  `xy/*`、`yaw/*`、`*_penalty/*`、`curriculum/*` 标量的回放均值。
- 回放冒烟 checkpoint 时要带上对应的 `--train_cfg ..._smoke`，否则 follower MLP
  宽度不匹配（这是原有的严格加载行为）。

### checkpoint 兼容策略

- checkpoint format marker 未改动（仍是 `hora_hierarchical_ppo_v1`），payload 新增
  可选字段：`master_action_dim`、`follower_action_dim`、`follower_obs_dim`、
  `env_action_dim`、`stage_dof_names`、`stage_curriculum_progress/ramp_steps`。
- 老 checkpoint 缺这些字段时，从 follower 权重形状反推维度，同样能被正确识别。
- **维度不匹配时明确报错**：把 2 维 XY follower checkpoint 交给 3 维任务会得到
  `follower_action_dim 2 != 3 / follower_obs_dim 159 != 164` 的显式错误。
  **没有**权重迁移器，也**绝不**用 `strict=False` 静默半加载。
- 已有的普通 21 维 master checkpoint 仍可通过 `--master_checkpoint` 严格热启动。

### 相关回归测试

```bash
PYTHONPATH=source/BrainCo_DexHand python -m pytest -q \
  tests/test_hora_hierarchical_xy.py \
  tests/test_hora_hierarchical_xyyaw.py
```

USD authoring 测试需要 `pxr` + `PhysxSchema` 绑定；在纯 conda 环境里它们会自动
skip。要在 Isaac Sim 的 USD 库上真正跑起来：

```bash
USDLIB=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*
PHYSXDIR=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.schema.physx-*
PXR_PLUGINPATH_NAME="$PHYSXDIR/plugins/PhysxSchema/resources" \
PYTHONPATH="$USDLIB:$PHYSXDIR:source/BrainCo_DexHand" \
LD_LIBRARY_PATH="$USDLIB/bin:$PHYSXDIR/bin:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
python -m pytest -q tests/test_hora_hierarchical_xyyaw.py
```

## Smoke 配置

| 配置 | 用途 |
|---|---|
| `Revo3HandScrewTactileSmoke` | screw/valve 快速冒烟 |
| `Revo3HandScrewTactileGRUSmoke` | GRU 学生冒烟 |
| `valvedriver_tactile_frame813_xy_smoke` | 层级主从策略冒烟（覆盖三个课程阶段） |
| `valvedriver_tactile_frame813_xyyaw_smoke` | XY + yaw 层级策略冒烟 |
| `valvedriver_tactile_frame813_yaw_smoke` | 只有 yaw 的层级策略冒烟 |

## 回放

```bash
python scripts/hora/play.py --task valvedriver_tactile \
  --checkpoint outputs/.../stage2_nn/model_best.ckpt \
  --train_cfg Revo3HandScrewTactile --num_envs 16
```

层级主从策略见[上文](#层级主从策略灵巧手--二维机械臂平移新增)：

```bash
python scripts/hora/play.py --task valvedriver_tactile_xy \
  --checkpoint outputs/.../hier_nn/best_speed.pth --num_envs 16
python scripts/hora/play.py --task valvedriver_tactile_xyyaw \
  --checkpoint outputs/.../hier_nn/best_speed.pth --num_envs 16
python scripts/hora/play.py --task valvedriver_tactile_yaw \
  --checkpoint outputs/.../hier_nn/best_speed.pth --num_envs 16
```

## 相关文档

- 网络结构：[`Model.md`](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/Model.md)
