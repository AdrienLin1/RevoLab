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
# 回放 / 评估（确定性主从策略）
python scripts/hora/train.py \
  --task valvedriver_tactile_xy \
  --algo HierarchicalPPO \
  --train_cfg valvedriver_tactile_frame813_xy \
  --checkpoint /ABS/PATH/TO/hier_nn/best_speed.pth \
  --num_envs 16 --test
```

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

## Smoke 配置

| 配置 | 用途 |
|---|---|
| `Revo3HandScrewTactileSmoke` | screw/valve 快速冒烟 |
| `Revo3HandScrewTactileGRUSmoke` | GRU 学生冒烟 |
| `valvedriver_tactile_frame813_xy_smoke` | 层级主从策略冒烟（覆盖三个课程阶段） |

## 回放

```bash
python scripts/hora/play.py --task valvedriver_tactile \
  --checkpoint outputs/.../stage2_nn/model_best.ckpt \
  --train_cfg Revo3HandScrewTactile --num_envs 16
```

## 相关文档

- 网络结构：[`Model.md`](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/Model.md)
