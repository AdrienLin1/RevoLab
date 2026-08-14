# hora_screw 网络架构说明（公开 MLP T-S 版）

对应代码：`BrainCo_DexHand/algo/hora/models/models.py`、
`BrainCo_DexHand/algo/hora/padapt/tactile_dagger.py`。

## 训练阶段

| 阶段 | 算法 | 模型 | Checkpoint |
|---|---|---|---|
| Stage1 教师 | PPO | `ActorCritic` + flat `priv_mlp` | `stage1_nn/*.pth` |
| Stage2 学生 | `TactileDAgger` | `TactileStudentPolicy` | `stage2_nn/*.ckpt` |

`scripts/hora/train.py` 在 `--algo ProprioAdapt` 且 `--task` 为触觉任务时自动切换为 `TactileDAgger`。

## Stage1：MLP 教师

- 观测：`obs` 141 维 + `priv_info`
- 特权编码：`env_mlp(priv_info)` → 32 维 extrin，与 `obs` 拼接后进 actor MLP
- screw/valve 与 tactile-rotation agent YAML 均设 `tactile_layout: estimated_official`，由 TacSL 按真实物理节点采样
- `priv_info_dim` 由 env 在运行时计算，`train.py` 会同步到 `ppo.priv_info_dim`

## Stage2：学生策略

- 输入：
  - `student_proprio_hist`：`3 × (21 joint_pos + 21 targets)`
  - `student_tactile_hist`：`10 × frame_dim` 二进制结构触觉历史
- 触觉编码器：
  - `conv1d`：`TactileHistoryEncoder`（默认）
  - `gru`：`TactileHistoryGRUEncoder`（`Revo3HandScrewTactileGRU.yaml`）
- 融合：可选 `GatedTactileFusion` 或 concat
- 蒸馏：仅动作均值；`distill_proj` 将学生 embedding 映射到 `distill_dim`

## 配置对照

| YAML | Layout | Teacher | Student |
|---|---|---|---|
| `Revo3HandScrewTactile.yaml` | estimated_official | mlp | conv1d |
| `Revo3HandScrewTactileGRU.yaml` | estimated_official | mlp | gru |
| `Revo3HandTactileRotate.yaml` | estimated_official | mlp | conv1d |
| `Revo3HandTactileRotateGRU.yaml` | estimated_official | mlp | gru |

## 兼容性

- Stage1 与 Stage2 必须使用同一 `--train_cfg` 族（同一 `priv_info_dim` 与任务掩码）
- 更换 student 编码器类型（conv1d ↔ gru）需重训 Stage2，但可复用同一 Stage1 checkpoint

## 层级主从策略（`--algo HierarchicalPPO`，任务 `valvedriver_tactile_xy`）

对应代码：`algo/hora/ppo/hierarchical_ppo.py`、`hierarchical_experience.py`、
`hierarchical_obs.py`，模型 `models.py::FollowerActorCritic`。

| 角色 | 模型 | 动作 | 输入 |
|---|---|---|---|
| master | 现有 `ActorCritic`（`finger_attention_gru`） | 21 维高斯 | 141 obs + priv_info + 触觉历史 |
| follower | `FollowerActorCritic` `[256,128,64]` | 2 维高斯 | 严格 159 维 |

### 单步主从调用链

```python
master_result   = master.act(obs_t)                        # 一次前向
sampled_hand    = master_result["actions"]
executed_hand   = clamp(sampled_hand, -1, 1)
tactile_latent  = master_result["tactile_latent"].detach()  # [B, 128] GRU hidden
follower_obs    = build_follower_obs(executed_hand, tactile_latent, xy_state_t)  # 159
follower_result = follower.act(follower_obs)
executed_xy     = clamp(follower_result["actions"], -1, 1)
obs_{t+1}, r, done, info = env.step(cat([executed_hand, executed_xy]))  # 唯一一次交互
```

两次前向之间没有任何 `env.step` / physics step / render / 传感器刷新；
触觉编码器每个控制周期只运行一次（latent 直接来自 master 本次前向）。

### follower 观测（159 维）

| 起始 | 维度 | 内容 |
|---|---|---|
| 0 | 21 | `executed_hand_action`（本周期裁剪后动作） |
| 21 | 128 | `tactile_latent`（master 同一次前向的 GRU hidden，detach） |
| 149 | 2 | `xy_position`（按固定 `xy_joint_limit` 归一化） |
| 151 | 2 | `xy_velocity`（按 `xy_velocity_limit` 归一化，截断 ±2） |
| 153 | 2 | `xy_target` |
| 155 | 2 | `previous_xy_action` |
| 157 | 2 | `xy_workspace_margin`（按**当前课程 workspace** 归一化） |

follower actor 不接收：master 的 141 维观测、21 维手指关节角、21 维手指目标、
master actor features、`priv_info`、1170 维原始触觉帧、160 维 attended finger tokens。
follower 的**中心化 critic**可以额外读取 `priv_info` 前 11 维基础特权信息，
但这部分永远不进入 actor。

### 独立性

- master / follower 各自的 PPO ratio、advantage、value、optimizer、学习率、
  entropy 系数、`RunningMeanStd` 与 value normalizer。
- 共享 team reward；两条 GAE 流各自用自己的 value 做 timeout bootstrap。
- follower loss 中的 hand action 与 tactile latent 都是 detach 的存储值，
  梯度不会回传到 master。

### Checkpoint 兼容

| 文件 | 内容 |
|---|---|
| `stage1_nn/*.pth`（原 PPO） | `--master_checkpoint` 严格加载到 master（权重 + 归一化） |
| `hier_nn/{best_reward,best_speed,last}.pth` | `--checkpoint` 完整恢复（含课程状态） |

层级 checkpoint 带 `format: "hora_hierarchical_ppo_v1"` 标记，
两条路径互相拒绝并给出明确报错。原 `PPO.restore_train` 完全未改动。
