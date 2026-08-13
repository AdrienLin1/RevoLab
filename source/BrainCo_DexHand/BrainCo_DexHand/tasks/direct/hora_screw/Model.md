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
