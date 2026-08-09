# Revo3 HORA Screw / Valve 任务

公开版使用 **MLP Stage1 教师 + conv1d/GRU Stage2 学生**，触觉 screw/valve 任务默认 **`tactile_layout: estimated_official`**（真实物理节点分布）。

## 任务

| CLI `--task` | 说明 |
|---|---|
| `nutbolt_tactile` | 三指螺母 |
| `screwdriver_tactile` | 四指螺丝刀 |
| `valvedriver_tactile` | 五指阀门 |
| `valvedriver_tactile_40` | 40° 阀门变体 |
| `rotate_ball_tactile` / `rotate_cylinder_tactile` | 连续旋转触觉任务 |

## Stage1：MLP Force Oracle

```bash
python scripts/hora/train.py --task valvedriver_tactile \
  --train_cfg Revo3HandScrewTactile \
  --num_envs 4096 --headless \
  --output_name valvedriver_tactile_mlp
```

- 配置：`Revo3HandScrewTactile.yaml`（含 `tactile_layout: estimated_official`）
- 教师：`tactile_encoder.type: mlp`，将 `priv_info` 展平后过 `priv_mlp`
- `ppo.priv_info_dim` 会由 `train.py` 按 env 自动同步（无需手填物理节点维度）
- 输出：`outputs/hora/revo3_right/<run>/stage1_nn/best.pth`

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

```bash
python scripts/hora/train.py --task rotate_cylinder_tactile \
  --train_cfg Revo3HandTactileRotate --num_envs 4096 --headless
```

## Smoke 配置

| 配置 | 用途 |
|---|---|
| `Revo3HandScrewTactileSmoke` | screw/valve 快速冒烟 |
| `Revo3HandScrewTactileGRUSmoke` | GRU 学生冒烟 |

## 回放

```bash
python scripts/hora/play.py --task valvedriver_tactile \
  --checkpoint outputs/.../stage2_nn/model_best.ckpt \
  --train_cfg Revo3HandScrewTactile --num_envs 16
```

## 相关文档

- 网络结构：[`Model.md`](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/Model.md)
