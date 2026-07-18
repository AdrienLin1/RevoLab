# RevoHoraNutBolt / RevoHoraScrewDriver 任务移植说明

> 最后更新：2026-07-12。当前版本已包含可视化警告修复、4096 环境 GUI 显存溢出诊断，
> 以及任意正整数 `num_envs` 的 PPO minibatch 自动适配。

本文档记录了将 DEXSCREW 项目（Isaac Gym）中的 `XHandHoraNutBolt` / `XHandHoraScrewDriver`
两个任务移植到 REVOLAB（Isaac Lab）后新增的两个任务：

| 新任务 | 来源任务 | Gym 注册 ID |
|---|---|---|
| RevoHoraNutBolt | XHandHoraNutBolt（三棱柱螺母旋转） | `BrainCo-Direct-Revo3-HoraNutBolt-v0`（别名 `RevoHoraNutBolt-v0`） |
| RevoHoraScrewDriver | XHandHoraScrewDriver（螺丝刀手柄旋转） | `BrainCo-Direct-Revo3-HoraScrewDriver-v0`（别名 `RevoHoraScrewDriver-v0`） |

两个任务均使用 Revo3 右手（21 DOF），完全复用 REVOLAB 已有的 HORA 训练栈
（`BrainCo_DexHand/algo/hora`：Stage1 PPO 教师 + Stage2 ProprioAdapt 学生），
不影响原有 `BrainCo-Direct-Revo3-HoraRotate-Cylinder/Ball-v0` 和
`BrainCo-Dexsuite-Revo3-Right-Lift-v0` 任务（已回归验证注册表完整）。

---

## 一、如何启动训练

### Stage 1（PPO 教师策略）

```bash
# 螺母-螺栓任务（NutBolt）
python scripts/hora/train.py --task nutbolt --num_envs 4096 --headless \
    --output_name nutbolt_teacher

# 螺丝刀任务（ScrewDriver）
python scripts/hora/train.py --task screwdriver --num_envs 4096 --headless \
    --output_name screwdriver_teacher
```

- 在 `brain_co` conda 环境下、仓库根目录执行。
- `--num_envs` 可使用任意正整数。配置中的 `minibatch_size=32768` 是期望上限；脚本会自动选择
  不超过该值、且能整除 `num_envs × horizon(8)` 的最大 minibatch，不丢弃 rollout 样本。
  例如 128/1000/2048/3000/4096 个环境分别得到 1024/8000/16384/24000/32768 的 minibatch。
  较少环境仍可训练，但每轮样本更少、梯度方差更大，训练吞吐和最终收敛效果可能下降。
- RTX 4080（16 GiB）已验证 4096 环境 headless 可完成场景和 PhysX 初始化；更大的环境数取决于
  可用显存。正式训练不要省略 `--headless`，GUI 可视化请使用下方 Viewer 配置。
- 默认训练配置自动选择 `Revo3HandScrew.yaml`（也可用 `--train_cfg` 覆盖）。
- Stage 1 首次使用某个 `output_name` 时保存到
  `outputs/hora/revo3_right/<output_name>/`；如果该目录已经存在，新实验会自动保存到
  `<output_name>_YYYYMMDD_HHMMSS/`，不会覆盖旧 checkpoint、TensorBoard 或配置文件。
  终端会打印本次实际使用的绝对路径。只有显式传入 `--force_overwrite` 才会复用旧目录并允许覆盖。
- 每个 Stage 1 实验目录内，`stage1_nn/` 保存 checkpoint，`stage1_tb/` 保存 TensorBoard 日志。

### 训练配置用途

| 配置 | 用途 | 默认关键参数 |
|---|---|---|
| `Revo3HandScrew.yaml` | 正式 Stage 1/2 训练 | horizon 8，minibatch 上限 32768，1B agent steps |
| `Revo3HandScrewViewer.yaml` | 小规模 GUI 长期训练/观察 | 128 envs，minibatch 1024，1B agent steps |
| `Revo3HandScrewSmoke.yaml` | 快速管线回归 | 128 envs，minibatch 1024，4096 agent steps |

可视化训练示例：

```bash
python scripts/hora/train.py --task nutbolt --train_cfg Revo3HandScrewViewer \
    --num_envs 128 --output_name nutbolt_viewer
```

### Stage 2（ProprioAdapt 学生策略）

```bash
python scripts/hora/train.py --task nutbolt --algo ProprioAdapt --num_envs 4096 --headless \
    --checkpoint outputs/hora/revo3_right/nutbolt_teacher/stage1_nn/best.pth
```

Stage 2 会自动设 `enable_contact_in_obs=False`（actor 观测中触觉清零，
`proprio_hist` 仍保留真实接触历史供 adapt_tconv 蒸馏）。默认输出到 checkpoint 所属实验目录下的
`stage2_runs/YYYYMMDD_HHMMSS/stage2_nn/`；每次蒸馏均使用独立时间戳目录，不覆盖旧学生模型。
显式使用 `--force_overwrite` 时才恢复为直接写入 checkpoint 所属实验目录的 `stage2_nn/`。

也可使用现有包装脚本：`scripts/hora/train_s1.sh` / `train_s2.sh`（追加 `--task nutbolt` 等参数）。

### 可视化回放（play.py）

```bash
python scripts/hora/play.py --task nutbolt \
    --checkpoint outputs/hora/revo3_right/<run_name>/stage1_nn/best.pth \
    --num_envs 16
```

- 打开 Isaac Sim 窗口（相机自动对准 env_0），以确定性策略（mu，不采样）回放 Stage1 checkpoint；
- 终端每 100 步打印滚动统计：每回合奖励/长度、螺母累计转角（rad 与圈数）、实时角速度；
- `--headless --steps 2000` 可做无窗口的纯指标评测；`--task` 同样支持 ball/cylinder/screwdriver；
- 只支持 Stage1 的 `.pth`（Stage2 的 `.ckpt` 会明确报错）。

---

## 二、任务设计（与 DEXSCREW 的对应关系）

### 场景几何（与 DEXSCREW 一致：物体立于地面，手悬浮在上方）

与 DEXSCREW 相同：螺丝物体 **直立固定在地面上**（底座圆盘 r=0.1 贴地、竖直螺杆、
顶端可旋转的三棱柱套筒/螺丝刀手柄），**手悬浮在物体正上方、掌心朝下**，
手指从上方包住套筒并绕竖直 Z 轴拨动它。

Revo3 手的悬浮姿态由已验证的 HORA 圆柱旋转抓握（掌心向上、物体在掌上方 0.135 m）
绕世界 Y 轴翻转 180° 得到：掌心变为朝正下方（-Z），抓握点位于手根下方 0.135 m、
-Y 方向 0.08 m 处，手-物相对抓握几何完全保持。手根位置 = 套筒中心 + (0, 0.08, 0.135)：

- NutBolt：套筒中心 (0, 0, 0.0687) → 手根 (0, 0.08, 0.204)
- ScrewDriver：手柄中心 (0, 0, 0.07) → 手根 (0, 0.08, 0.205)

几何探针实测（直立布局）：初始姿态下指尖即与套筒/手柄保持稳定接触
（螺母净接触力 ≈0.5–0.66 N，拇指/食指到抓握中心距离 0.035–0.051 m），
螺母关节静止无漂移。

### 核心机制逐项对照

| 机制 | DEXSCREW（XHand，12 DOF） | REVOLAB 移植（Revo3，21 DOF） |
|---|---|---|
| 场景布局 | 物体直立固定于地面，手悬浮上方掌心朝下 | 相同（手姿态为已验证抓握绕 Y 轴翻转 180°） |
| 物体 | URDF 关节体，`fix_base_link=True`，1 个被动旋转关节（摩擦 0.2） | 同一 URDF 经 Isaac Lab `UrdfFileCfg` 转换为固定基座 Articulation，被动关节（无驱动，摩擦 0.2） |
| 奖励主项 | `clip(螺母关节速度, -4, 4) × scale`（NutBolt 6.0 / Driver 2.5） | 相同（有限差分计算关节速度） |
| 超速惩罚 | 超过阈值部分 × (-0.3)；Driver 阈值课程 [7.5→15, 30M→60M steps] | 相同（含课程实现） |
| 姿态偏差惩罚 | 拇指关节屏蔽，NutBolt -0.5 / Driver -0.1 | 相同（按关节名含 "thumb" 屏蔽） |
| 力矩/做功惩罚 | torque² 与 (Σ\|τ\|·\|q̇\|)² | 相同 |
| 接近奖励 | 拇指+食指指尖到螺母距离，clamp(1-d/thr)×2.0 | 相同 |
| 手指动作屏蔽 | NutBolt 屏蔽 pinky+ring；Driver 屏蔽 pinky | NutBolt 屏蔽无名指+小指（8 DOF）；Driver 屏蔽小指（4 DOF） |
| 终止条件 | 拇指/食指远离螺母、螺母停滞（70/60 步方差）、螺母无接触、螺杆到上限、超时 800 步 | 全部相同 |
| 随机扰动力 | 每步 25% 概率对螺母施加随机力（2×质量），衰减 0.9 | 相同（`permanent_wrench_composer` 施加于螺母 link） |
| 控制器 PD 增益 | P=3（随机 2.7–3.3）、D=0.01（随机 0.009–0.011），严重欠阻尼以允许快速拨动 | 相同（**不要**用 hora_rotation 的 2.0/0.2——那个 D 约为临界阻尼 2 倍，会抑制 finger-gait，且 Driver 的力矩惩罚权重是按 P=3/D=0.01 的力矩量级调的） |
| 摩擦随机化 | 手和物体设为**同一个** U(0.5, 8.0) 值（橡胶级抓握摩擦） | 相同（**不要**用 hora_rotation 的 手 0.05–0.2 / 物 0.25–1.0——螺母关节摩擦 0.2 Nm 需要 ~10 N 切向力才转得动，低摩擦下物理上不可能，训练必然停在"静止拿接近奖励"的局部最优，表现为回合长度恒等于停滞窗口 70） |
| 其余域随机化 | 质量 0.04–0.06、COM、初始关节噪声 ±10% 半量程 | 相同 |
| 观测 | 3 帧滑窗（关节位置+目标） | 沿用 REVOLAB HORA 格式：3 帧 ×（21 位置+21 目标+5 指尖接触力）=141 维 |
| 特权信息 | 大量可选项（论文配置） | 11 维：螺母位置偏移(3)+摩擦(1)+质量(1)+COM(3)+sin/cos(螺母角)(2)+归一化角速度(1) |
| 重力 | -9.81 | -9.81（物体固定基座，无需重力课程） |

### 有意取舍（与 DEXSCREW 不完全一致处）

1. **指尖距离阈值 0.05 → 0.08**：DEXSCREW 用 URDF 的 `*_tip` 真指尖坐标系；Revo3 USD 只有
   DIP link 原点（距指腹约 2–3 cm，实测静止距离 0.035–0.051），阈值等比放宽，否则每步误触发终止。
2. **点云 z-extent 惩罚（pc_z_dist）未移植**：物体基座固定后点云 z 跨度是常数，
   只会给奖励加常数偏移，对梯度无意义。
3. **物体尺寸随机化（scale list）未移植**：Isaac Lab 中 Articulation 逐 env 缩放不便，
   如需可后续用多资产 spawn 的预缩放 USD 实现。
4. **手腕 20–30° 倾斜随机化、物体 ±5° 倾斜（Driver）暂未移植**：Revo3 手姿态目前固定为
   翻转后的已验证抓握（内含 -25° 倾斜）。如需该项域随机化，可在 `_reset_idx` 中对手根
   姿态叠加随机旋转（Isaac Lab 支持对固定基座写 root pose），是后续增强项。
5. **手根位置**：DEXSCREW 的 XHand 悬浮位置 (0, 0, 0.21) 附带 XHand 专属的物体 XY 偏移；
   Revo3 按"抓握中心 + (0, 0.08, 0.135)"重新推得（NutBolt 0.204 / Driver 0.205），
   与 DEXSCREW 的 0.21 高度基本一致。
6. **ScrewDriver 默认单资产**（8 面手柄 `driver_8.urdf`）；12 面变体 `driver_12.urdf` 已一并
   拷贝，可在 `hora_screw/assets.py` 中切换。
7. **奖励尺度**：新 yaml 中 `reward_scale: 1.0`（DEXSCREW 的 PPO 不缩放奖励；
   REVOLAB 旋转任务用 0.01）。
8. **初始抓握姿态**：在 HORA 圆柱抓握姿态基础上略增指弯（套筒比圆柱细），
   数值在 `hora_screw/assets.py`，是后续调参的首要入口。

---

## 三、对 REVOLAB 的修改清单

### 新增文件

```
assets/urdf/screw/
├── meshes/{tri.stl, screw_upper_8.stl, screw_upper_12.stl}     # 从 dexscrew 拷贝
├── trinut/trinut.urdf                                          # 与 dexscrew 几何一致
└── driver/{driver_8.urdf, driver_12.urdf}                      # 与 dexscrew 几何一致

source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/
├── __init__.py                        # gym 注册 4 个 ID（含 RevoHora* 别名）
├── assets.py                          # 悬浮掌心朝下的手姿态 + 直立螺丝物体 ArticulationCfg（URDF spawn）
├── revo3_hand_screw_env_cfg.py        # Revo3HandScrewEnvCfg + NutBolt/ScrewDriver 两个变体
├── revo3_hand_screw_env.py            # Revo3HandScrewEnv（DirectRLEnv，HORA 兼容接口，含地面）
└── agents/
    ├── __init__.py
    ├── Revo3HandScrew.yaml            # 正式训练配置（priv_info_dim=11, reward_scale=1.0）
    ├── Revo3HandScrewViewer.yaml      # 小规模 GUI 长期训练配置（128 envs）
    └── Revo3HandScrewSmoke.yaml       # 快速冒烟配置（128 envs，4096 agent steps）

REVO_HORA_SCREW_TASKS.md               # 本文档
```

### 修改的现有文件

- `scripts/hora/train.py`
  - `--task` 选项增加 `nutbolt` / `screwdriver`；
  - `--train_cfg` 默认值改为自动选择（ball/cylinder → `Revo3HandHora`，
    nutbolt/screwdriver → `Revo3HandScrew`），显式传参行为不变；
  - 训练 yaml 查找路径扩展到 `hora_screw/agents/`；
  - 按任务选择环境类（`Revo3HandHoraEnv` / `Revo3HandScrewEnv`）。
  - PPO 的 YAML `minibatch_size` 改为期望上限，按实际 `num_envs × horizon` 自动选择最大整除值；
  - 大规模 GUI 训练时输出显存风险提示，并推荐 headless 或 Viewer 配置；
  - 同名 Stage 1 实验自动追加时间戳，Stage 2 使用独立 `stage2_runs/<时间戳>/`，默认不覆盖模型。
- `source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/ppo.py`
  - 保留 minibatch 必须整除 rollout batch 的安全检查；错误由无上下文的 `assert` 改为明确的
    `ValueError`。正常入口会在创建 PPO 前自动解析出合法 minibatch。

---

## 四、冒烟测试记录（2026-07-12，RTX 4080，headless）

```bash
# Stage1 NutBolt：通过（exit 0，3 次 PPO 更新，checkpoint 已保存）
python scripts/hora/train.py --task nutbolt --train_cfg Revo3HandScrewSmoke \
    --num_envs 128 --headless --output_name smoke_nutbolt --force_overwrite

# Stage1 ScrewDriver：通过（exit 0，角速度惩罚课程阈值 7.5 生效）
python scripts/hora/train.py --task screwdriver --train_cfg Revo3HandScrewSmoke \
    --num_envs 128 --headless --output_name smoke_screwdriver --force_overwrite

# Stage2 ProprioAdapt（加载 stage1 checkpoint）：通过（exit 0，stage2_nn/ 已生成）
python scripts/hora/train.py --task nutbolt --algo ProprioAdapt \
    --train_cfg Revo3HandScrewSmoke --num_envs 128 --headless \
    --checkpoint outputs/hora/revo3_right/smoke_nutbolt/stage1_nn/best.pth --force_overwrite
```

验证要点（直立布局最终版）：

- 环境创建、URDF→USD 转换、接触传感器（指尖×5 + 螺母）均正常；
- 物体直立于地面、手悬浮上方掌心朝下；初始姿态下指尖与螺母/手柄保持接触
  （净接触力 0.46–0.66 N），无误触发终止
  （`reset/finger_dist = reset/nut_no_contact = reset/nut_stagnation = 0`）；
- 螺母关节静止无漂移，被动摩擦生效；
- Stage2 加载 stage1 checkpoint 正常，`stage2_nn/` 生成；
- gym 注册表回归：原 Lift / HoraRotate 任务 ID 全部完好。

## 五、待长时间训练验证的事项

冒烟测试只验证“管线能跑”，以下需要按硬件可承载的环境数完成数亿 agent steps 训练才能确认：

1. **策略能否学会持续旋转**（rotation_reward 上升、`screw/angular_position` 持续增长）；
2. **初始抓握姿态与奖励权重是否需要调参**（首要调参入口：`hora_screw/assets.py`
   的关节初始角、`revo3_hand_screw_env_cfg.py` 的 `torque_penalty_scale`
   ——Driver 的 -3.0 是 XHand 力矩量级下调出来的，Revo3 力矩量级不同可能需要减弱）；
3. **停滞/无接触终止的 70/60 步窗口** 在长回合中的触发频率是否合理
   （冒烟每 env 只跑了 32 步，未覆盖该分支的实际触发）；
4. Driver 的 **角速度惩罚课程**（30M→60M steps）只在长训中推进；
5. Stage2 学生策略蒸馏质量。

---

## 六、可视化训练警告修复（2026-07-12）

针对可视化训练启动时的 PhysX / DirectRLEnv 警告，完成以下修复：

1. `render_interval` 从 2 改为与 `decimation=12` 相同。渲染现在每个控制步执行一次，
   物理步长仍为 1/240 s，控制和渲染步长均为 0.05 s，避免一个环境步内重复渲染 6 次。
2. 开启 `enable_external_forces_every_iteration`，并将全局及两个 articulation 的
   `solver_velocity_iteration_count` 从 0 提高到 1。任务会持续向旋转 link 施加随机扰动力，
   此设置可提高 TGS 求解器的速度更新精度。
3. 删除 hand/object spawn 配置中无效的 `collision_props` 覆盖。两个资产本身已有完整碰撞几何；
   该嵌套覆盖无法写入 USD 的 instance prim，原先只产生警告且没有实际生效。
4. 保留 `replicate_physics=False`（质量、COM 和摩擦需要逐环境随机化），关闭不适用于该模式的
   自动碰撞过滤，并在环境克隆后显式调用
   `scene.filter_collisions(global_prim_paths=["/World/ground"])`，隔离不同环境且保留共享地面碰撞。

修复后重新运行 128 环境、4096 agent steps 的 NutBolt GPU 冒烟训练，正常退出（exit 0）。
上述四类警告均已消失；场景报告包含 `Global prim paths: ['/World/ground']`，前三轮训练中
`reset/finger_dist`、`reset/nut_no_contact`、`reset/nut_stagnation` 均为 0，接触行为正常。

### 4096 环境可视化时的 CUDA code 2

RTX 4080（16 GiB）上以 GUI 模式同时渲染 4096 个环境会耗尽显存。Kit 日志中的第一个根错误为：

```text
PxgCudaDeviceMemoryAllocator failed to allocate memory 67108864 bytes! Result = 2
```

后续 `compressContactStage*`、`computeArticulationData: CUDA error, code 2` 和
`Scene state is corrupted` 都是该显存分配失败的连锁错误，并非碰撞缓冲区不足或物理场景发散。

正式训练应关闭 GUI：

```bash
python scripts/hora/train.py --task nutbolt --num_envs 4096 --headless \
    --output_name nutbolt_teacher
```

如需观察训练过程，使用为小批量 GUI 训练新增的长期配置：

```bash
python scripts/hora/train.py --task nutbolt --train_cfg Revo3HandScrewViewer \
    --num_envs 128 --output_name nutbolt_viewer
```

`Revo3HandScrewViewer.yaml` 使用 `128 × horizon(8) = minibatch(1024)`，可以持续训练，
但吞吐量和样本多样性低于正式 headless 配置。实测 4096 环境 headless 可正常完成场景及 PhysX
初始化，128 环境 GUI 可完整运行 PPO；两者均未出现 CUDA/PhysX 错误。

### 非 4096 整数倍环境数

环境数限制来自 PPO minibatch 切分，并非 Isaac Lab 或 PhysX。旧配置固定
`horizon_length=8`、`minibatch_size=32768`，因此要求 `num_envs × 8` 可被 32768 整除。

当前 `train.py` 会将 YAML 中的 minibatch 视作期望上限，运行时选择不超过它、且能整除完整
rollout batch 的最大值。该过程不补零、不截断、不丢弃样本；4096/8192/16384 环境仍使用
32768，保持原训练行为。实测 500 环境配合 Smoke 配置时自动将 minibatch 从 1024 调整为
1000（rollout batch=4000），成功完成数据采集和 PPO 参数更新并正常退出。

### 实验目录防覆盖

默认情况下，同一 `--output_name` 再次启动 Stage 1 时不会复用已有目录，而是自动追加
`YYYYMMDD_HHMMSS`；若同一秒内仍发生重名，则继续追加 `_2`、`_3`。Stage 2 同样为每次运行
创建独立时间戳目录。实测用已有的 `warning_fix_smoke` 名称再次训练后，新 checkpoint 写入
`warning_fix_smoke_20260712_153840/stage1_nn/`，原目录中 `best.pth` 的 SHA-256 前后保持不变。

`--force_overwrite` 是唯一例外：该参数表示用户明确要求复用目标目录，已有同名 checkpoint
可能被替换，因此正式新实验通常不应添加此参数。

---

## 七、TacSL 阵列式触觉任务（2026-07-16 新增）

在不改动原 nutbolt / screwdriver 任务的前提下，参考 `tactile-revo3` 仓库的
TacSL 实现新增两个任务：`nutbolt_tactile`、`screwdriver_tactile`。

```bash
# Stage1 教师训练（触觉阵列只进教师观测 priv_info，actor obs 不变仍为 141 维）
python scripts/hora/train.py --task nutbolt_tactile     --num_envs 16384 --headless
python scripts/hora/train.py --task screwdriver_tactile --num_envs 16384 --headless
# 冒烟：--train_cfg Revo3HandScrewTactileSmoke --num_envs 128
```

设计要点：

1. **手部资产**：换用 `assets/usd/tactile_dexscrew/revo3_right_tactile.usda`
   （引用共享的 `revo3_right.usd`，仅在 5 个 `right_*_tip_Link` 下叠加
   `tactile_elastomer` Xform）。关节/刚体集合与原任务一致，其余配置全部继承。
2. **传感器**：每指一个 TacSL `VisuoTactileSensor`（力场模式，16×16 taxel，
   相机通道关闭），封装类 `hora_screw/tacsl_sensor.py::Revo3VisuoTactileSensor`
   从 tactile-revo3 移植（修复 elastomer 非刚体、平面 gel 网格两处上游假设）。
3. **SDF 碰撞**：TacSL 力场要求接触物体有 SDF 碰撞网格。URDF 导入的
   nut/handle 碰撞网格带有非恒等局部位姿（origin 偏移 + STL 缩放）且被标记
   instanceable，而 TacSL 以刚体系查询 SDF。因此环境在克隆前把 mesh→body
   变换烘焙进新建的碰撞网格 `<body>/tacsl_sdf_collision/mesh`（恒等位姿，
   approximation="sdf"，resolution=256），并禁用原 convexHull 复合碰撞体。
   注意：这使触觉任务中 nut/handle 的碰撞几何从凸包变为精确三角网格。
4. **教师观测**：每指 16×16×(法向+2 切向) 力场经 4×4 平均池化 → 48 维/指，
   5 指共 240 维，乘 `tactile_force_scale=200` 后 clip ±5，写入 priv_info 尾部：
   `priv_info_dim = 11 + 240 = 251`（`Revo3HandScrewTactile.yaml`）。
   监控指标：`tactile/priv_abs_mean`、`tactile/priv_abs_max`。
5. **Stage2 不变**：ProprioAdapt 的 adapt_tconv 仍以 47 维本体历史蒸馏教师的
   extrin 嵌入，教师 env_mlp 输入变宽对学生结构无影响。

冒烟记录（2026-07-16，128 envs，4096 agent steps，均 exit 0）：

- `nutbolt_tactile`：`tactile/priv_abs_max` 随训练增长（0.11→0.22→0.43），
  rotation_reward 为正，checkpoint env_mlp 输入 251 维；
- `screwdriver_tactile`：SDF 替换作用于 handle 网格，触觉特征非零（幅值较小，
  随机策略前 32 步指尖压入 handle 较浅）；
- 原 `nutbolt` 回归：行为不变（env_mlp 输入仍 11 维）。

待长训验证：触觉特征的稀疏度是否足以让教师利用（taxel 在 tip 垫上，
当前抓握姿态主要以 DIP 垫接触；如信号过稀，调参入口为
`revo3_hand_screw_tactile_env_cfg.py` 的 `tactile_force_scale` /
`tactile_array_pool`，或增大 `normal_contact_stiffness`）；SDF 碰撞对
finger-gaiting 动力学的影响（凸包→精确网格，接触更真实但求解更贵）。

---

## 八、五指六棱柱阀门任务 `valvedriver_tactile`

该任务独立复用 screw/tactile 环境框架，不修改已有任务的资产和任务参数：

```bash
# Stage 1
python scripts/hora/train.py --task valvedriver_tactile --num_envs 16384 --headless

# 小规模冒烟
python scripts/hora/train.py --task valvedriver_tactile \
    --train_cfg Revo3HandScrewTactileSmoke --num_envs 128 --headless

# 40 mm 外接圆半径版本；除阀柄半径外配置完全相同
python scripts/hora/train.py --task valvedriver_tactile_40 --num_envs 16384 --headless
```

- 阀柄资产位于 `assets/urdf/screw/vavledriver/`，是外接圆半径 30 mm、
  高 80 mm 的正六棱柱（对边 51.96 mm），通过 `valve_to_shaft` 被动转动关节
  绕竖直轴旋转。碰撞几何使用闭合三角网格，兼容触觉环境的 SDF 替换。
- 灵巧手 21 个关节全部接收策略动作，`masked_action_joint_names=[]`，并保留基类
  的 0.9 关节行程缩放。手根位置调整到 `(0, 0.078, 0.195)`，关节
  初值由已验证的半径 30 mm 圆柱抓握手型微调为五指分布在不同平面上的包裹姿态。
- 旋转、角速度课程、姿态、力矩、功率等权重与 `screwdriver_tactile` 一致。
  唯一奖励调整是接近奖励由拇指/食指平均距离改为五指平均距离，尺度仍为 2.0，
  目的是鼓励整手包裹和多指协同，而不是只形成两指夹持。
- 终止条件调整：两个阀门任务均移除了"拇指或食指指尖离抓取中心超过 0.08 m
  即回合重置"的条件（`enable_finger_dist_reset=False`，其余任务默认仍为 True）。
  停滞、无接触、转轴上限三种重置保持不变；0.08 m 阈值仍用于接近奖励的归一化。
- `valvedriver_tactile_40` 仅把阀柄外接圆半径从 30 mm 放大为 40 mm；手部初始
  位姿、动作与行程范围、奖励、触觉、随机化、终止条件和其余对象参数均不变。
  旧拼写 `vavledriver_tactile` 仍作为 30 mm 任务的兼容别名保留。
