# ICLR 2027 论文方案（三）：任务导向 · 代码核实版

> 日期：2026-09-05  ｜ 目标：Applications to Robotics, Autonomy, Planning
> 关系：本文取代 [paper_frame1.md](paper_frame1.md)（接口导向）与 [paper_frame2_task_driven.md](paper_frame2_task_driven.md)（任务导向初版）。
> 与前两版的区别：**本版核对过代码与 deploy 链路**，并据此把"可写的 claim"与"必须先补的工程"分开。
> 已确认的事实基线（作者答复，2026-09-05）：真机阀门成功 = **Stage-2 结构触觉 student，手部单独完成**；
> **follower 尚未上真机**，但计划在截稿前接上；资源为**多卡并行 + 真机基本可用**。

---

## 0. 三句话结论

1. **不需要多种真机灵巧手。** 是 `paper_frame1` 的措辞（interface / skill-preserving / 跨构型）自动生成了跨形态证据负担，而不是会议要求；而且 n=2 也买不到"通用接口"这个结论。
2. **单一硬件不是拒稿点，claim–证据错位才是。** 你当前最脆的地方不是"只有一只手"，是**"arm"在仿真里其实是 3 自由度有限力滑台**，以及 **follower 目前挂在 teacher latent 上、整栈不可部署**。
3. **采用任务导向，但必须附一条可证伪的知识结论**，否则会被读成系统 demo。推荐的结论是"接触结构充分性"（§3），它同时统一了触觉工作与主从工作，且能脱离阀门本身。

---

## 1. 回答两个问题

### 1.1 是否必须在真机上验证多种触觉灵巧手？

**不必须，而且不划算。**

证据负担来自 claim 的措辞。`paper_frame1.md` 里三个词各自触发一份跨形态实验：

| frame1 措辞 | 审稿人读到的承诺 | 触发的实验 |
|---|---|---|
| Touch as a **Coordination Interface** | 可替换、可插拔 | ≥2 种手 + 未见形态迁移协议 |
| **Skill-Preserving** expansion | 通用性质而非本例现象 | 重新锁臂回测 + KL 漂移 + 重适应样本 |
| 跨任务/跨构型**适用性**（C3） | 方法级泛化 | 3 任务族 × 多构型 |

关键在于：**做到 n=2 也不能写 interface**，审稿人只会说"两种手不叫通用"。所以第二只手花掉的时间，换回的 claim 增量约等于零。

低成本替代品（都在现有代码能做，攻击的是同一个"是不是只对你这套硬件有效"的疑虑）：

1. **传感退化扫描**：taxel 增益 ×[0.5,2]、偏置、随机死点 5–30%、阈值失配、额外延迟。**这是本文最有说服力的一条**，因为 student 本来就不看力幅值，理论上应该扛得住，而 force-based student 应该崩。
2. **触觉布局消融**：仿真里减少节点数 / 改分指划分后重训，检验结论是否依赖 115 节点这个具体布局。
3. **同一手上的第二类接触任务**：`hora_screw` 里 nut-bolt / screwdriver 环境与 deploy 注册项都已存在。

### 1.2 审稿人会不会认为单手撑不起创新点？

会问外部有效性，但单平台在 robotics application 方向是常态。真正致命的是下面五条，都与"几只手"无关：

1. **claim–证据错位**（说 interface，给一个装置）；
2. **缺机制证据**——"多给 3 个自由度当然更快"，没有接口消融这句话就足以打掉主贡献；
3. **弱基线**——缺同预算 warm-start joint PPO、缺规则式腕部反馈；
4. **真机 = 剪辑视频**，没有重复次数、没有保留条件；
5. **近邻工作没正面交代**：DexTouch（UR5e+Allegro+二值触觉+真机阀门）、DexScrew、AnyRotate。

**而按代码核实，当前最大的风险其实在"臂"这一侧**（详见 §4）。把辩护资源花在这里，比花在第二只手上高效得多。

---

## 2. 为什么任务导向更适合这个 track

| 维度 | frame1：接口导向 | 本方案：任务导向 + 接触结构结论 |
|---|---|---|
| 中心问题 | 如何串联特权学习 / 蒸馏 / 动作空间扩展 | 受约束持续旋转为什么难，需要什么信息才能做成 |
| 每个设计的辩护方式 | "这是框架的一环" | "因为换指瞬间力矩传递会断" —— 物理动机直接给出 |
| 已有真机成果的位置 | 结尾一段 demo | **主证据** |
| 实验负担 | 6 个 RQ + 3 任务族 + 技能保持 + 跨任务复用（估计 2–3 个月） | 4 个 RQ，20 天可闭环 |
| 需要先补的代码 | latent 蒸馏、技能保持回测、跨任务复用 | 只有一条：student → follower（且它同时是真机上臂的前置条件） |
| 主要风险 | claim 撑不住 | 被读成系统 demo |

**取任务导向。** 但风险要用 §3 的可证伪结论来消，而不是靠加实验量。

frame1 的技术资产**不丢，只是降级为 Method 章节的实现**：Frame813 编码器、容量感知协作奖励、三阶段课程、KL 锚定，全部保留；被删掉的只是"它们构成一个通用接口"这个说法。

---

## 3. 中心论点：接触结构充分性

这是把两份技术文档缝成一篇的接缝，也是脱离阀门后仍然成立的知识结论。

> **在受约束持续旋转中，驱动附加末端自由度所需要的手部信息，不是物体位姿、也不是接触力幅值，而是接触结构（接触通断事件、持续时间、接触斑块迁移）加上手部当前动作。这个变量同时是（i）多指换指的稠密 credit 来源、（ii）臂手协调的通信通道、（iii）唯一无需力标定因而可部署的信号。**

一个变量，三种角色——这正好对应你两份文档里已经存在的三块东西：

```
接触结构 ──(i) 容量感知协作奖励 H/G/q/S/E ──> 换指步态涌现        [tactie-enhancedRL §7]
         ──(ii) 128-D latent 条件化 follower ──> 末端补偿与换指同步 [master-slave1 §7]
         ──(iii) 去力幅值的 595 维 student ──> 真机免标定部署       [已在真机验证 ✔]
```

**为什么阀门任务能推出这个结论**：净转角的推进依赖一个不断变化的接触集合传递轴向力矩。在任一时刻，决定"下一步该怎么补偿"的不是当前抓得多紧，而是**哪些接触正在建立、哪些正在消失**——因为补偿动作要与换指的释放窗口对齐。力幅值描述"现在多用力"，接触结构描述"接触集合正在怎么变"，后者才是协调所需的量。

**这条结论是可证伪的**，反例很明确（见 §7 E2 的判定条件）：如果 follower 换成"零向量 / 原始力帧 / 手部本体观测"后性能无差，接缝断裂，全文必须换 claim。

### 标题建议

主推（假设真机臂手落地）：
> **Contact Structure Is Enough: Tactile Arm–Hand Coordination for Sustained Constrained Rotation**

保守版（真机仍为手部单独）：
> **Feeling the Handover: Contact-Structure Feedback for Sustained Valve Turning with a Tactile Hand**

不使用：universal、morphology-agnostic、skill-preserving、plug-and-play、first。

---

## 4. 系统事实核对（本次代码核实结果）

**这一节的每一条都必须在论文里按事实写，否则是可被直接抓住的失分点。**

| # | 事实 | 出处 | 对论文的约束 |
|---|---|---|---|
| F1 | 仿真"机械臂"= 世界系 X/Y 平动 + yaw 转动的**有限力 prismatic/revolute 关节**，共 3 DOF，无 IK、无臂体动力学、无冗余 | [xy_stage.py](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/xy_stage.py)、[yaw_stage.py](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/yaw_stage.py) | 全文不得出现无限定的 "robot arm"。仿真侧统一写 **force-limited task-space end-effector DoF**；只有真机执行在真臂上时，才写"经任务空间控制器在真实 6-DOF 臂上执行" |
| F2 | follower 观测里的 128-D latent **来自 master(teacher) 本次前向的 GRU 隐状态**，`validate_tactile_latent` 强制要求 teacher 提供 | [hierarchical_obs.py:18,152](source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/hierarchical_obs.py#L18) | **现有主从整栈不可部署。** 这是唯一的关键路径工程（§6） |
| F3 | deploy 链路是**纯手部** student：595 维结构触觉（无力幅值）+ proprio，21 维手指动作，20 Hz；注册表 7 个任务，无任何臂/follower | [TACTILE_DEPLOYMENT_ZH.md](deploy/revo3/TACTILE_DEPLOYMENT_ZH.md)、[deploy/revo3/revo3_deploy/](deploy/revo3/revo3_deploy/) | 真机已证据 = 手部 student 完成阀门。这条**本身就是强结果**，要单独成表 |
| F4 | teacher 通道 `[b, duration, Fn, Ft1, Ft2]`（丢 on/off/eta），student 通道 `[b, on, off, duration, eta]`（丢力）——**两者不是嵌套关系** | tactie-enhancedRL §4.3/§4.4 | 审稿人必问"teacher 都不用的事件通道，凭什么 student 靠它复现 teacher 行为"。要么附录给理由（teacher 有力幅值可隐式推断事件），要么补一个"teacher 也加 on/off/eta"的对照 |
| F5 | 工作区把 `joint_finetune_enable` 从 false 改成 **true** | [valvedriver_tactile_frame813_xy.yaml](source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/agents/valvedriver_tactile_frame813_xy.yaml) 未提交改动 | 每个结果必须记录实际使用的配置；Stage-2 联合微调开/关是两组不同实验，不能混报 |
| F6 | 双 PPO：独立 ratio、独立 GAE、follower 输入 detach、共享 team reward | master-slave1 §9 | 准确称 **independent/coordinate PPO with shared team reward**，不声称等价于联合 log-prob 的 PPO，不声称新的单调改进保证 |
| F7 | 存在 high-speed reward 开关 | master-slave1 §12.4 | 公平比较时全部关闭，并在附录声明 |
| F8 | 已有 `detect_handover_events` / phase 标注 / attention 汇总工具 | [algo/hora/analysis/signals.py](source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/analysis/signals.py) | 机制图（contact raster、handover 对齐）**不需要新写分析代码**，只需在主实验 run 里 dump 接触时间序列 |
| F9 | 仓库已有 UR5e / 天机 / 双臂 Revo3 资产 | [ur5e_revo3_right.py](source/BrainCo_DexHand/BrainCo_DexHand/assets/ur5e_revo3_right.py)、[tianji_revo3_right.py](source/BrainCo_DexHand/BrainCo_DexHand/assets/tianji_revo3_right.py) | 真臂仿真集成有现成资产，是 P1 而非 P2 |
| F10 | GRU student 自带隐状态出口（`hidden_dim` 256），Conv1d student 有 trunk 特征 | [models.py:786+](source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/models/models.py#L786) | **student latent 不需要新设计**，接一个投影层到 follower 即可 |

---

## 5. 问题定义：sustained constrained rotation

### 5.1 术语与范围

**Sustained constrained rotational manipulation（持续受约束旋转操作）**，由三条性质界定：

- 物体受环境约束（绕固定轴），不是自由物体重定向；
- 持续进展需克服外部阻力，必须**持续传递有效轴向力矩**；
- 手指行程有限，长时间操作必然耗尽运动范围，**必须发生接触交接**。

阀门是代表实例。**不要**未经验证地推广到插入、装配或"全部接触操作"。

### 5.2 物理动机（Intro 用，半页）

绕轴净转角由有效轴向力矩驱动：

$$
I\ddot{\theta} = \tau_{\mathrm{drive}} - \tau_{\mathrm{resist}},\qquad
\tau_{\mathrm{drive}} = \sum_i \hat{\mathbf e}^{\top}\big((\mathbf p_i - \mathbf p_{\mathrm{axis}})\times \mathbf f_i\big)
$$

粘着约束 $\|\mathbf f_{t,i}\| \le \mu_i f_{n,i}$ 与关节限位 $q_{\min}\le q \le q_{\max}$ 同时成立。**矛盾就在这里**：增大 $f_{n,i}$ 可提高可用切向力，但会锁死手指位形、加速行程耗尽；释放手指恢复行程，$\tau_{\mathrm{drive}}$ 立刻掉一块，且释放期间容易整体失去有效接触。

这是简化模型，用于解释设计动机，不是新理论，也不保证无滑移。

### 5.3 五个可检验的难点（Intro 的骨架）

| 难点 | 现象 | 本文机制 | 需要的证据 |
|---|---|---|---|
| D1 有效力矩不足 | 有接触但转动停滞 | 容量感知负载分担 H | 负载扫描、净转角、有效轴向力矩 |
| D2 接触不稳定 | 反转、失去有效接触 | 结构触觉历史反馈 | 同历史长度的无触觉/粗触觉对照 |
| D3 行程逐渐耗尽 | 关节饱和、有效力臂下降 | 末端补偿 + 换指 | 饱和率、力臂、workspace 利用率 |
| D4 **交接时进展中断** | 放开后抓不回，长时间停顿 | **动作+触觉条件化的补偿（本文核心）** | 交接成功率、恢复时间、末端动作与交接的时序对齐 |
| D5 训练/部署信号不一致 | 仿真成功真机不稳 | 去力幅值结构触觉 + 随机化 | 标定漂移扫描、真机重复试验 |

用词纪律：把"不打滑、不丢接触"改写成**"限制非期望滑移与整体失去有效接触，同时允许换指所需的局部释放与受控滑动"**。区分 contact handover / finger gaiting（部分手指通断）与 regrasping（全手重新建立抓握）——只观察到 $b_i$ 开关，只能叫前者。

---

## 6. 关键路径工程：student → follower（**必须最先做**）

这一项同时解决三件事，是全篇的瓶颈：

1. 它是**真机上臂的前置条件**（F2/F3：现在的 follower 需要 teacher latent，真机没有）；
2. 它让仿真实验与部署系统**是同一个系统**（否则 Method 图与真机图不一致，是硬伤）；
3. 它本身是 §7 的一组实验（teacher-latent follower vs student-latent follower）。

**推荐路线（frame2 的"路线一"，最省事）**：

```
Stage 0  触觉 teacher (Frame813, 特权)                       [已有]
Stage 0.5 TactileDAgger → 结构触觉 student                    [已有，真机已验证]
Stage 1' 冻结 student 作为手部策略，follower 条件于
         (student latent z_S, student 手部动作 a_H, 末端状态)  [需要新增]
Stage 2' 可选：KL 锚定联合微调                                [已有开关 F5]
```

实现要点：

- **latent 出口**：GRU student 直接取最后隐状态（256-D），Conv1d student 取 trunk 输出，加一层 `Linear→128` 与现有 follower 输入维度对齐；或直接把 follower 输入维改成 256。**不要**去做 teacher latent 对齐蒸馏——follower 从头就基于 $z_S$ 训练，就不存在"语义等价"问题。
- **分布问题**：student 手部策略在**末端会动**的条件下，状态分布与 Stage-0.5 训练时不同。要么在 Stage 1' 里允许 student 继续少量适应（记录为联合微调），要么冻结并明确报告手部性能变化。**两种都要在论文里说清用了哪种。**
- **验证**：Stage 1' 的手部+末端组合性能 vs 现有 teacher-latent 版本。若掉点，报告掉多少——这是诚实的 sim2real 代价，不是失败。

**工时估计**：latent 出口 + 接口对齐半天到一天；follower 重训（复用冻结手部 checkpoint，只训 follower）应与现有 Stage 1 同量级。

---

## 7. 实验矩阵（按多卡 + 20 天裁剪）

**排期核心杠杆：所有 follower 侧实验共享同一个冻结手部 checkpoint，只训 follower，成本远低于 teacher 侧重训。**
所以优先级是：**承载中心 claim 的 follower 侧消融（便宜）> teacher 侧触觉消融（贵）**。

### P0 —— 缺任何一组论文不成立

| ID | 要证明 | 设置 | 主指标 | 判定失败 |
|---|---|---|---|---|
| **E1** 主结果 | 方法能可靠完成持续旋转 | ① 触觉 hand-only（末端固定）② 规则式末端反馈 ③ flat joint PPO from scratch ④ **同 checkpoint warm-start 的 flat joint PPO** ⑤ 无通信双策略 ⑥ ours。全部 3–5 seed，同奖励、同随机化、同调参预算，high-speed reward 关闭 | 成功率、固定预算净转角 $\Delta\theta_{\mathrm{net}}$、首次失去持续能力的时间 | ④或⑤与⑥无差 → 协调结构不是贡献 |
| **E2** 接口消融（**money experiment**） | 接触结构充分性 | follower 输入 ∈ {末端状态；+手部动作；+触觉 latent；+两者(ours)；+原始触觉帧；+手部 141 维公开观测}。共享手部 checkpoint、同预算，各自重训 | 净转角提升、follower 收敛步数、**交接成功率**、workspace 利用率 | 后两个对照与 ours 无差 → 中心 claim 断裂，必须换 claim（见 §12 预案） |
| **E3** 末端自由度分解 | 收益不是"随便动动就行" | DOF ∈ {0, yaw, xy, xy+yaw} | 净转角增量、末端 effort/power、饱和比例 | yaw-only 与 xy 收益相同 → 需解释，否则被判为"加自由度红利" |
| **E4** 可部署整栈 | 仿真结论在可部署侧成立 | teacher-latent follower vs **student-latent follower**（§6） | 同 E1 指标 + 手部性能变化 | student 版完全不工作 → 真机臂手取消，退回保守标题 |
| **E5** 真机手部（**已有**） | 仅仿真训练即可部署 | 冻结 student，名义 + 保留阻力 + 未见几何/轴线偏差，多模型 × 多次重复 | 成功率区间、净转角、接触丢失、失败分类 | — |
| **E6** 标定漂移 | 去力幅值换来鲁棒而非损失 | structural student vs force-based student vs teacher，扫描增益 ×[0.5,2]、偏置、死点 5–30%、阈值失配、延迟 | 扰动–性能曲线、崩溃点 | force 版也不崩 → C 降级为"力幅值非必要"的弱主张 |

### P1 —— 决定分数 5→7

| ID | 内容 |
|---|---|
| E7 | **真机臂手**（见 §8，含 go/no-go） |
| E8 | 触觉奖励消融：去 $r_{\mathrm{coord}}$ / 去 handover $G$ / 去负载分担 $H$ / 去容量归一化 / 去 16 步窗；配 contact raster 图 |
| E9 | 触觉表示消融：去 $u,v$ 坐标 / 去跨指 attention / 去 GRU / 每指 MLP 改共享 |
| E10 | 负载与几何扫描：冻结策略在保留阻力、25/35/40 mm、轴线偏差上测试 |
| E11 | 执行期干预：latent 延迟、丢帧、阶段性置零（**只能支持"策略依赖此信号"，不能替代重训消融**） |

### P2 —— 有余力再做

E12 真臂仿真集成（UR5e/天机 + differential IK，F9 已有资产）；E13 第二任务（nut-bolt / screwdriver）；E14 触觉布局消融；E15 F4 的 teacher 通道对照。

### 公平性三原则（写进附录，逐条自查）

1. **比协调结构时保持奖励一致**——不能只给 ours 接触 shaping。
2. **比感知能力时保持训练监督一致**——"部署无触觉"可保留相同特权训练奖励；同时删触觉奖励要另列。
3. **比效率时计入预训练与蒸馏成本**——同时报总成本与给定 checkpoint 后的适应成本。

seed 纪律：仿真关键比较目标 5 seed，紧张时 3 并如实说明；**并行环境不是独立 seed**。真机理想 3 模型 × 每条件 10 次，顺序按条件与模型分块随机化。

---

## 8. 真机方案与 go/no-go

### 8.1 已有（E5，写成 Table 2 上半）

冻结的 Stage-2 结构触觉 student，21 维手指动作，20 Hz。要补录的是**协议而非新能力**：阀门尺寸/轴线/安装刚度/阻力、是否从预抓取开始、目标净转角与时间预算、停滞判据、人工介入计分、复位流程。

### 8.2 待接（E7，Table 2 下半）

集成清单：

1. follower 输出 = 世界系 XY(+yaw) 位置目标增量 → 真臂任务空间伺服（20 Hz，与仿真同频）；
2. **姿态约束**：仿真滑台只平移手部安装座、yaw 单独一轴，真臂需锁住其余姿态自由度；
3. **力/柔顺映射**：仿真是有限力 PD（120 N / 8 N·m·rad⁻¹），真臂需对应的力限或阻抗参数，并在论文里给出两者对照；
4. **延迟**：仿真动作延迟为 U(0,1) 控制周期，真臂伺服链路延迟通常更大——**先测量，再加进随机化重训**，否则这是最可能的 sim2real 失败源；
5. 安全：workspace 硬限位、急停、阀门轴线标定误差范围。

### 8.3 Go/No-Go 门（**9 月 19 日**）

- **Go**：真机臂手能在名义条件稳定完成 → 主标题用 arm–hand 版，Table 2 双层，Figure 1 用真机臂手连拍。
- **No-Go**：仍未打通 → **切保守版本，论文照样成立**：标题退到手部版，臂手部分定位为"在同一部署接口上的仿真验证（E1–E4）"，Limitations 明写"末端补偿尚未在真实臂上验证"。**不要在 9/19 之后还赌真机**，会同时丢掉真机数据和论文完成度。

---

## 9. 指标定义

**主指标**（$s\in\{-1,1\}$ 为目标方向）：

$$
\Delta\theta_{\mathrm{net}} = s\big(\theta_{\mathrm{unwrap}}(T) - \theta_{\mathrm{unwrap}}(0)\big)
$$

用净转角，**不要用累计绝对转角**（来回振荡会刷分）。报告：成功率+区间、固定预算净转角、首次失去持续能力时间、负载–成功率曲线。提前终止的回合按统一方式计入，不能只统计成功回合的角速度。

**机制指标**（对应 D1–D5，`analysis/signals.py` 已可支持）：

| 指标 | 定义 | 解释限制 |
|---|---|---|
| 整体接触丢失 | 全部有效接触低于阈值且超过预设时长 | 个别手指主动释放不算失败 |
| **交接成功率** | 事件后重建有效接触并在固定窗口恢复正向进展 | 报告事件判据、窗口、每回合事件数 |
| 交接停顿时间 | 事件开始到恢复预定进展 | 同回合多事件相关，按回合聚类 |
| 关节饱和比例 | 进入归一化极限邻域的时间比例 | 不等同力矩饱和 |
| 接触迁移 | 传感器坐标下斑块位移 | **不能直接标为滑移速度** |
| 有效轴向力矩 | 仿真接触力 × 轴向力臂 | 真机需独立标定，电流不算 |
| 末端补偿 | 位移/速度/边界利用率**及其与交接事件的时序关系** | 相关不等于因果 |

**力矩主张需要独立标定**：没有加载装置测量时，只能说"不同阻力设置"，不能说"更大力矩"。"转得更快"≠"力矩更大"。

---

## 10. 图表清单（对应已有基础设施）

| 编号 | 内容 | 优先级 | 数据来源 |
|---|---|---|---|
| F1 | Teaser：真机任务 + 一次完整接触交接连拍 + 触觉热力图，标出手指行程耗尽与末端补偿 | P0 | 真机录像 |
| F2 | 系统图：**训练（含特权，虚线）与部署（实线）分离**，明确 teacher/student/follower 各看什么 | P0 | — |
| **F3** | **Contact raster**：横轴时间 × 五指接触色块，w/ vs w/o $r_{\mathrm{coord}}$，直接看见步态涌现 | **P0 money figure** | `detect_handover_events` |
| F4 | 主结果：各方法净转角分布 + 成功率 | P0 | E1 |
| F5 | 接口消融柱状（6 个 follower 输入变体） | P0 | E2 |
| F6 | 交接事件对齐的多通道时序：净转角/五指接触/关节余量/末端位移 | P0 | `build_phase_labels` |
| F7 | 标定漂移曲线（两条 student） | P1 | E6 |
| T1 | **特权隔离表**：teacher / student / follower / 真机部署 各见什么 | P0 | F4 |
| T2 | 仿真主基线 + 总训练成本 | P0 | E1 |
| T3 | 真机：手部（已有）+ 臂手（待定），试验数、条件、失败分类 | P0 | E5/E7 |

---

## 11. 正文结构与写作骨架（9 页）

| 部分 | 页 | 内容 |
|---|---:|---|
| Introduction | 1.0 | D1–D5 矛盾 → 接触结构假设 → 三条贡献 |
| Related Work | 0.6 | 手内旋转、触觉 RL、臂手协同、residual/模块化；**每段末句写与本文区别**，主动写 "we deliberately do not claim temporal abstraction" |
| Problem Setup | 0.6 | 任务、观测、动作子空间（**按 F1 的措辞**）、持续进展定义 |
| Method | 2.3 | 部署闭环先行 → 结构触觉表示 → 容量感知协作奖励 → 条件化末端补偿 → 实际训练顺序 |
| Experiments | 4.0 | E1–E6（+E7 若 Go） |
| Limitations & Conclusion | 0.5 | 单平台、3-DOF 末端抽象、任务范围、失败条件 |

### Intro 五段

1. **矛盾**：持续旋转要求持续传递轴向力矩，但维持大法向力会锁死手指位形；释放手指恢复行程，力矩立刻中断。阀门给出一个可测量、可重复的代表场景。
2. **为什么局部技能不够**：手内可以形成旋转步态，但在特定负载与偏差下，交接期就是失效点（用基线数据支撑，不要空说）。
3. **设计假设**：交接期真正缺的信息是"接触集合正在怎么变"和"手部此刻在做什么"，而不是物体位姿或力幅值。据此让末端补偿策略显式条件于结构触觉历史与手部当前动作。
4. **学习与部署**：特权仿真训练 → 去力幅值 student → 真机；明确真实数据需求（无示范、无真机微调，但传感标定/控制器调参/人工初始化如实报告）。
5. **贡献与已验证结果**：只写正文能找到证据的数字。

### Abstract 骨架（写完实验后按证据回填方括号）

> Sustained rotation of a constrained object requires a dexterous hand to keep transmitting axial torque while repeatedly handing contacts over within a limited range of motion. We identify the handover as the failure point and ask what information an added end-effector degree of freedom actually needs from the hand. We show that **contact structure** — contact onset/offset events, durations, and patch migration — together with the hand's current action is sufficient, while object pose and force magnitudes are not required. The same variable serves three roles: it provides dense credit for multi-finger gaiting, it acts as the coordination channel for the added DoF, and, because it needs no force calibration, it is the signal that survives sim-to-real. On [tasks and held-out conditions] our policy achieves [numbers] against [baselines]; the interface ablation shows [result]; and a policy trained only in simulation transfers to a real tactile hand [and a real arm], sustaining [measured capability].

### 三条贡献（写法）

1. **面向持续受约束旋转的接触结构表示与协作奖励**：不是"用了 attention+GRU"，而是"为什么交接需要事件通道、为什么负载分担要按力臂容量归一化"。
2. **接触结构作为臂手协调通道**：条件化的同频双策略分解，由 E2 的接口消融支撑。
3. **免力标定的可部署路径与实证发现**：真机验证 + 标定漂移扫描 + 失效边界。

---

## 12. 声明边界

**可以说**（有对应实验时）：结构触觉 student 可部署；接触结构在本系统中是有效的臂手协调变量；末端补偿改善交接期表现；仅用仿真训练完成真机任务（说明标定/初始化的人工部分）。

**不能说**：universal / morphology-agnostic / plug-and-play / zero-shot 跨手或跨任务迁移；skill-preserving 作为通用性质（除非做了重新锁臂回测）；general hierarchical RL 或时间抽象；"首次"用触觉 RL 拧阀门（DexTouch 在先）；仿真里"完成了螺纹装配"（若螺母被建模为定轴刚体，只能叫旋转代理任务）。

**用词分层**：同一冻结策略测未见尺寸/阻力 = 条件泛化；同分布新样本 = 分布内，不叫 OOD；同方法分别训不同任务 = 方法适用性，不是零样本迁移；换臂 = 构型变化，不是跨手泛化；节点 dropout = 传感退化鲁棒性，不是跨传感器迁移。

---

## 13. 20 天排期（今天 9/5，摘要 9/18，全文 9/25 AoE）

| 日期 | 产出 | 备注 |
|---|---|---|
| **9/5–9/7** | ①§6 student→follower 打通并起跑第一批 ②冻结 E5 真机协议、补录已有真机数据 ③主实验 run 全部打开接触时间序列 dump | ③不做的话 F3/F6 要重跑，**这是最贵的遗漏** |
| 9/8–9/12 | E1 主基线 + E2 接口消融并行铺开（共享手部 checkpoint）；真臂集成同步推进 | follower 侧便宜，先占满卡 |
| 9/13–9/16 | E3/E4/E6 + E8/E9 择机；真机 E5 保留条件重复试验 | E6 只需推理，随时可插空 |
| **9/17–9/18** | **中期判定**：E2 是否成立 → 决定 claim；写摘要提交 | E2 若失败，按 §14 换 claim，摘要仍可提交 |
| **9/19** | **真机臂手 go/no-go** | 见 §8.3 |
| 9/20–9/23 | 出图出表、正文成稿、附录（全部超参/奖励常数/随机化表/复现命令） | |
| 9/24–9/25 | 数字与文字一致性核对、匿名与格式检查、留提交余量 | 全文 9/25 23:59 AoE = 北京时间 9/26 19:59 |

---

## 14. 审稿预案（含 claim 失败的备选）

| 质疑 | 应对 |
|---|---|
| 只有一种手 | 限定单平台；给传感退化扫描 + 条件泛化；不写跨手 |
| 阀门旋转已有人做（DexTouch） | 正面引用；区分**短角度成功 vs 持续净进展与多次交接**；不写 first |
| 加 3 个自由度当然更好 | E3 分解 + 同动作空间的 flat/warm-start/规则基线 |
| 奖励塑形才是性能来源 | 同奖励下比协调结构（E1 ⑤vs⑥）+ 独立奖励消融（E8） |
| 仿真只有 XY，凭什么叫 arm | 按 F1 措辞；真臂落地则说明任务空间控制器；否则 Limitations 明写 |
| 无触觉基线被削弱 | 同历史长度、同监督、同预算；给 learning curve |
| 只是选了成功视频 | 全 trial 统计、选例规则（取中位数成功 + 最常见失败）、展示 ours 的失败 |
| 交接其实是规则触发 | 报告规则与学习的边界，不称涌现规划 |

**如果 E2 失败（接口消融无差）**：不要硬撑。备选 claim 按证据强度排序：
1. 若 warm-start joint PPO 也能达到但**样本量高很多** → claim 改为"分阶段 + 冻结组合是把已训练手内技能扩展到附加自由度的高效路径"，主证据换成 steps-to-threshold；
2. 若收益主要来自触觉奖励 → 主贡献转向 E8/E9 的接触学习设计，臂手降为一个应用章节；
3. 若真机是最强的那块 → 转为"仅仿真训练、免力标定的触觉手部技能在真机上的持续操作"，标题用保守版。

三条都比"坚持一个被自己实验否掉的 claim"安全。

---

## 15. 立刻要做的事（按小时计）

1. **[今天]** 在主实验 run 里 dump 每指 $b_i$ 时间序列（F3/F6 的唯一数据源，漏了就得重跑）。
2. **[今天]** 提交或回滚 `joint_finetune_enable` 改动，并在实验记录里固定配置版本（F5）。
3. **[1–2 天]** student latent 出口 + follower 接口改造 + Stage 1' 起跑（§6）。
4. **[并行]** 冻结真机评估协议，把已完成的真机阀门试验按 §9 指标重新计分并补齐缺失元数据。
5. **[并行]** 真臂集成的延迟测量——这一项越早测越好，因为它可能要求重训（§8.2 第 4 条）。
