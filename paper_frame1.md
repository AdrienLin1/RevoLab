# ICLR 2027 论文框架：触觉技能学习与臂手协作

## 1. 投稿定位

- **Primary area**：Applications to Robotics, Autonomy, Planning
- **关键词**：tactile reinforcement learning、privileged learning、sim-to-real、dexterous manipulation、arm-hand coordination、policy expansion
- **核心问题**：如何从仿真特权信息中学习高性能触觉手部技能，将其蒸馏为真机可部署策略，并在不破坏原有技能的前提下扩展到臂手联合动作空间？

## 2. 一句话主线

> 将触觉从单纯的传感输入提升为可复用的策略接口：它连接仿真中的特权技能学习、真机可部署的教师—学生蒸馏，以及保持原有灵巧性的臂手协作扩展。

论文不是“触觉方法 + 臂手方法”的并列组合，而是一条连续链路：

```text
特权触觉 Teacher
    -> 结构触觉 Student
    -> 可部署手部技能与协调 latent
    -> 臂/腕补偿策略
    -> 保持手部技能的臂手联合控制
```

## 3. 推荐标题

首选：

> **Touch as a Coordination Interface: From Privileged Sim-to-Real Dexterity to Skill-Preserving Arm-Hand Control**

备选：

> **Touch-Scaffolded Dexterity: Deployable Tactile Skill Learning and Arm-Hand Policy Expansion**

## 4. 问题动机

本文解决三个递进问题。

### 4.1 可部署触觉难以直接学出高性能技能

真实触觉稀疏、有噪声且存在延迟，直接以可部署观测训练会限制探索效率和性能。仿真则可提供物体状态、动力学参数和节点三维接触力，用于训练高性能 Teacher。

### 4.2 特权触觉策略不能直接部署

Teacher 依赖真机不可获得的信息。需要将其行为蒸馏到只读取本体历史和结构触觉事件的 Student，并处理 sim-to-real 触觉差异。

### 4.3 固定手腕限制灵巧手性能上限

持续操作时，手指会接近关节极限、接触区域会漂移，有效力臂会下降。增加臂/腕自由度能够重新定位整只手，但从头训练高维联合策略探索困难，也可能破坏已有手部技能。

## 5. 统一方法

### 5.1 Phase A：特权触觉技能学习

训练触觉增强 Teacher：

\[
(z_T,a_T^H)\sim\pi_T(o^H,p,x^{force})
\]

- `o^H`：手部公开观测；
- `p`：物体状态、摩擦、质量、质心等训练期特权量；
- `x^{force}`：节点三维力及其时序变化；
- `z_T`：结构化时空触觉表示；
- `a_T^H`：21 维手部动作。

Teacher 使用：

1. 按手指组织的空间—时间触觉编码；
2. 多指负载分担、手指交接、有效轴向力矩和防滑奖励；
3. PPO 与动力学、几何、延迟和触觉随机化。

核心作用：利用训练期信息改善探索和接触技能，而不是将特权量带到部署阶段。

### 5.2 Phase B：可部署结构触觉蒸馏

Student 只读取真机可获得的信息：

\[
(z_S,a_S^H)=\pi_S(h^q,h^B)
\]

- `h^q`：关节位置和目标历史；
- `h^B`：接触状态、接触开始/结束、持续时间、接触迁移等结构触觉历史；
- 不读取物体真值、动力学参数或精确三维接触力。

Student 动作实际驱动环境，冻结 Teacher 在 Student 访问到的状态上提供标签：

\[
\mathcal L_{TS}
=\|a_S^H-a_T^H\|^2
+\lambda_z\|g(z_S)-\operatorname{sg}(z_T)\|^2
\]

第一项蒸馏动作，第二项使 Student latent 可替代 Teacher latent，供后续臂手协调使用。当前实现只有动作 DAgger，正式论文前需要启用并验证 coordination latent 蒸馏，或重新定义固定维度的可部署协调表示。

### 5.3 Phase C：保持技能的臂手动作空间扩展

用 Student 作为手部策略，以其动作和触觉 latent 条件化臂/腕策略：

\[
\pi(a_t^H,a_t^A\mid o_t)
=\pi_H(a_t^H\mid h_t^q,h_t^B)
 \pi_A(a_t^A\mid o_t^A,a_t^H,z_t^S)
\]

其中 `a^A` 为末端 Cartesian 或腕部动作。两策略同频，并在一个环境步内依次决策，因此这是**触觉条件化自回归联合策略**，不是经典时间层级策略。

训练分为：

1. **手部预训练**：获得稳定触觉技能；
2. **Follower adaptation**：冻结手部策略，只训练臂/腕补偿；
3. **Constrained co-adaptation**：小学习率联合微调，并约束手部策略偏移：

\[
\mathcal L
=\mathcal L_{RL}
+\lambda_{KL}D_{KL}(\pi_H\|\pi_H^{pretrained})
\]

Follower 根据手部能力达到阈值后激活，并逐渐放开 workspace 和动作尺度。

### 5.4 优化表述

当前双 PPO 分别计算 ratio 和 GAE，不等价于单一联合 PPO。论文应采用以下一种方式：

- 实现数学一致的 autoregressive joint PPO，联合 log-probability 为 `log pi_H + log pi_A`；或
- 将现有方法准确表述为 coordinate/independent PPO，并加入 autoregressive joint PPO 基线。

## 6. 论文贡献

### Contribution 1：触觉增强的特权学习与部署蒸馏

提出结合结构化时空触觉编码、接触协调奖励和 Student-distribution DAgger 的 T–S 框架，从仿真特权触觉中学习高性能技能，并蒸馏到仅依赖可部署结构触觉的策略。

### Contribution 2：触觉驱动的技能保持式臂手扩展

提出以手部动作和可部署触觉 latent 为协调接口的自回归策略分解，通过能力触发、冻结适应和 KL 约束联合微调，将已学习的手部技能扩展到臂手动作空间。

### Contribution 3：跨任务和 sim-to-real 证据

在多类接触操作、不同几何与动力学扰动、完整机械臂和真机条件下，验证方法能提高触觉 sim-to-real 性能、降低臂手适应样本量、保持原有手部技能，并突破固定手腕的性能上限。

第三项中的每个结论都必须有对应实验；未完成的部分不得写成已实现贡献。

## 7. 核心研究问题与实验

| 研究问题 | 核心证据 | 对应贡献 |
|---|---|---|
| RQ1：触觉增强是否改善技能学习？ | 触觉表示和触觉奖励消融 | C1 |
| RQ2：Teacher 能否转化为可部署 Student？ | BC/DAgger、动作/latent 蒸馏、真机 | C1 |
| RQ3：为何需要所提臂手分解？ | flat、independent、AR-JPPO、proposed | C2 |
| RQ4：扩展后是否保留手部能力？ | 冻结、联合微调、KL、重新锁臂测试 | C2 |
| RQ5：是否能快速扩展到新任务？ | 跨任务复用与 steps-to-threshold | C3 |
| RQ6：是否具备鲁棒性和跨构型适用性？ | OOD、完整机械臂、可选第二构型 | C3 |

## 8. 实验一：触觉技能学习

### 8.1 Baselines

- Proprioception only；
- 粗粒度 binary contact；
- 原始节点触觉 + MLP；
- 无空间坐标的时空编码；
- 无时间历史；
- 完整结构化触觉 Teacher。

### 8.2 Reward ablations

- 去掉负载分担；
- 去掉 finger handover；
- 去掉有效轴向力矩；
- 去掉防滑项；
- 去掉全部触觉协调奖励。

### 8.3 指标

- 成功率、角速度、累计转角；
- 学习曲线 AUC、steps-to-threshold、收敛 seed 比例；
- 接触丢失、滑移、关节饱和；
- 有效力矩、各指负载、单位转角能耗。

## 9. 实验二：T–S 蒸馏与 sim-to-real

### 9.1 Baselines

- Privileged Teacher：仿真上界；
- proprio-only Student；
- offline BC；
- action-only DAgger；
- action + latent DAgger；
- 完整结构触觉 Student。

### 9.2 触觉对齐指标

- contact-onset precision/recall/F1；
- 接触持续时间和激活节点比例分布；
- 触觉延迟、噪声和 dropout 敏感性；
- 仿真/真机 latent 分布差异；
- 真机成功率、连续操作时长和失败类型。

## 10. 实验三：臂手联合控制

### 10.1 必须比较的基线

1. Hand-only / fixed wrist；
2. 规则式 Cartesian/tactile servo；
3. Flat Joint PPO from scratch；
4. Flat Joint PPO with the same hand warm start；
5. Independent PPO，无动作/latent 通信；
6. Autoregressive Joint PPO，相同条件结构但无分阶段训练；
7. 完整方法。

若将方法定义为 MARL，再增加 MAPPO/HAPPO；否则不作为首要基线。

### 10.2 协调接口消融

做严格的 `2 x 2` 对比：

- follower 只有自身状态；
- 自身状态 + hand action；
- 自身状态 + tactile latent；
- 自身状态 + hand action + tactile latent。

进一步测试 latent 延迟、dropout、跨环境打乱和不同 latent 维数，证明策略确实利用触觉通信。

### 10.3 训练课程消融

- 全自由度从头同时训练；
- hand warm start 后同时训练；
- 冻结 hand，仅训练 follower；
- 无 KL 联合微调；
- KL 约束联合微调；
- 固定时间激活；
- 能力阈值激活。

### 10.4 技能保持测试

联合训练完成后重新锁定机械臂，在原 hand-only 环境测试：

- 原任务性能保留比例；
- hand policy KL；
- 接触模式和关节动作变化；
- 重新适应原任务所需样本量。

## 11. 如何证明“突破性能上限”

除最终任务性能外，还应报告：

- 手指关节接近极限的比例；
- 接触质心漂移和接触丢失次数；
- 有效力臂/轴向力矩；
- workspace 利用率；
- 臂动作与接触变化的时序关系；
- follower 置零、延迟、打乱后的性能下降。

这些结果用于证明机械臂主动维持了手部可操作区域，而不是仅因增加动作维度或奖励项获益。

## 12. 多任务与快速适应

推荐任务组合：

- 主任务：valve continuous rotation；
- 第二类：screwdriver 或 nut-bolt；
- 第三类：ball/cylinder in-hand rotation；
- valve 25/35/40 mm 用于同任务几何泛化，不计作三个独立任务。

“快速适应”需要报告：

- 从头训练与复用 tactile encoder 的样本量；
- 从头联合训练与复用 hand skill、仅训练 follower 的样本量；
- steps-to-threshold 和 wall-clock time；
- 新增可训练参数量；
- 是否使用同一组超参数。

若条件允许，增加跨任务 encoder transfer：在 valve 上训练触觉 encoder，在 screwdriver/nut-bolt 上冻结或轻微微调。

## 13. 机械臂与跨构型验证

当前 XY/yaw 系统是物理滑台/腕部等效自由度，不是完整机械臂。论文应优先增加：

```text
Franka + Revo3
follower Cartesian delta
    -> task-space controller / IK
    -> Franka joint commands
```

这能单独验证机械臂构型变化，同时保持手和触觉接口不变。

`Franka + Shadow Hand` 可进一步展示跨构型适用性，但同时改变手臂、手、动作维度和触觉布局，只能支撑“在两个臂手系统上适用”。除非设计形态无关的 tactile/action interface 并完成零样本或少样本迁移，否则不声称 morphology-agnostic 或 plug-and-play。

优先级：

1. Student 接入 follower；
2. 多任务核心实验；
3. 完整 Franka + Revo3；
4. 真机臂手实验；
5. Franka + Shadow Hand。

## 14. 鲁棒性与统计

### 14.1 OOD 条件

- 未见过的物体尺寸、质量、摩擦和负载；
- 初始位姿和轴线偏差；
- 触觉噪声、dropout、延迟；
- 动作延迟和 effort limit；
- 外部扰动与局部传感失效。

### 14.2 公平性

- 至少 5 个训练 seeds；
- 置信区间基于 seed 均值，不把并行环境当独立训练样本；
- 同时报告总训练样本和 hand checkpoint 后的 adaptation samples；
- warm-start 方法使用相同 hand checkpoint；
- 相同奖励、随机化、控制频率、动作限制和调参预算；
- 报告参数量、训练时间和推理延迟。

真机建议使用 3 个独立训练 seeds、每个 seed 每种条件 10 次试验，并随机化试验顺序。

## 15. 主结果组织

- **Figure 1**：Teacher -> Student -> arm-hand expansion 完整框架与特权信息隔离线；
- **Figure 2**：结构触觉编码和自回归臂手策略；
- **Figure 3**：触觉学习、蒸馏和臂手适应学习曲线；
- **Table 1**：触觉表示/奖励和 Student 蒸馏结果；
- **Table 2**：臂手主基线、多任务与真机结果；
- **Figure 4**：协调接口、技能保持、OOD 和失败分析。

## 16. 正文结构（9 页）

1. **Introduction，1.0 页**：中心问题、统一主线、三项贡献；
2. **Related Work，0.6 页**：触觉 RL、sim-to-real、privileged learning、臂手/whole-body control；
3. **Problem Formulation，0.5 页**：Teacher、Student、自回归联合策略；
4. **Method，2.7 页**：触觉 Teacher、结构触觉蒸馏、协调策略、训练课程；
5. **Experiments，3.6 页**：RQ1–RQ6；
6. **Limitations and Conclusion，0.6 页**。

详细网络尺寸、奖励公式、训练配置和补充结果放附录，但所有核心主张的关键证据必须出现在正文。

## 17. 摘要逻辑

摘要按五句话组织：

1. 固定手腕和不可部署的仿真触觉共同限制 dexterous RL；
2. 提出从特权触觉技能学习到可部署 Student 的统一框架；
3. 将 Student tactile latent 作为臂手协调接口；
4. 通过分阶段与 KL 约束扩展动作空间并保持手部技能；
5. 概括多任务、sim-to-real、样本效率和性能上限结果。

## 18. 当前必须修正的问题

1. 将可部署 Student 及其 latent 接入 arm/wrist follower；
2. 启用并验证 coordination latent 蒸馏；
3. 统一 Student 与 follower 的 latent 维度和接口定义；
4. 修正 XY 配置中 `joint_finetune_enable` 与文档/测试不一致；
5. 公平实验关闭额外 high-speed reward；
6. XY、yaw、XYYaw 不直接比较 raw return，改用物理指标；
7. 明确使用 autoregressive joint PPO，或准确说明 coordinate PPO；
8. 若没有完整机械臂实验，将表述限定为 hand-wrist/base coordination。

## 19. 声明边界

可以声称：

- deployable structural-tactile policy；
- tactile-conditioned arm-hand coordination；
- skill-preserving action-space expansion；
- 跨已验证任务、扰动和机器人构型的适用性。

没有相应证据时不能声称：

- universal dexterous manipulation；
- 任意臂手 plug-and-play；
- morphology-agnostic；
- zero-shot task/embodiment transfer；
- general temporal hierarchical RL。

## 20. 投稿前最低闭环

论文达到完整状态至少需要：

1. Student-driven arm/wrist coordination 能稳定运行；
2. 触觉部分有完整 Teacher/Student/真机对照；
3. 臂手部分有 flat、independent、AR-JPPO 和课程消融；
4. 至少三个语义任务族中的仿真结果；
5. 至少一个任务的真机臂手结果，或相应缩小论文声明；
6. 所有主要结果具有多 seed、置信区间和等预算对比。

## 21. 投稿规则提醒

- ICLR 2027 摘要截止：2026-09-18 AOE；全文截止：2026-09-25 AOE；
- 正文上限 9 页；
- 若原触觉稿已被正式接收、发表或仍在其他归档会议/期刊审稿，合并稿可能违反 substantial-overlap/dual-submission policy；仅内部稿、arXiv 或无正式 proceedings 的 workshop 通常不构成冲突。
