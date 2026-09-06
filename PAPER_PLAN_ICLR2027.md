# ICLR 2027 论文规划（robotics track）

> 来源：[tactie-enhancedRL.md](tactie-enhancedRL.md) + [master-slave1.md](master-slave1.md)
> Primary area: applications to robotics, autonomy, planning
> 规划日期：2026-09-04

---

## 0. 一句话主张（全文只允许有这一个 claim）

> **对于需要持续旋转的富接触手内操作，"接触结构"（contact structure）而非力幅值或物体真值，
> 是一个足够的共享接口：它既能为多指换指步态提供稠密 credit，又能作为把新增末端自由度
> 接入已训练手内策略的唯一通信通道，还能在丢弃力幅值后保持可标定、可部署。**

三份证据支撑同一个 claim，这就是把两篇文档缝成一篇的接缝：

```
接触结构 --(1) 内生协作奖励--> 换指步态涌现
         --(2) 128-D latent 作为通信接口--> 冻结 master + 新增末端 DOF follower
         --(3) 去力幅值的结构触觉--> 可部署 student，抗标定漂移
```

**方法命名建议**：`CoTaC`（Coordination through Tactile Contact-structure）或 `TacGait`。
**标题候选**：
- *Contact Structure as an Interface: Tactile Latents for Finger Gaiting and Arm–Hand Coordination*
- *Feel to Regrasp, Feel to Reach: A Tactile Representation that Elicits Gaiting and Coordinates Added Degrees of Freedom*

---

## 1. 贡献点盘点（含新颖性诚实评估）

| # | 贡献 | 新颖性 | 会被怎么打 | 必须给的证据 |
|---|---|---|---|---|
| **C1** | **容量感知的触觉内生协作奖励**：把每指接触转成沿阀门轴的有效正力矩，按力臂容量 `C_i = R_i·F_comfort·lever_ratio_i` 归一化，用凹效用 H 奖励"有能力的手指分担负载"，用 8 步窗口的 handover 分数 G 奖励换指，用 q/S 门控和 E 代价抑制作弊 | **最强**。不是 reward hacking，而是接触力学推导出来的结构：凹性→分担，时间窗→步态 credit assignment | "20 个魔数怎么调出来的""换掉常数还成立吗" | 去掉 r_coord 完全失败；α_H/α_G 扫描；去掉 16 步窗口→步态退化；关键常数（F_comfort、n_sat、α）灵敏度 |
| **C2** | **Frame813 结构化时空触觉编码器**：真实 taxel 坐标 `[u,v]` 入通道、每指独立 whole-finger MLP、finger-identity + 跨指 self-attention、10 帧 GRU → 128-D | 中等。注意力触觉编码不新，但"每指一个 token、跨指注意力建模指间协同"的归纳偏置与任务强耦合 | "换成大 MLP 是不是一样" | vs MLP force oracle；去 u/v；去跨指注意力；去 GRU（单帧）；共享 vs 每指独立 MLP |
| **C3** | **latent 条件化的具身扩展**：冻结 master，follower 只看 (执行手动作 21, detach 的 z_tac 128, 自身状态 10)，同频率、单次 env.step、共享 team reward、双 PPO/双 GAE、梯度隔离；能力触发式课程（ω EMA > 0.8 rad/s 连续 5 epoch 才放开末端） | **中上，且是 C2 的价值证明**。"表征好不好"由"第二个策略能否只靠它协同"来验证 | "这不是 hierarchy""为什么不直接学 23 维" | follower 输入消融（latent / 零 / 原始 1170 / 手部 obs / trunk feature）；vs 从零训 23 维 flat PPO；DOF 扫描 {0, yaw, xy, xy+yaw} |
| **C4** | **去力幅值的可部署结构触觉 student**：student 只见 `[b,on,off,duration,eta]`+finger context（595 维），完全不见 `Fn/Ft`，student-executed DAgger | 中等，但**是"无真机"的唯一挡箭牌** | "没有真机凭什么说 sim-to-real" | **标定漂移鲁棒性实验**：taxel 增益误差/偏置/死点/阈值失配/延迟下，force-based student 崩、structural student 稳 |

> **建议在 intro 里只列 3 条**：C1、C2+C3 合并成"表征作为协同接口"、C4。四条并列会稀释。

---

## 2. 论文骨架（9 页正文 + 附录）

### 1 Introduction（1 页）
- 动机递进：阀门/旋拧类任务的物体尺度**大于手的抓取跨度**，持续旋转必须换指（finger gaiting），
  而换指是稀疏奖励下几乎发现不了的行为；同时手内工作空间有限，末端自由度能进一步抬高速度上限。
- 指出两个真实困难：(a) 换指的 credit assignment；(b) 给已训练好的手内策略"加自由度"通常意味着重训。
- 亮出 claim：接触结构同时解决两者。
- 贡献 3 条 + 主要数字（例如 ω 提升 x%，加末端 DOF 再提升 y%，标定漂移下 student 保持 z%）。

### 2 Related Work（0.75 页）
四段，每段最后一句必须写"我们与之的区别"：
1. **手内旋转/灵巧操作**：HORA、RMA 谱系（本体感觉 + 特权蒸馏）→ 我们加入物理触觉与步态奖励。
2. **触觉感知与触觉 RL**：binary-touch 手内旋转、TacSL 类触觉仿真 → 我们区分 teacher/student 触觉接口，并把 latent 复用为协同通道。
3. **臂手/全身协同操作**：whole-body manipulation、sequential dexterity → 我们做的是**冻结手内策略的增量具身扩展**，不是联合重训。
4. **策略组合 / residual & 模块化 RL**：residual policy learning、options/HRL → **明确声明我们不是时间抽象层级**（见 §6 诚实性）。

### 3 Problem Setup（0.75 页）
- 任务族：35 mm 阀门手柄，半径 scale ∈ {0.8…1.2}；21 自由度 Revo3；20 Hz 控制 / 240 Hz 物理 / decimation 12；40 s episode。
- 动作接口：21 维**增量关节目标** `q_target ← clip(q_target + a/24)`，显式 PD (kp 3.0, kd 0.01)，per-env 动作延迟 U(0,1) 控制周期。
- 触觉硬件模型：115 物理节点（thumb 31 + 4×21），每节点 `[Fn, Ft1, Ft2]`，力 scale 200、裁剪 ±5。
- 末端自由度：**这里就要诚实定义**。世界系 X/Y prismatic + yaw revolute，有限力 PD（120 N / 8 N·m·rad⁻¹），
  速度 0.15 m/s、加速度 8 m/s²、yaw 1.2 rad/s，硬限位，与手共享动作延迟。写清"参数按真实臂腕规格设定，
  不含 IK/冗余/臂体动力学"，并在 Limitations 再声明一次。
- 评测指标定义：`ω̄`、`fraction_above_{0.8,1,2,4}`、停滞率、以及**新增的步态指标**（见 §4）。

### 4 Method（3 页）
- **4.1 触觉观测模型**：teacher 帧 1170 = 115×10 + 5×4；student 帧 595 = 115×5 + 5×4；
  结构通道 `[b, on, off, duration, eta]` 的定义（滞回阈值 0.001/0.0005、duration 对数归一化、eta 为接触斑块迁移投影）。
  用一张小表把"teacher 见什么 / student 见什么 / 部署见什么"讲清楚——**这张表是全文的特权隔离线**。
- **4.2 结构化编码器**：per-node 7 维 `[u,v,b,d,Fn,Ft1,Ft2]` → 每指独立 MLP(217/147→64→32) → +finger embedding
  → 4-head 跨指 attention → 展平 160 → GRU hidden 128。强调归纳偏置来自真实 taxel 几何而非 learned positional embedding。
- **4.3 容量感知协作奖励**：按顺序推 τ_axis,i → e_axis,i → τ_eff,i → C_i → load_i → H_inst → H(0.6/0.4, 16 步窗)
  → G(8 步释放窗) → q_guide / S / E → `Q = q·S·(0.4H + 0.6G − E)` → `r_coord = clip(0.12 g_τ + w·Q, −1, 1)`。
  **写作要点**：每一项都要给"为什么这样设计"的一句话力学理由，不要只贴公式，否则就是 reward engineering。
  凹性→边际收益递减→分担；16 步窗→顺序换指也能拿分；容量归一化→不强求各指出等力。
- **4.4 latent 条件化的末端扩展**：接口定义（159/154/164 维）、单次 env.step、双 GAE、梯度隔离（follower loss 不回传 master）、
  三阶段课程（能力触发 → workspace/action-scale ramp → 可选带 KL 锚的联合微调）。
  **画一个 interface box**，强调 follower 拿不到 141 维手部观测、拿不到原始触觉帧、拿不到 priv_info。
- **4.5 可部署蒸馏**：student-executed DAgger（环境执行 student μ，teacher 只打标签），
  Stage-2 显式关闭 r_coord，`L = MSE(μ_S, clip(μ_T))`。

### 5 Experiments（3 页）见 §3 实验矩阵

### 6 Limitations（0.4 页，**必须自己写，不要留给审稿人**）
1. 末端为受限自由度抽象，非全臂 IK/冗余控制。
2. 仿真评测；用标定漂移实验做真机代理，但不等价于真机。
3. 单一任务族（阀门/旋拧），未验证抓取、装配等其它接触模式。
4. 完整主从栈目前依赖特权 master；可部署版本见 §X。

---

## 3. Claim–实验–指标 对照表（核心）

| ID | 要证明的 claim | 实验设置 | 主指标 | 判定失败的条件 |
|---|---|---|---|---|
| **E0** | 触觉 + 协作奖励带来的手内性能 | flat PPO(无触觉) / HORA 式本体 teacher / MLP force oracle / **Frame813(ours)**，各 ≥3 seed | `ω̄`, `fraction_above_{1,2,4}`, 停滞率 | ours 不显著优于 MLP oracle → C2 站不住 |
| **E1** | r_coord 是换指步态的成因 | 去 r_coord / α_H=1 / α_G=1 / 去 16 步窗 / 去容量归一化（C_i 取常数）/ 去 S 门 | ω̄ + **换指率、接触集合切换率、平均接触时长、per-finger contact raster** | 去掉 r_coord 后 ω 不降或步态照样出现 → C1 站不住 |
| **E1b** | 奖励不是过拟合到魔数 | F_comfort ∈ {1.5,2.5,4.0}、n_sat ∈ {3,6,12}、α_H/α_G ∈ {0.2/0.8 … 0.8/0.2} | ω̄ 的方差带 | 性能对常数极敏感 → 必须在正文承认并给推荐区间 |
| **E2** | 结构化编码器的归纳偏置有用 | Frame813 去 u/v / 去跨指 attention / 去 GRU / 每指 MLP 改共享 | ω̄、样本效率（达到 1 rad/s 所需 steps） | 各变体差异 < seed 噪声 → 把 C2 降级为实现细节 |
| **E3** | **latent 足以充当协同接口（缝合线实验）** | follower 观测中 z_tac 替换为：① 零向量 ② 原始 1170 维触觉帧 ③ master 141 维公开观测 ④ master trunk feature ⑤ 保留（ours） | ω̄ 提升幅度、follower 收敛速度、xy/yaw 工作空间利用率 | ①②③④ 与 ⑤ 无差 → 整篇的接缝断裂，必须换 claim |
| **E3b** | 冻结组合优于联合重训 | 从零训 23 维 flat PPO（同 wall-clock / 同 samples）vs Stage0→Stage1 | ω̄、总样本量、是否破坏手内技能（回到 0-DOF 评测） | flat PPO 更好 → C3 只剩"省算力"的弱主张 |
| **E3c** | 末端自由度的边际收益可分解 | DOF ∈ {0, yaw only, xy, xy+yaw} | ω̄ 增量、末端 effort/power 代价 | yaw-only 与 xy 收益相同 → 说明是"随便动动就行"，需解释 |
| **E3d** | 能力触发课程是必要的 | 阈值 ∈ {不触发直接同时训, 0.8 rad/s(ours), 更高} | 训练稳定性、末端是否学出"代偿坏手策略"的行为 | 无差 → 课程降为附录 |
| **E4** | **去力幅值换来的是鲁棒性而非损失** | student(结构触觉, ours) vs student(加 Fn/Ft) vs teacher，在扰动下评测：taxel 增益 ×[0.5,2]、偏置、随机死点 5–30%、接触阈值失配、额外延迟、bit-flip | 扰动-性能曲线、崩溃点 | 加力幅值的 student 在扰动下也不崩 → C4 站不住，改成"力幅值非必要"的弱主张 |
| **E4b** | 泛化 | held-out 半径 scale、held-out 质量/摩擦区间、（若有）nut-bolt / screwdriver | 零样本 ω̄ | 只在训练分布内工作 → 必须在 Limitations 写明 |
| **E5** | 末端不是免费能源 | 报告 `xy/effort_norm`, `power`, `action_saturation_ratio`, `boundary_saturation_ratio`；再做 force limit 120 N → 40 N 的降级实验 | 是否仍在限内、性能下降幅度 | 性能全靠饱和在力限上 → 结论不可迁移到真臂 |

---

## 4. 现在就必须补的 logging（**时间敏感：不补就得重跑**）

当前 [master-slave1.md](master-slave1.md) §14 只记录了 ω 与 xy 统计，缺少能证明 C1 的行为指标。
在训练循环里补齐（都能从已有 `b_i`/接触状态直接算）：

```text
gait/handover_rate              单位时间内 (i 释放 → j 建立) 事件数
gait/contact_set_transitions    五指接触集合（32 种）的切换频率
gait/mean_contact_duration      每指平均连续接触时长
gait/active_finger_count        平均同时接触指数
gait/load_sharing_index         归一化 load 的熵或 Gini
gait/slip_velocity_mean         平均切向滑移速度
gait/contact_raster             周期性 dump 每指 b_i 时间序列（画 money figure 用）
```

> `gait/contact_raster` 是本文最有说服力的图的数据来源，务必在主实验 run 里就保存。

---

## 5. 图表清单

| 编号 | 内容 | 优先级 |
|---|---|---|
| F1 | Teaser：任务 + 触觉热力图 + 换指过程连拍 | P0 |
| F2 | 系统总图（三条链路：感知 / 协同 / 部署，用颜色区分特权与可部署） | P0 |
| **F3** | **Contact raster plot**：横轴时间、五行手指、色块表示接触；对比 w/ 和 w/o r_coord，直接看见步态涌现 | **P0（money figure）** |
| F4 | 主结果：各方法 ω 分布 + fraction_above 柱状 | P0 |
| F5 | E3 接口消融柱状（5 个 follower 输入变体） | P0 |
| F6 | 标定漂移鲁棒性曲线（横轴扰动强度，两条 student 曲线） | P1 |
| F7 | 触觉编码器结构图 | P1 |
| F8 | 奖励机制图（H/G/q/S/E 五条支路） | P1 |
| T1 | 特权隔离表（teacher / student / 部署 各见什么） | P0 |
| T2 | 消融汇总表 | P0 |
| A1–A5 | 附录：全部超参、全部奖励常数、域随机化表、网络维度表、复现命令 | P0 |

---

## 6. 三个必须先自己处理的诚实性问题

1. **不要叫 Hierarchical**。文档 §2.1 自己已经说清了：不是时间抽象、不是 option、同频率。
   论文里改称 **latent-conditioned auxiliary-DoF policy** 或 **compositional embodiment extension**，
   并在 Related Work 主动写一句"we deliberately do not claim temporal abstraction"。
   这一句能挡掉一个最容易致命的攻击。
2. **末端自由度的口径统一**。全文不要出现无限定的"robot arm"。要么接 UR5e（仓库已有
   [ur5e_revo3_right.py](source/BrainCo_DexHand/BrainCo_DexHand/assets/ur5e_revo3_right.py) 和
   [dexsuite_env_cfg_grasp_ur5e.py](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/dexsuite_env_cfg_grasp_ur5e.py)，
   follower 输出走 differential IK 即可）作为附加实验，要么全文统一为 "force-limited end-effector DoF"。
3. **完整栈的可部署性缺口**。现在 master 读特权、follower 依赖 teacher latent，整套不可部署。
   最低要求：在仿真里训一版 **follower 条件于 student latent `z_S`** 的版本，证明接口在可部署侧仍然成立。
   注意 [tactie-enhancedRL.md](tactie-enhancedRL.md) §10.4 指出 `_teacher_has_tactile_latent` 恒为 false、
   latent distillation 链路未接通——**这是需要先补代码的**。

### 附：一个需要解释的内部不一致
Teacher 的 Frame813 编码器用 `[b, duration, Fn, Ft1, Ft2]`，**丢弃了 on/off/eta**；
而 student 恰恰保留 `[b, on, off, duration, eta]`、丢弃力。两者不是嵌套关系。
审稿人会问"teacher 都不用的事件通道，凭什么认为 student 能靠它复现 teacher 行为"。
要么在附录给出理由（teacher 有力幅值可隐式推断事件），要么做一个"teacher 也加 on/off/eta"的对照。

---

## 7. 优先级排期

| 级别 | 任务 | 理由 |
|---|---|---|
| **P0** | 补 §4 的 gait logging → 重跑主实验 3–5 seed → E0 / E1 / E3 三组实验 + F3 raster 图 | 这三组撑起全部 claim，缺任何一组论文不成立 |
| **P0** | 统一末端口径 + 改掉 Hierarchical 命名 | 零成本，挡掉最致命攻击 |
| **P1** | E4 标定漂移鲁棒性（无真机的核心辩护）、E2 编码器消融、E1b 常数灵敏度 | 决定分数从 5 到 6 |
| **P1** | 至少一个外部基线（非自消融），如 binary-touch 手内旋转的编码器结构复现 | 审稿人明确要求 |
| **P2** | UR5e + differential IK 接入；follower 条件于 z_S 的可部署全栈；nut-bolt / screwdriver 跨任务 | 决定分数从 6 到 8 |
| **P2** | 真机 | 有则接收概率大幅上升；无则靠 P1 的鲁棒性实验硬撑 |
