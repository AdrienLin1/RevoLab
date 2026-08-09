# Research Proposal: TacRes-Hand — Contact-Event-Gated Tactile Residual Policies for Robust Dexterous Manipulation (v3, 执行版)

## Problem Anchor
- **Bottom-line problem**: 在 REVO3（21-DoF，全掌 3D 力触觉）+ RevoLab/IsaacLab 上，proprioception 训练的灵巧操作策略在接触异常（外力脉冲、摩擦骤降、质量/质心偏移）下成功率大幅下降；触觉信息可用，但"拼进观测让单一网络自己消化"的用法学习信号弱、无法保证触觉真正驱动修正动作。
- **Must-solve bottleneck**: 让触觉**因果地**驱动实时修正动作，且修正模块**完全在仿真 RL 中学得**——不依赖人工示教（vs CATFA）、不依赖手调控制律（vs DexTac）；并用受控实验证明"结构化触觉用法（门控残差）优于触觉作普通观测"。
- **Non-goals**: 不做世界模型；不做 GelSight 图像触觉仿真；本两周阶段不做真机实验；不做新仿真器/基建；不用视觉。
- **Constraints**: 单张 RTX 4080 16GB；本地 ~/IsaacLab + ~/RevoLab；两周产出 MVP 核心证据；目标 AAAI 2027（两周 MVP 后约 4 周完整实验窗口）。
- **Success condition**: 扰动矩阵下，TacRes 相比"同预算端到端触觉观测重训"成功率 ≥ +8–10 个百分点，**或**同等性能样本效率 ≥3×；门控激活与扰动事件时间对齐可量化；残差触觉输入置零后增益消失（因果性检验）。

## Technical Gap
- RevoLab 现状（本地源码核查）：HORA 把 5 指尖二值接触（含噪声/延迟建模）放进观测；dexsuite 有 `fingers_contact_force_b`（5 指尖 3D 力）观测函数；接触力用于奖励。全掌区域力分布、学习型触觉修正结构均缺失。
- 端到端"触觉拼观测"的失败模式：扰动是稀有事件（<10% 时间步），触觉多数时间步与最优动作无关 → 梯度信号稀释，网络学会忽略触觉；重训整策略破坏已收敛技能。
- Naive 修复不充分：更大网络/更长历史不解决信号稀释；触觉奖励塑造（Tac2Motion）只改目标不改信息流结构；手调修正（DexTac）不适应任务阶段；真实示教修正层（CATFA）需人工数据。
- 缺失机制：把触觉作用显式限制在"事件触发的有界残差修正"，稀有事件学习信号集中到专职小模块。

## Method Thesis
- 一句话：冻结 proprio 主策略，PPO 在扰动课程下学触觉事件门控的有界残差 a_env = clip(a_base + α·g·tanh(Δa), lo, hi)，门控经特权热启动（退火）+ 稀疏正则，迫使"必要时才介入"。
- 最小充分：仅两个小 MLP（π_res + g，共 ~0.6M 参数）；训练期辅助损失退火归零，测试期零附加组件。
- 时代性：residual adaptation / policy modulation + 非对称特权训练；刻意不用 LLM/VLM/Diffusion（瓶颈在控制侧信息流结构）。

## Contribution Focus
- 主贡献：触觉事件门控残差框架——"何时介入"（学习门控+特权热启动+防泄漏验证）与"如何介入"（有界残差）解耦，完全仿真 RL 习得；配套"结构化触觉闭环 vs 触觉观测"受控证据（训练级+eval 级双重因果检验、三层对齐分析）。
- 辅助贡献：全掌 vs 指尖触觉系统消融（完整阶段；未完成则按预案降级论文定位）。
- 显式非贡献：触觉表征学习、滑移预测头、世界模型、真机 sim2real。

## Proposed Method

### Complexity Budget
- 冻结/复用：RevoLab Dexsuite-Lift 环境与 proprio checkpoint、RSL-RL PPO、IsaacLab ContactSensor、HORA 噪声/延迟建模模式。
- 新增可训练：① π_res = MLP[512,256]；② g = MLP[64,32]（critic 为 PPO 标配不计）。
- 刻意排除：滑移预测头、GNN、双时间尺度、options/MoE、tiny GRU（仅 fallback）、r_slip（默认关，重开条件=恢复行为训不出）。

### System Overview
```
proprio s_t ─────────▶ π_base (FROZEN) ──▶ a_base ∈ R^21 (normalized)
tactile tensors ──▶ 预处理五步 ──┬──▶ e_t (~10d) ──▶ g = σ(MLP_g(e_t)) ∈ [0,1]
                                 └──▶ τ_hist (K=10, 250d) ─┐
proprio s_t, a_base ────────────────────────────────────────┴▶ Δa = tanh(MLP_res(·))
μ_t = a_base + α·g·Δa ;  π(a|s) = N(μ_t, diag(exp(logσ)))  [logσ 全局可学习, 初值−1.0]
a_env = clip(sample/μ_t, lo, hi) → 关节目标 (MIT mode)
```

### Actor–Critic Contract
1. **动作空间/单位**：残差加在 RSL-RL 归一化动作空间；α=0.1 归一化单位 ≈0.08 rad（按 RevoLab action scale 换算固化进配置）；先加和后 clip。
2. **随机性**：探索噪声加在合成均值 μ_t；g、Δa 为确定性函数，梯度经 PPO 目标通过 μ_t 传播；评估取 μ_t。
3. **非对称 critic（训练期特权）**：输入 = proprio + 当前触觉帧 + 扰动激活标志/脉冲向量 + 物体真实质量摩擦/位姿/速度；仅用于优势估计。**B1 使用同一 critic 类、同一隐层 [512,256]、同一特权输入集、同一 value 系数；全部配置由同一 YAML 基模板派生，diff 仅 actor 段。**
4. **触觉预处理五步**：a) `ContactSensor.data.net_forces_w` (E,B,3)，B=5 指尖（MVP）；b) body 四元数旋至 link 本体系；c) EMA(0.8) 平滑 + HORA 式噪声/延迟 DR；d) 每区域 5 维 [log(1+‖f̄‖), f̄_n/(‖f̄‖+ε), ‖f̄_t∥‖/(f̄_n+ε), 接触二值, 二值变化]；e) e_t = [max_b 力差分范数, Σ接触变化, max_b 切法比, 合力差分范数]（固定阈值入配置，~10 维）。
5. **τ_hist**：K=10 平铺（MVP 250 维）。

### Training Objective
L = L_PPO(clip=0.2) + c_v·L_value − c_e·H[π] + w(t)·L_gate-warm + λ_g(t)·E[g] + λ_r·E[‖Δa‖²]
- **L_gate-warm**：BCE(g_t, y_t)，y_t=1 于 [t_inj, t_inj+0.3s]；w(t): 1.0→0 线性退火（前 30% 步）。标签仅训练期；对齐评估防泄漏协议见 Validation。
- λ_g(t)：0→0.01（退火完成后启动）；λ_r=1e-3。
- 奖励 = 任务原生 + r_drop(−10) + r_force(−0.01·max(0,‖f‖_max−20N))。

### 扰动课程（MVP 单族）
p=0.7 每 episode 注入 1–2 次物体外力脉冲（0.1–0.2s，2N→8N 课程，方向随机）；质量/质心/摩擦保持 RevoLab 默认 DR。摩擦骤降为 **eval-only held-out 扰动族**（MVP 仅作门控泛化诊断，不作第二鲁棒性主张）。

### Training Plan 与吞吐降级预案
2048 envs，rollout 24，目标 1e8 步/run；D3 冒烟实测吞吐定档：
- **Tier A**（≤8h/run）：完整 MVP（8 runs）。
- **Tier B**（8–14h/run）：步数降 6e7，8 runs 保留，4 个证明点全保留（同减预算比较；样本效率主张更突出）。
- **Tier C**（>14h/run）：1024 envs 重测；仍超则砍 B2/B3 seed2（6 runs），证明点 3/4 降为单 seed 存在性证据 + 方差告警，完整阶段补 seeds。

### Failure Modes and Diagnostics
- 残差学成零：诊断=E[‖Δa‖] 曲线（窗口内/外分解）；缓解=升 p、增幅值、降 λ_g、开 r_slip（记录触发条件）。
- 门控坍缩：诊断=g 直方图+三层对齐；热启动对冲；仍失败 → fallback=固定阈值触发+学习残差（claim 收缩为"结构化触发优于无结构"）。
- B1 追平：claim 转样本效率 + B1 遗忘度 + 介入可解释性；负结果本身可报告。
- 吞吐不足：Tier B/C 决策树。

### Novelty and Positioning（含邻域 sanity pass 结论）
- **最近邻**：EquiTac (arXiv 2511.07381)——触觉等变残差**旋转**修正：SO(2) 单自由度、平行夹爪、测试期监督学习网络；我们是 21-DoF 关节空间残差 + 学习型介入时机 + 仿真 RL。CATFA (2509.23075)——修正层需**真实示教数据**训练；DexTac——修正律**手调**；Tac2Motion——奖励塑造不改信息流；PTLD (2603.04531)——特权触觉 latent 蒸馏（表征路线，无修正结构）；Contact-Grounded Policy (2603.05687)——IL 接触一致性映射；TacCoRL (2606.11743)——VLA 后训练；RESPRECT 等残差 RL——非触觉事件门控。
- **Novelty 措辞（收窄版）**："据我们所知，首个面向多指灵巧手、完全由仿真 RL 学得（无真机示教、无固定修正律）、带学习型介入时机的触觉事件门控关节空间残差修正框架。" 同时准备无 "first" 措辞版本：机制 + 四项受控证据主张自足。

## Claim-Driven Validation（MVP：4 个证明点）

**基线定义（公平性契约）**：
- **B0**：冻结 π_base，eval-only。
- **B1（端到端触觉观测重训，强基线）**：actor 从 proprio checkpoint 热启动，触觉输入零初始化新列并入首层（net2net 加宽，t=0 功能等价 π_base）；触觉特征/历史与 TacRes 完全相同；PPO 预算、扰动课程一致；critic 按上述架构匹配条款完全一致；额外评测无扰动原任务成功率（遗忘度）。
- **B2（always-on 残差）**：TacRes 同构但 g≡1；比较用修正能量 E[‖α·g·Δa‖]（扰动窗口内/外分解）与阈值化占空比（‖αgΔa‖>0.1α）。
- **B3（触觉因果消融）**：TacRes 同构同预算，训练与评估全程 τ_hist≡0 且 e_t≡0。另加 eval-time 探针：训好的 TacRes 评估时双路触觉置零 → 预期坍缩至 ≈B0。

**4 个证明点**：
1. 退化存在：B0 在 5N/8N 脉冲下成功率大幅下降（motivation）。
2. 结构优势：TacRes 扰动 AUC > B1 ≥8pp 或同性能样本效率 ≥3×；TacRes 无扰动原任务成功率不低于 π_base（结构性无遗忘），对照 B1 遗忘度。
3. 门控价值：TacRes 窗口外修正能量 ≪ B2、窗口内相当、性能不劣。
4. 触觉因果：B3 无增益；TacRes eval-time 触觉置零探针坍缩。

**统计报告规程（2-seed 预定义）**：结果按 seed 分别报告；每评估点 ≥256 episodes；成功率差 95% CI 用 episode 级 bootstrap（seed 内）；证明点通过判定 = 两 seed 方向一致且各自 CI 不跨零；报告绝对/相对效应量；正文注明 seeds=2 限制，完整阶段 3 seeds。

**门控对齐三层评估（防标签泄漏）**：
(a) 注入窗口对齐 P/R@±0.1s（分布内，降权）；
(b) 物理异常对齐：真值由仿真状态独立计算（滑移起始=接触点切向相对速度越阈；接触构型突变），与注入时间戳解耦；
(c) held-out 摩擦骤降泛化（训练未见此族；MVP 仅诊断）。

- 任务：Dexsuite-Revo3-Lift；评估：脉冲幅值 {2,5,8}N × ≥256 episodes。
- 运行矩阵：TacRes×2 + B1×2 + B2×2 + B3×2 = 8 runs。
- 完整阶段（+4 周）：全掌 vs 指尖轴（未完成则触发定位降级预案）、摩擦/质量扰动训练族、HoraRotate-Cylinder、no-warm-start、K=1、λ_g 扫描、3 seeds、留出物体。

## Experiment Handoff Inputs
- Must-prove claims: MVP 4 证明点。
- Must-run ablations（完整版）: 全掌 vs 指尖、no-warm-start、无历史、λ_g 扫描。
- Critical metrics: 扰动 AUC、样本效率、三层对齐 P/R、修正能量（窗口内/外）、遗忘度。
- Highest-risk assumptions: 稀有事件下残差可学（D5 go/no-go）；吞吐 Tier A/B（D3 实测）。

## Two-Week Execution Schedule
- **D1**：跑通 RevoLab Lift 训练/评估（现有 checkpoint 或短训复现）；确认 `fingers_contact_force_b` 数据流；实现力脉冲事件生成器 + 注入日志。
- **D2**：触觉预处理五步流水线 + 断言测试；滑移起始/接触突变独立真值计算器；全掌 ContactSensorCfg 代码就绪不启用；B0 扰动退化评估（motivation 数据）。
- **D3**：TacRes ActorCritic 包装器（冻结 base+残差+门控+非对称 critic+热启动损失）；YAML 基模板（TacRes/B1/B2/B3 派生，diff 仅 actor 段）；冒烟训练实测吞吐 → Tier 定档。
- **D4–5**：TacRes 首个完整 run + α/λ_g/热启动窗口小调；**D5 傍晚 go/no-go**（残差非零且扰动成功率回升 → 继续；否则 fallback 分支）。
- **D6–7**：夜间队列 B1/B2/B3（seed1）；白天写评估协议（幅值扫描、三层对齐、修正能量分解、bootstrap CI）；D5 通过则机会性全掌对照。
- **D8–9**：seed2 补齐；plotting 脚本。
- **D10**：全量评估 + 主表格 + held-out 摩擦骤降诊断。
- **D11**：四张主图：①幅值–成功率（B0–B3+TacRes，按 seed）；②同预算学习曲线；③修正能量/门控时间轴 vs 事件；④触觉置零探针坍缩。
- **D12**：机动日。
- **D13**：方法节+实验节草稿（含 related-work 定位段）。
- **D14**：缓冲 + 完整阶段 4 周矩阵修订。

## AAAI Paper Storyline
- 标题：*TacRes: Learning When and How to Correct — Contact-Event-Gated Tactile Residual Policies for Robust Dexterous Manipulation*。
- 摘要主线：触觉不应只被观测而应驱动修正；全仿真习得门控残差；扰动鲁棒 +X pp、样本效率 Y×、介入稀疏且与接触异常对齐（防泄漏验证）、双重因果检验；（若全掌轴完成）全掌 vs 指尖消融回应硬件趋势。
- 图故事：Fig.1 信息流对比；Fig.2 幅值–成功率；Fig.3 修正能量/门控时间轴；Fig.4 消融矩阵。

## Risks and Fallbacks
- 稀有事件学习失败 → D5 go/no-go + 三旋钮 + r_slip 备用 + fallback=固定触发。
- 门控坍缩 → 热启动对冲；仍败 → claim 收缩为"结构化触发优于无结构"。
- B1 追平 → 样本效率+遗忘度+可解释性主线；负结果可报告。
- 吞吐 → Tier A/B/C 决策树。
- 全掌轴缺失 → 定位降级预案：标题/摘要/贡献移除全掌措辞，定位为通用触觉事件门控残差机制（指尖阵列验证）。
- "first" 质疑 → novelty 收窄措辞 + 无 first 版本备用 + 邻域定位段（EquiTac/CATFA/PTLD/CGP/TacCoRL）。
- 热启动功劳质疑 → 完整阶段 no-warm-start 消融。

## Compute & Timeline Estimate
- MVP：8 runs × 6–8h（Tier A）≈ 48–64 GPU·h；Tier B ≈ 40h；数据成本零。
- 完整阶段：~25 runs ≈ 175 GPU·h（4 周夜间队列）。

## Immediate Next Actions
1. D1 上午：`cd ~/RevoLab && python scripts/rsl_rl/play.py --task BrainCo-Dexsuite-Revo3-Right-Lift-v0`。
2. D1 下午：`perturbation_events.py`（力脉冲 EventTerm + 注入日志）。
3. D2：`tactile_obs.py` 五步流水线 + 独立事件真值计算器 + 断言测试。

