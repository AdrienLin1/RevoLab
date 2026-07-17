你现在需要在当前代码库中实现一个全新的训练框架，用于 Dexsuite-Revo3-Lift 任务上的 TacRes 方法。请先阅读代码结构，再动手修改，不要直接大范围重构。

已知信息：

1. 当前环境中，下面命令可以在名为 brain_co 的 conda 环境里正常运行 Dexsuite-Revo3-Lift 任务：

   python scripts/rsl_rl/train.py \
     --task BrainCo-Dexsuite-Revo3-Right-Lift-v0 \
     --num_envs 4096 \
     --headless

2. 当前原始任务中，actor 和 critic 网络的输入都包含 5 指 3D 接触力信息。

3. FINAL_PROPOSAL_20260705_1547.md 是当前论文方法的具体实现方案。请优先阅读该文档，并以其中的方法设计为准。如果文档和现有代码存在冲突，请先遵循现有代码架构，在不破坏已有任务的前提下实现新任务。

总体目标：

请基于现有 BrainCo-Dexsuite-Revo3-Right-Lift-v0 任务，新增两个任务：

- TacRes-phase1
- TacRes-phase2

这两个任务要能通过 rsl_rl 的 train.py 正常启动，并且不要影响原有 BrainCo-Dexsuite-Revo3-Right-Lift-v0 任务的运行。

====================
阶段 1：TacRes-phase1
====================

目标：训练 base 策略，也就是 proprio-only base policy。

请实现一个新任务 TacRes-phase1，用于训练 base 策略。这个阶段的核心要求如下：

1. base 策略不应该使用触觉 / 接触力输入。
   - 原始任务中 actor 和 critic 都包含 5 指 3D 接触力。
   - TacRes-phase1 中需要构建“只使用本体感觉 proprioception”的 observation 配置。
   - 不要把 5 指 3D 接触力输入给 actor。
   - 如果 critic 默认也使用接触力或特权信息，请检查并显式处理，保证 base 阶段对应的是“没见过触觉和扰动的普通策略”。

2. TacRes-phase1 不启用扰动课程。
   - 不要启用扰动课程。
   - 不要启用触觉相关随机化。
   - 保持常规域随机化即可，除非 FINAL_PROPOSAL_20260705_1547.md 明确有不同要求。

3. TacRes-phase1 需要能够独立训练并保存 checkpoint。
   - 请确保任务注册、环境配置、PPO 配置、runner 配置都完整。
   - checkpoint 后续会被 TacRes-phase2 加载为 frozen base policy。

4. 不要破坏现有任务。
   - 原始 BrainCo-Dexsuite-Revo3-Right-Lift-v0 仍然应该可以用原命令运行。

完成 TacRes-phase1 后，请进行最小化 smoke test：
- 使用较小 num_envs，例如 64 或 128。
- 使用 headless。
- 尽量把训练迭代数降到 1～5 步；如果 CLI 参数名不确定，请先从代码中确认。
- 目标是验证任务能注册、环境能创建、PPO runner 能启动、一次训练循环能跑通。

====================
阶段 2：TacRes-phase2
====================

目标：冻结 phase1 得到的 base 策略，训练 residual policy 和 gating network。

请新增任务 TacRes-phase2，用于训练残差和门控网络。核心要求如下：

1. TacRes-phase2 需要加载 TacRes-phase1 的 base checkpoint。
   - base policy 从 phase1 checkpoint 初始化。
   - base policy 在 phase2 中必须冻结。
   - base 相关参数 requires_grad=False。
   - optimizer 不应该包含 base policy 参数。
   - base policy 只负责前向推理，不参与梯度更新。

2. TacRes-phase2 需要启用扰动课程环境。
   - phase2 是在开启扰动课程的环境中训练。
   - 具体扰动课程、调度、随机化方式请参考 FINAL_PROPOSAL_20260705_1547.md。
   - 如果文档中有明确的扰动强度、课程进度或配置项，请按文档实现。
   - 如果现有代码已有类似 perturbation / curriculum / randomization 机制，请尽量复用。

3. TacRes-phase2 需要训练以下可学习模块：
   - r_res：residual MLP / residual policy
   - g：gating MLP / gate network
   - critic：从零开始训练的 critic
   - log_std：如果当前 PPO actor 结构中 log_std 是可学习参数，则 phase2 中也需要正常训练它

4. phase2 的 critic 不能复用 phase1/base 的 critic。
   - 原因：base critic 没有见过触觉信息和扰动课程环境，它在 phase2 中的价值估计会不匹配。
   - 因此 phase2 的 critic 必须重新初始化并从零训练。
   - 不要加载 phase1 checkpoint 中的 critic 参数到 phase2 critic。
   - 如果为了兼容 checkpoint 读取需要过滤 state_dict，请明确实现过滤逻辑。

5. phase2 的 actor 行为应符合 TacRes 思路。
   - base action 由 frozen base policy 输出。
   - residual network 输出残差 action。
   - gate network 输出门控系数。
   - 最终 action 应该由 base action、residual action 和 gate 按文档定义组合。
   - 请注意 action scale、clip、normalization、log_prob 计算与 PPO 更新之间的一致性。
   - 如果现有 ActorCritic 结构需要扩展，请尽量用新增类实现，例如 TacResActorCritic，而不是破坏原有 ActorCritic。

6. phase2 的 observation 设计请严格区分：
   - base policy 前向时只能看到 phase1 中定义的 proprio-only observation。
   - residual / gate 可以使用文档中允许的输入，例如触觉、接触力、扰动相关信息或扩展 observation。
   - critic 可以使用 phase2 训练所需的 critic observation / privileged observation，但必须是新初始化的 critic。

7. 需要保证 PPO 更新只更新 phase2 目标参数。
   - 检查 optimizer 参数列表。
   - 检查 loss backward 后 base 参数梯度始终为 None 或 0。
   - 可以加入断言或 debug log，但不要让正常训练输出过于冗长。

完成 TacRes-phase2 后，请进行最小化 smoke test：
- 使用较小 num_envs，例如 64 或 128。
- 使用 headless。
- 使用 phase1 smoke test 产生的 checkpoint，或者在没有真实长训 checkpoint 时创建一个最小可用 checkpoint 用于加载验证。
- 尽量把训练迭代数降到 1～5 步。
- 目标是验证：
  1. TacRes-phase2 任务能注册；
  2. base checkpoint 能成功加载；
  3. base 参数被冻结；
  4. residual、gate、critic、log_std 能进入 optimizer；
  5. 一次 PPO 训练循环能正常跑通。

====================
实现要求
====================

请按以下顺序工作：

1. 阅读 FINAL_PROPOSAL_20260705_1547.md。
2. 阅读现有任务注册、环境配置、observation 构建、ActorCritic、PPO runner、train.py 参数解析相关代码。
3. 先给出你准备修改 / 新增的文件列表和实现计划。
4. 实现 TacRes-phase1。
5. 对 TacRes-phase1 做 smoke test。
6. 实现 TacRes-phase2。
7. 对 TacRes-phase2 做 smoke test。
8. 最后总结：
   - 新增了哪些文件；
   - 修改了哪些文件；
   - TacRes-phase1 如何启动训练；
   - TacRes-phase2 如何指定 base checkpoint 并启动训练；
   - smoke test 的实际命令和结果；
   - 还有哪些需要长时间训练才能验证的内容。

请注意：

- 不要删除或破坏原有 BrainCo-Dexsuite-Revo3-Right-Lift-v0 任务。
- 不要把 phase2 的 critic 从 phase1 checkpoint 中加载出来继续训练。
- phase1 的 base policy 必须是 proprio-only。
- phase2 的 base policy 必须 frozen。
- phase2 中只有 residual、gate、critic、log_std 等目标参数可以被 PPO 更新。
- 如果发现 FINAL_PROPOSAL_20260705_1547.md 中某些细节和代码结构不完全对应，请先采用最小侵入式实现，并在总结中说明取舍。