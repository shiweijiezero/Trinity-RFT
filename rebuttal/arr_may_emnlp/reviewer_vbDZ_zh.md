# 审稿人 vbDZ（中文翻译）

## 评分

- 置信度：4（相当确定）
- 可靠性（Soundness）：3（可接受）
- 兴奋度（Excitement）：3.5
- 总体评价：3（Findings）
- 可复现性：3
- 数据集：1
- 软件：1

---

## 论文总结

本文提出 R³L（Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit Assignment, and Positive Amplification），这是一种用于提升大语言模型在稀疏奖励环境中的推理和智能体任务表现的强化学习框架。

其核心思想是解决当前大语言模型强化学习方法的两个主要限制：

1. 随机采样成功率较低，导致探索效率低下；
2. 轨迹级信用分配以及失败信号占主导，导致利用过程不稳定。

作者为此引入三个关键组件：

1. **Reflect-then-Retry（语言引导的探索）。** 模型利用语言反馈分析失败轨迹，识别 pivot failure point，并从这些位置开始重新生成改进后的后缀。
2. **Pivotal Credit Assignment。** 不再对整条轨迹统一分配奖励，而是通过梯度 mask，使只有 pivot 之后产生分歧的后缀参与优化，从而保护正确的前缀。
3. **Positive Amplification。** 对成功轨迹进行增权，避免失败样本占主导的 batch 稀释正向梯度。

在智能体任务（ALFWorld、WebShop、ScienceWorld）和数学推理 benchmark（GSM8K、Math500 等）上的实验表明，该方法相对 GRPO、GSPO 和基于 critique 的 baseline 取得了一致提升；论文报告各设置上的提升为 5%–52%。论文还提供了关于熵坍塌、方差降低和收敛行为的理论分析。

## 优点总结

### 1. 新颖性与理论基础

论文识别并形式化了失败样本占主导的强化学习组中的“熵坍塌”和梯度不对称问题。这一观察在理论上合理，也很有洞见。所提出的机制优雅地针对这些特定瓶颈，而且不需要昂贵的外部过程奖励模型。

### 2. 方法设计简洁优雅

Pivotal Credit Assignment 是一种非常直观且计算高效的时序信用分配方法。它把共享前缀视为控制变量，并在梯度更新中将其 mask 掉，从而干净地隔离真正发生决策分歧的位置。

### 3. 实证表现强

论文使用现代基础模型（例如 Qwen2.5 和 Llama-3.2），在智能体任务和数学推理 benchmark 上进行了全面评测，有说服力地展示了该框架的有效性。相对 GRPO、GSPO 和 Critique-GRPO 等强 baseline 的一致提升，体现了方法的实际价值。

## 缺点总结

尽管论文有上述优点，它仍存在若干可能影响整体影响力和清晰度的限制。

### 1. 高度依赖 reflection 和 pivot localization 的正确性

该方法关键依赖于基于 reflection 的 pivot identification（$k_{pivot}$）是否准确。然而：

- 表 11 显示，即使在训练后期，误诊率仍为 18%–38%；
- 图 6 展示了 pivot 的不稳定性，例如 WebShop 的 pivot 又退回到较早的 step；
- Retry success 高度依赖定位是否正确：正确 pivot 的成功率为 85%，错误 pivot 的成功率为 43%（表 11）。

这形成了一条脆弱的依赖链：

`reflection → pivot → mask → credit assignment → update`

Reflection 中的错误会直接传播为错误的梯度更新。

### 2. 消融实验没有完全解耦组件之间的交互

表 2 的消融结果显示：

- w/o Reflect：下降最大（0.928→0.894）；
- w/o Credit：中等幅度下降；
- w/o Positive：中等幅度下降。

然而，移除 reflection 也会关闭 pivot detection，进而关闭 credit masking（第 5.3 节）。因此，“reflection ablation”同时混合了多个耦合组件的移除，无法明确地在以下因素之间进行因果归因：

- reflection；
- pivot masking；
- retry synthesis。

### 3. 评测仅限于具有可验证奖励的环境

所有实验均局限于 ALFWorld、WebShop、ScienceWorld 和数学 benchmark，没有评测：

- 开放式生成；
- 使用主观 preference 的 RLHF 任务。

因此，该方法能否推广到可验证奖励强化学习之外，仍未得到检验。

## 评论、建议与拼写问题

除上述主要问题外，审稿人还提出以下意见和建议，以帮助提升论文的清晰度和呈现质量。

1. Pivot point $k_{pivot}$ 的定义是方法的核心，但其识别过程描述得不够具体。第 4.1 节介绍了基于 reflection 的诊断，但当一条轨迹同时存在多个失败原因时，pivot extraction 是否确定、是否稳定并不清楚。给出更明确的规则或增加 pivot sensitivity ablation，会提高方法严谨性。
2. 论文称 retry 根据 reflection 的结果（成功、失败或低效）触发，但 decision boundary 没有完全形式化。Retry 究竟是确定性还是概率性的，以及如何处理带噪声的 reflection output，都不清楚。对 retry triggering 给出形式化定义会提高可复现性。

## 伦理问题

本投稿不存在伦理方面的担忧。

是否需要伦理审查：否。

## 其他字段

- 是否知道或推测作者身份：否。
- 是否从外部来源了解本文：不适用；审稿人未从外部来源了解本文。
- 是否知道论文来源：不适用；审稿人未从外部来源了解论文来源。
- 上述知识是否影响评审：不适用；审稿人未从外部来源了解本文。
- 审稿人认证：审稿人确认其评审准确反映了本人对该工作的评价。若使用了任何自动化工具，其用途仅限于改善语法和文风，评审实质内容来自审稿人本人或已注明的第二审稿人。
- 出版伦理政策合规：审稿人未使用任何生成式 AI 工具完成本评审。

## 回复草稿

感谢审稿人对方法设计和实验结果的认可，也感谢您围绕 reflection reliability、组件耦合和适用范围提出的具体问题。很抱歉回复得比较晚。由于实验资源有限，我们在 rebuttal 周期内优先完成了与这些问题直接相关的补充实验和检查。

> 关于 reflection 与 pivot localization 的可靠性

我们认同定位误差会影响 retry 从何处开始以及哪些 token 参与更新，但它不会不经验证地直接改写 reward 并转化为“错误方向的梯度”。对于任意预测位置 $k$，retry 都会回滚并复用 $\tau_{<k}$，因此 base 与 retry 在该前缀上按构造完全相同；两条轨迹使用同一个 Pivotal Credit mask，只更新分歧后缀。Pivot 偏早时，方法重新生成并训练更长的后缀，增加计算与方差，并逐渐退化为接近从头 rollout；pivot 偏晚时，方法可能复用真正的错误动作并遮蔽相应的学习信号，从而降低 retry 成功率。两种误差会分别引入额外方差或遗漏学习信号，但 base/retry 的 reward 仍由环境 verifier 独立决定，失败的 retry 会按照实际 reward 计算 advantage，只有验证后确实改进的 correction 才进入辅助 reflection/retry SFT。因此，pivot 误差主要通过样本效率和信号完整性影响训练，而不是将 reflection 的文本判断无条件地当作 reward label 传播。

我们在prompt中明确了 pivot 规则：存在多个问题时，`retry_from_step` 取根因首次显现、或最早可以通过修正决策改变结果的 turn；根因来自初始策略时取 0 。
另外表 11 中 correct/wrong pivot 的条件成功率确实只能说明相关性。

我们进一步在 Qwen2.5-1.5B-Instruct / ALFWorld 的 step 400 checkpoint 上补充了一组 pivot sensitivity 实验，直接检验 pivot 定位偏差会在多大程度上影响 retry success，以及这条依赖是否会使 R³L 对小幅定位误差过于敏感。对同一条失败轨迹，我们固定 reflection guidance，只改变 retry 的起点。模型原本预测的位置为 $k$，其他几组分别从 $k-2$、$k+2$、$k-5$、$k+5$ 和第 0 步开始，其余设置保持不变。超出轨迹边界的样本不计入对应 offset。

Start=0 表示忽略预测的 pivot、直接从头 retry，用来检验 localized restart 的作用。它与表 2 的 `w/o Credit` 不同：`w/o Credit` 仍然从 $k$ 开始 retry，只是不使用 prefix mask。表中的 Retry success rate 是指：在 base rollout 失败且 reflection 有效的轨迹中，retry 最终完成任务并获得成功 reward 的比例。

| Restart position | Predicted pivot $k$ | $k-2$ | $k+2$ | $k-5$ | $k+5$ | From start ($k=0$) |
|---|---:|---:|---:|---:|---:|---:|
| Retry success rate | 0.66 | 0.65 | 0.63 | 0.63 | 0.60 | 0.61 |

小幅提前 pivot 的影响很小，$k-2$ 只从 0.66 降到 0.65；偏晚的影响更明显，$k+5$ 降到 0.60，因为真正的错误动作可能已经被保留在 prefix 中。从第 0 步重新开始的成功率为 0.61，也低于模型预测的 pivot。说明 R³L 并不要求 pivot 完全精确，但较大的误差，尤其是偏晚定位，确实会降低 retry 的效果。

> 关于组件消融的耦合

`w/o Reflect` 这一行确实不是 reflection-only ablation。

由于 reflection 同时生成 guidance 和 pivot，移除 reflection 后，retry synthesis 和依赖 pivot 的 Credit 也会一起受到影响。因此，0.928 到 0.894 只能说明整条 Reflect-then-Retry 路径的联合贡献，不能单独作为 reflection 的贡献。

现有实验中，w/o Credit 保留相同的 reflection 和 retry，只关闭 prefix mask，因此 0.928 到 0.914 可以用来观察 Credit 在完整 R³L 中的增量。新补充的 pivot sensitivity 则固定 trajectory 和 guidance，只改变 restart position，用来检查 pivot localization 的影响。

不过，checkpoint 上的 pivot sensitivity 只能反映一次 retry 的变化，还不能代替完整训练后的 final score。我们因此在同样的 Qwen2.5-1.5B-Instruct / ALFWorld 设置下补了两组完整训练：`w/o Credit + From start` 保留相同的 reflection 和 guidance，但统一从第 0 步开始 retry；`w/o Credit + No guidance` 仍然使用 reflection 预测的 $k$，但在生成 retry 时不提供 guidance。两组都关闭 Credit，其余训练设置与 `w/o Credit` 相同。

| Variant | Guidance | Restart position | Credit mask | Final score |
|---|---:|---:|---:|---:|
| R³L | Yes | Predicted $k$ | Yes | 0.928 |
| w/o Credit | Yes | Predicted $k$ | No | 0.914 |
| w/o Credit + From start | Yes | $k=0$ | No | 0.908 |
| w/o Credit + No guidance | No | Predicted $k$ | No | 0.903 |
| w/o Reflect | No | No retry | No | 0.894 |

`w/o Credit` 从 0.914 降到 From start 的 0.908，说明 localized restart 不只提高单次 retry success，也会反映在最终训练结果上；去掉 guidance 后进一步降到 0.903，说明 language guidance 也有独立增量。这样，R³L 与 `w/o Credit` 比较 prefix masking，`w/o Credit` 与 From start 比较 restart position，`w/o Credit` 与 No guidance 比较 guidance。`w/o Reflect` 则仍然表示整条 Reflect-then-Retry 路径被移除，而不是 reflection-only ablation。

> 关于可验证奖励之外的适用范围

我们在 Limitations 中已经说明，当前实验只覆盖具有可验证 ground truth 的智能体和数学任务，尚未验证开放式生成或主观 preference RLHF。

主观 RLHF 仍然是业界很有挑战的场景，也很难直接套用 GRPO 的训练方式。

GRPO 通常需要针对同一个 prompt 采样多条 response，并根据可比较的 reward 计算组内 relative advantage；但真实用户反馈往往只来自一次 interaction，具有稀疏、噪声大、不同用户标准不一致等特点，也很难对同一个场景反复 rollout 并获得可比较的反馈。

R³L 目前同样需要可靠判断 base 和 retry 哪一个更好，因此还不能直接覆盖这类设置。如何利用不完整且带噪声的用户反馈构造稳定的 reflection、retry 和 preference signal，仍然需要进一步研究。

> 关于 retry trigger 的形式化定义

是否 retry 是模型自己通过 reflection 决定的。每条 base trajectory 都先由模型生成一份 reflection，其中包含 `trajectory_outcome` 和 `retry_from_step`。系统不根据 reward 另设 threshold，也不再做一次概率采样；给定模型生成的 reflection 后，只进行确定性的合法性检查。模型判断为 `failure` 或 `success_but_inefficient`，且 reflection 格式有效、`retry_from_step` 没有越界时，才会执行 retry。写成公式就是

\[
I_{\mathrm{retry}}(\tau,r)=
\mathbb{1}\!\left[
\operatorname{valid}(r)\ \land\
r_{\mathrm{outcome}}\in\{\texttt{failure},\texttt{success\_but\_inefficient}\}\ \land\
0\le r_{\mathrm{retry\_from\_step}}<|\tau|
\right].
\]

这里的 `valid` 是指 JSON 可以解析、必需字段完整且字段取值合法。模型判断为 `success` 时不进行 retry；无效 JSON、缺少字段或越界 pivot 也会被系统跳过，原来的 base trajectory 继续保留。因此，系统只负责检查模型的决定能否执行，并不替模型重新判断轨迹是否成功。如果 reflection 格式正确、语义判断却错了，仍然可能误触发或漏掉 retry，这部分噪声目前无法完全避免。

所以，trigger 没有额外的随机开关；但 reflection 本身由模型生成，不同采样仍可能给出不同判断。

这里的 trigger rate 是实际触发 retry 的数量除以 base rollout 数量，反映的是模型在当前 checkpoint 上通过 reflection 选择 retry 的比例，并不是一个固定超参数。以 Qwen2.5-1.5B-Instruct 在 ALFWorld 上的实验为例：

| Training step | Rollout reward | Retry trigger rate |
|---:|---:|---:|
| 0 | 0.01 | 0.55 |
| 50 | 0.03 | 0.65 |
| 100 | 0.08 | 0.80 |
| 150 | 0.50 | 0.78 |
| 200 | 0.80 | 0.35 |
| 250 | 0.87 | 0.13 |
| 300 | 0.90 | 0.10 |
| 350 | 0.91 | 0.08 |
| 400 | 0.91 | 0.07 |

前 100 steps 中，trigger rate 从 0.55 上升到 0.80，而 reward 仍然较低，说明 trigger rate 并不简单等于 $1-\text{reward}$，还取决于模型如何反思和判断轨迹。之后 reward 从 0.08 提高到 0.91，trigger rate 则从 0.80 降到 0.07。因此，模型还不会完成任务时会有更多 retry，成功率提高后 retry 和实际 rollout 数量都会自然减少。
