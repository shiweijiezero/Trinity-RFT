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
