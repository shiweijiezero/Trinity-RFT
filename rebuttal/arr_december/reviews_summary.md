# R3L Paper Reviews Summary / R3L论文审稿意见汇总

Paper: R³L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification
Submission Number: 4731

---

## Reviewer iAPy

**Scores / 评分:**
- Confidence: 3 (Pretty sure)
- Soundness: 3 (Acceptable)
- Excitement: 3 (Interesting)
- Overall Assessment: 2.5 (Borderline Findings)
- Reproducibility: 4

### Strengths / 优点

1. **Pivot-Based Credit Masking Effectively Addresses Prefix Mis-Credit.**
   The turn-level pivot identification and masking of shared prefixes provide a concrete and practical solution to trajectory-level mis-crediting. Results on long-horizon tasks support its effectiveness in preventing valid early reasoning from being penalized.

   **基于枢纽点的信用掩码有效解决了前缀误分配问题。**
   回合级枢纽点识别和共享前缀掩码为轨迹级信用误分配提供了具体且实用的解决方案。长时任务上的结果支持了其在防止有效早期推理被惩罚方面的有效性。

2. **Positive Amplification is Theoretically Grounded and Empirically Robust.**
   The gradient dominance analysis offers a clear explanation of entropy collapse and justifies the amplification factor. A single fixed alpha works across models and tasks, suggesting the mechanism addresses a structural imbalance rather than requiring heavy tuning.

   **正向放大具有理论基础且实证稳健。**
   梯度主导性分析为熵崩塌提供了清晰解释，并证明了放大因子的合理性。单一固定的alpha在不同模型和任务上均有效，表明该机制解决的是结构性不平衡，而非需要大量调参。

3. **Reflect-then-Retry Shows Sustained Exploration Gains During Training.**
   The reported retry improvement rate and reward gain over time demonstrate that retry trajectories consistently outperform base samples after warm-up, providing stronger evidence of exploration quality than final accuracy alone.

   **反思-重试在训练过程中展现了持续的探索增益。**
   报告的重试改进率和奖励增益随时间的变化表明，重试轨迹在热身后持续优于基础样本，比仅靠最终准确率提供了更强的探索质量证据。

### Weaknesses / 缺点

1. **W1: Novelty concerns - elements have precedents in related literature.**
   While the integration is coherent, elements such as retry-based refinement, advantage reweighting, and partial masking have precedents in related literature.

   **新颖性问题——各组件在相关文献中有先例。**
   虽然整合是连贯的，但基于重试的改进、优势权重重新分配和部分掩码在相关文献中都有先例。

2. **W2: Self-reflection reliability is not directly evaluated.**
   The framework assumes that the model can reliably identify the true error turn and generate an effective correction. However, self-reflection may mislocalize failures or produce superficial diagnoses, and the paper does not provide direct evaluation of pivot accuracy or correction validity beyond final task reward.

   **自我反思的可靠性未被直接评估。**
   该框架假设模型能可靠地识别真正的错误回合并生成有效的修正。然而，自我反思可能错误定位失败或产生肤浅的诊断，而论文并未提供除最终任务奖励之外的枢纽点准确率或修正有效性的直接评估。

3. **W3: Missing compute/cost analysis.**
   The reflection step requires an extra inference pass and auxiliary supervision. The paper does not provide a detailed wall-clock or cost analysis comparing R3L with strong baselines under equal compute budgets.

   **缺少计算/成本分析。**
   反思步骤需要额外的推理过程和辅助监督。论文未提供在相同计算预算下R3L与强基线的详细墙钟时间或成本分析。

### Suggestions / 建议

- Report metrics such as pivot identification accuracy, correction success rate conditioned on detected pivots, or agreement with oracle error locations to strengthen the claim that reflection reliably localizes and fixes errors.

  报告枢纽点识别准确率、基于检测到的枢纽点的修正成功率、或与oracle错误位置的一致性等指标，以加强反思可靠定位和修复错误的论证。

---

## Reviewer 4viW

**Scores / 评分:**
- Confidence: 3 (Pretty sure)
- Soundness: 3.5
- Excitement: 4 (Exciting)
- Overall Assessment: 3.5 (Borderline Conference)
- Reproducibility: 4

### Strengths / 优点

1. **Tackles a real RL pathology.**
   Penalizing valid prefixes is a big issue in long trajectories; pivot masking is a strong and intuitive fix.

   **解决了真实的RL病理问题。**
   在长轨迹中惩罚有效前缀是一个大问题；枢纽掩码是一个强大且直观的修复方案。

2. **Reduces wasted rollouts.**
   Restarting from a pivot can plausibly cut exploration cost substantially.

   **减少了浪费的rollout。**
   从枢纽点重新开始可以大幅降低探索成本。

3. **Strong empirical evaluation.**
   They test across both agentic and math reasoning tasks and multiple model families/sizes.

   **强大的实证评估。**
   在智能体和数学推理任务以及多种模型系列/规模上进行了测试。

### Weaknesses / 缺点

1. **W1: Additional system complexity and overhead.**
   Reflection + retry adds extra inference steps (even if cheaper than full rollouts).

   **额外的系统复杂性和开销。**
   反思+重试增加了额外的推理步骤（即使比完整rollout更便宜）。

2. **W2 (minor): Cold-start for small models.**
   The paper itself notes small models initially can't reflect well and need warm-up.

   **（次要）小模型的冷启动问题。**
   论文本身指出小模型最初无法很好地反思，需要热身。

3. **W3: Risk of reflection errors.**
   Incorrect pivot identification or low-quality reflections could reinforce wrong behavior.

   **反思错误的风险。**
   不正确的枢纽点识别或低质量的反思可能强化错误行为。

### Suggestions / 建议

1. Provide explicit compute comparisons: wall-clock, environment steps, and token budget vs baselines.

   提供明确的计算比较：墙钟时间、环境步骤和token预算与基线的对比。

2. Analyze pivot quality: where do reflections fail, and how often are pivots correct?

   分析枢纽点质量：反思在哪里失败，枢纽点正确的频率有多高？

3. Sensitivity analysis for alpha and group size, and for how many retries are allowed.

   对alpha和组大小进行敏感性分析，以及允许多少次重试。

---

## Reviewer XSY9

**Scores / 评分:**
- Confidence: 4 (Quite sure)
- Soundness: 3 (Acceptable)
- Excitement: 2 (Potentially Interesting)
- Overall Assessment: 2 (Resubmit next cycle)
- Reproducibility: 2

### Strengths / 优点

1. **Correctly identifies a critical bottleneck.**
   The "valid prefix penalization" problem is sound and addresses a known limitation of standard PPO/GRPO.

   **正确识别了关键瓶颈。**
   "有效前缀惩罚"问题是合理的，解决了标准PPO/GRPO的已知局限性。

2. **Reasonable credit assignment via pivot points.**
   The idea of using a "pivot point" to create a contrastive pair (a failed base trajectory vs. a successful retry trajectory) provides a clear signal for the model to learn exactly where the reasoning went astray.

   **通过枢纽点进行合理的信用分配。**
   使用"枢纽点"创建对比对（失败的基础轨迹与成功的重试轨迹）的想法为模型提供了清晰的信号，以准确学习推理在哪里出了错。

### Weaknesses / 缺点

1. **W1: Baseline results are significantly lower than official reports (MOST CRITICAL).**
   According to the Qwen2.5 technical report, Qwen2.5-1.5B-Instruct achieves 73.2 on GSM8K and 55.2 on MATH. However, the authors report a baseline (GRPO) of only 47.4 on GSM8K and 36.7 on MATH. For Qwen2.5-7B-Instruct, the official report is 91.6 on GSM8K, while the authors report 84.6. These discrepancies suggest the baseline models were either severely undertrained, evaluated using suboptimal prompts, or tested on a non-standard subset. The claimed improvements (e.g., 52% relative gain) are likely artifactual.

   **基线结果显著低于官方报告（最关键问题）。**
   根据Qwen2.5技术报告，Qwen2.5-1.5B-Instruct在GSM8K上达到73.2，在MATH上达到55.2。然而作者报告的基线（GRPO）在GSM8K上仅为47.4，在MATH上仅为36.7。对于Qwen2.5-7B-Instruct，官方报告GSM8K为91.6，而作者报告为84.6。这些差异表明基线模型要么严重训练不足，要么使用了次优提示进行评估，要么在非标准子集上测试。所声称的改进（如52%的相对增益）很可能是人为的。

2. **W2: Outdated model backbones.**
   The paper focuses on Qwen2.5 and Llama 3.2 (released in 2024). In 2026, these are considered legacy models. To demonstrate value, it is essential to evaluate on contemporary SOTA models such as Qwen3, Gemma-3, and Olmo-3.

   **过时的模型骨干。**
   论文聚焦于Qwen2.5和Llama 3.2（2024年发布）。在2026年，这些被视为遗留模型。为证明价值，有必要在当代SOTA模型如Qwen3、Gemma-3和Olmo-3上进行评估。

3. **W3: High complexity - collection of heuristics.**
   The method introduces several "moving parts" (reflection, retry, guidance distillation, pivotal masking, and positive amplification). It appears to be a collection of heuristics added atop existing ideas (like Critique-GRPO, VL-Rethinker). The "Positive Amplification" introduces an additional hyperparameter that likely requires task-specific tuning.

   **高复杂性——启发式方法的集合。**
   该方法引入了多个"活动部件"（反思、重试、引导蒸馏、枢纽掩码和正向放大）。它看起来是在现有想法（如Critique-GRPO、VL-Rethinker）之上添加的启发式方法集合。"正向放大"引入了一个可能需要任务特定调参的额外超参数。

4. **W4: Flawed compute-efficiency comparison.**
   The "Reflect-then-Retry" mechanism involves multiple inference passes. The authors claim to control for this by adjusting the number of trajectories, but they do not account for the increased sequence length (token count) from generating natural language reflections and guidance. A true comparison should be conducted under a strict total token budget.

   **有缺陷的计算效率比较。**
   "反思-重试"机制涉及多次推理过程。作者声称通过调整轨迹数量来控制这一点，但他们没有考虑生成自然语言反思和引导所增加的序列长度（token数量）。真正的比较应该在严格的总token预算下进行。

### Suggestions / 建议

1. **Address Baseline Discrepancy:** Provide a detailed explanation for why baseline numbers for Qwen2.5 are ~25-30 points lower on GSM8K/MATH than official figures. If due to specific evaluation protocol, it must be standardized.

   **解决基线差异问题：** 详细解释为什么Qwen2.5的基线数字在GSM8K/MATH上比官方数据低约25-30分。如果是由于特定评估协议，必须标准化。

2. **Modernize the Backbone:** For a 2026 publication, provide results on Qwen3 or Llama 4 to confirm the mechanism isn't just fixing flaws already solved by more advanced base models.

   **现代化模型骨干：** 对于2026年的发表，需要提供在Qwen3或Llama 4上的结果，以确认该机制不只是修复更先进基础模型已经解决的缺陷。

3. **Simplification Study:** Perform a "complexity-drop" study. Can similar gains be achieved by further refining the "Positive Amplification" on top of standard GRPO without the expensive reflection/retry loop?

   **简化研究：** 进行"复杂性递减"研究。是否可以通过在标准GRPO上进一步改进"正向放大"来实现类似的增益，而不需要昂贵的反思/重试循环？

4. **Detailed Token Accounting:** Include a table comparing the average tokens generated per training step for R3L vs. GRPO to ensure a fair comparison of exploration costs.

   **详细的Token核算：** 包含一个比较R3L与GRPO每个训练步骤平均生成token数的表格，以确保探索成本的公平比较。

---

## Summary Comparison / 总结对比

| | Reviewer iAPy | Reviewer 4viW | Reviewer XSY9 |
|---|---|---|---|
| Confidence | 3 | 3 | 4 |
| Soundness | 3 | 3.5 | 3 |
| Excitement | 3 | 4 | 2 |
| Overall | 2.5 | 3.5 | 2 |
| Reproducibility | 4 | 4 | 2 |

## Common Concerns Across Reviewers / 各审稿人共同关注点

1. **Compute/Cost analysis missing** (all three reviewers) — 缺少计算/成本分析（三位审稿人均提到）
2. **Pivot/reflection quality not directly evaluated** (iAPy + 4viW + XSY9) — 枢纽点/反思质量未直接评估（三位审稿人均提到）
3. **Novelty/complexity concerns** (iAPy + XSY9) — 新颖性/复杂性问题（iAPy和XSY9提到）
4. **Baseline discrepancy** (XSY9 only, but critical) — 基线差异（仅XSY9提到，但很关键）
5. **Outdated model backbones** (XSY9 only) — 过时的模型骨干（仅XSY9提到）
