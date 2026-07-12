# ARR May EMNLP Reviews Summary

Paper: R3L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification

Submission Number: 2874

Venue preference: EMNLP

Source snapshot: `openreview_raw.txt`

OpenReview dates in export: submitted on 22 May 2026, modified on 18 Jun 2026. Reviews shown were posted on 02 Jul 2026 and 05 Jul 2026, then modified on 09 Jul 2026.

## Score Overview

| Reviewer | Overall | Soundness | Excitement | Confidence | Reproducibility | Software | Main read |
|---|---|---|---|---|---|---|---|
| uCGY | 3, Findings | 3 | 3 | 4 | 3 | 3 | 正面但非常细，核心要求是 variance、KL/IS ablation、math guidance fairness、baseline 解释和 PA 公式一致性 |
| TC9w | 4, Conference | 3.5 | 4 | 4 | 5 | 4 | 明显支持，主要补充点是 KL/IS ablation 和更大模型 |
| vbDZ | 3, Findings | 3 | 3.5 | 4 | 3 | 1 | 正面但担心 pivot/reflection 可靠性、组件耦合、verifiable reward 范围限制 |

## Overall Situation

这轮比之前更积极。TC9w 给 conference，uCGY 和 vbDZ 都给 Findings。三位 reviewer 都认可问题分解和方法动机，TC9w 对 reproducibility 和 software 评价很高，uCGY 也承认 compute efficiency 是实质优势。最需要处理的是 uCGY 的八点弱点，因为它们覆盖了大部分可能影响最终讨论的问题。

共性关切集中在四类。第一，实验稳定性，尤其是 RL single-run 没有 seed 方差。第二，reflection 和 pivot 是否可靠，包含 oracle agreement、pivot localization error、retry trigger、pivot sensitivity。第三，方法组件和标准 RL safeguard 的关系，尤其是同时去掉 KL 和 importance sampling 的证据不足。第四，评测范围和公平性，包含数学任务是否使用 ground-truth-derived guidance、baseline 是否低于 zero-shot、模型规模是否够大。

## Reviewer uCGY

Scores: Confidence 4, Soundness 3, Excitement 3, Overall 3 Findings, Reproducibility 3, Datasets 2, Software 3.

Strengths:

- 问题分解清楚，三个 failure modes 和三个机制一一对应。
- 实验覆盖面广，包含多个 backbone、agentic 和 math 任务、多个 baseline 和多种 ablation。
- compute efficiency 是真实优势，Table 9-10 显示 rollout time 更低，multi-step tasks 的 training tokens 约减半。
- ablation 有信息量，retry count 说明一次 retry 已经捕获主要收益，更多 retry 在固定预算下反而伤害。

Weaknesses:

- W1, single-run results 没有 variance estimates，削弱 stability claim。Reviewer 要求至少主表或子集报告 3+ seeds 的 mean/std，最好有 seed-variance curves。
- W2, theoretical section 和实际训练耦合不够，应作为 intuition 而不是 guarantees。Reviewer 质疑 gradient-dominance 条件、$\alpha=3$ 覆盖范围、off-policy mismatch 和 pivot accuracy 影响。
- W3, 同时去掉 KL regularization 和 importance sampling 是强改动，但证据主要来自 ALFWorld 一个任务。Reviewer 要求 WebShop 和一个 math setting 的 KL、gradient norm、clip fraction 曲线，并最好 ablate re-add KL or IS。
- W4, math retry guidance 可能存在 fairness/leakage，因为 Appendix K.3 写到 guidance based on comparison with ground-truth answer。Reviewer 要求说明 baseline 是否得到 comparable ground-truth-conditioned signal，以及结果对 pivot-localization error 的敏感性。
- W5, baseline reproduction 需要解释低于 zero-shot 的情况，尤其是 Critique-GRPO 和 Reflect-GRPO 在 Qwen2.5-7B GSM8K 上低于 zero-shot reference。
- W6, Positive Amplification 规则存在定义不一致。Equation 11、Table 5 caption、Appendix K.3 对 max-reward trajectories 是给 $1$ 还是 $\alpha$ 不一致。
- W7, framing 略 promotional。52% relative improvement 来自 Qwen2.5-1.5B-Instruct GSM8K 对 GRPO，但该设置 Critique-GRPO 强于 R3L。Reviewer 建议报告 over strongest baseline 的 typical gains。另指出 w/o Reflect 同时 disable Pivotal Credit，ablation 耦合。
- W8, scope bounded。所有 backbone 不超过 7B，任务都有 verifiable rewards，general preference-signal domain 没有测试。

Additional suggestions:

- 小规模 seed study 是 rebuttal 最高价值补充。
- 无论结果如何都需要修正 PA rule。
- 报告 retry trigger rate，帮助解释 Table 9 的真实 rollout cost。
- 修正 checklist typo，包括 `voilate` 和 `not human-related data`。
- 首次使用 entropy collapse 时给出明确定义。
- 说明 Reflect-GRPO 是对 Reflect-Retry-Reward 的 reimplementation。

## Reviewer TC9w

Scores: Confidence 4, Soundness 3.5, Excitement 4, Overall 4 Conference, Reproducibility 5, Datasets 1, Software 4.

Strengths:

- C1-C3 问题分解清晰，方法组件匹配，Table 2 ablation 易解释。
- 实验和 ablation 支撑充分，认为 Reflect-then-Retry 是 dominant contributor。
- ALFWorld 和 DAPO case studies、pivot evolution analysis 提供了机制层面的 qualitative grounding。
- prompts、hyperparameters、dataset stats、baseline settings、anonymized code link 支撑 reproducibility。

Weaknesses:

- Dropping importance sampling and KL is justified by theory but not ablated against keeping them.
- 当前模型规模偏小，希望看到 large-size model 和 strong baseline 上的表现。

## Reviewer vbDZ

Scores: Confidence 4, Soundness 3, Excitement 3.5, Overall 3 Findings, Reproducibility 3, Datasets 1, Software 1.

Strengths:

- entropy collapse 和 failure-dominated gradient asymmetry 的识别有理论价值。
- Pivotal Credit Assignment 直观高效，通过 shared prefix masking 隔离真正分歧。
- agentic 和 math benchmarks 上的结果显示 practical utility。

Weaknesses:

- 对 reflection 和 pivot localization 正确性依赖高。Reviewer 引用 Table 11，认为 later training steps 仍有 18%-38% misdiagnosis，Fig. 6 显示 WebShop pivot drift，Table 11 显示 correct pivot retry success 85% vs incorrect pivot 43%。
- Ablation 没有完全 disentangle interaction effects，w/o Reflect 会同时 disable pivot detection 和 credit masking，因此 reflection、pivot masking、retry synthesis 的因果归因不清楚。
- Evaluation 限于 verifiable reward environments，没有 open-ended generation 或 subjective preference RLHF tasks，generalization 未测试。

Suggestions:

- 更清晰定义 pivot point extraction，尤其是多个 failure causes 同时存在时是否 deterministic/stable。
- 形式化 retry trigger decision boundary，说明 retry 是 deterministic 还是 probabilistic，以及 noisy reflection outputs 如何处理。
