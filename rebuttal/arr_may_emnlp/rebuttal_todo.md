# ARR May EMNLP Rebuttal Prep Todo

## Highest Priority

| Priority | Issue | Needed action | Reviewers |
|---|---|---|---|
| P0 | Positive Amplification rule inconsistency | 明确 max-reward trajectories 到底赋值为 $1$ 还是 $\alpha$，统一 Equation 11、Table 5 caption、Appendix K.3，并在 rebuttal 中承认这是表述错误还是实现错误 | uCGY |
| P0 | Seed variance and stability | 至少补一个小规模 3-seed study。优先选择主表中的关键 task/model subset，报告 mean/std，并说明已有大表是 single-run | uCGY |
| P0 | KL and importance sampling removal | 补 KL、gradient norm、clip fraction 或 entropy 曲线，至少 WebShop 加一个 math setting。若计算允许，补 re-add KL、re-add IS、re-add both 的 ablation | uCGY, TC9w |
| P0 | Math guidance fairness/leakage | 明确 math guidance 只指出 error type 或 reasoning issue，不泄露答案；说明 baselines 是否有 comparable information；如果没有，要解释这是 trajectory synthesis mechanism 而非 inference-time privilege，并提供控制实验或定性例子 | uCGY |
| P0 | Baselines below zero-shot | 针对 Critique-GRPO、Reflect-GRPO 低于 zero-shot 的现象给直接解释，说明所有数值是 post-RL training，不是 zero-shot，并写清 baseline tuning budget 和相同 training recipe | uCGY |

## Important But Smaller

| Priority | Issue | Needed action | Reviewers |
|---|---|---|---|
| P1 | Pivot quality and sensitivity | 汇总已有 Table 11、Fig. 6，补充 pivot oracle agreement、correct vs incorrect pivot success、pivot error robustness。若可以，补 pivot perturbation 或 wrong-pivot sensitivity | uCGY, vbDZ |
| P1 | Retry trigger rate | 报告 base trajectories 中触发 retry 的比例，分 ALFWorld、WebShop、ScienceWorld、DAPO 或至少代表性任务 | uCGY, vbDZ |
| P1 | Ablation coupling | 承认 w/o Reflect 会影响 pivot/credit，解释 Table 2 的目标是组件移除而非完全正交归因。若可能补 fixed-pivot/no-guidance 或 oracle-pivot ablation | uCGY, vbDZ |
| P1 | Larger model concern | 若已有 Qwen3-4B 结果，强调它是 newer backbone。若可能补一个更大模型或说明 due to compute budget, current evidence focuses on 1.5B-7B and cross-architecture check | TC9w, uCGY |
| P1 | Scope limitation | 承认当前方法面向 verifiable reward tasks，删除或软化 broad preference-signal claim，不把 open-ended RLHF 当作已验证结论 | uCGY, vbDZ |

## Camera-Ready / Revision Cleanups

| Priority | Issue | Needed action | Reviewers |
|---|---|---|---|
| P2 | Promotional framing | 把 5%-52% 改成 over strongest baseline 的 average/median gain，避免把 52% 作为核心卖点 | uCGY |
| P2 | Theory framing | 将 theorem wording 从 guarantee 改为 explanatory analysis 或 intuition，说明 assumptions and empirical ranges | uCGY |
| P2 | Entropy collapse definition | main text 首次出现时定义，不只放在 appendix | uCGY |
| P2 | Reflect-GRPO naming | 首次出现时说明是 Reflect-Retry-Reward 的 reimplementation | uCGY |
| P2 | Submission typos | 修正 `voilate` 为 `violate`，修正 `not human-related data` 为 `no human-related data` | uCGY |

## Suggested Rebuttal Structure

第一段先感谢并概括三点新增或澄清证据，避免防御性语气。重点放在 reviewer 最关心的可验证事项上，例如 seed variance、KL/IS ablation、math guidance fairness、PA rule clarification。

随后按问题块回应，而不是逐 reviewer 重复。建议顺序如下。

1. Stability and variance, 回应 uCGY W1。
2. KL/IS removal and training stability, 回应 uCGY W3 和 TC9w。
3. Math guidance fairness and baseline comparison, 回应 uCGY W4/W5。
4. Pivot quality, retry trigger, and ablation coupling, 回应 uCGY W4/W7 和 vbDZ。
5. PA rule and revision commitments, 回应 uCGY W6 和 typo/framing/scope。

最后补一句 revised manuscript 会收紧 scope claim，明确当前结论限于 verifiable rewards，并把 open-ended or subjective preference RL 留作 future work。
