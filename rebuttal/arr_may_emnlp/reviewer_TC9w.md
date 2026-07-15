# Reviewer TC9w

**Scores:**
- Confidence: 4 (Quite sure)
- Soundness: 3.5
- Excitement: 4 (Exciting)
- Overall Assessment: 4 (Conference)
- Reproducibility: 5
- Datasets: 1
- Software: 4

---

## Paper Summary

The paper proposes R³L (Reflect-then-Retry Reinforcement Learning), a method for RL post-training of LLMs on multi-turn agentic tasks and mathematical reasoning that targets three claimed failure modes of GRPO-style training: (C1) inefficient stochastic exploration on hard, sparse-reward tasks; (C2) trajectory-level credit assignment that penalizes valid prefixes when a later step errs; and (C3) gradient asymmetry in failure-dominated groups where destructive (error-suppressing) gradients overwhelm the rare constructive ones, driving entropy collapse.

R³L has three components:

Language-Guided Reflect-then-Retry: half the rollout budget is base sampling; each failed/inefficient base trajectory gets a structured JSON reflection (outcome class, root-cause, improvement suggestion, pivot turn), then a corrected suffix is regenerated from the pivot conditioned on the guidance.
Pivotal Credit Assignment: a binary mask zeros gradients for all turns before the pivot, focusing updates on the diverging suffix; motivated as a control-variate variance reduction.
Positive Amplification: after group-relative normalization, positive advantages are scaled by α (α=3), max-reward trajectories set to α, negatives unchanged. Importance sampling and KL are dropped entirely.
Contribution type: primarily a method/algorithm paper with substantial empirical analysis.

## Summary Of Strengths

Clear problem decomposition and matching design. C1–C3 are stated crisply and each component maps to one, making the method and the ablation (Table 2) easy to interpret.

Thorough empirical support in experiment and with ablation showing Reflect-then-Retry is the dominant contributor.

Concrete qualitative grounding. ALFWorld/DAPO case studies and pivot-evolution analysis, show what reflection changes mechanistically.

Full reproducibility by sharing full prompts, hyperparameter tables, dataset stats, baseline-specific settings, anonymized code link.

## Summary Of Weaknesses

Dropping importance sampling and KL is justified by theory but never ablated against keeping them.

We tested this approach using small size model, curious on the performance of the large size model with strong baseline.

## Comments, Suggestions And Typos

N/A

## Ethical Concerns

There are no concerns with this submission

Needs Ethics Review: No

## Other Fields

Knowledge Of Or Educated Guess At Author Identity: No

Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources

Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources

Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources

Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

## Author Response

Thank you for the positive assessment and the suggestions on KL, importance sampling, and model scale. We apologize for the late response. Despite limited resources, we completed the most relevant additional experiments during the rebuttal period.

> Ablation of KL and importance sampling

Thank you for suggesting a direct test of KL and importance sampling (IS). Using Qwen2.5-1.5B-Instruct on ALFWorld with the same setup, we ran four controlled experiments:

| Method | R³L | R³L+KL | R³L+IS | R³L+KL+IS |
|---|---|---|---|---|
| Final score | 0.928 | 0.927 | 0.928 | 0.929 |

All four runs remained stable. The gradient norm stayed around 0.12, KL showed no abnormal behavior, and the IS clip fraction stayed around 0.004, meaning clipping was rarely activated. The KL-only run had the lowest entropy loss, around 0.008, while the others were around 0.5. These differences did not affect the final score. In particular, R³L reached 0.928 without KL or IS and showed normal gradient norm, entropy, and KL curves. Thus, in this setting, R³L does not rely on either safeguard to avoid collapse. The low clip fraction also indicates small policy lag; stronger off-policy settings require separate validation.

> Model scale and stronger baselines

Thank you for raising model scale and baseline strength. Our experiments cover Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, the newer Qwen3-4B, and the cross-architecture Llama-3.2-3B-Instruct, with GSPO, Critique-GRPO, and other baselines evaluated under the same protocol across three agent environments and six mathematical benchmarks. On Qwen3-4B, R³L reaches 0.962 on ALFWorld versus 0.942 for Critique-GRPO. On GSM8K and Math500, it reaches 0.948 and 0.753, versus 0.934 for GRPO and 0.722 for GSPO, respectively.

These results do not replace direct validation above 7B. Even one full 1.5B run takes roughly 25 hours in our setup, and our rebuttal budget did not allow a sufficiently converged, fairly comparable larger-model run. We therefore limit our conclusions to the 1.5B–7B range and leave larger-scale evaluation for future work.
