# Reviewer XSY9

**Scores:**
- Confidence: 4 (Quite sure)
- Soundness: 3 (Acceptable)
- Excitement: 2 (Potentially Interesting)
- Overall Assessment: 2 (Resubmit next cycle)
- Reproducibility: 2
- Datasets: 2
- Software: 2

---

## Paper Summary

The paper proposes R³L (Reflect-then-Retry Reinforcement Learning), a framework designed to improve the exploration efficiency and training stability of Large Language Models (LLMs) in reinforcement learning (RL) settings, particularly for tasks with sparse rewards. The method is evaluated on agentic tasks (ALFWorld, WebShop, ScienceWorld) and mathematical reasoning benchmarks, utilizing Qwen2.5 and Llama 3.2 models.

## Summary Of Strengths

The paper correctly identifies a critical bottleneck in LLM Reinforcement Learning: the "valid prefix penalization" problem. The intuition that a failure in the final steps of a long reasoning chain should not lead to the penalization of the entire trajectory is sound and addresses a known limitation of standard PPO/GRPO.

The idea of using a "pivot point" to create a contrastive pair (a failed base trajectory vs. a successful retry trajectory) is a reasonable way to perform credit assignment. This structural symmetry provides a clear signal for the model to learn exactly where the reasoning went astray.

## Summary Of Weaknesses

The reported results for the baselines are significantly lower than that in Qwen2.5 technical report, casting doubt on the entire empirical evaluation. According to the Qwen2.5 technical report, Qwen2.5-1.5B-Instruct achieves 73.2 on GSM8K and 55.2 on MATH. However, the authors report a baseline (GRPO) of only 47.4 on GSM8K and 36.7 on MATH. For Qwen2.5-7B-Instruct, the official report is 91.6 on GSM8K, while the authors report 84.6. These discrepancies suggest that the baseline models were either severely undertrained, evaluated using suboptimal prompts, or tested on a non-standard subset. Since the "gains" of R³L are measured against these weakened baselines, the claimed improvements (e.g., 52% relative gain) are likely artifactual and do not reflect the method's true performance on a competitive, well-tuned model.

The paper focuses on Qwen2.5 and Llama 3.2, which were released in 2024. In the current landscape of 2026, these models are considered legacy. To demonstrate the value of a new RL framework, it is essential to evaluate it on contemporary SOTA models such as Qwen3, Gemma-3, and Olmo-3. If the method only shows gains on older, smaller models with poor baseline performance, its generalizability to modern, more capable architectures remains unproven.

The method introduces several "moving parts" (reflection, retry, guidance distillation, pivotal masking, and positive amplification). While the authors present this as a unified framework, it appears to be a collection of heuristics added atop existing ideas (like Critique-GRPO, VL-Rethinker). Given the unreliable baseline results mentioned in point #1, it is unclear whether this high degree of complexity is actually necessary or if a simple, well-tuned standard RL approach would outperform it. The "Positive Amplification" (α) introduces an additional hyperparameter that likely requires task-specific tuning, potentially making the training process more brittle rather than more stable.

Flawed Compute-Efficiency Comparison: The "Reflect-then-Retry" mechanism involves multiple inference passes (reflection + guided generation). The authors claim to control for this by adjusting the number of trajectories, but they do not seem to account for the increased sequence length (token count) incurred by generating natural language reflections and guidance. A true comparison of "Exploration Efficiency" should be conducted under a strict total token budget, as the current setup may be giving R³L an unfair advantage in terms of total compute used during the rollout phase.

References mentioned: [1] Critique-GRPO: Advancing LLM reasoning with natural language and numerical feedback. [2] VL-Rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning.

## Comments, Suggestions And Typos

Address Baseline Discrepancy: The authors must provide a detailed explanation for why their baseline numbers for Qwen2.5 are ~25-30 points lower (on GSM8K/MATH) than official figures. If this is due to a specific evaluation protocol (e.g., strict matching vs. flexible parsing), it must be standardized to allow comparison with existing literature.

Modernize the Backbone: For a 2026 publication, providing results on Qwen3 or Llama 4 is necessary to confirm that the "Reflect-then-Retry" mechanism isn't just fixing flaws that have already been solved by more advanced base models.

Simplification Study: I suggest the authors perform a "complexity-drop" study. For instance, can similar gains be achieved by further refining the "Positive Amplification" on top of standard GRPO without the expensive reflection/retry loop?

Detailed Token Accounting: Please include a table comparing the average tokens generated per training step for R³L vs. GRPO to ensure a fair comparison of exploration costs.
