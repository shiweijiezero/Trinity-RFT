# Reviewer vbDZ

**Scores:**
- Confidence: 4 (Quite sure)
- Soundness: 3 (Acceptable)
- Excitement: 3.5
- Overall Assessment: 3 (Findings)
- Reproducibility: 3
- Datasets: 1
- Software: 1

---

## Paper Summary

This paper proposes R3L (Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit Assignment, and Positive Amplification), a reinforcement learning framework for improving LLM reasoning and agentic task performance in sparse-reward environments. The core idea is to address two major limitations of current RL methods for LLMs: (1) inefficient exploration due to low success rates of stochastic sampling, and (2) unstable exploitation caused by trajectory-level credit assignment and dominance of failure signals. To address these issues, the authors introduce three key components: Reflect-then-Retry (language-guided exploration): The model analyzes failed trajectories using language feedback, identifies pivot failure points, and regenerates improved suffixes starting from these points. Pivotal Credit Assignment: Instead of assigning trajectory-level rewards uniformly, gradients are masked so that only the divergent suffix after the pivot contributes to optimization, preserving valid prefixes. Positive Amplification: Successful trajectories are upweighted to prevent gradient dilution caused by failure-dominated batches. Experiments on agentic tasks (ALFWorld, WebShop, ScienceWorld) and math reasoning benchmarks (GSM8K, Math500, etc.) show consistent improvements over GRPO, GSPO, and critique-based baselines, with reported gains of 5%–52% across settings. The paper also provides theoretical analysis on entropy collapse, variance reduction, and convergence behavior.

## Summary Of Strengths

The following summarizes the main strengths of this paper.

Novelty and Theoretical Grounding: The identification and formalization of "entropy collapse" and gradient asymmetry in failure-dominated RL groups is theoretically sound and highly insightful. The proposed mechanisms elegantly address these specific bottlenecks without requiring costly external process reward models.

Methodological Elegance: The Pivotal Credit Assignment is a highly intuitive and computationally efficient approach to the temporal credit assignment problem. By treating the shared prefix as a control variate and masking it out from gradient updates, the method cleanly isolates the actual decision divergence.

Strong Empirical Performance: The comprehensive evaluation across both agentic tasks and mathematical reasoning benchmarks using modern base models (e.g., Qwen2.5 and Llama-3.2) convincingly demonstrates the efficacy of the framework. The consistent performance gains over robust baselines like GRPO, GSPO, and Critique-GRPO highlight the practical utility of the method.

## Summary Of Weaknesses

Despite its strengths, the paper also has several limitations that may affect its overall impact and clarity. These concerns are outlined below.

High dependency on correctness of reflection and pivot localization: The method critically depends on accurate reflection-based pivot identification (k_pivot).However: Table 11 shows misdiagnosis rates remain 18%–38% even at later training steps. Fig. 6 shows pivot instability (WebShop pivot collapses back toward early steps). Retry success depends heavily on correctness:85% success with correct pivot vs 43% with incorrect pivot (Table 11). This creates a fragile dependency chain: reflection → pivot → mask → credit assignment → update. Errors in reflection propagate directly into incorrect gradient updates.

Ablation does not fully disentangle interaction effects: Ablation results (Table 2) show: w/o Reflect: largest drop (0.928 → 0.894) , w/o Credit: moderate drop , w/o Positive: moderate drop. However: Removing reflection also disables pivot detection → disables credit masking (Sec 5.3). Therefore, “reflection ablation” conflates multiple coupled components. This makes causal attribution unclear between: reflection, pivot masking, retry synthesis

Evaluation limited to verifiable reward environments: All experiments are restricted to: ALFWorld, WebShop, ScienceWorld, mathematical benchmarks. No evaluation on: open-ended generation, subjective preference RLHF tasks. Thus generalization beyond verifiable reward RL settings remains untested.

## Comments, Suggestions And Typos

In addition to the major concerns listed above, I provide several minor comments and suggestions that may help improve the clarity, a n d presentation of the paper.

The definition of the pivot point k pivot is central to the method, but its identification process is somewhat underspecified. While Section 4.1 describes reflection-based diagnosis, it is unclear how deterministic or stable the pivot extraction is, especially when multiple failure causes exist. Providing clearer rules or an ablation on pivot sensitivity would improve methodological rigor.

The paper states that retry is triggered based on reflection outcomes (success vs failure vs inefficient), but the decision boundary is not fully formalized. It is unclear whether retry is deterministic or probabilistic, and how noisy reflection outputs affect this decision. A formal definition of retry triggering would improve reproducibility.

## Ethical Concerns

There are no concerns with this submission

Needs Ethics Review: No

## Other Fields

Knowledge Of Or Educated Guess At Author Identity: No

Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources

Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources

Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources

Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
