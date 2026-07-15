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

## Author Response

Thank you for recognizing our method and results, and for the detailed questions on reflection reliability, component coupling, and scope. We apologize for the late response. With limited resources, we prioritized the experiments most directly related to these questions.

> Reliability of reflection and pivot localization

A localization error changes where retry begins and which tokens receive gradients, but it does not overwrite the reward with an unverified diagnosis. For any predicted $k$, retry reuses $\tau_{<k}$, so base and retry share an identical prefix and only their divergent suffixes are updated. An early pivot regenerates a longer suffix, increasing computation and variance and approaching a rollout from the beginning. A late pivot may retain an incorrect action while masking its learning signal, reducing retry success. In both cases, the environment verifier independently assigns the actual rewards; failed retries receive advantages from those rewards, and only verified corrections enter auxiliary SFT. Pivot errors therefore affect sample efficiency and signal coverage rather than directly becoming incorrect reward labels.

The prompt defines `retry_from_step` as the turn where the root cause first appears, or the earliest turn at which a changed decision could alter the outcome. If the cause lies in the initial policy, the pivot is 0. The conditional success rates in Table 11 show correlation, not causality.

We tested pivot sensitivity with the step-400 checkpoint of Qwen2.5-1.5B-Instruct on ALFWorld. For each failed trajectory, we fixed the reflection guidance and changed only the retry starting point from the predicted $k$ to $k-2$, $k+2$, $k-5$, $k+5$, or step 0. All other settings were fixed, and out-of-range shifts were excluded from the corresponding offset.

From start ($k=0$) ignores the predicted pivot and controls for localized restart. Unlike `w/o Credit` in Table 2, it changes the retry position rather than only removing the prefix mask. Retry success rate is the fraction of failed base rollouts with valid reflections whose retry receives a successful reward.

| Restart position | Predicted pivot $k$ | $k-2$ | $k+2$ | $k-5$ | $k+5$ | From start ($k=0$) |
|---|---|---|---|---|---|---|
| Retry success rate | 0.66 | 0.65 | 0.63 | 0.63 | 0.60 | 0.61 |

A small early shift has little effect: $k-2$ changes success from 0.66 to 0.65. The late shift $k+5$ reduces it to 0.60 because an incorrect action may remain in the prefix, while retrying from the beginning reaches 0.61. Thus, R³L tolerates small pivot errors, but larger errors, especially late ones, reduce retry effectiveness.

> Coupling in the component ablations

The `w/o Reflect` row is not a reflection-only ablation.

Reflection produces both guidance and the pivot, so removing it also removes retry synthesis and pivot-dependent Credit. The drop from 0.928 to 0.894 therefore measures the full Reflect-then-Retry path, not reflection alone.

`w/o Credit` keeps reflection and retry but removes the prefix mask, measuring Credit's incremental effect from 0.928 to 0.914. The pivot-sensitivity experiment instead fixes the trajectory and guidance and changes only the restart position.

Because checkpoint-level sensitivity does not show the final training effect, we added two full runs with Qwen2.5-1.5B-Instruct on ALFWorld. `w/o Credit + From start` keeps reflection and guidance but always retries from step 0. `w/o Credit + No guidance` uses the predicted $k$ without guidance. Both disable Credit and otherwise match `w/o Credit`.

| Variant | Guidance | Restart position | Credit mask | Final score |
|---|---|---|---|---|
| R³L | Yes | Predicted $k$ | Yes | 0.928 |
| w/o Credit | Yes | Predicted $k$ | No | 0.914 |
| w/o Credit + From start | Yes | $k=0$ | No | 0.908 |
| w/o Credit + No guidance | No | Predicted $k$ | No | 0.903 |
| w/o Reflect | No | No retry | No | 0.894 |

From start reduces the `w/o Credit` score from 0.914 to 0.908, showing that localized restart also affects final training. Removing guidance gives 0.903, showing a separate guidance gain. Thus, R³L versus `w/o Credit` tests prefix masking; `w/o Credit` versus From start tests restart position; and `w/o Credit` versus No guidance tests guidance. `w/o Reflect` still denotes removal of the full path, not reflection alone.

> Scope beyond verifiable rewards

As stated in Limitations, our experiments cover only agent and mathematical tasks with verifiable ground truth, not open-ended generation or subjective preference RLHF. Standard GRPO requires multiple responses to the same prompt and comparable rewards for within-group advantages, whereas real user feedback is often single-instance, sparse, noisy, inconsistent across users, and difficult to collect repeatedly. R³L also needs a reliable comparison between base and retry trajectories, so it cannot yet directly cover this setting. Building stable reflection, retry, and preference signals from incomplete, noisy feedback remains open.

> Formal definition of the retry trigger

The model decides whether to retry through a reflection containing `trajectory_outcome` and `retry_from_step`. There is no reward threshold or additional random draw; the system only applies deterministic validity checks. A retry occurs when the outcome is `failure` or `success_but_inefficient`, the format is valid, and `retry_from_step` is in range:

$$
I(\\tau,r)=\\mathbf{1}\\left[
v(r) \\land o(r) \\in \\{F,E\\}
\\land 0 \\le k(r) < |\\tau|
\\right].
$$

Here, $v(r)$ indicates a valid reflection, $o(r)$ is `trajectory_outcome`, $k(r)$ is `retry_from_step`, and $F$ and $E$ denote `failure` and `success_but_inefficient`. Valid means parseable JSON with all required fields and legal values. A predicted `success`, invalid JSON, missing field, or out-of-range pivot skips retry while retaining the base trajectory. The system checks only whether the decision is executable; it does not override the model's judgment. A well-formed but semantically wrong reflection can still trigger an unnecessary retry or miss a needed one. The trigger itself has no random switch, although sampled reflections may differ.

The trigger rate is the number of executed retries divided by base rollouts. It measures the model's retry decisions at the current checkpoint, not a fixed hyperparameter. For Qwen2.5-1.5B-Instruct on ALFWorld:

| Training step | Rollout reward | Retry trigger rate |
|---|---|---|
| 0 | 0.01 | 0.55 |
| 50 | 0.03 | 0.65 |
| 100 | 0.08 | 0.80 |
| 150 | 0.50 | 0.78 |
| 200 | 0.80 | 0.35 |
| 250 | 0.87 | 0.13 |
| 300 | 0.90 | 0.10 |
| 350 | 0.91 | 0.08 |
| 400 | 0.91 | 0.07 |

During the first 100 steps, the trigger rate rises from 0.55 to 0.80 while reward remains low, so it is not simply $1-\text{reward}$; it also depends on the model's reflection. As reward later rises from 0.08 to 0.91, the trigger rate falls from 0.80 to 0.07. Retries and actual rollout volume therefore decrease as the model improves.
