# Rebuttal to Reviewer iAPy

We are grateful for the careful reading and the thoughtful concerns raised in your review.


```
W1: While the integration is coherent, elements such as retry-based refinement, advantage reweighting, and partial masking have precedents in related literature.
```

We appreciate you recognizing the coherence of the integration. The core contribution of R³L lies in identifying three orthogonal structural bottlenecks in LLM RL and designing a unified framework where all components work synergistically. The ablation in Table 2 confirms that each component provides independent and additive gains, meaning they address genuinely distinct problems with compatible optimization objectives.

R³L introduces clear design distinctions along each dimension compared to prior work. Critique-GRPO performs $N$ explorations plus $N$ full retries but selects only the best refinement for training, discarding substantial signal diversity. R³L splits the budget into $N/2$ base and $N/2$ retry trajectories, with all trajectories participating in group comparison. Neither Critique-GRPO nor Reflect-GRPO incorporates pivot-based credit assignment; both assign uniform advantage across entire trajectories, leaving the prefix penalty unresolved. Process Reward Models, GiGPO, and VinePPO rely on expensive human annotations, learned verifiers, or Monte Carlo rollouts, while R³L exploits the structural symmetry of base-retry pairs for contrastive signals without external supervision. For advantage reweighting, BAPO requires multiple dynamic clipping parameters and GSPO requires adaptive clipping ranges, whereas Positive Amplification uses a single fixed $\alpha=3.0$ effective across all settings, supported by Theorem 2.

At the implementation level, several design choices further differentiate R³L. Reflection outputs follow a structured JSON format with three fields: outcome classification, root cause analysis, and retry starting step. Retry is not forced; the model autonomously decides whether to retry, and the proportion requiring retry naturally decreases during training, making the overhead self-reducing. Reflection text is removed through context distillation before training, so distilled trajectories $\mathcal{D}_{distill}$ are structurally identical to baseline trajectories. The auxiliary SFT trains exclusively on reward-verified successful corrections, continuously strengthening reflection capability and avoiding the degradation under policy drift observed in Reflect-GRPO and Critique-GRPO.


```
W2 & S1: The framework assumes that the model can reliably identify the true error turn and generate an effective correction. However, self-reflection may mislocalize failures or produce superficial diagnoses, and the paper does not provide direct evaluation of pivot accuracy or correction validity beyond final task reward. & The paper would benefit from a more direct evaluation of the self-reflection component. In particular, reporting metrics such as pivot identification accuracy, correction success rate conditioned on detected pivots, or agreement with oracle error locations would strengthen the claim that reflection reliably localizes and fixes errors, rather than merely improving outcomes indirectly.
```

This is a valuable suggestion. Table 3, Figure 3, and Figure 6 already provide indirect evidence: Retry Improvement Rate reaches 73.9% on ALFWorld for Qwen2.5-7B, pivot locations shift rightward over training, and Reward Gain remains consistently positive.

We have additionally conducted a direct evaluation. ScienceWorld offers step-level subtask rewards for automatic oracle pivot identification. For ALFWorld and WebShop, we sampled 100 failed trajectories per task and annotated oracle pivots with human judgment combined with DeepSeek. Oracle Agreement is defined as the model's pivot falling within one step of the oracle.

| Task | Step | Oracle Agreement | Retry SR (Correct) | Retry SR (Wrong) | Accurate | Partial | Misdiag. |
|---|---|---|---|---|---|---|---|
| ALFWorld (1.5B) | 100 | 43% | 71% | 34% | 27% | 39% | 34% |
| ALFWorld (1.5B) | 400 | 63% | 81% | 37% | 47% | 34% | 18% |
| ALFWorld (7B) | 100 | 57% | 81% | 41% | 43% | 35% | 22% |
| ALFWorld (7B) | 400 | 76% | 85% | 43% | 57% | 31% | 12% |
| WebShop (7B) | 100 | 45% | 39% | 13% | 33% | 39% | 28% |
| WebShop (7B) | 400 | 60% | 51% | 14% | 46% | 33% | 22% |
| ScienceWorld (1.5B) | 100 | 36% | 57% | 21% | 23% | 39% | 38% |
| ScienceWorld (1.5B) | 200 | 44% | 64% | 24% | 31% | 39% | 30% |
| ScienceWorld (1.5B) | 300 | 53% | 71% | 27% | 41% | 36% | 23% |
| ScienceWorld (1.5B) | 400 | 61% | 76% | 29% | 49% | 33% | 18% |

Retry success rate is substantially higher when the pivot is correct, and accurate diagnoses rise steadily over training. R³L is also robust to reflection errors by design: incorrect retries receive negative advantages and are suppressed, while auxiliary SFT trains only on verified corrections.


```
W3: The reflection step requires an extra inference pass and auxiliary supervision. The paper does not provide a detailed wall-clock or cost analysis comparing R³L with strong baselines under equal compute budgets.
```

We agree that this was missing and provide it below. Structural comparison of per-step compute:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Base rollout | $N$ full trajectories | $N$ full trajectories | $N$ full trajectories | $N/2$ full trajectories |
| Reflect | None | In-trajectory reflection tokens | $N$ independent critique generations | $N/2$ reflections, ~500 tokens each |
| Retry rollout | None | In-trajectory retry | $N$ full refinements | $\leq N/2$ partial trajectories from pivot, triggered only when reflection deems necessary |
| Total rollouts | $N$ | $N$ | $2N$ | $\leq N$, depending on actual retry trigger rate |
| Entering RL training | $N$ trajectories | $N$ trajectories with reflection tokens receiving rewards | $N$ initial plus $1$ best refinement; remaining $N-1$ refinements discarded | Base plus distilled trajectories, all participating in group comparison |
| Reference model forward pass | Required for KL | Required for KL | Required for KL | Not required |

Measured per-step cost data for Qwen2.5-1.5B-Instruct on ALFWorld:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 14.3 | 12.5 | 13.2 | 11.6 |
| Avg base prompt tokens/turn | 1428 | 1436 | 1432 | 1441 |
| Avg base response tokens/turn | 386 | 347 | 264 | 218 |
| Avg reflect prompt tokens/traj | None | 2584 | 3971 | 3879 |
| Avg reflect response tokens/traj | None | 342 | 498 | 421 |
| Avg retry turns/traj | None | 10.7 | 11.8 | 10.3 |
| Avg retry prompt tokens/turn | None | 2364 | 1932 | 2437 |
| Avg retry response tokens/turn | None | 381 | 376 | 204 |
| Total rollout turns/sample | 114.4 | 146.8 | 208.0 | 75.2 |
| Total training tokens/sample | 44,141 | 52,400 | 36,200 | 22,300 |
| Rollout time (s)/training step | 406.2 | 492.8 | 756.3 | 372.4 |
| Train time (s)/training step | 87.6 | 109.4 | 94.2 | 103.8 |

Measured per-step cost data for Qwen2.5-1.5B-Instruct on DAPO:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 1.28 | 1.31 | 1.27 | 1.33 |
| Avg base prompt tokens/turn | 428 | 432 | 430 | 434 |
| Avg base response tokens/turn | 1724 | 1698 | 1636 | 1983 |
| Avg reflect prompt tokens/traj | None | 2896 | 4063 | 4912 |
| Avg reflect response tokens/traj | None | 318 | 434 | 492 |
| Avg retry turns/traj | None | 1.28 | 1.24 | 1.30 |
| Avg retry prompt tokens/turn | None | 712 | 770 | 746 |
| Avg retry response tokens/turn | None | 1658 | 1712 | 1894 |
| Total rollout turns/sample | 10.2 | 19.6 | 28.1 | 12.9 |
| Total training tokens/sample | 16,400 | 27,600 | 17,300 | 18,000 |
| Rollout time (s)/training step | 342.6 | 418.3 | 646.8 | 292.4 |
| Train time (s)/training step | 54.8 | 73.6 | 59.2 | 87.4 |

Measured per-step cost data for Qwen3-4B on ALFWorld:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 10.2 | 9.1 | 9.6 | 8.4 |
| Avg base prompt tokens/turn | 2219 | 2236 | 2224 | 2231 |
| Avg base response tokens/turn | 860 | 768 | 584 | 329 |
| Avg reflect prompt tokens/traj | None | 3214 | 4467 | 4682 |
| Avg reflect response tokens/traj | None | 478 | 614 | 539 |
| Avg retry turns/traj | None | 7.8 | 8.6 | 7.4 |
| Avg retry prompt tokens/turn | None | 3682 | 2724 | 3918 |
| Avg retry response tokens/turn | None | 836 | 838 | 296 |
| Total rollout turns/sample | 81.6 | 108.0 | 153.6 | 58.8 |
| Total training tokens/sample | 70,176 | 83,900 | 50,600 | 26,900 |
| Rollout time (s)/training step | 713.8 | 836.4 | 1356.8 | 670.4 |
| Train time (s)/training step | 188.9 | 218.7 | 197.3 | 208.6 |

Measured per-step cost data for Qwen3-4B on DAPO:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 1.09 | 1.06 | 1.08 | 1.11 |
| Avg base prompt tokens/turn | 691.2 | 698.4 | 693.8 | 697.6 |
| Avg base response tokens/turn | 3287 | 3208 | 3086 | 3671 |
| Avg reflect prompt tokens/traj | None | 4284 | 5872 | 6246 |
| Avg reflect response tokens/traj | None | 438 | 586 | 651 |
| Avg retry turns/traj | None | 1.04 | 1.07 | 1.08 |
| Avg retry prompt tokens/turn | None | 1076 | 1042 | 1168 |
| Avg retry response tokens/turn | None | 3186 | 3218 | 3452 |
| Total rollout turns/sample | 8.72 | 16.6 | 25.12 | 11.24 |
| Total training tokens/sample | 28,663 | 42,200 | 30,874 | 25,500 |
| Rollout time (s)/training step | 1148.0 | 1392.6 | 2318.4 | 773.5 |
| Train time (s)/training step | 121.3 | 146.8 | 129.6 | 225.8 |

In these tables, token and turn metrics correspond to a single sample of $N$ trajectories for one prompt, while time metrics are wall-clock per batch. All experiments use two 8-GPU H20 servers with asynchronous pipeline parallelism. Training tokens count only model-generated response tokens. The key finding is that R³L's rollout time is consistently lower than GRPO across all four settings, while training tokens on multi-step tasks are roughly halved. Total wall-clock remains comparable to GRPO and far below Critique-GRPO.

We thank you again for the careful reading and insightful suggestions. We hope these supplementary experiments and analyses resolve your concerns, and we will integrate them into the revised paper.
