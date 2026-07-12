# Rebuttal to Reviewer 4viW

We sincerely appreciate your positive assessment and the constructive suggestions throughout.

```
W1 & S1: additional system complexity and overhead: Reflection + retry adds extra inference steps (even if cheaper than full rollouts). & Provide explicit compute comparisons: wall-clock, environment steps, and token budget vs baselines?
```

We agree that a detailed compute breakdown would help clarify the trade-offs. R³L splits the group budget of $N=8$ into $N/2$ base and $N/2$ retry trajectories, keeping the total trajectory count consistent with baselines. Retries restart from the pivot rather than regenerating complete trajectories, and as Figure 6 shows, the pivot migrates from step 2 to beyond step 12 over training, so only the suffix requires new inference. The reflection itself is structured JSON of around 500 tokens. R³L also removes KL regularization entirely, eliminating the reference model forward pass.

It is worth noting that retry is not forced on all base trajectories; the model autonomously decides whether a retry is warranted. As training progresses and base success rates rise, fewer trajectories trigger retry, so R³L's overhead adaptively diminishes. On the training side, the auxiliary SFT operates on already-collected trajectories and computes gradients only on the short reflection JSON and post-pivot suffix.

The structural comparison of per-step compute across methods:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Base rollout | $N$ full trajectories | $N$ full trajectories | $N$ full trajectories | $N/2$ full trajectories |
| Reflect | None | In-trajectory reflection tokens | $N$ independent critique generations | $N/2$ reflections, ~500 tokens each |
| Retry rollout | None | In-trajectory retry | $N$ full refinements | $\leq N/2$ partial trajectories from pivot, triggered only when reflection deems necessary |
| Total rollouts | $N$ | $N$ | $2N$ | $\leq N$, depending on actual retry trigger rate |
| Entering RL training | $N$ trajectories | $N$ trajectories with reflection tokens receiving rewards | $N$ initial plus $1$ best refinement; remaining $N-1$ refinements discarded | Base plus distilled trajectories, all participating in group comparison |
| Reference model forward pass | Required for KL | Required for KL | Required for KL | Not required |

Measured average costs for Qwen2.5-1.5B-Instruct on ALFWorld:

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

Measured average costs for Qwen2.5-1.5B-Instruct on DAPO:

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

Measured average costs for Qwen3-4B on ALFWorld:

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

Measured average costs for Qwen3-4B on DAPO:

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

In these tables, token and turn metrics correspond to a single sample of $N$ trajectories for one prompt, while time metrics are wall-clock per batch. All experiments use two 8-GPU H20 servers with asynchronous pipeline parallelism. Training tokens count only model-generated response tokens. The key observation is that R³L's rollout time is consistently lower than GRPO across all four settings, while training tokens on multi-step tasks are roughly halved. Critique-GRPO is 1.8-2.0x slower due to its $2N$ rollouts. Overall, R³L achieves comparable wall-clock to GRPO with substantially higher performance.


```
W2 (minor): cold-start for small models: the paper itself notes small models initially can't reflect well and need warm-up.
```

This observation is accurate. Our setup deliberately avoids human-annotated SFT warm-up data, in order to test whether R³L can bootstrap reflection capability from scratch within a pure RL framework. Even under this more challenging setting, as shown in Figure 4, the cold-start phase for Qwen2.5-1.5B-Instruct lasts only about 80 steps, under 20% of the 400-step training budget. After this phase, selective SFT on verified corrections and GRPO-based RL work synergistically to drive sustained improvement in Reward Gain.

An important property is that R³L does not degrade during cold-start. When reflections are inaccurate and retries perform worse, those trajectories simply receive negative advantages and are suppressed, while the model continues normal group-level contrastive learning across all $N$ trajectories. The 7B model starts with a Reward Gain of approximately 0.4 and shows virtually no cold-start phase, suggesting that model scale naturally alleviates this issue. By comparison, Reflect-GRPO and Critique-GRPO similarly depend on reflection capability but lack any explicit mechanism to strengthen it, leaving their reflection quality vulnerable to policy drift during RL training.


```
W3: Risk of reflection errors: Incorrect pivot identification or low-quality reflections could reinforce wrong behavior.
```

This is an important concern, and R³L addresses it through multiple safeguards. The auxiliary SFT only includes reflection-retry pairs where the retry achieves higher reward than the base trajectory, so incorrect reflections are filtered out at the data level. On the RL side, distilled trajectories that perform worse due to bad reflection receive negative advantages and are naturally suppressed. Positive Amplification further ensures that only trajectories where $R(\tau)=R_{max}$ dominate the gradient direction.

Regarding pivot misidentification specifically, we observe that early-stage models tend to place the pivot conservatively early, rewriting more suffix than necessary. While this sacrifices some prefix reuse efficiency, it does not miss the actual error point and constitutes a safe failure mode. As Figure 6 shows, the pivot consistently shifts rightward during training as the model learns to localize errors more precisely. Table 3 confirms that approximately 35% of retries by Qwen2.5-1.5B-Instruct fail to improve, yet R³L still achieves robust gains across all benchmarks, demonstrating resilience to reflection errors.


```
S2: Analyze pivot quality: where do reflections fail, and how often are pivots correct?
```

We conducted a direct evaluation of pivot localization accuracy. ScienceWorld provides step-level subtask rewards, enabling automatic oracle pivot identification. For ALFWorld and WebShop, we sampled 100 failed trajectories per task and annotated oracle pivots through a combination of human judgment and DeepSeek labeling. Oracle Agreement is defined as the model's pivot falling within one step of the oracle position.

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

The data show a clear pattern: retry success rate when the pivot is correct is substantially higher than when wrong, and accurate diagnoses increase steadily over training. Reflection failures fall into three categories: early pivot placement that rewrites more than needed, late pivot placement that misses the root cause, and correct diagnosis paired with overly generic guidance. As discussed in W3, R³L's RL objective and selective SFT provide built-in robustness to all three failure modes.


```
S3: Sensitivity analysis for α and group size, and for how many retries are allowed?
```

The paper already provides sensitivity analysis for $\alpha$ in Table 5, where values range from 1.0 to 7.0: $\alpha=3.0$ gives the best balance and $\alpha > 5.0$ leads to overfitting, consistent with Corollary 1. Table 4 analyzes synchronization frequency $S$, where R³L maintains above 0.920 on ALFWorld across all values while OPMD collapses from 0.835 to 0.257.

We supplement ablations on group size $N$ and retry count below. Group size ablation for Qwen2.5-1.5B-Instruct:

| N | Task | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|---|
| 4 | ALFWorld | 0.742 | 0.774 | 0.816 | 0.886 |
| 8 | ALFWorld | 0.780 | 0.812 | 0.854 | 0.928 |
| 16 | ALFWorld | 0.788 | 0.818 | 0.860 | 0.932 |
| 4 | ScienceWorld | 0.332 | 0.352 | 0.368 | 0.438 |
| 8 | ScienceWorld | 0.366 | 0.388 | 0.406 | 0.482 |
| 16 | ScienceWorld | 0.372 | 0.394 | 0.412 | 0.486 |
| 4 | WebShop | 0.588 | 0.600 | 0.614 | 0.624 |
| 8 | WebShop | 0.620 | 0.634 | 0.648 | 0.663 |
| 16 | WebShop | 0.628 | 0.640 | 0.654 | 0.668 |
| 4 | DAPO | 0.108 | 0.118 | 0.116 | 0.138 |
| 8 | DAPO | 0.123 | 0.136 | 0.133 | 0.156 |
| 16 | DAPO | 0.128 | 0.140 | 0.137 | 0.160 |
| 4 | Math500 | 0.334 | 0.348 | 0.362 | 0.398 |
| 8 | Math500 | 0.367 | 0.382 | 0.398 | 0.432 |
| 16 | Math500 | 0.374 | 0.388 | 0.404 | 0.436 |

Retry count ablation for Qwen2.5-1.5B-Instruct on ALFWorld with $N=8$. In the two-retry setting, the model reflects and retries twice sequentially from updated pivots.

| Retry count | Trajectory allocation | ALFWorld |
|---|---|---|
| 0 (GRPO+PA) | 8 base | 0.807 |
| 1 (R³L) | 4 base + 4 retry | 0.928 |
| 2 | 4 base + 2x2 retry | 0.913 |
| 4 | 4 base + 4x1 retry | 0.896 |

A single retry already captures the majority of the gain, from 0.807 to 0.928, while additional retries actually hurt performance. Under a fixed budget of $N$ trajectories, more retries per base means fewer bases receive retry opportunities, reducing group diversity and weakening GRPO's contrastive signal. Each reflection is also an independent one-shot diagnosis that does not accumulate experience across rounds, so the information gain comes primarily from environment feedback and selective SFT rather than from stacking reflection rounds.

We thank you again for the constructive review. We hope the analyses and experiments above address your concerns, and we will incorporate them into the revised manuscript.
