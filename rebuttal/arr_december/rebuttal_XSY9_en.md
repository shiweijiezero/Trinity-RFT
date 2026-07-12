# Rebuttal to Reviewer XSY9

We thank you for the detailed and rigorous review. We address each concern below with supplementary experiments and analysis.


```
W1 & S1: The reported results for the baselines are significantly lower than that in Qwen2.5 technical report, casting doubt on the entire empirical evaluation. According to the Qwen2.5 technical report, Qwen2.5-1.5B-Instruct achieves 73.2 on GSM8K and 55.2 on MATH. However, the authors report a baseline (GRPO) of only 47.4 on GSM8K and 36.7 on MATH. ... & The authors must provide a detailed explanation for why their baseline numbers for Qwen2.5 are ~25-30 points lower (on GSM8K/MATH) than official figures.
```

We apologize for the confusion caused by our presentation, which should have been stated more clearly in the paper. The numbers in Table 1 are post-RL-training results, not the pretrained model's zero-shot performance. Taking the GRPO score of 0.474 on GSM8K as an example, this is obtained after RL training on the DAPO training set starting from Qwen2.5-1.5B-Instruct, evaluated under the \<answer\> protocol. The 0.732 from the technical report is the same model's direct evaluation under the \boxed{} protocol without any RL training. These two numbers differ along two dimensions, and the apparent 0.26 gap is the sum of two independent factors: format discrepancy and training distribution shift.

Regarding format discrepancy, Qwen2.5-Instruct was extensively adapted to \boxed{} during training, and the technical report evaluates with the system prompt:

`"Please reason step by step, and put your final answer within \boxed{}."`

Our training and evaluation uniformly use the \<think\>/\<answer\> tag format, which has been widely adopted in RL training research since DeepSeek-R1 and was chosen as our unified protocol across agentic and mathematical reasoning tasks. The system prompt is:

> A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within \<think\> \</think\> and \<answer\> \</answer\> tags, respectively, i.e., \<think\> reasoning process here \</think\> \<answer\> answer here \</answer\>.

The impact of format switching on benchmark scores has been documented in prior work. `[1]` shows that answer extraction rules alone can cause fluctuations of tens of points on the same model, and `[2]` converts all training answers to integers specifically to avoid format parsing issues.

`[1] Jo, Hwiyeol, et al. "Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning." arXiv preprint arXiv:2510.14773 (2025).`

`[2] Yu, Qiying, et al. "Dapo: An open-source llm reinforcement learning system at scale." arXiv preprint arXiv:2503.14476 (2025).`

To rigorously quantify the format impact in our setting, we conducted zero-shot evaluations on four base models under both protocols. All evaluations use full test sets: GSM8K 1319 problems from openai/gsm8k, Math500 500 problems, MinervaMath 272, OlympiadBench 674, AMC23 40, DAPO 300.

Qwen2.5-1.5B-Instruct:

| Benchmark | \boxed{} | \<answer\> | Gap |
|---|---|---|---|
| GSM8K | 0.738 | 0.665 | -0.073 |
| Math500 | 0.440 | 0.422 | -0.018 |
| MinervaMath | 0.118 | 0.132 | +0.015 |
| OlympiadBench | 0.099 | 0.083 | -0.016 |
| AMC23 | 0.300 | 0.150 | -0.150 |
| DAPO | 0.133 | 0.107 | -0.027 |

Qwen2.5-7B-Instruct:

| Benchmark | \boxed{} | \<answer\> | Gap |
|---|---|---|---|
| GSM8K | 0.923 | 0.853 | -0.070 |
| Math500 | 0.620 | 0.612 | -0.008 |
| MinervaMath | 0.239 | 0.235 | -0.004 |
| OlympiadBench | 0.205 | 0.186 | -0.019 |
| AMC23 | 0.575 | 0.575 | 0 |
| DAPO | 0.337 | 0.337 | 0 |

Llama-3.2-3B-Instruct:

| Benchmark | \boxed{} | \<answer\> | Gap |
|---|---|---|---|
| GSM8K | 0.723 | 0.736 | +0.013 |
| Math500 | 0.366 | 0.380 | +0.014 |
| MinervaMath | 0.114 | 0.099 | -0.015 |
| OlympiadBench | 0.144 | 0.122 | -0.022 |
| AMC23 | 0.200 | 0.175 | -0.025 |
| DAPO | 0.137 | 0.130 | -0.007 |

Qwen3-4B:

| Benchmark | \boxed{} | \<answer\> | Gap |
|---|---|---|---|
| GSM8K | 0.932 | 0.938 | +0.006 |
| Math500 | 0.648 | 0.690 | +0.042 |
| MinervaMath | 0.254 | 0.283 | +0.029 |
| OlympiadBench | 0.432 | 0.504 | +0.073 |
| AMC23 | 0.625 | 0.600 | -0.025 |
| DAPO | 0.446 | 0.443 | -0.003 |

The format gap is concentrated on GSM8K for the Qwen2.5 series, at -0.073 and -0.070 for the 1.5B and 7B models respectively, while other benchmarks generally fall within ±0.020. Llama-3.2-3B shows nearly identical results under both formats, confirming that this sensitivity is Qwen2.5-specific and arises from its extensive \boxed{} adaptation during training. Qwen3-4B actually performs better under \<answer\> on several benchmarks, consistent with its native \<think\>/\<answer\> training paradigm.

For the specific Qwen2.5-1.5B-Instruct GSM8K case you highlighted, the 0.26 gap decomposes as follows. Our \boxed{} evaluation yields 0.738, consistent with the technical report's 0.732. Switching to \<answer\> gives 0.665, a format gap of 0.073 accounting for 28% of the total difference. The remaining 72% comes from training distribution shift: DAPO training causes GSM8K to drop from 0.665 to 0.474.

To confirm the distribution shift directly, we present evaluation results after training on different training sets.

Qwen2.5-1.5B-Instruct after training on the DAPO training set, evaluated under \<answer\>:

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.665 | 0.422 | 0.132 | 0.083 | 0.150 | 0.107 |
| GRPO | 0.474 | 0.368 | 0.099 | 0.114 | 0.250 | 0.123 |
| Reflect-GRPO | 0.672 | 0.376 | 0.102 | 0.130 | 0.300 | 0.136 |
| Critique-GRPO | 0.798 | 0.404 | 0.110 | 0.124 | 0.275 | 0.133 |
| R³L | 0.721 | 0.424 | 0.125 | 0.151 | 0.325 | 0.156 |

Qwen2.5-1.5B-Instruct after training on the DAPO training set, evaluated under \boxed{}:

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.738 | 0.440 | 0.118 | 0.099 | 0.300 | 0.133 |
| GRPO | 0.576 | 0.392 | 0.092 | 0.125 | 0.400 | 0.147 |
| Reflect-GRPO | 0.742 | 0.404 | 0.096 | 0.140 | 0.375 | 0.160 |
| Critique-GRPO | 0.793 | 0.428 | 0.103 | 0.135 | 0.375 | 0.153 |
| R³L | 0.788 | 0.446 | 0.114 | 0.162 | 0.425 | 0.173 |

Qwen2.5-7B-Instruct after DAPO training, evaluated under \<answer\>:

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.853 | 0.612 | 0.235 | 0.186 | 0.575 | 0.337 |
| GRPO | 0.846 | 0.572 | 0.239 | 0.277 | 0.675 | 0.393 |
| Reflect-GRPO | 0.765 | 0.532 | 0.194 | 0.250 | 0.550 | 0.396 |
| Critique-GRPO | 0.678 | 0.522 | 0.152 | 0.170 | 0.300 | 0.390 |
| R³L | 0.897 | 0.658 | 0.275 | 0.301 | 0.700 | 0.436 |

Qwen2.5-7B-Instruct after DAPO training, evaluated under \boxed{}:

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.923 | 0.620 | 0.239 | 0.205 | 0.575 | 0.337 |
| GRPO | 0.915 | 0.584 | 0.246 | 0.285 | 0.650 | 0.383 |
| Reflect-GRPO | 0.846 | 0.544 | 0.162 | 0.195 | 0.350 | 0.392 |
| Critique-GRPO | 0.872 | 0.550 | 0.205 | 0.264 | 0.550 | 0.407 |
| R³L | 0.943 | 0.658 | 0.276 | 0.305 | 0.675 | 0.424 |

The pattern is consistent across both models. DAPO, as the in-domain benchmark, sees stable improvements across all methods. Competition-level benchmarks close to DAPO's distribution also benefit broadly. GSM8K and Math500, distributionally distant from DAPO, show varying degrees of degradation, most notably GRPO on the 1.5B model where GSM8K drops from 0.665 to 0.474.

Notably, R³L matches or exceeds the zero-shot baseline on nearly all benchmarks for both models, indicating that the reflect-then-retry mechanism effectively mitigates cross-domain forgetting while improving target task performance. By contrast, Critique-GRPO on the 7B model drops GSM8K from 0.853 to 0.678 and AMC23 from 0.575 to 0.300, exhibiting severe capability forgetting.

We also provide in-domain training controls to verify that the degradation stems from distribution shift rather than methodological issues.

In-domain training on GSM8K training set, evaluated on GSM8K test set:

| Method | \<answer\> | \boxed{} |
|---|---|---|
| Zero-shot | 0.665 | 0.738 |
| GRPO | 0.814 | 0.798 |
| Reflect-GRPO | 0.822 | 0.830 |
| Critique-GRPO | 0.846 | 0.842 |
| R³L | 0.867 | 0.874 |

In-domain training on MATH training set, evaluated on Math500:

| Method | \<answer\> | \boxed{} |
|---|---|---|
| Zero-shot | 0.422 | 0.440 |
| GRPO | 0.481 | 0.493 |
| Reflect-GRPO | 0.505 | 0.498 |
| Critique-GRPO | 0.518 | 0.512 |
| R³L | 0.533 | 0.530 |

Under in-domain training, all methods achieve substantial improvements with fully consistent trends under both protocols. Comparing these results to the post-DAPO outcomes where GSM8K drops from 0.665 to 0.474 confirms that the low baselines in Table 1 stem from cross-domain distribution shift rather than methodological deficiencies.

Format comparison evaluations can be reproduced using:

`python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-1.5B-Instruct`

`python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-7B-Instruct`

`python eval_math/eval_format_comparison.py --model_path Qwen/Qwen3-4B`

`python eval_math/eval_format_comparison.py --model_path meta-llama/Llama-3.2-3B-Instruct`

RL training configurations are available under `examples/R³L/dapo/`, `examples/R³L/gsm8k/`, and `examples/R³L/math/`, where `*_qwen_1.5B.yaml` uses \boxed{} and `*_1.5B.yaml` uses \<answer\>. All compared methods share exactly the same training and evaluation conditions.

We will include the complete format comparison, in-domain controls, gap decomposition, and distribution shift discussion in the revised version.

```
W2 & S2: The paper focuses on Qwen2.5 and Llama 3.2, which were released in 2024. In the current landscape of 2026, these models are considered legacy. ... & Providing results on Qwen3 or Llama 4 is necessary to confirm that the "Reflect-then-Retry" mechanism isn't just fixing flaws that have already been solved by more advanced base models.
```

This is a fair concern. However, the problems R³L addresses — prefix penalty, entropy collapse under failure-dominated rewards, and inefficient random exploration — are structural challenges of RL training that do not disappear with newer base models. Stronger models still face the same bottlenecks on tasks at their difficulty frontier. We believe cross-model consistency provides stronger generalization evidence than results on any single new model, and we achieve the best or second-best across all 27 settings on three models spanning Qwen2.5-1.5B, Qwen2.5-7B, and Llama-3.2-3B.

That said, we have conducted additional experiments on Qwen3-4B. As a reasoning model, Qwen3 automatically trims `<think>` content from earlier turns in multi-turn conversations, which is incompatible with standard multi-turn RL pipelines that require gradients over complete interaction trajectories. We addressed this by decomposing multi-turn dialogues into single-turn training instances with step-level GRPO. Configurations are under `examples/R³L/*/step_*_4B.yaml` with workflows in `trinity/common/workflows/envs/R³L/*/step_*_workflow.py`.

| Task | RAFT | GRPO | OPMD | GSPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|---|---|---|
| ALFWorld | 0.886 | 0.912 | 0.878 | 0.916 | 0.938 | 0.942 | 0.962 |
| WebShop | 0.659 | 0.695 | 0.678 | 0.707 | 0.724 | 0.725 | 0.746 |
| ScienceWorld | 0.350 | 0.368 | 0.342 | 0.376 | 0.378 | 0.374 | 0.398 |
| GSM8K | 0.914 | 0.934 | 0.922 | 0.928 | 0.920 | 0.926 | 0.948 |
| MATH500 | 0.638 | 0.718 | 0.663 | 0.722 | 0.654 | 0.705 | 0.753 |
| MinervaMath | 0.326 | 0.350 | 0.358 | 0.364 | 0.366 | 0.368 | 0.383 |
| Olympiad | 0.516 | 0.546 | 0.492 | 0.538 | 0.552 | 0.544 | 0.571 |
| AMC23 | 0.650 | 0.725 | 0.675 | 0.800 | 0.650 | 0.775 | 0.800 |
| DAPO | 0.505 | 0.540 | 0.531 | 0.548 | 0.547 | 0.558 | 0.583 |

R³L continues to achieve the best performance on Qwen3-4B. This model has substantially higher baseline math performance than Qwen2.5-7B-Instruct, with GRPO reaching 0.718 on MATH500 compared to only 0.572 for the 7B model, yet R³L still delivers consistent improvements on this already high baseline. An interesting observation is that Reflect-GRPO falls below GRPO on math tasks, reaching only 0.654 on MATH500, a pattern also seen on Qwen2.5-7B. For models with already strong reasoning capabilities, simply appending reflection and retry may interfere with established reasoning pathways, whereas R³L converts reflection into structured credit assignment through pivot localization, maintaining advantages across all task types.

The four models now span 1.5B to 7B parameters, cover both Qwen and Llama architecture families, and include both instruction-tuned and reasoning-trained paradigms.

```
W3 & S3: The method introduces several "moving parts" (reflection, retry, guidance distillation, pivotal masking, and positive amplification). ... it appears to be a collection of heuristics added atop existing ideas (like [1,2]). ... The "Positive Amplification" (α) introduces an additional hyperparameter that likely requires task-specific tuning ... & Can similar gains be achieved by further refining the "Positive Amplification" on top of standard GRPO without the expensive reflection/retry loop?
```

We understand this concern, but these components form a coherent pipeline rather than independent heuristics stacked together. The core design reallocates GRPO's $N$ random explorations into $N/2$ base plus $\leq N/2$ targeted retries without increasing total trajectory count. Reflection text is removed through context distillation, leaving training sequences structurally identical to baselines. Pivotal credit assignment leverages the structural alignment of base-retry pairs — they share an identical prefix up to the pivot and diverge only afterward, forming a natural controlled experiment. This step-level contrastive signal can only be obtained after reflect-then-retry produces aligned trajectory pairs, making it the key output of the pipeline rather than an independently added module. Positive Amplification addresses successful trajectories being overwhelmed under sparse rewards, while simultaneously removing KL regularization and importance sampling clipping, eliminating the reference model forward pass. Table 2 confirms each component provides independent and stackable gains.

On the hyperparameter front, R³L actually reduces tuning burden: it removes $\beta$ and $\epsilon$ from GRPO, introducing only $\alpha=3.0$. GRPO requires both $\beta$ and $\epsilon$, and GSPO requires $\beta$ plus an adaptive clipping range, as shown in Table 7. Table 4 confirms that $\alpha=3.0$ works robustly across all models and tasks without per-task tuning, supported by Corollary 4.

Compared to Critique-GRPO that you mention, both involve reflection and retry but differ fundamentally in efficiency and credit assignment. Critique-GRPO generates $2N$ total rollouts and selects only the best single refinement from $N$, meaning computation for $N-1$ refinements is wasted. R³L keeps the budget within $\leq N$ rollouts with no wasted computation. More critically, Critique-GRPO still uses trajectory-level reward signals during training, whereas R³L achieves step-level credit assignment through pivot alignment, enabling the model to learn precisely where errors occurred. VL-Rethinker targets vision-language models with token-level rewards to incentivize reflection tokens, addressing a different application scenario.

For whether Positive Amplification (PA) alone can achieve similar gains, Table 5 already shows GRPO+PA improves from 0.747 to 0.807 on ALFWorld and from 0.474 to 0.504 on GSM8K — real but modest gains. These remain far from R³L's 0.928 and 0.721. The ablation in Table 2 further shows that removing reflect-then-retry drops ALFWorld from 0.928 to 0.894 and GSM8K from 0.721 to 0.562, confirming that it provides the single largest contribution.


```
W4 & S4: Flawed Compute-Efficiency Comparison: ... they do not seem to account for the increased sequence length (token count) incurred by generating natural language reflections and guidance. A true comparison of "Exploration Efficiency" should be conducted under a strict total token budget ... & Please include a table comparing the average tokens generated per training step for R³L vs. GRPO to ensure a fair comparison of exploration costs.
```

This is a well-taken point, and we provide the detailed per-step cost breakdowns below.

Structural comparison of per-step computation costs:

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Base rollout | $N$ full trajectories | $N$ full trajectories | $N$ full trajectories | $N/2$ full trajectories |
| Reflect | None | In-trajectory reflection tokens | $N$ independent critique generations | $N/2$ reflections, ~500 tokens each |
| Retry rollout | None | In-trajectory retry | $N$ full refinements | $\leq N/2$ partial trajectories from pivot, triggered only when reflection deems necessary |
| Total rollouts | $N$ | $N$ | $2N$ | $\leq N$, depending on actual retry trigger rate |
| Entering RL training | $N$ trajectories | $N$ trajectories with reflection tokens receiving rewards | $N$ initial plus $1$ best refinement; remaining $N-1$ refinements discarded | Base plus distilled trajectories, all participating in group comparison |
| Reference model forward pass | Required for KL | Required for KL | Required for KL | Not required |

Qwen2.5-1.5B-Instruct on ALFWorld:

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

Qwen2.5-1.5B-Instruct on DAPO:

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

Qwen3-4B on ALFWorld:

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

Qwen3-4B on DAPO:

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

In these tables, token and turn metrics correspond to a single sample of $N$ trajectories for one prompt, while time metrics are wall-clock per batch. All experiments use two 8-GPU H20 servers with asynchronous pipeline parallelism. Training tokens count only model-generated response tokens. The data confirm that R³L's rollout time is consistently lower than GRPO across all four settings, while Critique-GRPO is 1.8-2.0x slower due to its $2N$ rollouts. Training tokens on multi-step tasks are roughly halved. Combined with removing the reference model forward pass, R³L achieves comparable total wall-clock to GRPO with substantially higher performance.

We thank you again for raising these important questions. We hope the extensive experiments and analysis above adequately address your concerns, and we look forward to incorporating them into the revised version.
