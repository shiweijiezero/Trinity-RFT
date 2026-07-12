# ARR May EMNLP Rebuttal Experiments Todo

这个文件只整理需要为 rebuttal 补的实验和分析。目标不是扩展论文全部实验，而是在 rebuttal 期内用最少计算量直接回应 reviewer 的关键疑问。

优先级含义如下。

| Priority | Meaning |
|---|---|
| P0 | rebuttal 里最好必须有数据，否则 uCGY 的核心弱点很难压住 |
| P1 | 很有价值，若 GPU 和时间允许应补，至少可以承诺放 camera-ready |
| P2 | 低成本分析、日志抽取或文本修订支撑，不一定需要新训练 |

当前最关键的 reviewer concern 映射如下。

| Concern | Reviewer | Needed evidence |
|---|---|---|
| single-run 没有 variance | uCGY W1 | 3 seeds mean/std，最好覆盖一个 agentic、一个 math、一个 close-margin setting |
| 去掉 KL 和 importance sampling 证据不足 | uCGY W3, TC9w | stability curves 和 re-add KL/IS ablation |
| math guidance 可能有 ground-truth leakage | uCGY W4 | 控制实验、guidance 审计、不给答案的样例 |
| pivot localization 质量和错误敏感性 | uCGY W4, vbDZ | oracle agreement、correct/incorrect pivot success、pivot perturbation |
| baselines 低于 zero-shot | uCGY W5 | same-prompt zero-shot eval、baseline tuning/selection 说明、best checkpoint vs final |
| compute efficiency 是否真正公平 | uCGY comments, previous reviews | token/wall-clock/env-step 统计，包含 reflection tokens 和 retry trigger rate |
| 更大或更新模型 | TC9w, uCGY W8 | Qwen3-4B 或更大模型上的 R3L vs strong baseline |

## Common Run Rules

所有新跑实验尽量遵守同一套规则，避免 rebuttal 里被追问不可比。

1. 新实验统一使用 seed set `[0, 1, 2]`。如果已有 seed 0 可复用，只补 seed 1 和 seed 2。
2. 每个 seed 必须改三个地方，`name`、`explorer.rollout_model.seed`、`buffer.trainer_input.experience_buffer.path`。sqlite path 不加 seed 会互相覆盖。
3. 如果有全局 python/torch/ray seed 配置，应一并设置。若当前 Trinity 只有 rollout seed，也要在 rebuttal 中明确 seed 控制范围。
4. 每个 run 同时保存 final checkpoint 和 best eval checkpoint。表里主报 final，括号或附表给 best eval，防止被认为挑 checkpoint。
5. 训练和 baseline 使用相同 `total_epochs`、`batch_size`、`repeat_times`、eval split 和 temperature。除非实验本身就是 ablation，不要改学习率和资源。
6. wandb/tensorboard run name 统一带 `arrmay_rebuttal`、task、model、method、seed，例如 `arrmay_webshop_qwen25_15b_r3l_seed1`。
7. 每个 run 结束后导出一份机器可读结果，建议放在 `rebuttal/arr_may_emnlp/results/`，文件名如 `webshop_qwen25_15b_r3l_seed1.json`。
8. 每张 rebuttal 表都记录训练硬件、GPU 数、模型路径、commit hash、config path、是否复用已有 run。

最常用配置路径如下。

| Task | R3L full | GRPO | Critique-GRPO | Reflect-GRPO | w/o credit | w/o PA |
|---|---|---|---|---|---|---|
| ALFWorld 1.5B | `examples/R3L/alfworld/opmd_R3L_1.5B.yaml` | `examples/R3L/alfworld/grpo_1.5B.yaml` | `examples/R3L/alfworld/critique_grpo_1.5B.yaml` | `examples/R3L/alfworld/reflect_grpo_1.5B.yaml` | `examples/R3L/alfworld/opmd_R3L_w_o_credit_1.5B.yaml` | `examples/R3L/alfworld/opmd_R3L_w_o_reweight_1.5B.yaml` |
| WebShop 1.5B | `examples/R3L/webshop/opmd_R3L_1.5B.yaml` | `examples/R3L/webshop/grpo_1.5B.yaml` | `examples/R3L/webshop/critique_grpo_1.5B.yaml` | `examples/R3L/webshop/reflect_grpo_1.5B.yaml` | `examples/R3L/webshop/opmd_R3L_w_o_credit_1.5B.yaml` | `examples/R3L/webshop/opmd_R3L_w_o_reweight_1.5B.yaml` |
| ScienceWorld 7B | `examples/R3L/scienceworld/opmd_R3L_7B.yaml` | `examples/R3L/scienceworld/grpo_7B.yaml` | `examples/R3L/scienceworld/critique_grpo_7B.yaml` | `examples/R3L/scienceworld/reflect_grpo_7B.yaml` | `examples/R3L/scienceworld/opmd_R3L_w_o_credit_7B.yaml` | `examples/R3L/scienceworld/opmd_R3L_w_o_reweight_7B.yaml` |
| DAPO 1.5B | `examples/R3L/dapo/opmd_R3L_1.5B.yaml` | `examples/R3L/dapo/grpo_1.5B.yaml` | `examples/R3L/dapo/critique_grpo_1.5B.yaml` | `examples/R3L/dapo/reflect_grpo_1.5B.yaml` | `examples/R3L/dapo/opmd_R3L_w_o_credit_1.5B.yaml` | `examples/R3L/dapo/opmd_R3L_w_o_reweight_1.5B.yaml` |
| DAPO Qwen3-4B | `examples/R3L/dapo/step_opmd_R3L_4B.yaml` | `examples/R3L/dapo/step_grpo_4B.yaml` | `examples/R3L/dapo/step_critique_grpo_4B.yaml` | `examples/R3L/dapo/step_reflect_grpo_4B.yaml` | 无现成 | 无现成 |

## P0-E0, Positive Amplification Rule Audit

Reviewer: uCGY W6.

Type: no GPU, code/config/paper consistency check.

Why: reviewer 指出 Equation 11、Table 5 caption、Appendix K.3 对 max-reward trajectory 的 advantage 赋值不一致。这是 correctness issue，不需要新训练也必须先定下来。

Current code evidence:

- `trinity/algorithm/advantage_fn/opmd_advantage.py` 中 `OPMDReweightAdvGroupAdvantage` 逻辑是 `if exp.reward >= 1.0: score = 1.0; elif score >= 0: score = score * 3`。
- `trinity/algorithm/advantage_fn/multi_step_grpo_advantage.py` 中 `StepWiseOPMDReweightAdvAdvantageFn` 逻辑也是 max reward 设为 `1.0`，其他 positive score 乘以 `alpha=3.0`。

Need output:

| Item | Current implementation | Submitted text | Decision |
|---|---|---|---|
| max-reward trajectory advantage | 1.0 | TBD | TBD |
| positive non-max advantage | $\alpha A$ | TBD | TBD |
| negative advantage | unchanged | TBD | TBD |

Rebuttal use:

承认这是 manuscript 表述不一致，说明实现和所有实验采用同一规则。若我们决定 camera-ready 改为 max reward 设为 $1$，就明确说 revised manuscript 会统一 Equation 11、Table 5 caption、Appendix K.3。不要在 rebuttal 里含糊写 “does not affect results”，除非确认所有提交结果都来自同一实现。

## P0-E1, Seed Variance Study

Reviewer: uCGY W1.

Type: new training runs.

Goal: 用 3 seeds 证明 R3L 的结论不是 single-run artifact。最小目标是给出 mean/std，并说明 stability claim 有 across-seed 支撑。

Minimal version:

| Task | Model | Methods | Seeds | Why this subset |
|---|---|---|---|---|
| ALFWorld | Qwen2.5-1.5B-Instruct | R3L, GRPO, Critique-GRPO | 0, 1, 2 | flagship agentic task，R3L gains 大，能回应 stability 和 strong baseline |
| DAPO train, math eval suite | Qwen2.5-1.5B-Instruct | R3L, GRPO, Critique-GRPO | 0, 1, 2 | math guidance fairness 和 baseline 比较都集中在这里 |
| ScienceWorld | Qwen2.5-7B-Instruct | R3L, Critique-GRPO | 0, 1, 2 | uCGY 点名 close-margin concern，ScienceWorld-7B 差距小 |

Run count: 如果 seed 0 已有且可复用，最小补 10 个 runs。若全部重跑，是 24 个 runs。

Expanded version:

| Task | Model | Methods | Seeds | Use if |
|---|---|---|---|---|
| WebShop | Qwen2.5-1.5B-Instruct | R3L, GRPO, Critique-GRPO | 0, 1, 2 | 需要同时支撑 WebShop stability curves 和 pivot drift 回应 |
| ALFWorld | Qwen2.5-7B-Instruct | R3L, strongest baseline | 0, 1, 2 | 如果 reviewer 继续质疑 1.5B 太小 |

Metrics:

- final eval success or accuracy
- best eval success or accuracy
- mean, std, min, max over seeds
- training stability curves, at least reward/success, actor grad norm, actor KL if available, entropy if available
- number of failed runs, NaN runs, early stops

Result table template:

| Task | Model | Method | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Best eval mean |
|---|---|---|---|---|---|---|---|---|
| ALFWorld | Qwen2.5-1.5B-Instruct | R3L | TBD | TBD | TBD | TBD | TBD | TBD |
| ALFWorld | Qwen2.5-1.5B-Instruct | GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| ALFWorld | Qwen2.5-1.5B-Instruct | Critique-GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO test | Qwen2.5-1.5B-Instruct | R3L | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO test | Qwen2.5-1.5B-Instruct | GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO test | Qwen2.5-1.5B-Instruct | Critique-GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| ScienceWorld | Qwen2.5-7B-Instruct | R3L | TBD | TBD | TBD | TBD | TBD | TBD |
| ScienceWorld | Qwen2.5-7B-Instruct | Critique-GRPO | TBD | TBD | TBD | TBD | TBD | TBD |

Rebuttal acceptance bar:

- R3L mean remains higher than strongest baseline on ALFWorld and DAPO.
- For close-margin ScienceWorld-7B, if intervals overlap, be honest and soften “best on all 27” to “competitive and consistently strong”; do not overclaim.
- If variance is large, use this as motivation for adding variance reporting in camera-ready and avoid saying stability is fully solved.

## P0-E2, KL and Importance Sampling Ablation

Reviewers: uCGY W3, TC9w.

Type: new training runs plus logging check.

Goal: isolate the effect of dropping KL regularization and importance sampling/clipping. Reviewer specifically asks for curves on WebShop and one math setting, plus ablation that re-adds KL or IS.

Precheck before running:

1. Confirm whether submitted R3L config actually sets `kl_loss_fn: none`.
2. Current `opmd_R3L_*.yaml` files do not explicitly set `kl_loss_fn`, while `OPMDReweightAdvAlgorithm.default_config()` currently has `kl_loss_fn: k2` and `use_reference: True`.
3. Before writing rebuttal, inspect actual run logs for `actor/kl_loss` and config dump. If KL was active in submitted runs, this needs to be treated as a manuscript/config inconsistency, not only an ablation question.

Minimal experiment grid:

| Task | Model | Variant | Config action | Seeds |
|---|---|---|---|---|
| WebShop | Qwen2.5-1.5B-Instruct | R3L-default | submitted config | 0, 1, 2 if possible |
| WebShop | Qwen2.5-1.5B-Instruct | R3L + KL | add `kl_loss_fn: k3`, `kl_loss_fn_args.kl_coef: 0.01` if default was none | 0 |
| WebShop | Qwen2.5-1.5B-Instruct | R3L + PPO clip/IS | use PPO-style policy loss with `clip_range: 0.2`, or implement an OPMD+clip policy loss | 0 |
| DAPO | Qwen2.5-1.5B-Instruct | R3L-default | submitted config | 0, 1, 2 if possible |
| DAPO | Qwen2.5-1.5B-Instruct | R3L + KL | same as above | 0 |
| DAPO | Qwen2.5-1.5B-Instruct | R3L + PPO clip/IS | same as above | 0 |

If compute permits, add `R3L + KL + PPO clip/IS` for WebShop seed 0 only.

Required curves:

- eval success or accuracy over training
- actor KL loss or approx KL
- actor grad norm
- entropy
- `actor/pg_clipfrac` for variants with PPO-style clipping
- reward mean and second reward mean

Implementation notes:

- PPO policy loss already logs `pg_clipfrac` and `ppo_kl` in `trinity/algorithm/policy_loss_fn/ppo_policy_loss.py`.
- OPMD policy loss currently logs only `opmd_loss`, so default R3L cannot produce clip fraction unless we add a diagnostic or switch to a clipping variant.
- KL loss is computed in `trinity/trainer/verl/dp_actor.py` through `kl_loss_fn.calculate_kl_loss`; use run logs to verify metric names.

Result table template:

| Task | Model | Variant | Final score | Best score | KL mean | Max grad norm | Entropy final | Clipfrac mean | Wall-clock/step |
|---|---|---|---|---|---|---|---|---|---|
| WebShop | Qwen2.5-1.5B-Instruct | R3L-default | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| WebShop | Qwen2.5-1.5B-Instruct | R3L + KL | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| WebShop | Qwen2.5-1.5B-Instruct | R3L + PPO clip | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO | Qwen2.5-1.5B-Instruct | R3L-default | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| DAPO | Qwen2.5-1.5B-Instruct | R3L + KL | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| DAPO | Qwen2.5-1.5B-Instruct | R3L + PPO clip | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Rebuttal acceptance bar:

- If default R3L is stable without KL/IS and re-adding them does not improve score, say the safeguards are not necessary in our setting and may add cost or dampen positive updates.
- If re-adding KL helps stability but lowers score slightly, say PA handles the main instability while KL is optional; revise paper wording.
- If current submitted runs used KL by mistake, fix the claim immediately. Do not argue we dropped KL if config/logs contradict it.

## P0-E3, Math Guidance Fairness and Leakage Control

Reviewer: uCGY W4.

Type: new math runs plus offline audit.

Goal: show math gains do not come from leaking ground-truth answers through retry guidance.

Minimal training/control grid:

| Variant | Description | Config/workflow action | Model | Seeds |
|---|---|---|---|---|
| R3L-submitted | current math guidance | `examples/R3L/dapo/opmd_R3L_1.5B.yaml` | Qwen2.5-1.5B-Instruct | 0, 1, 2 if reused for E1 |
| R3L-self-reflect-only | reflection gets no answer comparison, only model trajectory and final verifier result | add workflow flag or duplicate DAPO R3L workflow | Qwen2.5-1.5B-Instruct | 0 |
| R3L-binary-verifier-only | guidance only says final answer is wrong, no error type | duplicate prompt/workflow | Qwen2.5-1.5B-Instruct | 0 |
| Critique-GRPO-matched | baseline gets same non-answer error-type critique budget | adapt critique prompt to same information budget | Qwen2.5-1.5B-Instruct | 0 |

Offline guidance audit:

- Sample 100 generated math guidance items from submitted R3L.
- For each item, label whether it contains the final answer, an intermediate numeric answer that trivially reveals final answer, only error type, or generic reasoning advice.
- Include 5 representative examples in internal notes; in rebuttal include 1-2 short paraphrased examples, not long prompt dumps.

Audit table template:

| Category | Count / 100 | Action |
|---|---|---|
| Reveals final answer | TBD | must be 0 or fix workflow |
| Reveals equivalent intermediate value | TBD | inspect manually |
| Error type only | TBD | acceptable |
| Generic process advice | TBD | acceptable |
| Invalid/unparseable guidance | TBD | report if nontrivial |

Training result table template:

| Training variant | GSM8K | Math500 | MinervaMath | OlympiadBench | AMC23 | DAPO test | Avg |
|---|---|---|---|---|---|---|---|
| R3L-submitted | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R3L-self-reflect-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R3L-binary-verifier-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Critique-GRPO-matched | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Rebuttal acceptance bar:

- If no-answer controls retain most of the gain, we can say gains are not driven by answer leakage.
- If submitted guidance is materially stronger than no-answer controls, rebuttal must concede this and narrow the claim for math tasks.
- Regardless of result, revise Appendix K.3 to explicitly state the guidance construction rule and include examples that do not reveal the answer.

## P0-E4, Baselines Below Zero-Shot and Checkpoint Selection

Reviewer: uCGY W5.

Type: evaluation plus selective reruns if needed.

Goal: directly explain cases where Critique-GRPO or Reflect-GRPO fall below zero-shot, especially Qwen2.5-7B-Instruct on GSM8K.

Required evaluations:

| Model | Eval set | Methods/checkpoints | Why |
|---|---|---|---|
| Qwen2.5-7B-Instruct | GSM8K | zero-shot base, GRPO, Critique-GRPO, Reflect-GRPO, R3L | reviewer explicitly cited this case |
| Qwen2.5-1.5B-Instruct | GSM8K and Math500 | zero-shot base, GRPO, Critique-GRPO, R3L | checks whether 52% gain framing is misleading |
| Qwen2.5-7B-Instruct | DAPO test | zero-shot base, GRPO, Critique-GRPO, R3L | separate in-distribution train/eval from cross-dataset degradation |

For each trained method, report both final checkpoint and best eval checkpoint using the same selection rule.

Result table template:

| Model | Eval set | Method | Zero-shot/base | Final ckpt | Best ckpt | Training data | Notes |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | GSM8K | GRPO | TBD | TBD | TBD | DAPO | TBD |
| Qwen2.5-7B-Instruct | GSM8K | Critique-GRPO | TBD | TBD | TBD | DAPO | TBD |
| Qwen2.5-7B-Instruct | GSM8K | Reflect-GRPO | TBD | TBD | TBD | DAPO | TBD |
| Qwen2.5-7B-Instruct | GSM8K | R3L | TBD | TBD | TBD | DAPO | TBD |

If baselines remain below zero-shot:

- Check whether drop is mainly on out-of-distribution eval sets while DAPO test improves.
- Check whether final checkpoint overfits relative to best checkpoint.
- Check whether Critique-GRPO gamma or KL settings are unusually bad. Current 1.5B Critique config has `gamma: 0.1`, `clip_range: 0.2`; some 4B step configs use KL. If tuning is limited, state tuning budget explicitly.

Optional baseline tuning grid:

| Method | Parameter | Values | Task/model |
|---|---|---|---|
| Critique-GRPO | `gamma` | 0.05, 0.1, 0.2 | DAPO Qwen2.5-1.5B or 7B |
| Critique-GRPO | `kl_coef` | none, 0.01 | DAPO Qwen2.5-1.5B or 7B |
| Reflect-GRPO | KL | none, 0.01 | DAPO Qwen2.5-1.5B or 7B |

Rebuttal acceptance bar:

- We need a clear sentence saying all baseline numbers are post-RL training, not zero-shot.
- If a baseline hurts zero-shot on GSM8K but improves DAPO test, the explanation should be cross-distribution specialization, not undertraining.
- If a tuned baseline closes the gap, use the tuned baseline in rebuttal and camera-ready.

## P1-E5, Pivot Quality and Sensitivity

Reviewers: uCGY W4, vbDZ.

Type: offline labeling plus targeted retry rollouts.

Goal: quantify whether pivot errors actually break R3L, and whether performance is robust to modest pivot noise.

Part A, oracle agreement:

| Task | Sample size | Model/checkpoint | Labels |
|---|---|---|---|
| ALFWorld | 100 or 200 failed base trajectories | initial, mid, final checkpoint | model pivot, human/LLM oracle pivot, exact match, within ±1 |
| WebShop | 100 or 200 failed base trajectories | initial, mid, final checkpoint | same |
| DAPO | 100 or 200 incorrect solutions | initial, mid, final checkpoint | same |

Metrics:

- exact pivot agreement
- within-1-step agreement
- invalid reflection rate
- retry success when pivot agrees
- retry success when pivot disagrees
- mean reward improvement conditioned on pivot correctness

Part B, pivot perturbation:

For each sampled failed trajectory, run retry with several pivot choices while keeping guidance fixed or using matched guidance.

| Pivot source | Meaning |
|---|---|
| model pivot | default R3L |
| oracle pivot | upper bound |
| pivot - 1 | tests early restart robustness |
| pivot + 1 | tests late restart robustness |
| first step | similar to full retry from scratch |
| random valid step | negative control |

Result table template:

| Task | Pivot source | Retry success | Reward improvement | Avg retry tokens | Notes |
|---|---|---|---|---|---|
| ALFWorld | model pivot | TBD | TBD | TBD | TBD |
| ALFWorld | oracle pivot | TBD | TBD | TBD | TBD |
| ALFWorld | pivot - 1 | TBD | TBD | TBD | TBD |
| ALFWorld | pivot + 1 | TBD | TBD | TBD | TBD |
| WebShop | model pivot | TBD | TBD | TBD | TBD |
| DAPO | model pivot | TBD | TBD | TBD | TBD |

Rebuttal acceptance bar:

- If model pivot is close to oracle or within-1 agreement is high, emphasize tolerant localization rather than exact-match only.
- If pivot error hurts but default R3L still improves, say the mechanism is not perfect and add sensitivity analysis.
- If WebShop pivot drifts early, analyze whether early pivots are still useful because WebShop failures often originate from early product/search decisions.

## P1-E6, Retry Trigger Rate and Compute Accounting

Reviewers: uCGY comments, previous ARR concerns.

Type: instrumentation plus log extraction.

Goal: make cost comparison fully concrete, including reflection tokens and retry trigger rate.

Need instrumentation:

In each R3L workflow, log per training step or per task:

- number of base rollouts
- number of valid reflections
- number of retries actually triggered
- number of retries skipped because reflection says success/perfect
- number of invalid reflection JSONs
- base rollout tokens
- reflection prompt tokens
- reflection completion tokens
- retry rollout tokens
- distilled training tokens
- wall-clock seconds for base/reflection/retry separately if easy

Existing useful metrics:

- `success`
- `reward`
- `second_success`
- `second_reward`
- `second_improve`
- `second_reward_diff`
- `pivot_point`

Derived metrics:

- `retry_trigger_rate = retry_triggered / base_rollouts`
- `valid_reflection_rate = valid_reflections / base_rollouts`
- `retry_success_rate = second_success / retry_triggered`
- `avg_retry_token_saving = 1 - retry_tokens / full_rollout_tokens`
- `total_generation_tokens = base_tokens + reflection_tokens + retry_tokens`
- `training_tokens = tokens with action_mask == 1`

Minimal task set:

| Task | Model | Methods |
|---|---|---|
| ALFWorld | Qwen2.5-1.5B-Instruct | R3L, GRPO, Critique-GRPO |
| WebShop | Qwen2.5-1.5B-Instruct | R3L, GRPO, Critique-GRPO |
| DAPO | Qwen2.5-1.5B-Instruct | R3L, GRPO, Critique-GRPO |

Result table template:

| Task | Method | Success/accuracy | Wall-clock sec/step | Total gen tokens/step | Train tokens/step | Env steps/step | Retry trigger rate |
|---|---|---|---|---|---|---|---|
| ALFWorld | R3L | TBD | TBD | TBD | TBD | TBD | TBD |
| ALFWorld | GRPO | TBD | TBD | TBD | TBD | TBD | N/A |
| ALFWorld | Critique-GRPO | TBD | TBD | TBD | TBD | TBD | N/A |
| WebShop | R3L | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO | R3L | TBD | TBD | TBD | TBD | N/A | TBD |

Rebuttal acceptance bar:

- Include reflection tokens in total generation tokens. Do not report only training tokens.
- If R3L is cheaper in training tokens but not total generation tokens on some math setting, state the distinction clearly.
- For agentic tasks, emphasize wall-clock and env-step savings if pivot restart reduces long failed rollouts.

## P1-E7, Ablation Disentanglement

Reviewers: uCGY W7, vbDZ.

Type: new targeted ablations, possibly workflow edits.

Goal: address concern that `w/o Reflect` removes reflection, pivot detection, retry synthesis, and credit masking together.

Existing ablations:

- full R3L
- `w/o credit`, using R3L workflow without Pivotal Credit masking
- `w/o reweight`, no Positive Amplification

Missing disentanglement experiments:

| Variant | What it isolates | Implementation idea | Task/model |
|---|---|---|---|
| retry without pivot credit | retry synthesis effect without prefix masking | use full reflection/retry but no action-mask adjustment | already close to `w/o credit` |
| pivot credit with oracle/fixed pivot but no language guidance | credit masking effect independent from reflection quality | retry from oracle or fixed pivot, no natural-language improvement suggestion | ALFWorld/WebShop 1.5B subset |
| reflection guidance with fixed restart point | language guidance effect independent from pivot localization | always retry from first failure/halfway/step 0 | ALFWorld/WebShop 1.5B subset |
| Positive Amplification only | PA independent from R3L exploration | use `opmd_reweight_adv_*` or `grpo_reweight_adv_*` without R3L workflow | existing configs available |

Minimal run:

| Task | Model | Variants | Seeds |
|---|---|---|---|
| ALFWorld | Qwen2.5-1.5B-Instruct | full R3L, w/o credit, PA only, fixed-pivot retry | 0 |
| WebShop | Qwen2.5-1.5B-Instruct | full R3L, w/o credit, PA only, fixed-pivot retry | 0 |

Rebuttal acceptance bar:

- We do not need perfect orthogonal decomposition for rebuttal, but one extra targeted ablation can show the original `w/o Reflect` row is not the only attribution evidence.
- If no time to run, explicitly acknowledge the coupling and say revised manuscript will label it as component-removal ablation rather than causal isolation.

## P1-E8, Larger or Newer Backbone Check

Reviewer: TC9w, uCGY W8.

Type: new training or reuse existing Qwen3-4B step experiments.

Goal: answer “small size model” concern and show R3L works on a newer backbone.

Already available script:

- `run_experiments.sh` contains 28 Qwen3-4B step-level experiments across ALFWorld, WebShop, ScienceWorld, DAPO, with 7 methods each.

Minimal rebuttal subset:

| Task | Model | Methods | Configs |
|---|---|---|---|
| ALFWorld | Qwen3-4B | R3L, Critique-GRPO, GRPO | `examples/R3L/alfworld/step_opmd_R3L_4B.yaml`, `step_critique_grpo_4B.yaml`, `step_grpo_4B.yaml` |
| DAPO | Qwen3-4B | R3L, Critique-GRPO, GRPO | `examples/R3L/dapo/step_opmd_R3L_4B.yaml`, `step_critique_grpo_4B.yaml`, `step_grpo_4B.yaml` |

Expanded:

- Run all 28 in `run_experiments.sh` if the cluster is available.
- If reviewer insists on “larger” not just “newer,” choose one 7B or 14B model and run only R3L vs strongest baseline on ALFWorld and DAPO. There is no existing 14B config in the current examples, so this would need new config copies.

Result table template:

| Task | Model | GRPO | Critique-GRPO | R3L | R3L gain over strongest baseline |
|---|---|---|---|---|---|
| ALFWorld | Qwen3-4B | TBD | TBD | TBD | TBD |
| DAPO test | Qwen3-4B | TBD | TBD | TBD | TBD |
| WebShop | Qwen3-4B | TBD | TBD | TBD | TBD |
| ScienceWorld | Qwen3-4B | TBD | TBD | TBD | TBD |

Rebuttal acceptance bar:

- If Qwen3-4B is already in submitted Table 1, emphasize it as contemporary backbone and avoid overclaiming “large model”.
- If new Qwen3-4B results are strong, use them to address TC9w directly.

## P2-E9, Theory Assumption Diagnostics

Reviewer: uCGY W2.

Type: log extraction, no full new training if logs already have advantages and retry metrics.

Goal: soften theorem framing and show empirical ranges for assumptions used in the analysis.

Metrics to extract:

- fraction of positive advantages per group
- positive vs negative advantage magnitude ratio
- retry success probability or retry trigger probability, $p_{\text{retry}}$
- gradient norm ratio if available
- number of failure-dominated groups per training step

Result table template:

| Task | Model | Positive advantage fraction | $|\bar{A}_{-}| / \bar{A}_{+}$ | Retry trigger rate | Retry success | Notes |
|---|---|---|---|---|---|---|
| ALFWorld | Qwen2.5-1.5B-Instruct | TBD | TBD | TBD | TBD | TBD |
| WebShop | Qwen2.5-1.5B-Instruct | TBD | TBD | TBD | TBD | TBD |
| DAPO | Qwen2.5-1.5B-Instruct | TBD | TBD | TBD | TBD | TBD |

Rebuttal use:

Use this to say the theory is intended to explain the observed behavior under measurable conditions, not to give a global guarantee. This directly addresses uCGY W2.

## P2-E10, Framing and Strongest-Baseline Gains

Reviewer: uCGY W7.

Type: table recomputation, no new training unless missing results.

Goal: replace “5% to 52% relative improvement” with less promotional and more defensible statistics.

Needed calculations:

- absolute gain over strongest baseline for each task/model cell
- relative gain over strongest baseline for each task/model cell
- mean, median, min, max gain over strongest baseline
- count of first-place, second-place, and lower placements
- separate agentic and math summaries

Result table template:

| Split | Metric | R3L vs strongest baseline |
|---|---|---|
| Agentic | mean absolute gain | TBD |
| Agentic | median absolute gain | TBD |
| Agentic | first-place count | TBD |
| Math | mean absolute gain | TBD |
| Math | median absolute gain | TBD |
| Math | first-or-second count | TBD |
| All | worst relative gain | TBD |
| All | best relative gain | TBD |

Rebuttal use:

If the 52% number comes from a weak-baseline comparison where Critique-GRPO beats R3L, do not lead with it. Say we will revise the abstract to report gains over the strongest baseline or report average gains.

## Suggested Execution Order

Start with the experiments that can change the rebuttal argument most.

1. P0-E0, PA rule audit. This is immediate and affects wording.
2. P0-E4, zero-shot/baseline evaluation. This may reuse existing checkpoints and directly addresses an easy-to-understand reviewer concern.
3. P0-E1, seed variance. Launch long runs early.
4. P0-E2, KL/IS ablation. Do WebShop first, then DAPO.
5. P0-E3, math guidance audit and control. Start offline audit immediately, launch controls if workflow edits are ready.
6. P1-E6, compute accounting instrumentation. Add logging before any reruns if possible, so P0 runs also produce cost numbers.
7. P1-E5, pivot sensitivity. Use saved trajectories from P0/E6 runs.
8. P1-E8, Qwen3-4B subset. Run if cluster has spare capacity.
9. P2-E9 and P2-E10, log/table analysis.

## Minimal Rebuttal Package If Time Is Tight

If we only have time for a small package, prioritize the following deliverables.

| Deliverable | Source |
|---|---|
| 3-seed mean/std on ALFWorld 1.5B and DAPO 1.5B for R3L vs GRPO vs Critique-GRPO | P0-E1 |
| WebShop and DAPO stability curves with KL/grad norm/entropy, plus one KL re-add ablation | P0-E2 |
| math guidance audit showing no final answer leakage | P0-E3 |
| same-prompt zero-shot vs post-RL table for Qwen2.5-7B GSM8K | P0-E4 |
| retry trigger rate and total token accounting for ALFWorld/WebShop/DAPO | P1-E6 |
| PA rule clarification table | P0-E0 |

This package should be enough to respond to the main uCGY concerns while reinforcing TC9w and vbDZ.
