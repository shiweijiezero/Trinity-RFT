# ARR May Rebuttal Experiment Runbook

This runbook contains the reduced experiment package chosen for the limited rebuttal window.
Run every command from the repository root on the GPU cluster and record the active commit hash.
All newly prepared runs use WandB project `weijie_r3l`; the config `group` separates each
experiment family and `name` identifies the method and seed.

The eight ALFWorld training configs below require a two-node Ray cluster with 8 GPUs per node.
They reserve one complete node (8 GPUs) for rollout and one complete node (8 GPUs) for training.
Before starting a run, verify that `ray status` reports exactly the intended two nodes and 16 GPUs.

## 0. New Machine and ALFWorld Setup

Use Python 3.10 and CUDA 12.6 or newer. Run these commands from the repository root. If the two
nodes do not share the same environment and repository path, repeat the package installation on
both nodes.

```bash
conda create -n r3l python=3.10 -y
conda activate r3l
python -m pip install --upgrade pip setuptools wheel
pip install -e .
pip install flash-attn==2.8.1 --no-build-isolation
pip install 'alfworld==0.4.2'
```

This R3L workflow uses ALFWorld's TextWorld environment, not the THOR visual environment, so an
X server and `DISPLAY` are not required. Download the ALFWorld games to shared CPFS storage and
keep `ALFWORLD_DATA` exported on both nodes:

```bash
export ALFWORLD_DATA=/mnt/cpfs/shiweijie/alfworld
mkdir -p "$ALFWORLD_DATA"
alfworld-download --data-dir "$ALFWORLD_DATA"
python examples/R3L/alfworld/get_alfworld_data.py
wc -l examples/R3L/alfworld/alfworld_data/train.jsonl
wc -l examples/R3L/alfworld/alfworld_data/test.jsonl
```

The expected full taskset contains 3553 training games and 140 `valid_seen` evaluation games.
Every generated JSONL record contains an absolute game path under `/mnt/cpfs`, so both Ray nodes
must mount that path identically. Configure persistent outputs and WandB before launching:

```bash
export TRINITY_CHECKPOINT_ROOT_DIR=/mnt/cpfs/shiweijie/checkpoints/r3l_rebuttal
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
wandb login
```

Verify the runtime before allocating both nodes:

```bash
python -c "import torch, ray, vllm, verl, alfworld, textworld; print(torch.__version__, torch.version.cuda); print(ray.__version__, vllm.__version__)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
test -f /mnt/cpfs/shiweijie/hf_cache/qwen2.5-1.5b-ins/config.json
test -f examples/R3L/alfworld/alfworld_data/train.jsonl
```

## 1. Focused Three-Seed Study

The submitted seed-0 runs are reused only if their code, config, training budget, evaluation split,
and checkpoint-selection rule match the new runs. Otherwise rerun seed 0 as well.
Each command below occupies two 8-GPU nodes: 8 GPUs for trainer and 8 GPUs for rollout.

```bash
trinity run --config examples/R3L/alfworld/rebuttal/opmd_R3L_1.5B_seed1.yaml
trinity run --config examples/R3L/alfworld/rebuttal/opmd_R3L_1.5B_seed2.yaml
trinity run --config examples/R3L/alfworld/rebuttal/grpo_1.5B_seed1.yaml
trinity run --config examples/R3L/alfworld/rebuttal/grpo_1.5B_seed2.yaml
```

Copy `rebuttal/arr_may_emnlp/results/alfworld_seed_results_template.csv`, fill in final and
best scores for seeds 0, 1, and 2, and summarize final scores:

```bash
python rebuttal/arr_may_emnlp/scripts/summarize_seed_results.py \
  rebuttal/arr_may_emnlp/results/alfworld_seed_results.csv
```

Report final-score mean and sample standard deviation. Keep best-checkpoint scores as a separate
column; do not select a different reporting rule per seed.

## 2. ALFWorld KL x IS Safeguard Ablation

Run the four Qwen2.5-1.5B configurations with the same seed and training budget. Each run uses
16 GPUs across two nodes (8 trainer + 8 rollout) and logs to WandB project `weijie_r3l`, group
`alfworld_qwen25_1.5b_kl_is_ablation`.

```bash
trinity run --config examples/R3L/alfworld/rebuttal/r3l_1.5B_no_kl_no_is_seed0.yaml
trinity run --config examples/R3L/alfworld/rebuttal/r3l_1.5B_kl_only_seed0.yaml
trinity run --config examples/R3L/alfworld/rebuttal/r3l_1.5B_is_only_seed0.yaml
trinity run --config examples/R3L/alfworld/rebuttal/r3l_1.5B_kl_is_seed0.yaml
```

Use eight separate 8-GPU nodes to run all four concurrently, or run them sequentially on one
two-node allocation.
Fill `rebuttal/arr_may_emnlp/results/alfworld_kl_is_ablation_template.csv` with final and best
evaluation scores. Before claiming that the variants are similar, inspect reward, gradient norm,
entropy, KL loss for KL variants, and `actor/is_clipfrac` for IS variants. Record failed runs,
NaNs, and early stops rather than silently excluding them.

## 3. ALFWorld Pivot Perturbation

Point `TRINITY_MODEL_PATH` to a fixed R3L checkpoint. A mid-training checkpoint is preferable if
the final checkpoint yields too few failed base trajectories. The config evaluates up to 200 tasks;
the summarizer uses at most 100 eligible failed trajectories with valid reflections.

```bash
export TRINITY_MODEL_PATH=/path/to/r3l/checkpoint
trinity run --config examples/R3L/alfworld/rebuttal/pivot_perturbation_1.5B.yaml
python rebuttal/arr_may_emnlp/scripts/summarize_pivot_perturbation.py \
  rebuttal/arr_may_emnlp/results/pivot_perturbation_alfworld \
  --max-samples 100
```

The workflow generates one reflection per base trajectory, removes `retry_from_step` from the
shared guidance, and varies only the actual restart point. Duplicate tested pivots are evaluated
once per task to save inference cost.

## 4. Math Guidance Leakage Audit

Process 150 deterministic DAPO tasks without starting a trainer. This oversamples because
successful bases and invalid reflections do not produce retry guidance; the audit sheet selects
exactly 100 eligible records.

```bash
export TRINITY_MODEL_PATH=/path/to/submitted/model/or/checkpoint
trinity run --config examples/R3L/dapo/rebuttal/math_guidance_audit_1.5B.yaml
```

Prepare a reproducible manual audit sheet:

```bash
python rebuttal/arr_may_emnlp/scripts/math_guidance_audit.py prepare \
  rebuttal/arr_may_emnlp/results/math_guidance_audit/raw/train \
  --sample-size 100 \
  --output rebuttal/arr_may_emnlp/results/math_guidance_audit/audit.csv
```

The prepare command prints both the candidate count and selected count. If it finds fewer than
100 candidates, increase `buffer.total_steps` and recollect before labeling; do not report a
smaller denominator as a 100-record audit.

Label every row using exactly one of:

- `final_answer`
- `equivalent_intermediate`
- `error_type_only`
- `generic_advice`
- `invalid`

Then summarize it:

```bash
python rebuttal/arr_may_emnlp/scripts/math_guidance_audit.py summarize \
  rebuttal/arr_may_emnlp/results/math_guidance_audit/audit.csv \
  --output rebuttal/arr_may_emnlp/results/math_guidance_audit/summary.json
```

The automatic exact-match flag is only a screening aid. The final leakage claim must use the
manual labels, especially for equivalent intermediate values.

## 5. Retry Trigger and Cost Metrics

New R3L runs log the following per-base metrics, which become rates after aggregation:

- `reflection_valid_rate`
- `retry_trigger_rate`
- `retry_skip_success_rate`
- `retry_skip_invalid_rate`
- `retry_completed_rate`

They also log response/completion-token counts, excluding prompt tokens:

- `base_completion_tokens`
- `reflection_completion_tokens`
- `retry_completion_tokens`
- `total_generation_tokens`

Use the ALFWorld R3L seed runs for the rebuttal trigger-rate table. Do not average the trigger
metrics from GRPO because retry does not apply to that baseline.

## Deferred Experiments

The reduced package intentionally defers KL/IS ablations, models larger than 7B, open-ended RLHF,
and a fully orthogonal reflection/pivot/retry training ablation. The response should acknowledge
these as limitations or revision commitments rather than imply that they were run.
