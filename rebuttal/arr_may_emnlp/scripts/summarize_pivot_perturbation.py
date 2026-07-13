"""Aggregate paired pivot perturbation JSON records."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _bootstrap_delta(
    records: List[Dict],
    variant: str,
    metric: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [records[rng.randrange(len(records))] for _ in records]
        deltas.append(
            mean(
                float(record["variants"][variant][metric])
                - float(record["variants"]["model"][metric])
                for record in selected
            )
        )
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def load_records(input_dir: Path, max_samples: int) -> List[Dict]:
    records = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open(encoding="utf-8") as input_file:
            record = json.load(input_file)
        if record.get("variants") and "model" in record["variants"]:
            records.append(record)
        if max_samples and len(records) >= max_samples:
            break
    return records


def summarize(records: List[Dict], bootstrap_samples: int, seed: int) -> List[Dict]:
    variants = list(records[0]["variants"])
    rows = []
    for variant in variants:
        # Fixed-offset comparisons require the requested shift to be feasible.
        # Otherwise, boundary clamping can make (for example) early_5 identical
        # to start and would overstate the effective perturbation size.
        paired_records = [
            record
            for record in records
            if variant in record["variants"]
            and not bool(record["variants"][variant].get("clipped", False))
        ]
        if not paired_records:
            print(f"Skipping {variant}: no unclipped paired records.")
            continue
        success = [
            float(record["variants"][variant]["retry_success"])
            for record in paired_records
        ]
        rewards = [
            float(record["variants"][variant]["retry_reward"])
            for record in paired_records
        ]
        improvements = [
            float(record["variants"][variant]["reward_improvement"])
            for record in paired_records
        ]
        steps = [
            float(record["variants"][variant]["retry_steps"])
            for record in paired_records
        ]
        success_delta = mean(success) - mean(
            float(record["variants"]["model"]["retry_success"])
            for record in paired_records
        )
        reward_delta = mean(rewards) - mean(
            float(record["variants"]["model"]["retry_reward"])
            for record in paired_records
        )
        ci_low, ci_high = _bootstrap_delta(
            paired_records,
            variant,
            "retry_success",
            samples=bootstrap_samples,
            seed=seed,
        )
        rows.append(
            {
                "variant": variant,
                "n": len(paired_records),
                "success_rate": mean(success),
                "mean_reward": mean(rewards),
                "mean_reward_improvement": mean(improvements),
                "mean_retry_steps": mean(steps),
                "success_delta_vs_model": success_delta,
                "success_delta_ci95_low": ci_low,
                "success_delta_ci95_high": ci_high,
                "reward_delta_vs_model": reward_delta,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    records = load_records(args.input_dir, args.max_samples)
    if not records:
        raise SystemExit(f"No eligible pivot records found in {args.input_dir}")
    rows = summarize(records, args.bootstrap_samples, args.seed)
    output_prefix = args.output_prefix or args.input_dir / "pivot_summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, ensure_ascii=False, indent=2)
    with output_prefix.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("| Pivot | N | Retry SR | Mean reward | Delta SR vs model | 95% CI | Steps |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['variant']} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['mean_reward']:.3f} | {row['success_delta_vs_model']:+.3f} | "
            f"[{row['success_delta_ci95_low']:+.3f}, "
            f"{row['success_delta_ci95_high']:+.3f}] | "
            f"{row['mean_retry_steps']:.1f} |"
        )


if __name__ == "__main__":
    main()
