"""Compute mean/std for the focused ALFWorld seed study."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    args = parser.parse_args()

    with args.results_csv.open(encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(float(row["final_score"]))

    print("| Method | Seeds | Scores | Mean | Std |")
    print("|---|---:|---|---:|---:|")
    for method, scores in sorted(grouped.items()):
        if len(scores) < 3:
            raise SystemExit(
                f"{method} has only {len(scores)} seeds; at least 3 are required"
            )
        print(
            f"| {method} | {len(scores)} | "
            f"{', '.join(f'{score:.3f}' for score in scores)} | "
            f"{mean(scores):.3f} | {stdev(scores):.3f} |"
        )


if __name__ == "__main__":
    main()
