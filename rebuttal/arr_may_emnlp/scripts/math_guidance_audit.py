"""Prepare and summarize a manual audit of math retry guidance."""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


LABELS = {
    "final_answer",
    "equivalent_intermediate",
    "error_type_only",
    "generic_advice",
    "invalid",
}


def _iter_json_records(path: Path) -> Iterator[tuple[str, Dict]]:
    paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    for json_path in paths:
        with json_path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
        records = data if isinstance(data, list) else [data]
        for index, record in enumerate(records):
            if isinstance(record, dict):
                yield f"{json_path}:{index}", record


def _extract_guidance(record: Dict) -> Optional[str]:
    for key in (
        "guidance_prompt",
        "guidance",
        "reflection_text",
        "reflection_report",
        "reflection_data",
    ):
        value = record.get(key)
        if value:
            return (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )

    for message in record.get("trajectory", []):
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", ""))
        if "Previous Attempt Analysis & Guidance" in content:
            return content.split("Previous Attempt Analysis & Guidance", 1)[-1].lstrip(
                "\n# "
            )
    return None


def _normalize_answer(text: str) -> str:
    text = re.sub(r"\\(?:boxed|text|mathrm)\s*\{([^{}]*)\}", r"\1", str(text))
    text = re.sub(r"[^0-9A-Za-z.+\-/]", "", text)
    return text.lower()


def _exact_answer_match(ground_truth: str, guidance: str) -> bool:
    normalized_truth = _normalize_answer(ground_truth)
    normalized_guidance = _normalize_answer(guidance)
    return bool(normalized_truth and normalized_truth in normalized_guidance)


def prepare(inputs: Iterable[Path], output: Path, sample_size: int, seed: int) -> None:
    candidates = []
    for input_path in inputs:
        for source, record in _iter_json_records(input_path):
            ground_truth = record.get("ground_truth") or record.get("answer")
            guidance = _extract_guidance(record)
            if not ground_truth or not guidance:
                continue
            candidates.append(
                {
                    "record_id": record.get("task_id", source),
                    "source": source,
                    "ground_truth": str(ground_truth),
                    "exact_answer_match": _exact_answer_match(
                        str(ground_truth), guidance
                    ),
                    "label": "",
                    "notes": "",
                    "guidance": guidance,
                }
            )
    if not candidates:
        raise SystemExit(
            "No records containing both ground truth and retry guidance were found"
        )
    if len(candidates) < sample_size:
        raise SystemExit(
            f"Found only {len(candidates)} eligible records, but {sample_size} were "
            "requested. Increase buffer.total_steps and recollect before labeling."
        )

    random.Random(seed).shuffle(candidates)
    selected = candidates[:sample_size]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(
        f"Prepared {len(selected)} audit rows from {len(candidates)} candidates: {output}"
    )


def summarize(audit_csv: Path, output: Optional[Path]) -> None:
    with audit_csv.open(encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    invalid_labels = sorted({row["label"].strip() for row in rows} - LABELS - {""})
    if invalid_labels:
        raise SystemExit(
            f"Unknown labels: {invalid_labels}; allowed labels: {sorted(LABELS)}"
        )
    unlabeled = sum(not row["label"].strip() for row in rows)
    counts = Counter(row["label"].strip() for row in rows if row["label"].strip())
    result = {
        "total_rows": len(rows),
        "labeled_rows": len(rows) - unlabeled,
        "unlabeled_rows": unlabeled,
        "counts": {label: counts.get(label, 0) for label in sorted(LABELS)},
        "automatic_exact_match_flags": sum(
            row["exact_answer_match"].lower() == "true" for row in rows
        ),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("inputs", nargs="+", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--sample-size", type=int, default=100)
    prepare_parser.add_argument("--seed", type=int, default=2026)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("audit_csv", type=Path)
    summarize_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.inputs, args.output, args.sample_size, args.seed)
    else:
        summarize(args.audit_csv, args.output)


if __name__ == "__main__":
    main()
