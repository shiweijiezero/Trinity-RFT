import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Optional


def _default_data_root() -> Path:
    """Return the directory containing ALFWorld's train/valid_seen splits."""
    project_data_dir = Path(__file__).resolve().parent / "alfworld_data"
    data_dir = Path(os.environ.get("ALFWORLD_DATA", str(project_data_dir))).expanduser()
    if data_dir.name == "json_2.1.1":
        return data_dir
    return data_dir / "json_2.1.1"


def create_dataset_files(
    output_dir: Path,
    alfworld_data_root: Path,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = 42,
) -> None:
    """Create Trinity FILE tasksets containing absolute ALFWorld game paths."""
    train_pattern = str(alfworld_data_root / "train" / "*" / "*" / "game.tw-pddl")
    test_pattern = str(alfworld_data_root / "valid_seen" / "*" / "*" / "game.tw-pddl")

    train_game_files = sorted(os.path.abspath(path) for path in glob.glob(train_pattern))
    test_game_files = sorted(os.path.abspath(path) for path in glob.glob(test_pattern))

    print(f"ALFWorld data root: {alfworld_data_root}")
    print(f"Total train game files found: {len(train_game_files)}")
    print(f"Total test game files found: {len(test_game_files)}")

    if not train_game_files or not test_game_files:
        raise FileNotFoundError(
            "ALFWorld game files were not found. Install `alfworld[full]`, then run "
            f"`alfworld-download --data-dir {alfworld_data_root.parent}`."
        )

    train_size = len(train_game_files) if train_size is None else train_size
    test_size = len(test_game_files) if test_size is None else test_size
    if train_size > len(train_game_files):
        raise ValueError(f"train_size {train_size} > available {len(train_game_files)}")
    if test_size > len(test_game_files):
        raise ValueError(f"test_size {test_size} > available {len(test_game_files)}")

    rng = random.Random(seed)
    selected_train_files = rng.sample(train_game_files, train_size)
    selected_test_files = rng.sample(test_game_files, test_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [{"game_file": path, "target": ""} for path in selected_train_files],
        "test": [{"game_file": path, "target": ""} for path in selected_test_files],
    }
    for split, records in splits.items():
        output_file = output_dir / f"{split}.jsonl"
        with output_file.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataset_info = {
        "citation": "",
        "description": "ALFWorld TextWorld game-file taskset for Trinity-RFT",
        "splits": {
            "train": {"name": "train", "num_examples": len(splits["train"])},
            "test": {"name": "test", "num_examples": len(splits["test"])},
        },
    }
    with (output_dir / "dataset_dict.json").open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, indent=2, ensure_ascii=False)

    print(
        f"Created dataset with {len(splits['train'])} train and "
        f"{len(splits['test'])} test examples in {output_dir}."
    )


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare ALFWorld tasksets for R3L experiments.")
    parser.add_argument(
        "--alfworld-data-root",
        type=Path,
        default=_default_data_root(),
        help="Path to ALFWorld's json_2.1.1 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=current_dir / "alfworld_data",
        help="Output directory for train.jsonl and test.jsonl.",
    )
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset_files(
        output_dir=args.output_dir.resolve(),
        alfworld_data_root=args.alfworld_data_root.expanduser().resolve(),
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
