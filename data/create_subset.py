#!/usr/bin/env python3
"""Create a random subset from a path-list file."""

import argparse
import random
from pathlib import Path


def create_subset(input_file, output_file, ratio, seed):
    with open(input_file, "r", encoding="utf-8") as file:
        paths = [line for line in file.readlines() if line.strip()]

    subset_size = max(1, int(len(paths) * ratio))
    subset = random.Random(seed).sample(paths, subset_size)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(subset)

    print(f"Saved {subset_size}/{len(paths)} paths to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create a random subset from a DRFusion path list.")
    parser.add_argument("--input", default="data/visible_train_paths.txt")
    parser.add_argument("--output", default="data/visible_train_paths_subset.txt")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_subset(
        input_file=Path(args.input),
        output_file=Path(args.output),
        ratio=args.ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
