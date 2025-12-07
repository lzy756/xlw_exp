#!/usr/bin/env python3
"""Plot avg_acc, worst_acc, and variance curves from multiple metrics.json files.

Usage:
    uv run python scripts/plot_fed_metrics.py \
        --metrics outputs/2025-12-06-12-20-pacs_cnn5_baseline_fedbn/metrics.json \
                         outputs/2025-12-05-17-59-pacs_cnn5_baseline/metrics.json \
                         outputs/2025-12-06-12-55-pacs_cnn5_fedrep_lora/metrics.json \
        --labels FedBN FedAvg FedRep-LoRA \
        --out figs/pacs_cnn5_comparison.png

Notes:
- Labels are optional; when omitted, the script uses the metrics filename stem.
- Handles runs with different round counts (curves truncated to their own length).
- Outputs a single figure with two subplots: avg_acc (left) and worst_acc (right).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("avg_acc", []), data.get("worst_acc", []), data.get("variance", [])


def plot_curves(metric_paths: List[Path], labels: List[str], out_path: Path):
    plt.figure(figsize=(14, 4))

    # Subplot 1: avg_acc
    plt.subplot(1, 3, 1)
    for p, lbl in zip(metric_paths, labels):
        avg_acc, _, _ = load_metrics(p)
        plt.plot(range(1, len(avg_acc) + 1), avg_acc, label=lbl)
    plt.title("Average Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Avg Acc (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: worst_acc
    plt.subplot(1, 3, 2)
    for p, lbl in zip(metric_paths, labels):
        _, worst_acc, _ = load_metrics(p)
        plt.plot(range(1, len(worst_acc) + 1), worst_acc, label=lbl)
    plt.title("Worst Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Worst Acc (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: variance
    plt.subplot(1, 3, 3)
    for p, lbl in zip(metric_paths, labels):
        _, _, variance = load_metrics(p)
        plt.plot(range(1, len(variance) + 1), variance, label=lbl)
    plt.title("Variance")
    plt.xlabel("Round")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot avg_acc and worst_acc from metrics.json files")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Paths to metrics.json files"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels matching metrics; defaults to filename stems"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figs/metrics_comparison.png",
        help="Output PNG path"
    )
    args = parser.parse_args()

    metric_paths = [Path(m) for m in args.metrics]
    for p in metric_paths:
        if not p.exists():
            raise FileNotFoundError(f"Metrics file not found: {p}")

    if args.labels:
        if len(args.labels) != len(metric_paths):
            raise ValueError("Number of labels must match number of metrics files")
        labels = args.labels
    else:
        labels = [p.stem for p in metric_paths]

    out_path = Path(args.out)
    plot_curves(metric_paths, labels, out_path)


if __name__ == "__main__":
    main()
