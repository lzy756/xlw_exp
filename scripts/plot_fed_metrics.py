#!/usr/bin/env python3
"""Plot avg_acc, worst_acc, and variance curves from multiple metrics.json files.

Usage:
    uv run python scripts/plot_fed_metrics.py \
        --metrics outputs/2025-12-06-12-20-pacs_cnn5_baseline_fedbn/metrics.json \
                         outputs/2025-12-05-17-59-pacs_cnn5_baseline/metrics.json \
                         outputs/2025-12-06-12-55-pacs_cnn5_fedrep_lora/metrics.json \
        --labels FedBN FedAvg FedRep-LoRA \
        --out figs/pacs_cnn5_comparison.png

    # With per-domain accuracy plots (one subplot per domain):
    uv run python scripts/plot_fed_metrics.py \
        --metrics outputs/exp1/metrics.json outputs/exp2/metrics.json \
        --labels Method1 Method2 \
        --out figs/comparison.png \
        --per-domain

Notes:
- Labels are optional; when omitted, the script uses the metrics filename stem.
- Handles runs with different round counts (curves truncated to their own length).
- Outputs a single figure with three subplots: avg_acc, worst_acc, variance.
- Use --per-domain to generate additional per-domain accuracy plots (2x2 grid for 4 domains).
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("avg_acc", []), data.get("worst_acc", []), data.get("variance", [])


def load_per_domain_acc(path: Path):
    """Load per-domain accuracy from metrics.json.

    Returns:
        Dictionary mapping domain name to list of accuracies per round.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("per_domain_acc", {})


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


def plot_per_domain_curves(metric_paths: List[Path], labels: List[str], out_path: Path):
    """Plot per-domain accuracy curves with one subplot per domain.

    Each domain gets its own subplot, showing accuracy curves for all methods.
    Layout is a single row to save vertical space for single-column papers.
    """
    # Collect all domains from all metrics files
    all_domains = set()
    for p in metric_paths:
        per_domain = load_per_domain_acc(p)
        all_domains.update(per_domain.keys())

    domains = sorted(all_domains)
    n_domains = len(domains)

    if n_domains == 0:
        print("No per_domain_acc data found in metrics files.")
        return

    # Single row layout for paper-friendly format
    nrows, ncols = 1, n_domains

    # Use consistent color scheme with distinct markers for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Compact figure size: narrower per subplot, shorter height
    _, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8))

    # Flatten axes array for easy iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each domain in its own subplot
    for idx, domain in enumerate(domains):
        ax = axes[idx]

        for method_idx, (p, lbl) in enumerate(zip(metric_paths, labels)):
            per_domain = load_per_domain_acc(p)
            if domain not in per_domain:
                continue

            acc = per_domain[domain]
            rounds = range(1, len(acc) + 1)

            color = colors[method_idx % len(colors)]
            marker = markers[method_idx % len(markers)]
            # Only the first method uses dashed line, others use solid
            linestyle = '--' if method_idx == 0 else '-'

            # Plot with markers at intervals to avoid clutter
            marker_interval = max(1, len(acc) // 10)
            ax.plot(
                rounds, acc,
                label=lbl,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=marker_interval,
                markersize=5,
                linewidth=1.5,
                alpha=0.85
            )

        # Format domain name for title (capitalize, replace underscore)
        domain_title = domain.replace('_', ' ').title()
        ax.set_title(f"{domain_title}", fontsize=10)
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        # Only show legend on first subplot to save space
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for idx in range(n_domains, len(axes)):
        axes[idx].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved per-domain plot to {out_path}")


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
    parser.add_argument(
        "--per-domain",
        action="store_true",
        dest="per_domain",
        help="Plot per-domain accuracy curves (one subplot per domain)"
    )
    parser.add_argument(
        "--per-domain-out",
        type=str,
        dest="per_domain_out",
        default=None,
        help="Output path for per-domain plot (default: append '_per_domain' to --out)"
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

    # Plot per-domain curves if requested
    if args.per_domain:
        if args.per_domain_out:
            per_domain_path = Path(args.per_domain_out)
        else:
            # Default: append '_per_domain' before extension
            per_domain_path = out_path.parent / f"{out_path.stem}_per_domain{out_path.suffix}"
        plot_per_domain_curves(metric_paths, labels, per_domain_path)


if __name__ == "__main__":
    main()
