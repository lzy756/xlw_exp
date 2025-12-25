#!/usr/bin/env python3
"""Compute final round statistics (mean ± std) for avg_acc and worst_acc.

Usage:
    # Single method with multiple runs (default: last 20 rounds)
    uv run python scripts/compute_final_stats.py \
        --method FedRep \
        --metrics outputs/run1/metrics.json outputs/run2/metrics.json outputs/run3/metrics.json

    # Multiple methods (compare different approaches)
    uv run python scripts/compute_final_stats.py \
        --method FedRep \
        --metrics outputs/fedrep_run1/metrics.json outputs/fedrep_run2/metrics.json \
        --method FedBN \
        --metrics outputs/fedbn_run1/metrics.json outputs/fedbn_run2/metrics.json \
        --method FedRep-LoRA \
        --metrics outputs/fedrep_lora_run1/metrics.json outputs/fedrep_lora_run2/metrics.json

    # Custom number of final rounds to average
    uv run python scripts/compute_final_stats.py \
        --method FedRep \
        --metrics outputs/fedrep_run1/metrics.json outputs/fedrep_run2/metrics.json \
        --last-n-rounds 20

    # Output to file
    uv run python scripts/compute_final_stats.py \
        --method FedRep \
        --metrics outputs/fedrep_run1/metrics.json outputs/fedrep_run2/metrics.json \
        --out results/final_stats.txt

Notes:
- Use multiple --method and --metrics pairs to specify different methods.
- Each method can have multiple runs (different seeds).
- The script computes mean ± std over the last N rounds across all runs.
- Default: last 20 rounds for better variance estimation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


def load_final_metrics(path: Path, last_n_rounds: int = 20) -> Tuple[List[float], List[float]]:
    """Load last N rounds of avg_acc and worst_acc from metrics.json.

    Args:
        path: Path to metrics.json file.
        last_n_rounds: Number of final rounds to extract.

    Returns:
        Tuple of (avg_acc_list, worst_acc_list) for the last N rounds.
    """
    with open(path, "r") as f:
        data = json.load(f)

    avg_acc = data.get("avg_acc", [])
    worst_acc = data.get("worst_acc", [])

    if not avg_acc or not worst_acc:
        raise ValueError(f"Missing avg_acc or worst_acc in {path}")

    # Take last N rounds (or all if fewer than N)
    n = min(last_n_rounds, len(avg_acc))
    return avg_acc[-n:], worst_acc[-n:]


def compute_stats(metric_paths: List[Path], last_n_rounds: int = 20) -> Dict:
    """Compute mean ± std for final rounds across multiple runs.

    Args:
        metric_paths: List of paths to metrics.json files.
        last_n_rounds: Number of final rounds to use for statistics.

    Returns:
        Dictionary with statistics and raw values.
    """
    all_avg_accs = []
    all_worst_accs = []

    for path in metric_paths:
        avg_list, worst_list = load_final_metrics(path, last_n_rounds)
        all_avg_accs.extend(avg_list)
        all_worst_accs.extend(worst_list)

    # Compute statistics over all collected values (population std, ddof=0)
    avg_mean = float(np.mean(all_avg_accs))
    avg_std = float(np.std(all_avg_accs))
    worst_mean = float(np.mean(all_worst_accs))
    worst_std = float(np.std(all_worst_accs))

    return {
        "avg_acc": (avg_mean, avg_std),
        "worst_acc": (worst_mean, worst_std),
        "n_samples": len(all_avg_accs),
        "n_runs": len(metric_paths),
        "raw_avg_accs": all_avg_accs,
        "raw_worst_accs": all_worst_accs
    }


def format_stats_table(method_stats: Dict[str, Dict], last_n_rounds: int) -> str:
    """Format statistics as a readable table.

    Args:
        method_stats: Dictionary mapping method name to statistics.
        last_n_rounds: Number of rounds used for statistics.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("=" * 90)
    lines.append(f"Final Round Statistics (Mean ± Std over last {last_n_rounds} rounds)")
    lines.append("=" * 90)
    lines.append(f"{'Method':<20} {'Avg Acc (%)':<25} {'Worst Acc (%)':<25} {'Runs':<10} {'Samples':<10}")
    lines.append("-" * 90)

    for method, stats in method_stats.items():
        avg_mean, avg_std = stats["avg_acc"]
        worst_mean, worst_std = stats["worst_acc"]
        n_runs = stats["n_runs"]
        n_samples = stats["n_samples"]

        avg_str = f"{avg_mean:.2f} ± {avg_std:.2f}"
        worst_str = f"{worst_mean:.2f} ± {worst_std:.2f}"

        lines.append(f"{method:<20} {avg_str:<25} {worst_str:<25} {n_runs:<10} {n_samples:<10}")

    lines.append("=" * 90)

    # Add summary statistics
    lines.append("\nSummary:")
    lines.append("-" * 90)
    for method, stats in method_stats.items():
        avg_mean, avg_std = stats["avg_acc"]
        worst_mean, worst_std = stats["worst_acc"]
        lines.append(f"\n{method}:")
        lines.append(f"  Avg Acc:   {avg_mean:.2f} ± {avg_std:.2f}  (n={stats['n_samples']})")
        lines.append(f"  Worst Acc: {worst_mean:.2f} ± {worst_std:.2f}  (n={stats['n_samples']})")
        lines.append(f"  Runs: {stats['n_runs']}, Samples per metric: {stats['n_samples']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute final round statistics from multiple metrics.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method name (can be specified multiple times)"
    )
    parser.add_argument(
        "--metrics",
        action="append",
        nargs="+",
        dest="metrics_lists",
        help="Paths to metrics.json files for the corresponding method"
    )
    parser.add_argument(
        "--last-n-rounds",
        type=int,
        default=20,
        dest="last_n_rounds",
        help="Number of final rounds to use for statistics (default: 20)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.methods or not args.metrics_lists:
        parser.error("At least one --method and --metrics pair is required")

    if len(args.methods) != len(args.metrics_lists):
        parser.error("Number of --method and --metrics arguments must match")

    # Process each method
    method_stats = {}

    for method, metric_paths in zip(args.methods, args.metrics_lists):
        # Convert to Path objects and validate
        paths = [Path(m) for m in metric_paths]
        for p in paths:
            if not p.exists():
                print(f"Error: Metrics file not found: {p}", file=sys.stderr)
                sys.exit(1)

        # Compute statistics
        stats = compute_stats(paths, args.last_n_rounds)
        method_stats[method] = stats

    # Format and output results
    output = format_stats_table(method_stats, args.last_n_rounds)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output)
        print(f"Statistics saved to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
