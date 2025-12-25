"""Baseline comparison script.

Compares the main method (FedRep+LoRA with local adapters) against
the baseline where adapters are aggregated globally with backbone.
"""

import json
import numpy as np
from pathlib import Path


def load_metrics_json(json_path: str, last_n: int = 20) -> dict:
    """Load metrics from JSON file and extract last N rounds.

    Args:
        json_path: Path to metrics.json file
        last_n: Number of last rounds to extract

    Returns:
        Dictionary with lists of avg_acc, worst_acc, variance
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    return {
        'avg_acc': data['avg_acc'][-last_n:],
        'worst_acc': data['worst_acc'][-last_n:],
        'variance': data['variance'][-last_n:]
    }


def compute_summary(metrics: dict) -> dict:
    """Compute summary statistics from metrics.

    Args:
        metrics: Dictionary with lists of metrics

    Returns:
        Dictionary with mean and std values
    """
    avg_arr = np.array(metrics['avg_acc'])
    worst_arr = np.array(metrics['worst_acc'])
    var_arr = np.array(metrics['variance'])

    return {
        'avg_acc_mean': np.mean(avg_arr),
        'avg_acc_std': np.std(avg_arr),
        'worst_acc_mean': np.mean(worst_arr),
        'worst_acc_std': np.std(worst_arr),
        'variance_mean': np.mean(var_arr),
        'variance_std': np.std(var_arr),
    }


def main():
    base_dir = Path('/root/xlw_exp/outputs')

    experiments = {
        '主方法 (本地化适配器)': base_dir / '2025-12-06-12-55-pacs_cnn5_fedrep_lora' / 'metrics.json',
        '全局聚合适配器': base_dir / '2025-12-05-17-59-pacs_cnn5_baseline' / 'metrics.json',
    }

    last_n = 20
    results = {}

    # Load and compute summaries
    for name, json_path in experiments.items():
        if json_path.exists():
            metrics = load_metrics_json(str(json_path), last_n)
            results[name] = compute_summary(metrics)
        else:
            print(f"Warning: File not found: {json_path}")

    # Print summary table
    print("=" * 90)
    print("域适配器本地化 vs 全局聚合 对比 (最后20轮)")
    print("=" * 90)
    print()
    print(f"{'方法':<25} {'Avg Acc (%)':<20} {'Worst Acc (%)':<20} {'Variance':<20}")
    print("-" * 90)

    for name, s in results.items():
        avg_str = f"{s['avg_acc_mean']:.2f} ± {s['avg_acc_std']:.2f}"
        worst_str = f"{s['worst_acc_mean']:.2f} ± {s['worst_acc_std']:.2f}"
        var_str = f"{s['variance_mean']:.2f} ± {s['variance_std']:.2f}"
        print(f"{name:<25} {avg_str:<20} {worst_str:<20} {var_str:<20}")

    # Print difference analysis
    print()
    print("=" * 90)
    print("差距分析 (相对于主方法)")
    print("=" * 90)
    print(f"{'对比方法':<25} {'Avg Acc Δ':<20} {'Worst Acc Δ':<20} {'Variance Δ':<20}")
    print("-" * 90)

    main_result = results.get('主方法 (本地化适配器)')
    if main_result:
        for name, s in results.items():
            if name != '主方法 (本地化适配器)':
                avg_delta = s['avg_acc_mean'] - main_result['avg_acc_mean']
                worst_delta = s['worst_acc_mean'] - main_result['worst_acc_mean']
                var_delta = s['variance_mean'] - main_result['variance_mean']
                print(f"{name:<25} {avg_delta:+.2f}%{'':<13} {worst_delta:+.2f}%{'':<13} {var_delta:+.2f}")

    print()
    print("=" * 90)


if __name__ == '__main__':
    main()
