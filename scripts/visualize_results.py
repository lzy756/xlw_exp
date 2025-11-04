#!/usr/bin/env python3
"""Visualize federated learning experiment results.

This script creates visualizations including:
1. Aggregator selection timeline heatmap
2. Worst-domain accuracy curve
3. Domain drift heatmap
"""

import argparse
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_experiment_log(log_file):
    """Parse experiment log for metrics.

    Args:
        log_file: Path to log file

    Returns:
        Dictionary with parsed metrics
    """
    data = {
        'rounds': [],
        'avg_acc': [],
        'worst_acc': [],
        'variance': [],
        'per_domain_acc': {},
        'selected_aggregators': [],
        'drift_scores': [],
        'domains': []
    }

    with open(log_file, 'r') as f:
        current_round = None

        for line in f:
            # Extract round number
            round_match = re.search(r'Round (\d+)/', line)
            if round_match:
                current_round = int(round_match.group(1))

            # Extract metrics
            if 'Average accuracy:' in line:
                acc_match = re.search(r'Average accuracy: ([\d.]+)%', line)
                if acc_match and current_round:
                    if current_round not in data['rounds']:
                        data['rounds'].append(current_round)
                    data['avg_acc'].append(float(acc_match.group(1)))

            if 'Worst accuracy:' in line:
                acc_match = re.search(r'Worst accuracy: ([\d.]+)%', line)
                if acc_match:
                    data['worst_acc'].append(float(acc_match.group(1)))

            if 'Variance:' in line:
                var_match = re.search(r'Variance: ([\d.]+)', line)
                if var_match:
                    data['variance'].append(float(var_match.group(1)))

            # Extract per-domain accuracy
            domain_acc_match = re.search(r'Domain (\w+): val_acc=([\d.]+)%', line)
            if domain_acc_match:
                domain = domain_acc_match.group(1)
                acc = float(domain_acc_match.group(2))
                if domain not in data['per_domain_acc']:
                    data['per_domain_acc'][domain] = []
                    if domain not in data['domains']:
                        data['domains'].append(domain)
                data['per_domain_acc'][domain].append(acc)

            # Extract aggregator selection
            if 'Selected aggregator:' in line:
                agg_match = re.search(r'Selected aggregator: (\w+)', line)
                if agg_match:
                    data['selected_aggregators'].append(agg_match.group(1))

            # Extract drift scores
            if 'Domain drift scores' in line:
                dict_match = re.search(r"\{[^}]+\}", line)
                if dict_match:
                    try:
                        drift_dict = eval(dict_match.group(0))
                        data['drift_scores'].append(drift_dict)
                    except:
                        pass

    return data


def plot_aggregator_timeline(data, output_path=None):
    """Plot aggregator selection timeline heatmap.

    Args:
        data: Parsed experiment data
        output_path: Optional path to save figure
    """
    if not data['selected_aggregators'] or not data['domains']:
        print("No aggregator selection data found")
        return

    # Create matrix: rows are domains, columns are selection rounds
    domains = sorted(data['domains'])
    n_selections = len(data['selected_aggregators'])

    matrix = np.zeros((len(domains), n_selections))

    for i, selected in enumerate(data['selected_aggregators']):
        if selected in domains:
            domain_idx = domains.index(selected)
            matrix[domain_idx, i] = 1

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='Blues', origin='lower')

    # Set ticks
    ax.set_xticks(range(n_selections))
    ax.set_xticklabels(range(1, n_selections + 1))
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains)

    # Labels
    ax.set_xlabel('Selection Round', fontsize=12)
    ax.set_ylabel('Domain', fontsize=12)
    ax.set_title('Aggregator Selection Timeline', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Not Selected', 'Selected'])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregator timeline to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_worst_domain_curve(data, output_path=None):
    """Plot worst-domain accuracy over rounds.

    Args:
        data: Parsed experiment data
        output_path: Optional path to save figure
    """
    if not data['worst_acc'] or not data['rounds']:
        print("No worst accuracy data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot curves
    ax.plot(data['rounds'], data['worst_acc'], 'r-', linewidth=2, label='Worst Domain')
    if data['avg_acc']:
        ax.plot(data['rounds'], data['avg_acc'], 'b--', linewidth=2, label='Average')

    # Labels
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Worst-Domain Accuracy Trajectory', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved worst-domain curve to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_drift_heatmap(data, output_path=None):
    """Plot domain drift heatmap.

    Args:
        data: Parsed experiment data
        output_path: Optional path to save figure
    """
    if not data['drift_scores'] or not data['domains']:
        print("No drift data found")
        return

    domains = sorted(data['domains'])

    # Build matrix
    drift_matrix = np.zeros((len(data['drift_scores']), len(domains)))

    for i, drift_dict in enumerate(data['drift_scores']):
        for j, domain in enumerate(domains):
            drift_matrix[i, j] = drift_dict.get(domain, 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create heatmap
    im = ax.imshow(drift_matrix.T, aspect='auto', cmap='YlOrRd', origin='lower')

    # Set ticks
    ax.set_xticks(range(0, len(data['drift_scores']), max(1, len(data['drift_scores']) // 20)))
    ax.set_xticklabels([i + 1 for i in range(0, len(data['drift_scores']), max(1, len(data['drift_scores']) // 20))])
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains)

    # Labels
    ax.set_xlabel('Selection Round', fontsize=12)
    ax.set_ylabel('Domain', fontsize=12)
    ax.set_title('Domain Drift Scores (Δ_e)', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Drift Score', rotation=270, labelpad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved drift heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize FL experiment results')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to experiment log file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: current directory)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots instead of saving')

    args = parser.parse_args()

    # Check log file
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log}")
        return

    # Parse data
    print(f"Parsing experiment log from {args.log}...")
    data = parse_experiment_log(args.log)

    print(f"Found {len(data['rounds'])} rounds")
    print(f"Found {len(data['domains'])} domains: {', '.join(data['domains'])}")
    print(f"Found {len(data['selected_aggregators'])} aggregator selections")

    # Prepare output paths
    if args.output_dir and not args.show:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregator_path = output_dir / 'aggregator_timeline.png'
        worst_path = output_dir / 'worst_domain_curve.png'
        drift_path = output_dir / 'drift_heatmap.png'
    else:
        aggregator_path = worst_path = drift_path = None

    # Generate plots
    print("\nGenerating visualizations...")

    print("1. Aggregator selection timeline...")
    plot_aggregator_timeline(data, aggregator_path)

    print("2. Worst-domain accuracy curve...")
    plot_worst_domain_curve(data, worst_path)

    print("3. Domain drift heatmap...")
    plot_drift_heatmap(data, drift_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
