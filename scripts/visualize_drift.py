#!/usr/bin/env python3
"""Visualize drift heatmap from training logs.

This script parses training logs and creates a heatmap showing drift scores
(Δ_e) for each domain across rounds.
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_drift_from_log(log_file):
    """Parse drift scores from training log.

    Args:
        log_file: Path to training log file

    Returns:
        Tuple of (rounds, domains, drift_matrix)
    """
    rounds = []
    drift_data = []
    domains_set = set()

    with open(log_file, 'r') as f:
        for line in f:
            # Look for drift score lines
            # Expected format: "Domain drift scores (Δ_e): {'clipart': 0.123, 'real': 0.456, ...}"
            if 'Domain drift scores' in line or 'drift_map' in line:
                # Extract dictionary part
                match = re.search(r"\{[^}]+\}", line)
                if match:
                    dict_str = match.group(0)
                    try:
                        # Convert string dict to actual dict
                        drift_dict = eval(dict_str)

                        # Extract round number from previous lines if available
                        # This is a simple approach - could be improved
                        drift_data.append(drift_dict)
                        domains_set.update(drift_dict.keys())
                    except:
                        continue

    if not drift_data:
        print("Warning: No drift data found in log file")
        return None, None, None

    # Sort domains for consistent ordering
    domains = sorted(list(domains_set))

    # Build matrix
    drift_matrix = np.zeros((len(drift_data), len(domains)))

    for i, drift_dict in enumerate(drift_data):
        rounds.append(i + 1)  # Round numbering starts at 1
        for j, domain in enumerate(domains):
            drift_matrix[i, j] = drift_dict.get(domain, 0.0)

    return rounds, domains, drift_matrix


def plot_drift_heatmap(rounds, domains, drift_matrix, output_path=None):
    """Plot drift heatmap.

    Args:
        rounds: List of round numbers
        domains: List of domain names
        drift_matrix: Matrix of drift scores (rounds × domains)
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(drift_matrix.T, aspect='auto', cmap='YlOrRd', origin='lower')

    # Set ticks
    ax.set_xticks(range(0, len(rounds), max(1, len(rounds) // 20)))
    ax.set_xticklabels([rounds[i] for i in range(0, len(rounds), max(1, len(rounds) // 20))])
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains)

    # Labels
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Domain', fontsize=12)
    ax.set_title('Domain Drift Scores (Δ_e) Over Training', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Drift Score (Δ_e)', rotation=270, labelpad=20, fontsize=12)

    # Grid
    ax.grid(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved drift heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize domain drift scores from training logs')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to training log file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for heatmap (default: show plot)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figure (default: 300)')

    args = parser.parse_args()

    # Check if log file exists
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log}")
        return

    # Parse drift data
    print(f"Parsing drift data from {args.log}...")
    rounds, domains, drift_matrix = parse_drift_from_log(args.log)

    if rounds is None:
        print("Error: Could not parse drift data from log file")
        return

    print(f"Found drift data for {len(rounds)} selection rounds across {len(domains)} domains")

    # Plot heatmap
    print("Generating drift heatmap...")
    plot_drift_heatmap(rounds, domains, drift_matrix, args.output)

    # Print summary statistics
    print("\nDrift Statistics:")
    print(f"  Mean drift: {drift_matrix.mean():.4f}")
    print(f"  Max drift: {drift_matrix.max():.4f}")
    print(f"  Min drift: {drift_matrix.min():.4f}")

    print("\nPer-domain average drift:")
    for i, domain in enumerate(domains):
        avg_drift = drift_matrix[:, i].mean()
        print(f"  {domain}: {avg_drift:.4f}")


if __name__ == '__main__':
    main()
