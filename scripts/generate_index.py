#!/usr/bin/env python3
"""Generate index.json from DomainNet txt files.

This script reads the {domain}_train.txt and {domain}_test.txt files
from the DomainNet dataset and creates a unified index.json file
that can be used by the DomainNetDataset class.

Usage:
    python scripts/generate_index.py [--max-classes 126] [--output /path/to/index.json]
"""

import json
import os
import argparse
from pathlib import Path
from collections import Counter


def generate_index(
    root_dir: str,
    output_path: str,
    max_classes: int = None,
    domains: list = None
):
    """Generate index.json from {domain}_train.txt files.

    Args:
        root_dir: Root directory containing domainnet data
        output_path: Path to save index.json
        max_classes: Optional maximum number of classes to include (for subset experiments)
        domains: List of domains to include (default: all 6 domains)
    """
    if domains is None:
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    print(f"Generating index from: {root_dir}")
    print(f"Domains: {domains}")
    if max_classes:
        print(f"Filtering to first {max_classes} classes (labels 0-{max_classes-1})")

    samples = []
    all_labels = []
    domain_stats = {}
    skipped_samples = 0

    for domain in domains:
        domain_samples = 0
        for split in ['train', 'test']:
            txt_file = os.path.join(root_dir, f'{domain}_{split}.txt')

            if not os.path.exists(txt_file):
                print(f"Warning: {txt_file} not found, skipping")
                continue

            print(f"Reading {domain}_{split}.txt...")

            with open(txt_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 2:
                        print(f"Warning: Invalid line format at {txt_file}:{line_num}")
                        continue

                    path, label_str = parts
                    try:
                        label = int(label_str)
                    except ValueError:
                        print(f"Warning: Invalid label '{label_str}' at {txt_file}:{line_num}")
                        continue

                    # Filter by max_classes if specified
                    if max_classes is not None and label >= max_classes:
                        skipped_samples += 1
                        continue

                    samples.append({
                        'path': path,
                        'label': label,
                        'domain': domain,
                        'split': split
                    })

                    all_labels.append(label)
                    domain_samples += 1

        domain_stats[domain] = domain_samples
        print(f"  {domain}: {domain_samples} samples")

    # Calculate statistics
    unique_labels = sorted(set(all_labels))
    actual_num_classes = len(unique_labels)
    label_distribution = Counter(all_labels)

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Skipped samples (label >= {max_classes}): {skipped_samples}" if max_classes else "")
    print(f"  Unique labels found: {actual_num_classes}")
    print(f"  Label range: {min(unique_labels)} to {max(unique_labels)}")
    print(f"  Samples per class (min/avg/max): {min(label_distribution.values())}/{len(samples)//actual_num_classes}/{max(label_distribution.values())}")

    # Verify label continuity
    expected_labels = set(range(max(unique_labels) + 1))
    missing_labels = expected_labels - set(unique_labels)
    if missing_labels:
        print(f"  Warning: Missing labels: {sorted(missing_labels)[:10]}{'...' if len(missing_labels) > 10 else ''}")

    # Create index structure
    index = {
        'domains': domains,
        'num_classes': max_classes if max_classes else actual_num_classes,
        'actual_num_classes': actual_num_classes,
        'label_range': {
            'min': min(unique_labels),
            'max': max(unique_labels)
        },
        'domain_stats': domain_stats,
        'total_samples': len(samples),
        'samples': samples
    }

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n✓ Successfully generated index.json")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Print warning if using subset
    if max_classes and max_classes < actual_num_classes:
        print(f"\n⚠️  WARNING: You specified max_classes={max_classes} but dataset has {actual_num_classes} classes")
        print(f"   Filtered out {skipped_samples} samples with labels >= {max_classes}")
        print(f"   Make sure your config also sets num_classes={max_classes}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate index.json from DomainNet txt files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full dataset (345 classes)
  python scripts/generate_index.py

  # Generate subset with first 126 classes only
  python scripts/generate_index.py --max-classes 126

  # Generate for specific domains only
  python scripts/generate_index.py --domains clipart real sketch

  # Custom output path
  python scripts/generate_index.py --output /custom/path/index.json
        """
    )

    parser.add_argument(
        '--root',
        type=str,
        default='/root/domainnet',
        help='Root directory containing DomainNet data (default: /root/domainnet)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for index.json (default: <root>/index.json)'
    )

    parser.add_argument(
        '--max-classes',
        type=int,
        default=None,
        help='Maximum number of classes to include (e.g., 126 for subset). '
             'Samples with label >= max_classes will be filtered out.'
    )

    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=None,
        help='Domains to include (default: all 6 domains)'
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = os.path.join(args.root, 'index.json')

    # Verify root directory exists
    if not os.path.exists(args.root):
        print(f"Error: Root directory not found: {args.root}")
        return 1

    # Generate index
    generate_index(
        root_dir=args.root,
        output_path=args.output,
        max_classes=args.max_classes,
        domains=args.domains
    )

    return 0


if __name__ == '__main__':
    exit(main())
