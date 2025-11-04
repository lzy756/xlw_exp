#!/usr/bin/env python3
"""Command-line tool for managing experiments."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import yaml


def find_experiments(output_dir: str = './outputs') -> List[Dict]:
    """Find all experiments in the output directory.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        List of experiment info dictionaries
    """
    experiments = []
    
    if not os.path.exists(output_dir):
        return experiments
    
    for entry in os.listdir(output_dir):
        exp_path = os.path.join(output_dir, entry)
        
        # Skip if not a directory or is the 'latest' symlink
        if not os.path.isdir(exp_path) or entry == 'latest':
            continue
        
        # Try to load experiment info
        info_file = os.path.join(exp_path, 'experiment_info.json')
        config_file = os.path.join(exp_path, 'config_effective.yaml')
        
        exp_info = {
            'name': entry,
            'path': exp_path,
            'has_info': os.path.exists(info_file),
            'has_config': os.path.exists(config_file),
        }
        
        if exp_info['has_info']:
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    exp_info.update(info)
            except (json.JSONDecodeError, IOError):
                pass
        
        experiments.append(exp_info)
    
    # Sort by start time (newest first)
    experiments.sort(
        key=lambda x: x.get('start_time', '0000-00-00 00:00:00'),
        reverse=True
    )
    
    return experiments


def list_experiments(args):
    """List all experiments."""
    experiments = find_experiments(args.output_dir)
    
    if not experiments:
        print(f"No experiments found in {args.output_dir}")
        return
    
    print(f"Found {len(experiments)} experiments:\n")
    print(f"{'Name':<25} {'Status':<12} {'Start Time':<20} {'Duration':<12}")
    print("=" * 80)
    
    for exp in experiments:
        name = exp['name']
        status = exp.get('status', 'unknown')
        start_time = exp.get('start_time', 'N/A')
        
        # Format duration
        if 'duration_seconds' in exp:
            duration_sec = exp['duration_seconds']
            hours = duration_sec // 3600
            minutes = (duration_sec % 3600) // 60
            duration = f"{hours}h {minutes}m"
        else:
            duration = 'N/A'
        
        # Color code status
        status_colored = status
        if args.color:
            if status == 'completed':
                status_colored = f"\033[92m{status}\033[0m"  # Green
            elif status == 'failed':
                status_colored = f"\033[91m{status}\033[0m"  # Red
            elif status == 'running':
                status_colored = f"\033[93m{status}\033[0m"  # Yellow
        
        print(f"{name:<25} {status_colored:<12} {start_time:<20} {duration:<12}")
    
    # Show summary
    completed = sum(1 for e in experiments if e.get('status') == 'completed')
    failed = sum(1 for e in experiments if e.get('status') == 'failed')
    running = sum(1 for e in experiments if e.get('status') == 'running')
    
    print("\n" + "=" * 80)
    print(f"Summary: {completed} completed, {failed} failed, {running} running")


def show_experiment(args):
    """Show detailed information about a specific experiment."""
    exp_path = os.path.join(args.output_dir, args.name)
    
    if not os.path.exists(exp_path):
        print(f"Error: Experiment '{args.name}' not found in {args.output_dir}")
        return
    
    info_file = os.path.join(exp_path, 'experiment_info.json')
    config_file = os.path.join(exp_path, 'config_effective.yaml')
    
    print(f"Experiment: {args.name}")
    print(f"Path: {exp_path}")
    print("=" * 80)
    
    # Show experiment info
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print("\nExperiment Information:")
        print(f"  Status: {info.get('status', 'unknown')}")
        print(f"  Start: {info.get('start_time', 'N/A')}")
        print(f"  End: {info.get('end_time', 'N/A')}")
        
        if 'duration_seconds' in info:
            duration_sec = info['duration_seconds']
            hours = duration_sec // 3600
            minutes = (duration_sec % 3600) // 60
            seconds = duration_sec % 60
            print(f"  Duration: {hours}h {minutes}m {seconds}s")
        
        print(f"\nCommand:")
        print(f"  {info.get('command', 'N/A')}")
        
        if 'git_info' in info:
            git = info['git_info']
            print(f"\nGit Information:")
            print(f"  Branch: {git.get('branch', 'unknown')}")
            print(f"  Commit: {git.get('commit_hash', 'unknown')}")
            print(f"  Dirty: {git.get('is_dirty', False)}")
        
        if 'environment' in info:
            env = info['environment']
            print(f"\nEnvironment:")
            print(f"  Python: {env.get('python_version', 'unknown')}")
            print(f"  PyTorch: {env.get('pytorch_version', 'unknown')}")
            print(f"  CUDA: {env.get('cuda_version', 'N/A') if env.get('cuda_available') else 'Not available'}")
            print(f"  Hostname: {env.get('hostname', 'unknown')}")
    
    # Show configuration summary
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\nConfiguration Summary:")
        if 'data' in config:
            print(f"  Domains: {config['data'].get('domains', [])}")
        if 'training' in config:
            print(f"  Rounds: {config['training'].get('total_rounds', 'N/A')}")
            print(f"  Batch size: {config['training'].get('batch_size', 'N/A')}")
        if 'model' in config and 'lora' in config['model']:
            print(f"  LoRA rank: {config['model']['lora'].get('rank', 'N/A')}")
    
    # Show metrics summary if available
    metrics_file = os.path.join(exp_path, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nFinal Metrics:")
        if 'avg_acc' in metrics and metrics['avg_acc']:
            print(f"  Average accuracy: {metrics['avg_acc'][-1]:.2f}%")
        if 'worst_acc' in metrics and metrics['worst_acc']:
            print(f"  Worst accuracy: {metrics['worst_acc'][-1]:.2f}%")
        if 'variance' in metrics and metrics['variance']:
            print(f"  Variance: {metrics['variance'][-1]:.4f}")


def diff_experiments(args):
    """Compare configurations of two experiments."""
    exp1_path = os.path.join(args.output_dir, args.exp1)
    exp2_path = os.path.join(args.output_dir, args.exp2)
    
    if not os.path.exists(exp1_path):
        print(f"Error: Experiment '{args.exp1}' not found")
        return
    if not os.path.exists(exp2_path):
        print(f"Error: Experiment '{args.exp2}' not found")
        return
    
    config1_file = os.path.join(exp1_path, 'config_effective.yaml')
    config2_file = os.path.join(exp2_path, 'config_effective.yaml')
    
    if not os.path.exists(config1_file):
        print(f"Error: Config not found for '{args.exp1}'")
        return
    if not os.path.exists(config2_file):
        print(f"Error: Config not found for '{args.exp2}'")
        return
    
    with open(config1_file, 'r') as f:
        config1 = yaml.safe_load(f)
    with open(config2_file, 'r') as f:
        config2 = yaml.safe_load(f)
    
    print(f"Comparing configurations:")
    print(f"  {args.exp1} vs {args.exp2}\n")
    
    # Find differences
    diffs = _find_config_diffs(config1, config2)
    
    if not diffs:
        print("No differences found in configurations.")
        return
    
    print(f"Found {len(diffs)} differences:\n")
    for path, val1, val2 in diffs:
        print(f"  {path}:")
        print(f"    {args.exp1}: {val1}")
        print(f"    {args.exp2}: {val2}")


def _find_config_diffs(dict1, dict2, path=''):
    """Recursively find differences between two dictionaries."""
    diffs = []
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        
        if key not in dict1:
            diffs.append((current_path, '<missing>', dict2[key]))
        elif key not in dict2:
            diffs.append((current_path, dict1[key], '<missing>'))
        else:
            val1, val2 = dict1[key], dict2[key]
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                diffs.extend(_find_config_diffs(val1, val2, current_path))
            elif val1 != val2:
                diffs.append((current_path, val1, val2))
    
    return diffs


def clean_experiments(args):
    """Clean up failed or incomplete experiments."""
    experiments = find_experiments(args.output_dir)
    
    to_delete = []
    
    # Filter experiments to clean
    for exp in experiments:
        should_delete = False
        
        if args.failed and exp.get('status') == 'failed':
            should_delete = True
        if args.incomplete and not exp.get('has_info'):
            should_delete = True
        if args.all:
            should_delete = True
        
        if should_delete:
            to_delete.append(exp)
    
    if not to_delete:
        print("No experiments to clean.")
        return
    
    print(f"Found {len(to_delete)} experiments to delete:")
    for exp in to_delete:
        status = exp.get('status', 'unknown')
        print(f"  - {exp['name']} ({status})")
    
    if not args.yes:
        response = input("\nProceed with deletion? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Delete experiments
    import shutil
    deleted = 0
    for exp in to_delete:
        try:
            shutil.rmtree(exp['path'])
            deleted += 1
            print(f"Deleted: {exp['name']}")
        except Exception as e:
            print(f"Error deleting {exp['name']}: {e}")
    
    print(f"\nDeleted {deleted} experiments.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Manage FL-DomainNet experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory containing experiments'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.add_argument(
        '--color',
        action='store_true',
        help='Use colored output'
    )
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('name', help='Experiment name')
    
    # Diff command
    diff_parser = subparsers.add_parser('diff', help='Compare two experiments')
    diff_parser.add_argument('exp1', help='First experiment name')
    diff_parser.add_argument('exp2', help='Second experiment name')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up experiments')
    clean_parser.add_argument(
        '--failed',
        action='store_true',
        help='Delete failed experiments'
    )
    clean_parser.add_argument(
        '--incomplete',
        action='store_true',
        help='Delete incomplete experiments (missing metadata)'
    )
    clean_parser.add_argument(
        '--all',
        action='store_true',
        help='Delete all experiments'
    )
    clean_parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'list':
        list_experiments(args)
    elif args.command == 'show':
        show_experiment(args)
    elif args.command == 'diff':
        diff_experiments(args)
    elif args.command == 'clean':
        clean_experiments(args)


if __name__ == '__main__':
    main()
