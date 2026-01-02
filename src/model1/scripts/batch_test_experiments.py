#!/usr/bin/env python3
"""
Batch test script for testing multiple experiments with their best checkpoints.
Usage: python batch_test_experiments.py --data_config config.json
"""

import os
import json
import argparse
from test_best_model import test_model


def batch_test_experiments(outputs_dir: str, data_config: dict, max_samples: int = None):
    """Test all experiments in the outputs directory."""

    if not os.path.exists(outputs_dir):
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return

    # Find all experiment directories
    experiments = []
    for item in os.listdir(outputs_dir):
        exp_path = os.path.join(outputs_dir, item)
        if os.path.isdir(exp_path):
            checkpoint_dir = os.path.join(exp_path, 'checkpoints')
            best_checkpoint = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            if os.path.exists(best_checkpoint):
                experiments.append(exp_path)

    if not experiments:
        print(f"❌ No experiments with best checkpoints found in {outputs_dir}")
        return

    print(f"🔍 Found {len(experiments)} experiments to test:")
    for exp in experiments:
        print(f"  📁 {os.path.basename(exp)}")
    print("")

    # Test each experiment
    results_summary = []

    for i, experiment_dir in enumerate(experiments, 1):
        exp_name = os.path.basename(experiment_dir)
        print(f"\n{'='*80}")
        print(f"🧪 Testing experiment {i}/{len(experiments)}: {exp_name}")
        print(f"{'='*80}")

        try:
            results = test_model(experiment_dir, data_config, max_samples)

            # Add to summary
            summary_item = {
                'experiment_name': exp_name,
                'experiment_path': experiment_dir,
                'status': 'success',
                'best_epoch': results['checkpoint_info']['best_epoch'],
                'trained_epochs': results['checkpoint_info']['epoch'],
                'best_val_metric': results['checkpoint_info']['best_metric'],
                'primary_metric': results['checkpoint_info']['primary_metric'],
                'test_metrics': results['test_metrics']
            }
            results_summary.append(summary_item)

            print(f"✅ {exp_name}: Test completed successfully")

        except Exception as e:
            print(f"❌ {exp_name}: Test failed - {str(e)}")

            summary_item = {'experiment_name': exp_name, 'experiment_path': experiment_dir, 'status': 'failed', 'error': str(e)}
            results_summary.append(summary_item)

    # Save batch results summary
    summary_path = os.path.join(outputs_dir, 'batch_test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print("📊 BATCH TEST SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results_summary if r['status'] == 'success']
    failed = [r for r in results_summary if r['status'] == 'failed']

    print(f"✅ Successful: {len(successful)}/{len(experiments)}")
    print(f"❌ Failed: {len(failed)}/{len(experiments)}")

    if successful:
        print(f"\n🏆 Top performers by test metrics:")

        # Sort by different metrics if available
        test_metrics_keys = set()
        for result in successful:
            test_metrics_keys.update(result['test_metrics'].keys())

        for metric in sorted(test_metrics_keys):
            print(f"\n📈 Best {metric}:")
            metric_results = [(r['experiment_name'], r['test_metrics'].get(metric, 0)) for r in successful if metric in r['test_metrics']]
            metric_results.sort(key=lambda x: x[1], reverse=True)

            for i, (name, score) in enumerate(metric_results[:3]):
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "📊"
                print(f"  {medal} {name}: {score:.4f}")

    if failed:
        print(f"\n❌ Failed experiments:")
        for result in failed:
            print(f"  • {result['experiment_name']}: {result['error']}")

    print(f"\n📁 Summary saved to: {summary_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Batch test all experiments')
    parser.add_argument('--outputs_dir', type=str, default='outputs', help='Directory containing experiment folders (default: outputs)')
    parser.add_argument('--data_config', type=str, required=True, help='JSON file with data paths (features_path, image_ids_path, annotations_path)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of test samples per experiment (default: all)')

    args = parser.parse_args()

    # Load data configuration
    if not os.path.exists(args.data_config):
        raise FileNotFoundError(f"Data config file not found: {args.data_config}")

    with open(args.data_config, 'r') as f:
        data_config = json.load(f)

    # Validate required keys
    required_keys = ['features_path', 'image_ids_path', 'annotations_path']
    for key in required_keys:
        if key not in data_config:
            raise ValueError(f"Missing required key in data config: {key}")

    print(f"📁 Data config loaded from: {args.data_config}")
    print(f"📊 Testing experiments in: {args.outputs_dir}")
    if args.max_samples:
        print(f"🔢 Max samples per test: {args.max_samples}")

    # Run batch test
    batch_test_experiments(args.outputs_dir, data_config, args.max_samples)


if __name__ == "__main__":
    main()
