#!/usr/bin/env python3
"""
Test script using the best checkpoint for final evaluation.
Usage: python test_best_model.py --experiment_dir outputs/experiment_name
"""

import torch
import os
import json
import argparse
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.model import ImageTextModel
from lib.dataset import create_dataloaders
from lib.eval import CaptionEvaluator
from lib.logger import TrainingLogger


def load_best_checkpoint(experiment_dir: str, device: torch.device):
    """Load the best checkpoint from experiment directory."""
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')

    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")

    print(f"Loading best checkpoint from: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)

    return checkpoint


def create_model_from_checkpoint(checkpoint: dict, device: torch.device) -> ImageTextModel:
    """Create and load model from checkpoint."""
    model_config = checkpoint['model_config']

    # Create model
    model = ImageTextModel(clip_dim=model_config['clip_dim'],
                           hidden_dim=model_config['hidden_dim'],
                           t5_model_name=model_config['t5_model_name'],
                           aggregation_method=model_config['aggregation_method'],
                           tokens_per_image=model_config.get('tokens_per_image', 4),
                           attention_layers=model_config.get('attention_layers', 1),
                           attention_heads=model_config.get('attention_heads', 4),
                           use_visual_mlp=model_config.get('use_visual_mlp', False),
                           use_lora=model_config['use_lora'],
                           lora_r=model_config['lora_r'],
                           lora_alpha=model_config['lora_alpha'],
                           lora_dropout=model_config['lora_dropout']).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best {checkpoint.get('primary_metric', 'metric')}: {checkpoint['best_metric']:.4f} (epoch {checkpoint['best_epoch']})")

    return model


def test_model(experiment_dir: str, data_config: dict, max_samples: int = None):
    """Test the best model and save results."""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = load_best_checkpoint(experiment_dir, device)

    # Create model
    model = create_model_from_checkpoint(checkpoint, device)

    # Load config
    config_path = os.path.join(experiment_dir, 'checkpoints', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create data loaders
    print("Creating data loaders...")
    tag_cache_dir = config.get('tag_cache_dir') or '../../data/model1_data/clip_tags'
    if not os.path.isabs(tag_cache_dir):
        base_dir = os.path.dirname(config_path)
        tag_cache_dir = os.path.abspath(os.path.join(base_dir, tag_cache_dir))

    image_root = config.get('image_root')
    if image_root and not os.path.isabs(image_root):
        base_dir = os.path.dirname(config_path)
        image_root = os.path.abspath(os.path.join(base_dir, image_root))

    tag_generation_method = str(config.get('tag_generation_method', 'blip_noun_phrases')).lower()

    _, _, test_loader = create_dataloaders(features_path=data_config['features_path'],
                                           image_ids_path=data_config['image_ids_path'],
                                           annotations_path=data_config['annotations_path'],
                                           batch_size=config['batch_size'],
                                           num_workers=config['num_workers'],
                                           tokenizer_name=config['t5_model_name'],
                                           max_length=config['max_length'],
                                           split_ratio=config['split_ratio'],
                                           use_tags=config.get('use_tags', False),
                                           tag_top_k=config.get('tag_top_k', 4),
                                           max_tags_per_sample=config.get('max_tags_per_sample', 12),
                                           clip_text_model_name=config.get('clip_text_model_name', 'openai/clip-vit-base-patch32'),
                                           tag_cache_dir=tag_cache_dir,
                                           tag_generation_method=tag_generation_method,
                                           blip_model_name=config.get('blip_model_name', 'Salesforce/blip-image-captioning-base'),
                                           image_root=image_root,
                                           tag_device=config.get('tag_device'))

    # Create evaluator
    evaluator = CaptionEvaluator()

    # Create test results directory
    test_results_dir = os.path.join(experiment_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)

    # Generate timestamp for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Test the model
    print(f"Running test evaluation on {len(test_loader.dataset)} samples...")
    samples_path = os.path.join(test_results_dir, f'best_model_test_samples_{timestamp}.json')

    test_metrics, test_samples = evaluator.evaluate_dataset(model=model, dataloader=test_loader, device=device, save_samples=True, samples_path=samples_path, max_samples=max_samples)

    # Save test results
    results = {
        'timestamp': timestamp,
        'experiment_dir': experiment_dir,
        'checkpoint_info': {
            'epoch': checkpoint['epoch'],
            'best_epoch': checkpoint['best_epoch'],
            'best_metric': checkpoint['best_metric'],
            'primary_metric': checkpoint.get('primary_metric', 'unknown')
        },
        'test_metrics': test_metrics,
        'model_config': checkpoint['model_config'],
        'data_config': data_config,
        'test_samples_file': samples_path
    }

    # Save results to JSON
    results_path = os.path.join(test_results_dir, f'best_model_test_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("🎯 BEST MODEL TEST RESULTS")
    print("=" * 60)
    print(f"📁 Experiment: {os.path.basename(experiment_dir)}")
    print(f"📊 Best checkpoint from epoch {checkpoint['best_epoch']} (trained to epoch {checkpoint['epoch']})")
    print(f"🏆 Best validation {checkpoint.get('primary_metric', 'metric')}: {checkpoint['best_metric']:.4f}")
    print("")
    print("📈 Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("")
    print("📁 Results saved to:")
    print(f"  📊 Metrics: {results_path}")
    print(f"  📝 Samples: {samples_path}")
    print("=" * 60)

    return results


def find_latest_experiment(outputs_dir: str = "outputs") -> str:
    """Find the most recent experiment directory."""
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    experiments = []
    for item in os.listdir(outputs_dir):
        exp_path = os.path.join(outputs_dir, item)
        if os.path.isdir(exp_path) and os.path.exists(os.path.join(exp_path, 'checkpoints')):
            experiments.append((exp_path, os.path.getmtime(exp_path)))

    if not experiments:
        raise FileNotFoundError(f"No experiment directories found in {outputs_dir}")

    # Return the most recent experiment
    latest_exp = sorted(experiments, key=lambda x: x[1], reverse=True)[0][0]
    return latest_exp


def main():
    parser = argparse.ArgumentParser(description='Test best model checkpoint')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Path to experiment directory (default: latest in outputs/)')
    parser.add_argument('--features_path', type=str, required=True, help='Path to image_features.npy')
    parser.add_argument('--image_ids_path', type=str, required=True, help='Path to image_ids.json')
    parser.add_argument('--annotations_path', type=str, required=True, help='Path to image_groups_with_1caption.json')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of test samples (default: all)')
    parser.add_argument('--outputs_dir', type=str, default='outputs', help='Outputs directory to search for experiments (default: outputs)')

    args = parser.parse_args()

    # Determine experiment directory
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    else:
        print("No experiment directory specified, finding latest...")
        experiment_dir = find_latest_experiment(args.outputs_dir)
        print(f"Using latest experiment: {experiment_dir}")

    # Prepare data configuration
    data_config = {'features_path': args.features_path, 'image_ids_path': args.image_ids_path, 'annotations_path': args.annotations_path}

    # Run test
    try:
        results = test_model(experiment_dir, data_config, args.max_samples)
        print(f"\n✅ Testing completed successfully!")

    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
