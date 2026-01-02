#!/usr/bin/env python3
"""
Standalone evaluation script for trained models.
Usage: python evaluate.py --checkpoint path/to/checkpoint.pth
"""

import argparse
import os
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.eval import evaluate_model_checkpoint
import torch


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Image-to-Text model')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth file)')

    # Data paths
    parser.add_argument('--features_path', type=str, default='../../data/test_data/model1/features/image_features.npy', help='Path to image_features.npy')
    parser.add_argument('--image_ids_path', type=str, default='../../data/test_data/model1/features/image_ids.json', help='Path to image_ids.json')
    parser.add_argument('--annotations_path', type=str, default='../../data/test_data/model1/annotations/image_groups_with_1caption.json', help='Path to image_groups_with_1caption.json')

    # Evaluation parameters
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate (None for all)')
    parser.add_argument('--no_samples', action='store_true', help='Do not save individual sample results')
    parser.add_argument('--samples_dir', type=str, default='outputs/evaluation_samples', help='Directory to save sample results')

    # Device
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda/cpu, auto-detect if None)')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Check if data files exist
    for path, name in [(args.features_path, 'Features'), (args.image_ids_path, 'Image IDs'), (args.annotations_path, 'Annotations')]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("-" * 50)

    try:
        # Run evaluation
        metrics = evaluate_model_checkpoint(model_path=args.checkpoint,
                                            features_path=args.features_path,
                                            image_ids_path=args.image_ids_path,
                                            annotations_path=args.annotations_path,
                                            device=device,
                                            batch_size=args.batch_size,
                                            split_type=args.split,
                                            save_samples=not args.no_samples,
                                            samples_dir=args.samples_dir,
                                            max_samples=args.max_samples)

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
