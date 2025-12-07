#!/usr/bin/env python3
"""
Inference script for generating captions from image features.
Usage: python inference.py --checkpoint path/to/checkpoint.pth --group_ids 123 456 789
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.model import ImageTextModel
from lib.dataset import BLIPNounPhraseTagGenerator, ZeroShotTagGenerator


def load_data(features_path, image_ids_path, annotations_path):
    """Load the dataset files."""
    # Load features
    features = np.load(features_path)

    # Load image IDs
    with open(image_ids_path, 'r') as f:
        image_ids = json.load(f)

    # Load annotations
    with open(annotations_path, 'r') as f:
        groups = json.load(f)

    # Create mappings
    image_id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
    group_id_to_group = {group['group_id']: group for group in groups}

    return features, image_id_to_idx, group_id_to_group


def get_image_features_for_group(group, features, image_id_to_idx):
    """Extract features for a specific group."""
    image_indices = []
    for img_id in group['image_ids']:
        if img_id in image_id_to_idx:
            image_indices.append(image_id_to_idx[img_id])
        else:
            raise ValueError(f"Image ID {img_id} not found")

    # Extract features for the 3 images: shape (3, 512)
    image_features = features[image_indices]
    return torch.tensor(image_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


def main():
    parser = argparse.ArgumentParser(description='Generate captions using trained model')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth file)')

    # Input specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--group_ids', type=int, nargs='+', help='Group IDs to generate captions for')
    group.add_argument('--all_groups', action='store_true', help='Generate captions for all groups')

    # Data paths
    parser.add_argument('--features_path', type=str, default='../../data/test_data/model1/features/image_features.npy', help='Path to image_features.npy')
    parser.add_argument('--image_ids_path', type=str, default='../../data/test_data/model1/features/image_ids.json', help='Path to image_ids.json')
    parser.add_argument('--annotations_path', type=str, default='../../data/test_data/model1/annotations/image_groups_with_1caption.json', help='Path to image_groups_with_1caption.json')

    # Generation parameters
    parser.add_argument('--max_length', type=int, default=128, help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy/beam search')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output JSON file to save results (optional)')

    # Device
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda/cpu, auto-detect if None)')
    parser.add_argument('--image_root', type=str, default=None, help='Root directory containing raw images (required if tags use BLIP).')

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

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('model_config', {})

    model = ImageTextModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Aggregation method: {model.get_aggregation_method()}")

    # Load training config for tag settings
    config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json')
    tag_generator = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        if train_config.get('use_tags', False):
            base_dir = os.path.dirname(config_path)
            cache_dir = train_config.get('tag_cache_dir') or '../../data/model1_data/clip_tags'
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.abspath(os.path.join(base_dir, cache_dir))

            tag_method = str(train_config.get('tag_generation_method', 'blip_noun_phrases')).lower()
            tag_device = train_config.get('tag_device')

            if tag_method == 'blip_noun_phrases':
                image_root = args.image_root or train_config.get('image_root')
                if image_root is None:
                    raise ValueError("image_root must be provided (via training config or --image_root) when using BLIP-based tagging.")
                if not os.path.isabs(image_root):
                    image_root = os.path.abspath(os.path.join(base_dir, image_root))

                tag_generator = BLIPNounPhraseTagGenerator(cache_dir=cache_dir,
                                                           image_root=image_root,
                                                           blip_model_name=train_config.get('blip_model_name', 'Salesforce/blip-image-captioning-base'),
                                                           max_tags=train_config.get('max_tags_per_sample', 12),
                                                           device=tag_device)
                print(f"Semantic tag generator: BLIP noun phrases (max_tags={train_config.get('max_tags_per_sample', 12)})")
                print(f"  image_root: {image_root}")
            else:
                tag_generator = ZeroShotTagGenerator(cache_dir=cache_dir,
                                                     clip_model_name=train_config.get('clip_text_model_name', 'openai/clip-vit-base-patch32'),
                                                     top_k=train_config.get('tag_top_k', 4),
                                                     max_tags=train_config.get('max_tags_per_sample', 12),
                                                     device=tag_device)
                print(f"Semantic tag generator: CLIP zero-shot (top_k={train_config.get('tag_top_k', 4)}, max_tags={train_config.get('max_tags_per_sample', 12)})")
    else:
        print("Warning: config.json not found next to checkpoint; running without tag prompts.")

    # Load data
    print("Loading data...")
    try:
        features, image_id_to_idx, group_id_to_group = load_data(args.features_path, args.image_ids_path, args.annotations_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Determine which groups to process
    if args.all_groups:
        target_group_ids = list(group_id_to_group.keys())
        print(f"Processing all {len(target_group_ids)} groups")
    else:
        target_group_ids = args.group_ids
        # Verify all group IDs exist
        missing_ids = [gid for gid in target_group_ids if gid not in group_id_to_group]
        if missing_ids:
            print(f"Error: Group IDs not found: {missing_ids}")
            sys.exit(1)
        print(f"Processing {len(target_group_ids)} groups: {target_group_ids}")

    # Generate captions
    results = []

    print("\nGenerating captions...")
    print("-" * 50)

    with torch.no_grad():
        for i, group_id in enumerate(target_group_ids):
            group = group_id_to_group[group_id]

            # Get image features
            try:
                image_features = get_image_features_for_group(group, features, image_id_to_idx).to(device)
            except Exception as e:
                print(f"Error processing group {group_id}: {e}")
                continue

            prompt_text = None
            tags = []
            if tag_generator is not None:
                tags = tag_generator(image_features=image_features.squeeze(0).cpu().numpy(), image_ids=group['image_ids'])
                if tags:
                    prompt_text = [f"tags: {', '.join(tags)}. generate English caption: "]
            if prompt_text is None:
                prompt_text = ["generate English caption: "]

            # Generate caption
            generation_results = model.generate(image_features=image_features,
                                                max_length=args.max_length,
                                                num_beams=args.num_beams,
                                                temperature=args.temperature,
                                                do_sample=args.do_sample,
                                                prompt_text=prompt_text)

            generated_caption = generation_results['generated_texts'][0]

            # Store result
            result = {'group_id': group_id,
                      'image_ids': group['image_ids'],
                      'generated_caption': generated_caption,
                      'reference_caption': group['merged_caption'],
                      'tags': tags}
            results.append(result)

            # Print result
            print(f"Group {group_id}:")
            print(f"  Images: {', '.join(group['image_ids'])}")
            print(f"  Generated: {generated_caption}")
            print(f"  Reference: {group['merged_caption']}")
            print()

            # Progress update
            if len(target_group_ids) > 10 and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(target_group_ids)} groups...")

    print(f"Generated captions for {len(results)} groups")

    # Save results if output file specified
    if args.output:
        output_data = {
            'checkpoint': args.checkpoint,
            'generation_params': {
                'max_length': args.max_length,
                'num_beams': args.num_beams,
                'temperature': args.temperature,
                'do_sample': args.do_sample
            },
            'model_config': model_config,
            'num_results': len(results),
            'results': results
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
