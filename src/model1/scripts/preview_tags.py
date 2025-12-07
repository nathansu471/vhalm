#!/usr/bin/env python3
"""Preview CLIP zero-shot tags for a few dataset samples."""

import argparse
import os
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import ImageCaptionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview cached zero-shot tags with corresponding image IDs.")
    parser.add_argument("--features_path", default="../../data/model1_data/image_features.npy", help="Path to image_features.npy")
    parser.add_argument("--image_ids_path", default="../../data/model1_data/image_ids.json", help="Path to image_ids.json")
    parser.add_argument("--annotations_path", default="../../data/model1_data/merged_captions_async_augmented.json", help="Path to merged captions JSON")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to inspect")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to preview (ignored when --indices is given)")
    parser.add_argument("--indices", type=int, nargs="*", help="Explicit dataset indices to preview (uses --split)")
    parser.add_argument("--tag_cache_dir", default="../../data/model1_data/clip_tags", help="Directory containing the tag cache")
    parser.add_argument("--clip_text_model_name", default="openai/clip-vit-base-patch32", help="CLIP text encoder identifier")
    parser.add_argument("--blip_model_name", default="Salesforce/blip-image-captioning-base", help="BLIP model identifier")
    parser.add_argument("--image_root", default="../../data/model1_data/images", help="Root directory containing raw images for BLIP tagging")
    parser.add_argument("--tag_generation_method", default="blip_noun_phrases", choices=["clip_zero_shot", "blip_noun_phrases"], help="Tag extraction method to preview")
    parser.add_argument("--tag_device", default=None, help="Device override for tag generators (e.g. 'cuda')")
    parser.add_argument("--tag_top_k", type=int, default=4, help="Top K tags used when employing CLIP zero-shot tagging")
    parser.add_argument("--max_tags", type=int, default=12, help="Maximum tags per sample")
    return parser.parse_args()


def format_row(index: int, image_ids: List[str], tags: List[str], caption: str) -> str:
    joined_ids = ", ".join(image_ids)
    joined_tags = ", ".join(tags) if tags else "<no tags>"
    preview_caption = caption.replace("\n", " ")
    if len(preview_caption) > 120:
        preview_caption = preview_caption[:117] + "..."
    return f"[idx={index}] image_ids=[{joined_ids}]\n  tags: {joined_tags}\n  caption: {preview_caption}\n"


def main() -> None:
    args = parse_args()

    dataset = ImageCaptionDataset(
        features_path=args.features_path,
        image_ids_path=args.image_ids_path,
        annotations_path=args.annotations_path,
        tokenizer_name="t5-base",
        max_length=128,
        split_type=args.split,
        use_tags=True,
        tag_top_k=args.tag_top_k,
        max_tags_per_sample=args.max_tags,
        clip_text_model_name=args.clip_text_model_name,
        tag_cache_dir=args.tag_cache_dir,
        tag_generation_method=args.tag_generation_method,
        blip_model_name=args.blip_model_name,
        image_root=args.image_root,
        tag_device=args.tag_device,
    )

    if args.indices:
        indices = [idx for idx in args.indices if 0 <= idx < len(dataset)]
    else:
        count = max(1, min(args.count, len(dataset)))
        indices = list(range(count))

    print("=========================================")
    print(f"Previewing tags for split='{args.split}'")
    print(f"Dataset size: {len(dataset)}")
    print(f"Tag cache dir: {os.path.abspath(args.tag_cache_dir)}")
    print(f"Tag method   : {args.tag_generation_method}")
    if args.tag_generation_method == "blip_noun_phrases":
        print(f"Image root   : {os.path.abspath(args.image_root)}")
        print(f"BLIP model   : {args.blip_model_name}")
    else:
        print(f"CLIP model   : {args.clip_text_model_name}")
    print("=========================================")

    for idx in indices:
        sample = dataset[idx]
        print(format_row(idx, sample['image_ids'], sample['tags'], sample['caption']))


if __name__ == "__main__":
    main()
