#!/usr/bin/env python3
"""Utility script to precompute tag caches for different strategies."""

import argparse
import json
import os
import shutil
import sys
import time

# Ensure project root (src/model1) is on the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import BLIPNounPhraseTagGenerator, ZeroShotTagGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute tag caches for the specified method.")
    parser.add_argument(
        "--method",
        default="blip_noun_phrases",
        choices=["clip_zero_shot", "blip_noun_phrases"],
        help="Tag generation method to precompute.",
    )
    parser.add_argument(
        "--cache_dir",
        default="../../data/model1_data/clip_tags",
        help="Directory where the tag cache will be stored.",
    )
    parser.add_argument(
        "--clip_model_name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model identifier for the CLIP text encoder (clip_zero_shot).",
    )
    parser.add_argument(
        "--blip_model_name",
        default="Salesforce/blip-image-captioning-base",
        help="BLIP model identifier (blip_noun_phrases).",
    )
    parser.add_argument(
        "--image_root",
        default="../../data/model1_data/images",
        help="Directory containing raw images for BLIP tagging.",
    )
    parser.add_argument(
        "--image_ids_path",
        default="../../data/model1_data/image_ids.json",
        help="JSON file listing image filenames (blip_noun_phrases).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Number of top tokens to collect per image (clip_zero_shot).",
    )
    parser.add_argument(
        "--max_tags",
        type=int,
        default=12,
        help="Maximum tag count to keep per sample.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (e.g. 'cuda', 'cuda:0', or 'cpu').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cache even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    start = time.time()
    print("=========================================")
    print(f"Precomputing tag cache ({args.method})")
    print("=========================================")
    print(f"Cache directory : {cache_dir}")
    if args.device:
        print(f"Device override : {args.device}")
    print(f"Force rebuild   : {args.force}")

    if args.method == "clip_zero_shot":
        safe_model_name = args.clip_model_name.replace('/', '_')
        cache_path = os.path.join(cache_dir, f"clip_tags_{safe_model_name}.npz")

        if args.force and os.path.exists(cache_path):
            print(f"Force flag set: removing existing cache at {cache_path}")
            os.remove(cache_path)

        print(f"CLIP model      : {args.clip_model_name}")
        generator = ZeroShotTagGenerator(
            cache_dir=cache_dir,
            clip_model_name=args.clip_model_name,
            top_k=args.top_k,
            max_tags=args.max_tags,
            device=args.device,
        )

        duration = time.time() - start
        print("-----------------------------------------")
        print(f"Cached tags     : {len(generator.tag_strings)}")
        print(f"Embedding shape : {tuple(generator.text_embeddings.shape)}")
        print(f"Cache file      : {cache_path}")
        print(f"Completed in    : {duration:.2f}s")

    else:
        safe_model_name = args.blip_model_name.replace('/', '_')
        store_dir = os.path.join(cache_dir, f"blip_np_{safe_model_name}")
        if args.force and os.path.isdir(store_dir):
            print(f"Force flag set: removing existing cache directory {store_dir}")
            shutil.rmtree(store_dir)

        image_root = os.path.abspath(args.image_root)
        image_ids_path = os.path.abspath(args.image_ids_path)
        print(f"BLIP model      : {args.blip_model_name}")
        print(f"Image root      : {image_root}")
        print(f"Image IDs path  : {image_ids_path}")

        with open(image_ids_path, 'r', encoding='utf-8') as f:
            image_ids = json.load(f)

        generator = BLIPNounPhraseTagGenerator(
            cache_dir=cache_dir,
            image_root=image_root,
            blip_model_name=args.blip_model_name,
            max_tags=args.max_tags,
            device=args.device,
        )

        for idx, image_id in enumerate(image_ids, 1):
            generator.ensure_cached(image_id)
            if idx % 50 == 0 or idx == len(image_ids):
                print(f"Processed {idx}/{len(image_ids)} images", end="\r")

        duration = time.time() - start
        print("\n-----------------------------------------")
        print(f"Images cached   : {len(image_ids)}")
        print(f"Cache directory : {store_dir}")
        print(f"Completed in    : {duration:.2f}s")

    print("=========================================")


if __name__ == "__main__":
    main()
