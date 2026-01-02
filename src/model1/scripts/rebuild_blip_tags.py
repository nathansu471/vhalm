#!/usr/bin/env python3
"""Rebuild BLIP noun tags from existing cached captions."""

import argparse
import json
import os
import sys
from typing import List, Tuple

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk

from lib.dataset import BLIPNounPhraseTagGenerator


def ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are available."""
    requirements: List[Tuple[str, str]] = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for resource, package in requirements:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def rebuild_tags(source_dir: str, target_dir: str, max_tags: int) -> None:
    ensure_nltk_resources()

    os.makedirs(target_dir, exist_ok=True)
    entries = [name for name in os.listdir(source_dir) if name.endswith(".json")]
    entries.sort()

    total = len(entries)
    if total == 0:
        print(f"No JSON files found in {source_dir}")
        return

    processed = 0
    skipped = 0

    for idx, filename in enumerate(entries, start=1):
        src_path = os.path.join(source_dir, filename)
        tgt_path = os.path.join(target_dir, filename)

        try:
            with open(src_path, "r", encoding="utf-8") as src_file:
                data = json.load(src_file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {filename}: failed to read ({exc})")
            skipped += 1
            continue

        caption = data.get("caption")
        if not isinstance(caption, str):
            print(f"Skipping {filename}: missing caption")
            skipped += 1
            continue

        tags = BLIPNounPhraseTagGenerator.extract_noun_tags(caption)
        tags = tags[:max_tags] if max_tags > 0 else tags

        payload = {"caption": caption, "tags": tags}
        try:
            with open(tgt_path, "w", encoding="utf-8") as tgt_file:
                json.dump(payload, tgt_file, ensure_ascii=False)
        except OSError as exc:
            print(f"Failed to write {filename}: {exc}")
            skipped += 1
            continue

        processed += 1
        if idx % 500 == 0 or idx == total:
            print(f"Processed {idx}/{total} files (success: {processed}, skipped: {skipped})")

    print(f"Done. Rebuilt tags for {processed} file(s); skipped {skipped}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-extract noun tags from cached BLIP captions.")
    parser.add_argument(
        "--source_dir",
        default="../../data/model1_data/blip_tags/blip_np_Salesforce_blip-image-captioning-base_prev",
        help="Directory containing original BLIP caption JSON files.",
    )
    parser.add_argument(
        "--target_dir",
        default="../../data/model1_data/blip_tags/blip_np_Salesforce_blip-image-captioning-base",
        help="Directory to write updated noun-only tag files.",
    )
    parser.add_argument(
        "--max_tags",
        type=int,
        default=12,
        help="Maximum number of tags to keep per image (0 means keep all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    rebuild_tags(source_dir, target_dir, max(0, args.max_tags))


if __name__ == "__main__":
    main()
