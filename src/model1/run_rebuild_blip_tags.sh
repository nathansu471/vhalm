#!/bin/bash

# Rebuild noun-only BLIP tags from cached captions.
# Usage: ./run_rebuild_blip_tags.sh [source_dir] [target_dir]
# Environment overrides:
#   BLIP_TAG_SOURCE_DIR, BLIP_TAG_TARGET_DIR, BLIP_TAG_MAX_TAGS

set -euo pipefail

# =================== 环境配置 ===================
export NLTK_DATA=${NLTK_DATA:-/root/autodl-tmp/nltk_data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# 默认路径（相对于 run 脚本所在目录）
DEFAULT_SOURCE="../../data/model1_data/blip_tags/blip_np_Salesforce_blip-image-captioning-base_prev"
DEFAULT_TARGET="../../data/model1_data/blip_tags/blip_np_Salesforce_blip-image-captioning-base"

SOURCE_DIR=${1:-${BLIP_TAG_SOURCE_DIR:-$DEFAULT_SOURCE}}
TARGET_DIR=${2:-${BLIP_TAG_TARGET_DIR:-$DEFAULT_TARGET}}
MAX_TAGS=${BLIP_TAG_MAX_TAGS:-12}

echo "========================================="
echo "Rebuilding BLIP noun tags"
echo "========================================="
echo "Source directory : $SOURCE_DIR"
echo "Target directory : $TARGET_DIR"
echo "Max tags per file: $MAX_TAGS"
echo "NLTK data        : $NLTK_DATA"
echo "HF cache         : $HF_HOME"
echo "HF endpoint      : $HF_ENDPOINT"
echo "========================================="

python scripts/rebuild_blip_tags.py \
  --source_dir "$SOURCE_DIR" \
  --target_dir "$TARGET_DIR" \
  --max_tags "$MAX_TAGS"

echo "Done. Updated tags written to: $TARGET_DIR"
