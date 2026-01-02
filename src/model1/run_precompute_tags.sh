#!/bin/bash

# Tag Cache Precomputation Script
# Usage: ./run_precompute_tags.sh [cache_dir]
# Environment overrides:
#   TAG_CACHE_DIR, TAG_METHOD, CLIP_MODEL_NAME, BLIP_MODEL_NAME,
#   TAG_IMAGE_ROOT, TAG_IMAGE_IDS_PATH,
#   TAG_TOP_K, TAG_MAX_TAGS, TAG_DEVICE, TAG_FORCE_REBUILD

# TAG METHOD: clip_zero_shot, blip_noun_phrases
set -euo pipefail

# =================== 环境配置 ===================
export NLTK_DATA=${NLTK_DATA:-/root/autodl-tmp/nltk_data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# 允许通过第一个参数或环境变量覆盖缓存目录
DEFAULT_CACHE_DIR="../../data/model1_data/blip_tags"
CACHE_DIR=${1:-${TAG_CACHE_DIR:-$DEFAULT_CACHE_DIR}}

TAG_METHOD=${TAG_METHOD:-blip_noun_phrases}
CLIP_MODEL_NAME=${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}
BLIP_MODEL_NAME=${BLIP_MODEL_NAME:-Salesforce/blip-image-captioning-base}
DEFAULT_IMAGE_ROOT="../../../../../root/data/train2017"
DEFAULT_IMAGE_IDS_PATH="../../data/model1_data/image_ids.json"
TAG_IMAGE_ROOT=${TAG_IMAGE_ROOT:-$DEFAULT_IMAGE_ROOT}
TAG_IMAGE_IDS_PATH=${TAG_IMAGE_IDS_PATH:-$DEFAULT_IMAGE_IDS_PATH}
TAG_TOP_K=${TAG_TOP_K:-4}
TAG_MAX_TAGS=${TAG_MAX_TAGS:-12}
TAG_DEVICE=${TAG_DEVICE:-}
FORCE_REBUILD=${TAG_FORCE_REBUILD:-}

echo "========================================="
echo "Preparing tag cache"
echo "========================================="
echo "Method         : $TAG_METHOD"
echo "Cache directory: $CACHE_DIR"
[[ $TAG_METHOD == "clip_zero_shot" ]] && echo "CLIP model      : $CLIP_MODEL_NAME"
[[ $TAG_METHOD == "blip_noun_phrases" ]] && echo "BLIP model      : $BLIP_MODEL_NAME"
[[ $TAG_METHOD == "blip_noun_phrases" ]] && echo "Image root      : $TAG_IMAGE_ROOT"
[[ $TAG_METHOD == "blip_noun_phrases" ]] && echo "Image IDs path  : $TAG_IMAGE_IDS_PATH"
[[ -n "$TAG_DEVICE" ]] && echo "Device override : $TAG_DEVICE"
[[ -n "$FORCE_REBUILD" ]] && echo "Force rebuild   : $FORCE_REBUILD"
echo "NLTK data       : $NLTK_DATA"
echo "HF cache        : $HF_HOME"
echo "HF endpoint     : $HF_ENDPOINT"
echo "========================================="

mkdir -p "$CACHE_DIR"

ARGS=(
  --method "$TAG_METHOD"
  --cache_dir "$CACHE_DIR"
  --clip_model_name "$CLIP_MODEL_NAME"
  --blip_model_name "$BLIP_MODEL_NAME"
  --image_root "$TAG_IMAGE_ROOT"
  --image_ids_path "$TAG_IMAGE_IDS_PATH"
  --top_k "$TAG_TOP_K"
  --max_tags "$TAG_MAX_TAGS"
)

if [[ -n "$TAG_DEVICE" ]]; then
  ARGS+=(--device "$TAG_DEVICE")
fi

if [[ -n "$FORCE_REBUILD" ]]; then
  ARGS+=(--force)
fi

python scripts/precompute_tags.py "${ARGS[@]}"

echo "Cache ready at: $CACHE_DIR"
