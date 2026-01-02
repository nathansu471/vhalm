#!/bin/bash

# Tag Preview Script
# Usage: ./run_preview_tags.sh [SPLIT] [COUNT] [additional argparse flags]
# Environment overrides:
#   FEATURES_PATH, IMAGE_IDS_PATH, ANNOTATIONS_PATH,
#   TAG_CACHE_DIR, CLIP_MODEL_NAME, BLIP_MODEL_NAME,
#   TAG_IMAGE_ROOT, TAG_TAG_DEVICE,
#   TAG_TOP_K, TAG_MAX_TAGS,
#   TAG_PREVIEW_SPLIT, TAG_PREVIEW_COUNT, TAG_PREVIEW_INDICES,
#   TAG_METHOD

set -euo pipefail

# =================== 环境配置 ===================
export NLTK_DATA=${NLTK_DATA:-/root/autodl-tmp/nltk_data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# 数据路径可通过环境变量覆盖
DEFAULT_FEATURES_PATH="../../data/model1_data/image_features.npy"
DEFAULT_IMAGE_IDS_PATH="../../data/model1_data/image_ids.json"
DEFAULT_ANNOTATIONS_PATH="../../data/model1_data/merged_captions_async_augmented.json"

temp_split=${1:-}
temp_count=${2:-}

FEATURES_PATH=${FEATURES_PATH:-$DEFAULT_FEATURES_PATH}
IMAGE_IDS_PATH=${IMAGE_IDS_PATH:-$DEFAULT_IMAGE_IDS_PATH}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-$DEFAULT_ANNOTATIONS_PATH}
TAG_CACHE_DIR=${TAG_CACHE_DIR:-../../data/model1_data/clip_tags}
CLIP_MODEL_NAME=${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}
BLIP_MODEL_NAME=${BLIP_MODEL_NAME:-Salesforce/blip-image-captioning-base}
TAG_IMAGE_ROOT=${TAG_IMAGE_ROOT:-../../data/model1_data/images}
TAG_METHOD=${TAG_METHOD:-clip_zero_shot}
TAG_DEVICE=${TAG_DEVICE:-}
TAG_TOP_K=${TAG_TOP_K:-4}
TAG_MAX_TAGS=${TAG_MAX_TAGS:-12}

SPLIT=${TAG_PREVIEW_SPLIT:-${temp_split:-train}}
COUNT=${TAG_PREVIEW_COUNT:-${temp_count:-5}}
INDICES=${TAG_PREVIEW_INDICES:-}

shift $(( $# > 0 ? 1 : 0 ))
if [[ $# -gt 0 ]]; then
  shift $(( $# > 0 ? 1 : 0 ))
fi

echo "========================================="
echo "Previewing cached tags"
echo "========================================="
echo "Features path  : $FEATURES_PATH"
echo "Image IDs path  : $IMAGE_IDS_PATH"
echo "Annotations     : $ANNOTATIONS_PATH"
echo "Tag cache dir   : $TAG_CACHE_DIR"
echo "Tag method      : $TAG_METHOD"
[[ $TAG_METHOD == "clip_zero_shot" ]] && echo "CLIP model      : $CLIP_MODEL_NAME"
[[ $TAG_METHOD == "blip_noun_phrases" ]] && echo "BLIP model      : $BLIP_MODEL_NAME"
[[ $TAG_METHOD == "blip_noun_phrases" ]] && echo "Image root      : $TAG_IMAGE_ROOT"
echo "Split           : $SPLIT"
if [[ -n "$INDICES" ]]; then
  echo "Indices         : $INDICES"
else
  echo "Sample count    : $COUNT"
fi
echo "Top-K           : $TAG_TOP_K"
echo "Max tags/sample : $TAG_MAX_TAGS"
[[ -n "$TAG_DEVICE" ]] && echo "Tag device      : $TAG_DEVICE"
echo "NLTK data       : $NLTK_DATA"
echo "HF cache        : $HF_HOME"
echo "HF endpoint     : $HF_ENDPOINT"
echo "========================================="

ARGS=(
  --features_path "$FEATURES_PATH"
  --image_ids_path "$IMAGE_IDS_PATH"
  --annotations_path "$ANNOTATIONS_PATH"
  --tag_cache_dir "$TAG_CACHE_DIR"
  --clip_text_model_name "$CLIP_MODEL_NAME"
  --blip_model_name "$BLIP_MODEL_NAME"
  --image_root "$TAG_IMAGE_ROOT"
  --tag_generation_method "$TAG_METHOD"
  --tag_top_k "$TAG_TOP_K"
  --max_tags "$TAG_MAX_TAGS"
  --split "$SPLIT"
)

if [[ -n "$TAG_DEVICE" ]]; then
  ARGS+=(--tag_device "$TAG_DEVICE")
fi

if [[ -n "$INDICES" ]]; then
  read -r -a INDEX_ARRAY <<< "$INDICES"
  ARGS+=(--indices "${INDEX_ARRAY[@]}")
else
  ARGS+=(--count "$COUNT")
fi

python scripts/preview_tags.py "${ARGS[@]}" "$@"
