#!/bin/bash

# Image2Haiku Model Training Script
# Usage: ./run_training.sh [experiment_name]

# =================== 配置区域 ===================
export NLTK_DATA=/root/autodl-tmp/nltk_data
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false  # 避免多进程冲突

# 可选环境覆盖（留空则使用 config.json）
FEATURES_PATH=${FEATURES_PATH:-}
IMAGE_IDS_PATH=${IMAGE_IDS_PATH:-}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-}
TAG_CACHE_DIR=${TAG_CACHE_DIR:-}
TAG_METHOD=${TAG_METHOD:-}
BLIP_MODEL_NAME=${BLIP_MODEL_NAME:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
TAG_DEVICE=${TAG_DEVICE:-}
T5_MODEL_NAME_OR_PATH=${T5_MODEL_NAME_OR_PATH:-}

# 训练配置
CONFIG_PATH="config.json"
OUTPUT_DIR="outputs"

# 实验名称
EXPERIMENT_NAME=${1:-"image2haiku"}
# ============================================

CONFIG_LINES=$(python - "$CONFIG_PATH" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
flat = {}
for key, value in cfg.items():
    if isinstance(value, dict):
        flat.update(value)
    else:
        flat[key] = value
keys = [
    'features_path',
    'image_ids_path',
    'annotations_path',
    'tag_cache_dir',
    'tag_generation_method',
    'blip_model_name',
    'image_root',
    't5_model_name',
    'use_tags',
    'tag_top_k',
    'max_tags_per_sample'
]
for key in keys:
    print(f"{key}={flat.get(key)}")
PY
)

declare -A CONFIG_DEFAULTS=()
while IFS='=' read -r key value; do
    [[ -z "$key" ]] && continue
    CONFIG_DEFAULTS["$key"]="$value"
done <<< "$CONFIG_LINES"

echo "Starting Image2Haiku Model Training"
echo "=================================="
if [[ -n "$FEATURES_PATH" ]]; then
    echo "Features override: $FEATURES_PATH"
else
    echo "Features: ${CONFIG_DEFAULTS[features_path]}"
fi
if [[ -n "$IMAGE_IDS_PATH" ]]; then
    echo "Image IDs override: $IMAGE_IDS_PATH"
else
    echo "Image IDs: ${CONFIG_DEFAULTS[image_ids_path]}"
fi
if [[ -n "$ANNOTATIONS_PATH" ]]; then
    echo "Annotations override: $ANNOTATIONS_PATH"
else
    echo "Annotations: ${CONFIG_DEFAULTS[annotations_path]}"
fi
echo "Config: $CONFIG_PATH"
if [[ -n "$T5_MODEL_NAME_OR_PATH" ]]; then
    echo "T5 Model override: $T5_MODEL_NAME_OR_PATH"
else
    echo "T5 Model: ${CONFIG_DEFAULTS[t5_model_name]}"
fi
echo "Experiment: $EXPERIMENT_NAME"
if [[ -n "$TAG_CACHE_DIR" ]]; then
    echo "Tag cache override: $TAG_CACHE_DIR"
else
    echo "Tag cache dir: ${CONFIG_DEFAULTS[tag_cache_dir]}"
fi
if [[ -n "$TAG_METHOD" ]]; then
    echo "Tag method override: $TAG_METHOD"
else
    echo "Tag method   : ${CONFIG_DEFAULTS[tag_generation_method]}"
fi
if [[ -n "$BLIP_MODEL_NAME" ]]; then
    echo "BLIP model override: $BLIP_MODEL_NAME"
else
    echo "BLIP model   : ${CONFIG_DEFAULTS[blip_model_name]}"
fi
if [[ -n "$IMAGE_ROOT" ]]; then
    echo "Image root override: $IMAGE_ROOT"
else
    echo "Image root   : ${CONFIG_DEFAULTS[image_root]}"
fi
[[ -n "$TAG_DEVICE" ]] && echo "Tag device override: $TAG_DEVICE"
echo ""
echo "All training parameters are defined in: $CONFIG_PATH"
echo "=================================="

# Pre-download NLTK data to avoid runtime errors
echo "Checking NLTK data..."
python <<'PY'
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print('NLTK data already available')
except LookupError:
    print('Downloading missing NLTK data...')
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('NLTK data downloaded')
PY

# Assemble dataset arguments (allow overrides via env)
CLI_ARGS=()
if [[ -n "$FEATURES_PATH" ]]; then
    CLI_ARGS+=(--features_path "$FEATURES_PATH")
fi
if [[ -n "$IMAGE_IDS_PATH" ]]; then
    CLI_ARGS+=(--image_ids_path "$IMAGE_IDS_PATH")
fi
if [[ -n "$ANNOTATIONS_PATH" ]]; then
    CLI_ARGS+=(--annotations_path "$ANNOTATIONS_PATH")
fi
if [[ -n "$TAG_CACHE_DIR" ]]; then
    CLI_ARGS+=(--tag_cache_dir "$TAG_CACHE_DIR")
fi
if [[ -n "$TAG_METHOD" ]]; then
    CLI_ARGS+=(--tag_generation_method "$TAG_METHOD")
fi
if [[ -n "$BLIP_MODEL_NAME" ]]; then
    CLI_ARGS+=(--blip_model_name "$BLIP_MODEL_NAME")
fi
if [[ -n "$IMAGE_ROOT" ]]; then
    CLI_ARGS+=(--image_root "$IMAGE_ROOT")
fi
if [[ -n "$TAG_DEVICE" ]]; then
    CLI_ARGS+=(--tag_device "$TAG_DEVICE")
fi
if [[ -n "$T5_MODEL_NAME_OR_PATH" ]]; then
    CLI_ARGS+=(--t5_model_name_or_path "$T5_MODEL_NAME_OR_PATH")
fi

# Run training (所有关键参数现由 config.json 控制，可选覆盖才会注入)
python scripts/train.py "${CLI_ARGS[@]}" \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME"

echo "Training completed!"
echo "Check results in: $OUTPUT_DIR/$EXPERIMENT_NAME*/"