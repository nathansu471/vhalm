#!/bin/bash

# Image2Haiku Model Testing Script
# Usage: ./run_test.sh <experiment_dir> [max_samples]

set -euo pipefail

# =================== 配置区域 ===================
export NLTK_DATA=${NLTK_DATA:-/root/autodl-tmp/nltk_data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}  # 避免多进程冲突

EXPERIMENT_DIR=${1:-${EXPERIMENT_DIR:-}}  # 优先使用命令行参数，其次环境变量
MAX_SAMPLES_ARG=${2:-${MAX_SAMPLES:-}}

if [[ -z "$EXPERIMENT_DIR" ]]; then
    echo "Usage: ./run_test.sh <experiment_dir> [max_samples]"
    echo "Provide the experiment directory that contains checkpoints."
    exit 1
fi

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: experiment directory not found: $EXPERIMENT_DIR" >&2
    exit 1
fi

# 数据路径可通过环境变量覆盖；默认指向完整训练集
DEFAULT_FEATURES_PATH="../../data/model1_data/image_features.npy"
DEFAULT_IMAGE_IDS_PATH="../../data/model1_data/image_ids.json"
DEFAULT_ANNOTATIONS_PATH="../../data/model1_data/merged_captions_async_augmented.json"

FEATURES_PATH=${FEATURES_PATH:-$DEFAULT_FEATURES_PATH}
IMAGE_IDS_PATH=${IMAGE_IDS_PATH:-$DEFAULT_IMAGE_IDS_PATH}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-$DEFAULT_ANNOTATIONS_PATH}

ARGS=(
    --experiment_dir "$EXPERIMENT_DIR"
    --features_path "$FEATURES_PATH"
    --image_ids_path "$IMAGE_IDS_PATH"
    --annotations_path "$ANNOTATIONS_PATH"
)

if [[ -n "$MAX_SAMPLES_ARG" ]]; then
    ARGS+=(--max_samples "$MAX_SAMPLES_ARG")
fi

python scripts/test_best_model.py "${ARGS[@]}"
