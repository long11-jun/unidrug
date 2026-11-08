#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/unidrug/configs/ppi/ppi.json"
DATASET_PATH="${PROJECT_ROOT}/unidrug/datasets/ppi"

DATASET_NAME=${1:-"SHS27k"}
SPLIT_METHOD=${2:-"BFS"}
MODE=${3:-"single"}
CUSTOM_SEEDS="${4:-}"
GPU_ID=${5:-"${CUDA_VISIBLE_DEVICES:-0}"}
ABLATION=${6:-""}

if [ -n "${ABLATION}" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/ckpts/ppi/${DATASET_NAME,,}_${SPLIT_METHOD,,}/ablation/${ABLATION//,/_}"
else
    OUTPUT_DIR="${PROJECT_ROOT}/ckpts/ppi/${DATASET_NAME,,}_${SPLIT_METHOD,,}"
fi
mkdir -p "${OUTPUT_DIR}"

if [ "${DATASET_NAME}" = "SHS148k" ] && [ "${SPLIT_METHOD}" = "DFS" ]; then
    LR="4e-5"
    WEIGHT_DECAY="0.01"
    LABEL_SMOOTHING="0.03"
    EPOCHS="30"
    PATIENCE="10"
    BATCH_SIZE="48"
    ACCUMULATION_STEPS="2"
    MAX_GRAD_NORM="1.0"
    WARMUP_RATIO="0.08"
    LOSS_TYPE="bce"
elif [ "${DATASET_NAME}" = "SHS148k" ] && [ "${SPLIT_METHOD}" = "BFS" ]; then
    LR="3e-5"
    WEIGHT_DECAY="0.015"
    LABEL_SMOOTHING="0.01"
    EPOCHS="30"
    PATIENCE="10"
    BATCH_SIZE="48"
    ACCUMULATION_STEPS="2"
    MAX_GRAD_NORM="1.0"
    WARMUP_RATIO="0.1"
    LOSS_TYPE="bce"
elif [ "${DATASET_NAME}" = "SHS27k" ] && [ "${SPLIT_METHOD}" = "DFS" ]; then
    LR="4e-5"
    WEIGHT_DECAY="0.01"
    LABEL_SMOOTHING="0.02"
    EPOCHS="50"
    PATIENCE="12"
    BATCH_SIZE="48"
    ACCUMULATION_STEPS="2"
    MAX_GRAD_NORM="1.0"
    WARMUP_RATIO="0.12"
    LOSS_TYPE="bce"
elif [ "${DATASET_NAME}" = "SHS27k" ]; then
    LR="3e-5"
    WEIGHT_DECAY="0.02"
    LABEL_SMOOTHING="0.01"
    EPOCHS="50"
    PATIENCE="12"
    BATCH_SIZE="48"
    ACCUMULATION_STEPS="2"
    MAX_GRAD_NORM="1.0"
    WARMUP_RATIO="0.12"
    LOSS_TYPE="bce"
else
    echo "Unknown dataset: ${DATASET_NAME}"
    echo "Supported: SHS148k, SHS27k"
    exit 1
fi

if [ "${MODE}" = "ensemble" ]; then
    if [ -n "${CUSTOM_SEEDS}" ]; then
        SEEDS="${CUSTOM_SEEDS}"
    else
        SEEDS="42,123,456,789,2024"
    fi
else
    SEEDS="42"
fi

CMSA_WEIGHT="${CMSA_WEIGHT:-0.3}"
CMON_WEIGHT="${CMON_WEIGHT:-0.5}"
ABLATION_ARG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_ARG="--ablation ${ABLATION}"
fi

echo "${DATASET_NAME} ${SPLIT_METHOD} ${MODE} lr=${LR} bs=${BATCH_SIZE}x${ACCUMULATION_STEPS}"

cd "${PROJECT_ROOT}/unidrug"
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u tasks/prot_task/ppi.py \
    --config_path "${CONFIG_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --split_method "${SPLIT_METHOD}" \
    --device "cuda:0" \
    --batch_size ${BATCH_SIZE} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --label_smoothing ${LABEL_SMOOTHING} \
    --num_classes 7 \
    --warmup_ratio ${WARMUP_RATIO} \
    --loss_type "${LOSS_TYPE}" \
    --lora_r 32 \
    --dropout 0.15 \
    --cmsa_weight "${CMSA_WEIGHT}" \
    --cmon_weight "${CMON_WEIGHT}" \
    --seeds "${SEEDS}" \
    ${ABLATION_ARG} \
    --output_path "${OUTPUT_DIR}/best_model.pt" \
    --num_workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
