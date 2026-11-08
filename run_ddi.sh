#!/bin/bash
set -e

SCENARIO=${1:-"LUO"}
MODE=${2:-"single"}
ABLATION=${3:-""}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/unidrug/configs/ddi/ddi.json"
DATASET_PATH="${PROJECT_ROOT}/unidrug/datasets/ddi"

if [ -n "${ABLATION}" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/ckpts/ddi/${SCENARIO,,}/ablation/${ABLATION}"
else
    OUTPUT_DIR="${PROJECT_ROOT}/ckpts/ddi/${SCENARIO,,}"
fi
mkdir -p "${OUTPUT_DIR}"

if [ "${SCENARIO}" = "S1" ]; then
    LR="2e-5"
    WEIGHT_DECAY="0.01"
    LABEL_SMOOTHING="0.08"
    EPOCHS="30"
    PATIENCE="8"
    BATCH_SIZE="16"
    ACCUMULATION_STEPS="4"
    MAX_GRAD_NORM="1.0"
    WARMUP_RATIO="0.1"
    LOSS_TYPE="focal"
    FOCAL_GAMMA="2.0"
elif [ "${SCENARIO}" = "S2" ]; then
    LR="1e-5"
    WEIGHT_DECAY="0.02"
    LABEL_SMOOTHING="0.10"
    EPOCHS="40"
    PATIENCE="10"
    BATCH_SIZE="8"
    ACCUMULATION_STEPS="8"
    MAX_GRAD_NORM="0.8"
    WARMUP_RATIO="0.15"
    LOSS_TYPE="focal"
    FOCAL_GAMMA="2.5"
elif [ "${SCENARIO}" = "S3" ]; then
    LR="8e-6"
    WEIGHT_DECAY="0.03"
    LABEL_SMOOTHING="0.12"
    EPOCHS="50"
    PATIENCE="12"
    BATCH_SIZE="8"
    ACCUMULATION_STEPS="8"
    MAX_GRAD_NORM="0.5"
    WARMUP_RATIO="0.2"
    LOSS_TYPE="focal"
    FOCAL_GAMMA="3.0"
    NUM_CLASSES="4"
elif [ "${SCENARIO}" = "LUO" ]; then
    LR="2e-5"
    WEIGHT_DECAY="0.01"
    LABEL_SMOOTHING="0.05"
    EPOCHS="30"
    PATIENCE="8"
    BATCH_SIZE="48"
    ACCUMULATION_STEPS="2"
    MAX_GRAD_NORM="1.0"
    WARMUP_RATIO="0.1"
    LOSS_TYPE="focal"
    FOCAL_GAMMA="2.0"
    NUM_CLASSES="2"
    LORA_R="16"
    DROPOUT="0.15"
else
    echo "Unknown scenario: ${SCENARIO}"
    echo "Supported: S1, S2, S3, LUO"
    exit 1
fi

if [ -z "${NUM_CLASSES}" ]; then
    NUM_CLASSES="4"
fi

SEEDS="54"
ABLATION_ARG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_ARG="--ablation ${ABLATION}"
fi

echo "${SCENARIO} ${MODE} classes=${NUM_CLASSES} lr=${LR} bs=${BATCH_SIZE}x${ACCUMULATION_STEPS}"

cd "${PROJECT_ROOT}/unidrug"
python -u tasks/mol_task/ddi.py \
    --config_path "${CONFIG_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --scenario "${SCENARIO}" \
    --device "cuda:0" \
    --batch_size ${BATCH_SIZE} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --label_smoothing ${LABEL_SMOOTHING} \
    --num_classes ${NUM_CLASSES} \
    --warmup_ratio ${WARMUP_RATIO} \
    --loss_type "${LOSS_TYPE}" \
    --focal_gamma ${FOCAL_GAMMA} \
    --lora_r "${LORA_R:-16}" \
    --dropout "${DROPOUT:-0.15}" \
    --seeds "${SEEDS}" \
    ${ABLATION_ARG} \
    --output_path "${OUTPUT_DIR}/best_model.pt" \
    --num_workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
