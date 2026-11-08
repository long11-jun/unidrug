#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}/unidrug"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_NAME="${1:-SIDER}"
MODE="${2:-single}"
CUSTOM_SEEDS="${3:-}"
ABLATION="${4:-}"

LORA_R="16"
DROPOUT="0.15"
CONFIG_PATH="${PROJECT_ROOT}/unidrug/configs/dp/dp.json"
DATASET_PATH="${PROJECT_ROOT}/unidrug/datasets/dp/moleculenet"

if [ "$DATASET_NAME" == "HIV" ]; then
    LR="2.5e-5"
    WEIGHT_DECAY="0.15"
    LABEL_SMOOTHING="0.03"
    EPOCHS="150"
    PATIENCE="20"
    BATCH_SIZE="24"
    ACCUMULATION_STEPS="3"
    LOSS_TYPE="asl"
    MAX_GRAD_NORM="0.5"
    LORA_R="48"
    DROPOUT="0.20"
elif [ "$DATASET_NAME" == "ClinTox" ]; then
    LR="2e-5"
    WEIGHT_DECAY="0.15"
    LABEL_SMOOTHING="0.0"
    EPOCHS="100"
    PATIENCE="20"
    BATCH_SIZE="16"
    ACCUMULATION_STEPS="2"
    LOSS_TYPE="asl"
    MAX_GRAD_NORM="1.0"
    LORA_R="16"
    DROPOUT="0.15"
elif [ "$DATASET_NAME" == "Bace" ]; then
    LR="1e-5"
    WEIGHT_DECAY="0.20"
    LABEL_SMOOTHING="0.0"
    EPOCHS="250"
    PATIENCE="35"
    BATCH_SIZE="24"
    ACCUMULATION_STEPS="3"
    LOSS_TYPE="asl"
    MAX_GRAD_NORM="0.15"
    WARMUP_RATIO="0.15"
    LR_SCHEDULER="cosine"
    LORA_R="16"
    DROPOUT="0.24"
elif [ "$DATASET_NAME" == "SIDER" ]; then
    LR="2.5e-5"
    WEIGHT_DECAY="0.18"
    LABEL_SMOOTHING="0.03"
    EPOCHS="120"
    PATIENCE="25"
    BATCH_SIZE="16"
    ACCUMULATION_STEPS="2"
    LOSS_TYPE="asl"
    MAX_GRAD_NORM="0.4"
    LORA_R="20"
    DROPOUT="0.18"
else
    echo "Paper datasets: SIDER ClinTox HIV Bace"
    exit 1
fi

ABLATION_ARG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_ARG="--ablation ${ABLATION}"
    OUTPUT_DIR="${PROJECT_ROOT}/ckpts/dp/${DATASET_NAME,,}/ablation/${ABLATION}"
    MODEL_FILE="${OUTPUT_DIR}/model.pt"
else
    OUTPUT_DIR="${PROJECT_ROOT}/ckpts/dp/${DATASET_NAME,,}/vis"
    MODEL_FILE="${OUTPUT_DIR}/best_model.pt"
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "${DATASET_NAME} ${MODE} lr=${LR} bs=${BATCH_SIZE}x${ACCUMULATION_STEPS} epochs=${EPOCHS}"
echo "log ${LOG_FILE}"

COMMON_ARGS=(
    --config_path "${CONFIG_PATH}"
    --dataset MoleculeNet
    --dataset_path "${DATASET_PATH}"
    --dataset_name "${DATASET_NAME}"
    --device cuda:0
    --batch_size "${BATCH_SIZE}"
    --accumulation_steps "${ACCUMULATION_STEPS}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --patience "${PATIENCE}"
    --loss_type "${LOSS_TYPE}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --label_smoothing "${LABEL_SMOOTHING}"
    --lora_r "${LORA_R}"
    --dropout "${DROPOUT}"
    --eval_test_each_epoch
    --freeze_structure_encoder
    --unfreeze_graphmvp_layers 2
    --use_rdrop
    --output_path "${MODEL_FILE}"
)

if [ "$MODE" == "ensemble" ]; then
    if [ -n "${CUSTOM_SEEDS}" ]; then
        SEEDS="${CUSTOM_SEEDS}"
    else
        SEEDS="42,137,256,521,1024"
    fi
    python tasks/mol_task/dp.py "${COMMON_ARGS[@]}" --seeds "${SEEDS}" ${ABLATION_ARG} 2>&1 | tee "${LOG_FILE}"
elif [ "$MODE" == "multi" ]; then
    NUM_GPUS="${3:-2}"
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
        tasks/mol_task/dp.py "${COMMON_ARGS[@]}" ${ABLATION_ARG} 2>&1 | tee "${LOG_FILE}"
else
    python tasks/mol_task/dp.py "${COMMON_ARGS[@]}" ${ABLATION_ARG} 2>&1 | tee "${LOG_FILE}"
fi
