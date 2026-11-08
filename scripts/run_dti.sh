#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset> [split] [fold] [strict_cold] [seed] [ablation]"
    echo "dataset: yamanishi08, bmkg-dti"
    echo "split: warm, cold_drug, cold_protein, cold_cluster"
    echo "fold: 0-4, all, or N+"
    exit 1
fi

DATASET_NAME="${1:-yamanishi08}"
SPLIT_STRATEGY="${2:-warm}"
FOLD_IDX="${3:-all}"
STRICT_COLD="${4:-false}"
SEED="${5:-42}"
ABLATION="${6:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOKENIZERS_PARALLELISM=false

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EPOCHS="100"
BATCH_SIZE="48"
ACCUMULATION_STEPS="2"
LR="2e-5"
WEIGHT_DECAY="1e-5"
LOSS_TYPE="focal"
MAX_GRAD_NORM="1.0"
PATIENCE="15"

if [ "$DATASET_NAME" == "yamanishi08" ]; then
    DATASET_PATH="${PROJECT_ROOT}/unidrug/datasets/dti/Yamanishi08"
    LR="2e-5"
    WEIGHT_DECAY="2e-4"
    EPOCHS="100"
    PATIENCE="10"
    BATCH_SIZE="48"
    ACCUMULATION_STEPS="2"
    LOSS_TYPE="bce"
    MAX_GRAD_NORM="1.0"
    LABEL_SMOOTHING="0.05"
    WARMUP_RATIO="0.20"
    LR_SCHEDULER="cosine"
    DROPOUT="0.20"
    LORA_R="12"
elif [ "$DATASET_NAME" == "bmkg-dti" ]; then
    DATASET_PATH="${PROJECT_ROOT}/unidrug/datasets/dti/BMKG_DTI"
    LR="1e-5"
    WEIGHT_DECAY="2e-4"
    EPOCHS="100"
    PATIENCE="15"
    BATCH_SIZE="32"
    ACCUMULATION_STEPS="3"
    LOSS_TYPE="bce"
    MAX_GRAD_NORM="0.5"
    LABEL_SMOOTHING="0.05"
    WARMUP_RATIO="0.20"
    LR_SCHEDULER="cosine"
    DROPOUT="0.18"
    LORA_R="32"
    CMSA_WEIGHT="0.3"
    CMON_WEIGHT="0.3"
else
    echo "Unknown dataset: $DATASET_NAME"
    exit 1
fi

if [ "$SPLIT_STRATEGY" == "warm" ]; then
    if [ "$DATASET_NAME" == "bmkg-dti" ]; then
        LR="3e-5"
        DROPOUT="0.22"
        LORA_R="32"
        WARMUP_RATIO="0.15"
        EPOCHS="80"
        PATIENCE="10"
    else
        LR="3e-5"
        LORA_R="14"
        DROPOUT="0.2"
        WARMUP_RATIO="0.15"
        EPOCHS="100"
        WEIGHT_DECAY="1e-4"
        PATIENCE="15"
    fi
fi

if [ "$SPLIT_STRATEGY" == "cold_protein" ]; then
    if [ "$DATASET_NAME" == "bmkg-dti" ]; then
        LR="2.5e-5"
        WARMUP_RATIO="0.15"
        WEIGHT_DECAY="1.5e-4"
        MAX_GRAD_NORM="0.3"
        EPOCHS="100"
        PATIENCE="15"
        LORA_R="32"
        DROPOUT="0.25"
        LABEL_SMOOTHING="0.06"
    else
        LR="2e-5"
        WARMUP_RATIO="0.15"
        MAX_GRAD_NORM="0.5"
        EPOCHS="100"
        WEIGHT_DECAY="3e-4"
        PATIENCE="15"
        LORA_R="14"
        DROPOUT="0.25"
        LABEL_SMOOTHING="0.08"
    fi
fi

if [ "$SPLIT_STRATEGY" == "cold_drug" ]; then
    if [ "$DATASET_NAME" == "bmkg-dti" ]; then
        LR="3e-5"
        WARMUP_RATIO="0.15"
        WEIGHT_DECAY="1e-4"
        MAX_GRAD_NORM="0.3"
        EPOCHS="100"
        PATIENCE="15"
        LORA_R="32"
        DROPOUT="0.22"
        LABEL_SMOOTHING="0.05"
    else
        LR="3e-5"
        WARMUP_RATIO="0.10"
        WEIGHT_DECAY="2e-4"
        MAX_GRAD_NORM="0.5"
        EPOCHS="100"
        PATIENCE="15"
        LORA_R="14"
        DROPOUT="0.22"
    fi
fi

if [ "$SPLIT_STRATEGY" == "cold_cluster" ]; then
    if [ "$DATASET_NAME" == "bmkg-dti" ]; then
        LR="3.2e-5"
        WARMUP_RATIO="0.15"
        WEIGHT_DECAY="2.5e-4"
        MAX_GRAD_NORM="0.5"
        EPOCHS="100"
        PATIENCE="15"
        LORA_R="32"
        DROPOUT="0.24"
        LABEL_SMOOTHING="0.01"
        CMSA_WEIGHT="0.3"
        CMON_WEIGHT="0.2"
    else
        LR="2e-5"
        WARMUP_RATIO="0.15"
        EPOCHS="200"
        PATIENCE="25"
        LORA_R="14"
        MAX_GRAD_NORM="0.5"
        WEIGHT_DECAY="3e-4"
        LABEL_SMOOTHING="0.08"
        DROPOUT="0.25"
    fi
fi

run_single_fold() {
    local fold=$1
    EXTRA_ARGS=""
    if [ "$STRICT_COLD" == "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --strict_cold_start"
    fi
    if [ -n "${ABLATION}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --ablation ${ABLATION}"
    fi
    cd "${PROJECT_ROOT}/unidrug"
    CUDA_VISIBLE_DEVICES=$GPU_ID python tasks/mol_task/dti.py \
        --config_path configs/dti/dti.json \
        --dataset_name "$DATASET_NAME" \
        --dataset_path "$DATASET_PATH" \
        --split_strategy "$SPLIT_STRATEGY" \
        --fold_idx "$fold" \
        --gpu 0 \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --accumulation_steps "$ACCUMULATION_STEPS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --loss_type "$LOSS_TYPE" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --patience "$PATIENCE" \
        --label_smoothing "${LABEL_SMOOTHING:-0.0}" \
        --warmup_ratio "${WARMUP_RATIO:-0.05}" \
        --lr_scheduler "${LR_SCHEDULER:-linear}" \
        --dropout "${DROPOUT:-0.1}" \
        --lora_r "${LORA_R:-8}" \
        --cmsa_weight "${CMSA_WEIGHT:-0.3}" \
        --cmon_weight "${CMON_WEIGHT:-1.0}" \
        --seed "$SEED" \
        --num_workers 4 \
        --output_path "${PROJECT_ROOT}/ckpts/dti" \
        $EXTRA_ARGS
}

summarize_folds() {
    python << EOF
import os
import re
import numpy as np

base_dir = "${PROJECT_ROOT}/ckpts/dti/${DATASET_NAME}/${SPLIT_STRATEGY}"
auroc_list = []
aupr_list = []
num_folds = ${NUM_FOLDS}

for fold in range(num_folds):
    log_dir = os.path.join(base_dir, f"fold_{fold}")
    if not os.path.exists(log_dir):
        print(f"fold {fold}: no directory")
        continue
    log_files = [f for f in os.listdir(log_dir) if f.startswith("train_") and f.endswith(".log")]
    if not log_files:
        print(f"fold {fold}: no log")
        continue
    log_file = os.path.join(log_dir, sorted(log_files)[-1])
    best_auroc = 0
    best_aupr = 0
    with open(log_file) as f:
        for line in f:
            match = re.search(r"Best Test AUROC:\s*([\d.]+)", line)
            if match:
                best_auroc = float(match.group(1))
            match = re.search(r"AUPR=([\d.]+)", line)
            if match:
                best_aupr = max(best_aupr, float(match.group(1)))
    if best_auroc > 0:
        auroc_list.append(best_auroc)
        aupr_list.append(best_aupr)
        print(f"fold {fold}: AUROC={best_auroc:.4f} AUPR={best_aupr:.4f}")

if auroc_list:
    print(f"AUROC {np.mean(auroc_list):.4f} +/- {np.std(auroc_list):.4f}")
    print(f"AUPR  {np.mean(aupr_list):.4f} +/- {np.std(aupr_list):.4f}")
EOF
}

NUM_FOLDS=5

echo "${DATASET_NAME} ${SPLIT_STRATEGY} fold=${FOLD_IDX} lr=${LR} bs=${BATCH_SIZE}x${ACCUMULATION_STEPS}"

if [ "$FOLD_IDX" == "all" ]; then
    for fold in $(seq 0 $((NUM_FOLDS-1))); do
        run_single_fold $fold
    done
    summarize_folds
elif [[ "$FOLD_IDX" == *"+" ]]; then
    START_FOLD="${FOLD_IDX%+}"
    for fold in $(seq $START_FOLD $((NUM_FOLDS-1))); do
        run_single_fold $fold
    done
    summarize_folds
else
    run_single_fold $FOLD_IDX
fi
