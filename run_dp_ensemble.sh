#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}/unidrug"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

DATASET_NAME="${1:-BBBP}"
NUM_SEEDS="${2:-5}"
SEEDS=(42 137 256 521 1024)

ENSEMBLE_DIR="${PROJECT_ROOT}/ckpts/ensemble_${DATASET_NAME,,}"
mkdir -p "${ENSEMBLE_DIR}"

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=${SEEDS[$i]}
    OUTPUT_PATH="${ENSEMBLE_DIR}/model_seed${SEED}.pt"
    echo "seed ${SEED} -> ${OUTPUT_PATH}"
    python tasks/mol_task/dp.py \
      --config_path "${PROJECT_ROOT}/unidrug/configs/dp/dp.json" \
      --dataset MoleculeNet \
      --dataset_path "${PROJECT_ROOT}/unidrug/datasets/dp/moleculenet" \
      --dataset_name "${DATASET_NAME}" \
      --device "cuda:0" \
      --batch_size 16 \
      --epochs 120 \
      --lr 8e-6 \
      --weight_decay 0.08 \
      --patience 30 \
      --label_smoothing 0.12 \
      --seed ${SEED} \
      --output_path "${OUTPUT_PATH}"
done
