#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


    python ensemble_eval.py --ensemble_dir /path/to/ensemble --dataset BBBP
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_repo_root = os.path.abspath(os.path.join(_pkg_root, '..'))
sys.path.insert(0, _repo_root)
sys.path.insert(0, _pkg_root)

from datasets.dp_dataset import MoleculeNetDataset
from models.task_model.dp import DPModel
from utils.collators import ToDevice

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _build_kge_bank(dataset) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
    
    kge_bank = None
    kge_keys = None
    try:
        if hasattr(dataset, "kge") and isinstance(dataset.kge, dict) and len(dataset.kge) > 0:
            keys = sorted([k for k in dataset.kge.keys() if isinstance(k, str) and k and k[0] == "D"])
            if not keys:
                keys = sorted([k for k in dataset.kge.keys() if isinstance(k, str)])
            kge_keys = list(keys)
            vecs = [torch.as_tensor(dataset.kge[k]) for k in keys]
            kge_bank = torch.stack([v.float() for v in vecs], dim=0)
            logger.info(f"[KGE] Global bank ready: N={kge_bank.size(0)} dim={kge_bank.size(1)}")
    except Exception as e:
        logger.warning(f"[KGE] Failed to build global bank: {e}. KG completion will be disabled.")
    return kge_bank, kge_keys


def load_model(
    ckpt_path: str,
    config_path: str,
    device: str,
    num_tasks: int,
    test_dataset,
) -> DPModel:
    
    from utils.paths import expand_asset_paths
    cfg = expand_asset_paths(json.load(open(config_path)))
    try:
        cfg["network"]["dataset_name"] = getattr(test_dataset, "dataset_name", cfg["network"].get("dataset_name"))
    except Exception:
        pass
    try:
        cfg["network"]["task_names"] = getattr(test_dataset, "targets", None)
    except Exception:
        pass

    kge_bank, kge_keys = _build_kge_bank(test_dataset)
    model = DPModel(cfg, num_tasks, kge_bank=kge_bank, kge_keys=kge_keys)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {ckpt_path}")
    return model


def ensemble_predict(
    models: List[DPModel],
    loader: DataLoader,
    device: str,
    ensemble_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    
    Args:
    
    Returns:
    """
    all_probs_list = [[] for _ in models]
    all_labels = []
    
    with torch.no_grad():
        for drug, label in loader:
            drug = ToDevice(drug, device)
            label = label.numpy()
            
            for i, model in enumerate(models):
                output, _ = model(drug)
                probs = torch.sigmoid(output["classification"]).cpu().numpy()
                all_probs_list[i].append(probs)
            
            all_labels.append(label)
    
    all_probs_list = [np.concatenate(probs, axis=0) for probs in all_probs_list]
    all_labels = np.concatenate(all_labels, axis=0)
    
    stacked = np.stack(all_probs_list, axis=0)  # (num_models, N, num_tasks)
    
    if ensemble_method == "mean":
        ensemble_probs = stacked.mean(axis=0)
    elif ensemble_method == "median":
        ensemble_probs = np.median(stacked, axis=0)
    elif ensemble_method == "max":
        ensemble_probs = stacked.max(axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    return ensemble_probs, all_labels


def compute_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    
    valid_cols = []
    aucs = []
    
    for col in range(labels.shape[1]):
        mask = ~np.isnan(labels[:, col])
        if mask.sum() < 10:
            continue
        y_true = labels[mask, col]
        y_pred = probs[mask, col]
        
        if len(np.unique(y_true)) < 2:
            continue
        
        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            valid_cols.append(col)
        except:
            continue
    
    return np.mean(aucs) if aucs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")
    parser.add_argument("--ensemble_dir", type=str, required=True,
                        help="Directory containing ensemble models")
    parser.add_argument("--config_path", type=str, 
                        default="configs/dp/dp.json")
    parser.add_argument("--dataset", type=str, default="BBBP")
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/dp/moleculenet")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ensemble_method", type=str, default="mean",
                        choices=["mean", "median", "max"])
    args = parser.parse_args()
    
    model_paths = sorted(glob.glob(os.path.join(args.ensemble_dir, "*.pt")))
    if not model_paths:
        logger.error(f"No models found in {args.ensemble_dir}")
        return
    
    logger.info(f"Found {len(model_paths)} models for ensemble")
    for path in model_paths:
        logger.info(f"  - {os.path.basename(path)}")
    
    logger.info(f"Loading dataset: {args.dataset}")
    from utils.paths import expand_asset_paths
    cfg = expand_asset_paths(json.load(open(args.config_path)))
    
    test_dataset = MoleculeNetDataset(
        data_path=args.dataset_path,
        dataset_name=args.dataset,
        split="test",
        drug_featurizer_cfg=cfg["data"]["drug"]["featurizer"],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_fn,
    )
    
    models = []
    individual_aucs = []
    
    for path in model_paths:
        model = load_model(path, args.config_path, args.device, num_tasks=len(test_dataset.targets), test_dataset=test_dataset)
        models.append(model)
        
        probs, labels = ensemble_predict([model], test_loader, args.device)
        auc = compute_auc(probs, labels)
        individual_aucs.append(auc)
        logger.info(f"  {os.path.basename(path)}: Test AUC = {auc:.4f}")
    
    logger.info(f"\nRunning ensemble prediction ({args.ensemble_method})...")
    ensemble_probs, all_labels = ensemble_predict(
        models, test_loader, args.device, args.ensemble_method
    )
    ensemble_auc = compute_auc(ensemble_probs, all_labels)
    
    print("\n" + "=" * 60)
    print(" Ensemble Results")
    print("=" * 60)
    print(f"   Number of models: {len(models)}")
    print(f"   Ensemble method:  {args.ensemble_method}")
    print(f"   Individual AUCs:  {np.mean(individual_aucs):.4f} ± {np.std(individual_aucs):.4f}")
    print(f"   Best single:      {max(individual_aucs):.4f}")
    print(f"   Ensemble AUC:     {ensemble_auc:.4f}")
    print(f"   Improvement:      +{(ensemble_auc - np.mean(individual_aucs)) * 100:.2f}%")
    print("=" * 60)
    
    result_path = os.path.join(args.ensemble_dir, "ensemble_result.json")
    result = {
        "dataset": args.dataset,
        "num_models": len(models),
        "ensemble_method": args.ensemble_method,
        "individual_aucs": individual_aucs,
        "individual_mean": float(np.mean(individual_aucs)),
        "individual_std": float(np.std(individual_aucs)),
        "best_single": float(max(individual_aucs)),
        "ensemble_auc": float(ensemble_auc),
        "improvement": float(ensemble_auc - np.mean(individual_aucs)),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()

