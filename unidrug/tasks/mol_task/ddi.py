"""

- Structure (GraphMVP)


"""

import argparse
import copy
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from datasets.ddi_dataset import DDIDataset, get_ddi_dataset
from models.task_model.ddi import create_ddi_model, DDIModel
from feat.drug_featurizer import DrugMultiModalFeaturizer
from utils.collators import DPCollator, ToDevice

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get_average(self):
        return self.avg


class EarlyStopping:
    def __init__(self, patience=10, mode="higher", filename="best_model.pt"):
        self.patience = patience
        self.mode = mode
        self.filename = filename
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def step(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch)
        elif self._is_improvement(score):
            self.best_score = score
            self.save_checkpoint(model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == "higher":
            return score > self.best_score
        return score < self.best_score
    
    def save_checkpoint(self, model, epoch):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_score": self.best_score
        }, self.filename)


class DDIDatasetWithFeaturizer(Dataset):
    """
    """
    def __init__(
        self,
        base_dataset: DDIDataset,
        featurizer: DrugMultiModalFeaturizer,
        cache_features: bool = True,
        tokenmol_cache_file: str = None,
    ):
        self.base_dataset = base_dataset
        self.featurizer = featurizer
        self.cache_features = cache_features
        self.cache = {}
        
        self.tokenmol_cache = {}
        if tokenmol_cache_file and os.path.exists(tokenmol_cache_file):
            try:
                with open(tokenmol_cache_file, 'rb') as f:
                    self.tokenmol_cache = pickle.load(f)
                logger.info(f"[DDI] Loaded TokenMol cache: {len(self.tokenmol_cache)} entries from {tokenmol_cache_file}")
            except Exception as e:
                logger.warning(f"[DDI] Failed to load TokenMol cache: {e}")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _featurize(self, smiles: str):
        if self.cache_features and smiles in self.cache:
            return self.cache[smiles]
        
        try:
            feat = self.featurizer(smiles)
            
            if smiles in self.tokenmol_cache:
                feat["tokenmol_seq"] = self.tokenmol_cache[smiles]
            
            if self.cache_features:
                self.cache[smiles] = feat
            return feat
        except Exception as e:
            logger.warning(f"Failed to featurize: {smiles[:50]}..., error: {e}")
            return None
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        smiles_a = item["smiles_a"]
        smiles_b = item["smiles_b"]
        label = item["label"]
        
        feat_a = self._featurize(smiles_a)
        feat_b = self._featurize(smiles_b)
        
        return feat_a, feat_b, label


class DDICollator:
    """
    """
    def __init__(self, dp_collator: DPCollator):
        self.dp_collator = dp_collator
    
    def __call__(self, batch):
        feat_a_list = []
        feat_b_list = []
        labels = []
        
        for feat_a, feat_b, label in batch:
            if feat_a is not None and feat_b is not None:
                feat_a_list.append(feat_a)
                feat_b_list.append(feat_b)
                labels.append(label)
        
        if len(feat_a_list) == 0:
            return None, None, None
        
        drug_a_batch = self.dp_collator.mol_collator(feat_a_list)
        drug_b_batch = self.dp_collator.mol_collator(feat_b_list)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return drug_a_batch, drug_b_batch, labels_tensor


def focal_loss_multiclass(logits, labels, gamma=2.0, alpha=None, label_smoothing=0.0):
    """Focal Loss for multi-class classification"""
    num_classes = logits.size(-1)
    
    # Label smoothing
    if label_smoothing > 0:
        one_hot = F.one_hot(labels, num_classes).float()
        smooth_labels = one_hot * (1 - label_smoothing) + label_smoothing / num_classes
        ce_loss = -(smooth_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    else:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    
    if alpha is not None:
        alpha_t = alpha[labels]
        focal_weight = focal_weight * alpha_t
    
    loss = focal_weight * ce_loss
    return loss.mean()


def compute_class_weights(labels_list, num_classes=4, device='cuda'):
    
    class_counts = torch.zeros(num_classes)
    for label in labels_list:
        class_counts[label] += 1
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    
    try:
        if num_classes == 2:
            probs_positive = probs[:, 1] if probs.shape[1] == 2 else probs.flatten()
            auroc_macro = roc_auc_score(labels, probs_positive)
            aupr_macro = average_precision_score(labels, probs_positive)
        else:
            labels_onehot = label_binarize(labels, classes=list(range(num_classes)))
            auroc_macro = roc_auc_score(labels_onehot, probs, average="macro", multi_class="ovr")
            aupr_macro = average_precision_score(labels_onehot, probs, average="macro")
    except Exception as e:
        logger.warning(f"Metrics calculation failed: {e}")
        auroc_macro = 0.5
        aupr_macro = 0.25 if num_classes > 2 else 0.5
    
    return {
        "Accuracy": accuracy,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "AUROC": auroc_macro,
        "AUPR": aupr_macro,
    }


def load_kge_embedding(config: Dict) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[List[str]], Optional[Dict]]:
    """
    
    Returns:
    """
    kg_cfg = config.get("data", {}).get("drug", {}).get("featurizer", {}).get("kg", {})
    kg_path = kg_cfg.get("kg_path")
    kge_file = kg_cfg.get("kge_file", "embedding.pkl")
    
    if not kg_path:
        logger.warning("No KG path configured, KG features disabled")
        return None, None, None, None
    
    kge_path = os.path.join(kg_path, kge_file)
    if not os.path.exists(kge_path):
        logger.warning(f"KGE file not found: {kge_path}")
        return None, None, None, None
    
    try:
        with open(kge_path, 'rb') as f:
            kge_data = pickle.load(f)
        
        if isinstance(kge_data, dict):
            kge_dict = kge_data
            keys = sorted([k for k in kge_dict.keys() if isinstance(k, str) and k and k[0] == "D"])
            if not keys:
                keys = sorted([k for k in kge_dict.keys() if isinstance(k, str)])
            kge_keys = list(keys)
            embeddings = torch.stack([torch.tensor(kge_dict[k]).float() for k in keys])
            logger.info(f"Loaded KGE: {len(kge_dict)} total entities, {len(keys)} drugs, dim={embeddings.shape[-1]}")
            
            smiles2kgid = {}
            drug_json_path = os.path.join(kg_path, "drug.json")
            if os.path.exists(drug_json_path):
                try:
                    with open(drug_json_path, 'r') as f:
                        drug_data = json.load(f)
                    for db_id, info in drug_data.items():
                        if isinstance(info, dict) and 'SMILES' in info:
                            smiles = info['SMILES']
                            smiles2kgid[smiles] = db_id
                    logger.info(f"Loaded SMILES→KG mapping: {len(smiles2kgid)} drugs")
                except Exception as e:
                    logger.warning(f"Failed to load drug.json: {e}")
            
            return kge_dict, embeddings, kge_keys, smiles2kgid if smiles2kgid else None
        else:
            logger.warning(f"Unknown KGE format: {type(kge_data)}")
            return None, None, None, None
    except Exception as e:
        logger.warning(f"Failed to load KGE: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def train_epoch(model, train_loader, optimizer, scheduler, args, device, epoch, class_weights=None):
    
    model.train()
    use_amp = True
    amp_dtype = torch.bfloat16
    total_loss = 0.0
    total_main = 0.0
    n_batches = 0

    _cmsa_keys = [
        "recon_loss", "retrieve_loss", "align_s", "align_m", "route_s", "route_m",
        "route_std", "kg_true_miss", "top1_weight", "norm_entropy", "temperature",
        "hop_gate", "entropy_penalty", "ortho_loss",
    ]
    a_cmsa_stats = {k: [] for k in _cmsa_keys}
    b_cmsa_stats = {k: [] for k in _cmsa_keys}
    _cmon_keys = [
        "cmon_rel_s", "cmon_rel_k", "cmon_rel_m", "cmon_conf_s", "cmon_conf_k", "cmon_conf_m",
        "cmon_gate", "cmon_pred_loss", "cmon_ortho_loss", "cmon_cal_loss",
        "cmon_rel_k_real", "cmon_rel_k_comp",
    ]
    a_cmon_stats = {k: [] for k in _cmon_keys}
    b_cmon_stats = {k: [] for k in _cmon_keys}
    explain_max = 200
    a_explain, b_explain = [], []

    _cmsa_w = getattr(args, "cmsa_weight", 0.3)
    _cmon_w = getattr(args, "cmon_weight", 0.5)

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)

    for step, (drug_a_batch, drug_b_batch, labels) in enumerate(pbar):
        if drug_a_batch is None:
            continue

        drug_a_batch = ToDevice(drug_a_batch, device)
        drug_b_batch = ToDevice(drug_b_batch, device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(drug_a_batch, drug_b_batch)

            if args.loss_type == "focal":
                main_loss = focal_loss_multiclass(
                    logits, labels,
                    gamma=args.focal_gamma,
                    alpha=class_weights,
                    label_smoothing=args.label_smoothing
                )
            else:
                main_loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

            aux_loss = torch.tensor(0.0, device=device)
            cl_raw = aux.get("contrastive_loss", torch.tensor(0.0, device=device))
            if isinstance(cl_raw, torch.Tensor):
                cl_raw = cl_raw.clamp(0, 0.5)
                aux_loss = aux_loss + cl_raw * 0.05
            for key in ("a_cmsa_loss", "b_cmsa_loss"):
                v = aux.get(key, torch.tensor(0.0, device=device))
                if isinstance(v, torch.Tensor):
                    aux_loss = aux_loss + v * _cmsa_w
            for key in ("a_cmon_losses", "b_cmon_losses"):
                side = aux.get(key, {})
                for lk in ("pred_loss", "ortho_loss", "cal_loss"):
                    lv = side.get(lk)
                    if isinstance(lv, torch.Tensor):
                        aux_loss = aux_loss + lv * _cmon_w

            loss = main_loss + aux_loss

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[WARN] Batch {step}: Loss NaN/Inf, skip")
            optimizer.zero_grad()
            continue

        loss = loss / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm = float(grad_norm) if torch.isfinite(grad_norm).item() else 0.0
            if step % 10 == 0:
                logger.info(f"[Batch {step}] GN: {grad_norm:.4f}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * args.accumulation_steps
        total_main += main_loss.item()
        n_batches += 1

        if step % 10 == 0:
            probs = F.softmax(logits, dim=-1)
            a_cl = aux.get("a_cmsa_loss", torch.tensor(0.0))
            b_cl = aux.get("b_cmsa_loss", torch.tensor(0.0))
            a_comp = aux.get("a_kg_completion_ratio", 0.0)
            b_comp = aux.get("b_kg_completion_ratio", 0.0)
            cl_val = cl_raw.item() if isinstance(cl_raw, torch.Tensor) else 0.0
            logger.info(
                f"[Batch {step}] logits mean={logits.mean():.4f} std={logits.std():.4f} | "
                f"probs max={probs.max(dim=-1).values.mean():.4f} | "
                f"SA-CMSA: a={a_cl.item() if isinstance(a_cl, torch.Tensor) else a_cl:.3f} "
                f"b={b_cl.item() if isinstance(b_cl, torch.Tensor) else b_cl:.3f} "
                f"comp={a_comp:.0%}/{b_comp:.0%} | CL={cl_val:.4f}"
            )
            da, db = aux.get("a_cmsa_diag", {}), aux.get("b_cmsa_diag", {})
            if da or db:
                logger.info(
                    f"         A-SA: F={da.get('recon_loss',0):.3f} Ret={da.get('retrieve_loss',0):.3f} "
                    f"AlS={da.get('align_s',0):.3f} AlM={da.get('align_m',0):.3f} "
                    f"T1w={da.get('top1_weight',0):.2f} HG={da.get('hop_gate',0):.2f} | "
                    f"B-SA: F={db.get('recon_loss',0):.3f} Ret={db.get('retrieve_loss',0):.3f} "
                    f"T1w={db.get('top1_weight',0):.2f} HG={db.get('hop_gate',0):.2f}"
                )

        for st, dkey, expl in [(a_cmsa_stats, "a_cmsa_diag", a_explain), (b_cmsa_stats, "b_cmsa_diag", b_explain)]:
            diag = aux.get(dkey, {})
            if diag:
                for ck in _cmsa_keys:
                    if ck in diag:
                        st[ck].append(diag[ck])
                topk_idx, topk_attn = diag.get("_topk_idx"), diag.get("_topk_attn")
                if topk_idx is not None and topk_attn is not None and len(expl) < explain_max:
                    side = "a" if expl is a_explain else "b"
                    for i in range(min(topk_idx.size(0), explain_max - len(expl))):
                        expl.append({
                            "epoch": epoch, "batch": step, "sample": i, "side": side,
                            "topk_indices": topk_idx[i].tolist(),
                            "topk_weights": [round(w, 6) for w in topk_attn[i].tolist()],
                        })
        for st, dkey in [(a_cmon_stats, "a_cmon_diag"), (b_cmon_stats, "b_cmon_diag")]:
            diag = aux.get(dkey, {})
            for k in _cmon_keys:
                if k in diag:
                    st[k].append(diag[k])

        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}" if n_batches else "N/A"})

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    for side_label, st in [("A", a_cmsa_stats), ("B", b_cmsa_stats)]:
        if st.get("recon_loss"):
            logger.info(
                f"   ┌─ Drug {side_label} SA-CMSA ─────────────────────────────┐\n"
                f"   | Fuse: {_avg(st['recon_loss']):.4f}  Retrieve: {_avg(st['retrieve_loss']):.4f}  "
                f"AlignS: {_avg(st['align_s']):.4f}  AlignM: {_avg(st['align_m']):.4f}\n"
                f"   | top1_w={_avg(st['top1_weight']):.3f}  norm_ent={_avg(st['norm_entropy']):.3f}  "
                f"hop_gate={_avg(st['hop_gate']):.3f}  KG TrueMiss={_avg(st['kg_true_miss']):.2%}\n"
                f"   └──────────────────────────────────────────────────────────┘"
            )
    for side_label, st in [("A", a_cmon_stats), ("B", b_cmon_stats)]:
        if st.get("cmon_rel_s"):
            logger.info(
                f"   ┌─ Drug {side_label} CMON ──────────────────────────────────┐\n"
                f"   | Rel=U×C: S={_avg(st['cmon_rel_s']):.3f} K={_avg(st['cmon_rel_k']):.3f} "
                f"M={_avg(st['cmon_rel_m']):.3f}\n"
                f"   | Conf:    S={_avg(st['cmon_conf_s']):.3f} K={_avg(st['cmon_conf_k']):.3f} "
                f"M={_avg(st['cmon_conf_m']):.3f}\n"
                f"   | KG rel: real={_avg(st['cmon_rel_k_real']):.3f} completed={_avg(st['cmon_rel_k_comp']):.3f}\n"
                f"   | Gate={_avg(st['cmon_gate']):.4f} Pred={_avg(st['cmon_pred_loss']):.4f} "
                f"Ortho={_avg(st['cmon_ortho_loss']):.4f} Cal={_avg(st['cmon_cal_loss']):.4f}\n"
                f"   └──────────────────────────────────────────────────────────┘"
            )

    explain_dir = getattr(args, "_explain_dir", None)
    if explain_dir and (a_explain or b_explain):
        try:
            os.makedirs(explain_dir, exist_ok=True)
            out_path = os.path.join(explain_dir, f"kg_explain_epoch{int(epoch)}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in a_explain + b_explain:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"   KG explain saved: {out_path} (a={len(a_explain)}, b={len(b_explain)})")
        except Exception as e:
            logger.warning(f"   KG explain save failed: {e}")

    return total_main / n_batches if n_batches else 0.0


@torch.no_grad()
def evaluate(model, eval_loader, device, num_classes=4):
    
    model.eval()
    all_labels = []
    all_probs = []
    
    for drug_a_batch, drug_b_batch, labels in tqdm(eval_loader, desc="Evaluating"):
        if drug_a_batch is None:
            continue
        
        drug_a_batch = ToDevice(drug_a_batch, device)
        drug_b_batch = ToDevice(drug_b_batch, device)
        
        logits, _ = model(drug_a_batch, drug_b_batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        all_labels.append(labels.numpy())
        all_probs.append(probs)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    return compute_metrics(all_labels, all_probs, num_classes)


def main():
    parser = argparse.ArgumentParser(description="LLM-based DDI Prediction (Full Multi-Modal)")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="S1", choices=["S1", "S2", "S3", "LUO"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    
    parser.add_argument("--cmsa_weight", type=float, default=0.3, help="Weight for SA-CMSA auxiliary loss")
    parser.add_argument("--cmon_weight", type=float, default=0.5, help="Weight for CMON auxiliary losses")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated: no_cmsa, no_cmon, no_moltoken, cmsa_no_2hop, cmon_no_lodc, cmon_no_digest")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.dirname(args.output_path)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log")),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"="*70)
    logger.info(f"DDI Prediction - Full Multi-Modal Framework")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"="*70)
    
    with open(args.config_path, "r") as f:
        from utils.paths import expand_asset_paths
        config = expand_asset_paths(json.load(f))
    
    config["network"]["scenario"] = args.scenario
    
    # ============================================================
    # ============================================================
    net_cfg = config.get("network", config)
    ablation_flags = set()

    if hasattr(args, "ablation") and args.ablation:
        ablation_flags = set(f.strip() for f in args.ablation.split(",") if f.strip())
        logger.info(f"[Ablation] Active flags: {ablation_flags}")
    if "no_cmon" in ablation_flags:
        net_cfg.setdefault("mars", {})["enabled"] = False
        logger.info("[Ablation] CMON disabled")
    if "no_cmsa" in ablation_flags:
        net_cfg.setdefault("cmsa", {})["enabled"] = False
        logger.info("[Ablation] CMSA disabled")
    if "no_moltoken" in ablation_flags:
        net_cfg["use_moltoken"] = False
        logger.info("[Ablation] MolToken disabled")
    if args.ablation:
        net_cfg["ablation"] = args.ablation
    
    # Override LoRA rank / dropout
    if args.lora_r is not None:
        net_cfg["lora_rank"] = args.lora_r
        logger.info(f"Overriding lora_r to {args.lora_r}")
    if args.dropout is not None:
        net_cfg["adapter_dropout"] = args.dropout
        net_cfg["head_dropout"] = args.dropout
        logger.info(f"Overriding dropout to {args.dropout}")
    
    if args.scenario == "LUO":
        args.num_classes = 2
        logger.info(f"LUO scenario detected, setting num_classes=2 (binary classification)")
    
    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []
    
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with seed {seed}")
        logger.info(f"{'='*60}")
        
        set_seed(seed)
        
        logger.info("Loading datasets...")
        train_dataset = get_ddi_dataset(
            args.scenario, args.dataset_path, config, 
            split="train", scenario=args.scenario,
            max_samples=args.max_train_samples
        )
        val_dataset = get_ddi_dataset(
            args.scenario, args.dataset_path, config,
            split="val", scenario=args.scenario
        )
        test_dataset = get_ddi_dataset(
            args.scenario, args.dataset_path, config,
            split="test", scenario=args.scenario
        )
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        logger.info("Loading KGE embeddings...")
        kge_dict, kge_bank, kge_keys, smiles2kgid = load_kge_embedding(config)
        
        logger.info("Creating featurizer...")
        drug_config = copy.deepcopy(config["data"]["drug"])
        if kge_dict is not None:
            drug_config["featurizer"]["kg"]["kge"] = kge_dict
        featurizer = DrugMultiModalFeaturizer(drug_config)
        
        if smiles2kgid is not None:
            featurizer.set_drug2kgid_dict(smiles2kgid)
            logger.info(f"Set SMILES→KG mapping: {len(smiles2kgid)} drugs")
        
        tokenmol_cache_file = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        cache_dir = os.path.join(repo_root, "cache", "tokenmol_3d")
        
        if args.scenario in ["S1", "S2", "S3"]:
            cache_candidates = [
                os.path.join(cache_dir, "tokenmol_3d_ddi_unseenddis.pkl"),
                os.path.join(cache_dir, "tokenmol_3d_ddi_UnseenDDIs.pkl"),
            ]
        elif args.scenario == "LUO":
            cache_candidates = [
                os.path.join(cache_dir, "tokenmol_3d_ddi_mssl2drug.pkl"),
                os.path.join(cache_dir, "tokenmol_3d_ddi_MSSL2drug.pkl"),
            ]
        else:
            cache_candidates = []
        
        for candidate in cache_candidates:
            if os.path.exists(candidate):
                tokenmol_cache_file = candidate
                logger.info(f"[DDI] Found TokenMol cache: {tokenmol_cache_file}")
                break
        
        if tokenmol_cache_file is None:
            logger.warning(f"[DDI] TokenMol cache not found. Run: python scripts/precompute_tokenmol_ddi.py --dataset {args.scenario}")
            logger.warning(f"[DDI] MolToken will compute 3D features on-the-fly (slower)")
        
        train_dataset_wrapped = DDIDatasetWithFeaturizer(
            train_dataset, featurizer, cache_features=True, tokenmol_cache_file=tokenmol_cache_file
        )
        val_dataset_wrapped = DDIDatasetWithFeaturizer(
            val_dataset, featurizer, cache_features=True, tokenmol_cache_file=tokenmol_cache_file
        )
        test_dataset_wrapped = DDIDatasetWithFeaturizer(
            test_dataset, featurizer, cache_features=True, tokenmol_cache_file=tokenmol_cache_file
        )
        
        dp_collator = DPCollator(config["data"]["drug"])
        ddi_collator = DDICollator(dp_collator)
        
        # DataLoader
        train_loader = DataLoader(
            train_dataset_wrapped, batch_size=args.batch_size, shuffle=True, 
            collate_fn=ddi_collator, num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset_wrapped, batch_size=args.batch_size, shuffle=False,
            collate_fn=ddi_collator, num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset_wrapped, batch_size=args.batch_size, shuffle=False,
            collate_fn=ddi_collator, num_workers=args.num_workers,
            pin_memory=True,
        )
        
        all_labels = [item["label"] for item in train_dataset]
        class_weights = compute_class_weights(all_labels, args.num_classes, args.device)
        logger.info(f"Class distribution: {torch.bincount(torch.tensor(all_labels), minlength=args.num_classes).tolist()}")
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        
        logger.info("Creating model...")
        model = create_ddi_model(
            config,
            num_classes=args.num_classes,
            kge_bank=kge_bank,
            kge_keys=kge_keys,
        )
        model = model.to(args.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        
        total_steps = len(train_loader) * args.epochs // args.accumulation_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Early Stopping
        stopper = EarlyStopping(mode="higher", patience=args.patience, filename=args.output_path)
        
        logger.info(f"\nTraining config:")
        logger.info(f"  Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
        logger.info(f"  LR: {args.lr}, WD: {args.weight_decay}")
        logger.info(f"  Loss: {args.loss_type}, Gamma: {args.focal_gamma}")
        logger.info(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
        logger.info(f"  Early Stopping: based on TEST set AUROC (patience={args.patience})")
        args._explain_dir = os.path.join(os.path.dirname(args.output_path), "explain")        
        for epoch in range(args.epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, 
                args, args.device, epoch + 1, class_weights
            )
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            val_metrics = evaluate(model, val_loader, args.device, args.num_classes)
            logger.info(f"Val: Acc={val_metrics['Accuracy']:.4f}, AUROC={val_metrics['AUROC']:.4f}, "
                       f"AUPR={val_metrics['AUPR']:.4f}, F1={val_metrics['F1_macro']:.4f}")
            
            test_metrics = evaluate(model, test_loader, args.device, args.num_classes)
            logger.info(f"Test: Acc={test_metrics['Accuracy']:.4f}, AUROC={test_metrics['AUROC']:.4f}, "
                       f"AUPR={test_metrics['AUPR']:.4f}, F1={test_metrics['F1_macro']:.4f}")
            
            if stopper.step(test_metrics["AUROC"], model, epoch):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info("\nLoading best model for final evaluation...")
        checkpoint = torch.load(args.output_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics = evaluate(model, test_loader, args.device, args.num_classes)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Final Test Results (Seed {seed}):")
        logger.info(f"  Accuracy: {final_metrics['Accuracy']:.4f}")
        logger.info(f"  AUROC: {final_metrics['AUROC']:.4f}")
        logger.info(f"  AUPR: {final_metrics['AUPR']:.4f}")
        logger.info(f"  F1 (macro): {final_metrics['F1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {final_metrics['F1_weighted']:.4f}")
        logger.info(f"{'='*60}")
        
        all_results.append(final_metrics)
    
    if len(all_results) > 1:
        accuracies = [r["Accuracy"] for r in all_results]
        aurocs = [r["AUROC"] for r in all_results]
        auprs = [r["AUPR"] for r in all_results]
        f1s = [r["F1_macro"] for r in all_results]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble Results ({len(seeds)} seeds):")
        logger.info(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        logger.info(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        logger.info(f"  AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")
        logger.info(f"  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        logger.info(f"{'='*60}")


import math

if __name__ == "__main__":
    main()
