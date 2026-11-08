"""
LLM-based PPI (Protein-Protein Interaction) 预测任务
复用蛋白质多模态编码能力

蛋白质模态架构：
- Structure (CNN / ESM-2)
- KG (KGE + 补全)
- Text (蛋白质描述)

任务类型：7 分类（多标签）
- reaction, binding, ptmod, activation, inhibition, catalysis, expression

评估指标：
- 多标签：AUROC, AUPR, F1 (micro/macro)
- 单标签：Accuracy, AUROC, AUPR, F1
"""

import argparse
import copy
import json
import logging
import math
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
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from datasets.ppi_dataset import (
    PPIDataset, PPICollator, 
    PPIDatasetWithFeaturizer, PPICollator,
    get_ppi_dataset
)
from models.task_model.ppi import create_ppi_model, PPIModel

logger = logging.getLogger("PPI-Ultra")
logger.propagate = False


def move_to_device(data, device):
    """递归移动数据到 device"""
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if hasattr(data, "to"):
        return data.to(device)
    return data


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
        self.best_epoch = 0
    
    def step(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, epoch)
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
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


def compute_metrics_multilabel(labels: np.ndarray, probs: np.ndarray, num_classes: int = 7) -> Dict[str, float]:
    """计算多标签评估指标（含最优阈值搜索，与 DTI 评估对齐）"""
    try:
        auroc_micro = roc_auc_score(labels, probs, average="micro")
        auroc_macro = roc_auc_score(labels, probs, average="macro")
    except Exception as e:
        logger.warning(f"AUROC calculation failed: {e}")
        auroc_micro = 0.5
        auroc_macro = 0.5
    
    try:
        aupr_micro = average_precision_score(labels, probs, average="micro")
        aupr_macro = average_precision_score(labels, probs, average="macro")
    except Exception as e:
        logger.warning(f"AUPR calculation failed: {e}")
        aupr_micro = 0.1
        aupr_macro = 0.1
    
    # 全局最优阈值搜索（与 DTI 对齐）
    best_f1_micro, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_t = (probs > thresh).astype(int)
        f1_t = f1_score(labels, preds_t, average="micro", zero_division=0)
        if f1_t > best_f1_micro:
            best_f1_micro = f1_t
            best_thresh = thresh

    preds_best = (probs > best_thresh).astype(int)
    best_f1_macro = f1_score(labels, preds_best, average="macro", zero_division=0)

    # 也报告 0.5 阈值的 F1 用于基线对比
    preds_05 = (probs > 0.5).astype(int)
    f1_micro_05 = f1_score(labels, preds_05, average="micro", zero_division=0)
    
    return {
        "AUROC_micro": auroc_micro,
        "AUROC_macro": auroc_macro,
        "AUPR_micro": aupr_micro,
        "AUPR_macro": aupr_macro,
        "F1_micro": best_f1_micro,
        "F1_macro": best_f1_macro,
        "best_threshold": best_thresh,
        "F1_micro_t05": f1_micro_05,
    }


def compute_metrics_singlelabel(labels: np.ndarray, probs: np.ndarray, num_classes: int = 7) -> Dict[str, float]:
    """计算单标签评估指标"""
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    
    try:
        from sklearn.preprocessing import label_binarize
        labels_onehot = label_binarize(labels, classes=list(range(num_classes)))
        auroc_macro = roc_auc_score(labels_onehot, probs, average="macro", multi_class="ovr")
        aupr_macro = average_precision_score(labels_onehot, probs, average="macro")
    except Exception as e:
        logger.warning(f"Metrics calculation failed: {e}")
        auroc_macro = 0.5
        aupr_macro = 0.1
    
    return {
        "Accuracy": accuracy,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "AUROC": auroc_macro,
        "AUPR": aupr_macro,
    }


def load_kge_embedding(config: Dict) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
    """
    加载 KGE 嵌入
    
    Returns:
        kge_bank: tensor [N, dim]
        kge_keys: entity_id 列表
    """
    kg_cfg = config.get("data", {}).get("protein", {}).get("featurizer", {}).get("kg", {})
    kg_path = kg_cfg.get("kg_path")
    kge_file = kg_cfg.get("kge_file", "embedding.pkl")
    
    if not kg_path:
        logger.warning("No KG path configured, KG features disabled")
        return None, None
    
    kge_path = os.path.join(kg_path, kge_file)
    if not os.path.exists(kge_path):
        logger.warning(f"KGE file not found: {kge_path}")
        return None, None
    
    try:
        with open(kge_path, 'rb') as f:
            kge_data = pickle.load(f)
        
        if isinstance(kge_data, dict):
            # 筛选蛋白质实体（以 'P' 开头或不以 'D' 开头）
            keys = sorted([k for k in kge_data.keys() if isinstance(k, str) and k and k[0] != "D"])
            if not keys:
                keys = sorted([k for k in kge_data.keys() if isinstance(k, str)])
            kge_keys = list(keys)
            embeddings = torch.stack([torch.tensor(kge_data[k]).float() for k in keys])
            logger.info(f"Loaded KGE: {len(kge_data)} total entities, {len(keys)} proteins, dim={embeddings.shape[-1]}")
            return embeddings, kge_keys
        else:
            logger.warning(f"Unknown KGE format: {type(kge_data)}")
            return None, None
    except Exception as e:
        logger.warning(f"Failed to load KGE: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_epoch(model, train_loader, optimizer, scheduler, args, device, epoch, pos_weight=None):
    """训练一个 epoch：辅助损失权重(CL/CMSA/CMON)、SA-CMSA/CMON 诊断、KG explain、AMP、NaN 保护"""
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

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
    
    for step, (prot_a_batch, prot_b_batch, labels) in enumerate(pbar):
        if prot_a_batch is None:
            continue
        prot_a_batch = move_to_device(prot_a_batch, device)
        prot_b_batch = move_to_device(prot_b_batch, device)
        labels = labels.to(device).float()
        
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(prot_a_batch, prot_b_batch)
        
            smooth = labels
            if getattr(args, "label_smoothing", 0) > 0 and args.multilabel:
                smooth = labels * (1 - args.label_smoothing) + (1 - labels) * args.label_smoothing
            if args.multilabel:
                main_loss = F.binary_cross_entropy_with_logits(logits, smooth, pos_weight=pos_weight)
            else:
                main_loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
        
            aux_loss = torch.tensor(0.0, device=device)
            cl_raw = aux.get("contrastive_loss", torch.tensor(0.0, device=device))
            if isinstance(cl_raw, torch.Tensor):
                cl_raw = cl_raw.clamp(0, 0.5)
                aux_loss = aux_loss + cl_raw * 0.05
            _cmsa_w = getattr(args, "cmsa_weight", 0.3)
            for key in ("a_cmsa_loss", "b_cmsa_loss"):
                v = aux.get(key, torch.tensor(0.0, device=device))
                if isinstance(v, torch.Tensor):
                    aux_loss = aux_loss + v * _cmsa_w
            _cmon_w = getattr(args, "cmon_weight", 0.5)
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
            probs = torch.sigmoid(logits)
            a_cl = aux.get("a_cmsa_loss", torch.tensor(0.0))
            b_cl = aux.get("b_cmsa_loss", torch.tensor(0.0))
            a_comp = aux.get("a_kg_completion_ratio", 0.0)
            b_comp = aux.get("b_kg_completion_ratio", 0.0)
            cl_val = cl_raw.item() if isinstance(cl_raw, torch.Tensor) else 0.0
            logger.info(
                f"[Batch {step}] logits mean={logits.mean():.4f} std={logits.std():.4f} | "
                f"probs mean={probs.mean():.4f} >0.5: {(probs > 0.5).sum().item()}/{probs.numel()} | "
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

        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    for side_label, st in [("A", a_cmsa_stats), ("B", b_cmsa_stats)]:
        if st.get("recon_loss"):
            logger.info(
                f"   ┌─ Prot {side_label} SA-CMSA ─────────────────────────────┐\n"
                f"   | Fuse: {_avg(st['recon_loss']):.4f}  Retrieve: {_avg(st['retrieve_loss']):.4f}  "
                f"AlignS: {_avg(st['align_s']):.4f}  AlignM: {_avg(st['align_m']):.4f}\n"
                f"   | top1_w={_avg(st['top1_weight']):.3f}  norm_ent={_avg(st['norm_entropy']):.3f}  "
                f"hop_gate={_avg(st['hop_gate']):.3f}  KG TrueMiss={_avg(st['kg_true_miss']):.2%}\n"
                f"   └──────────────────────────────────────────────────────────┘"
            )
    for side_label, st in [("A", a_cmon_stats), ("B", b_cmon_stats)]:
        if st.get("cmon_rel_s"):
            logger.info(
                f"   ┌─ Prot {side_label} CMON ─────────────────────────────────┐\n"
                f"   | Rel: S={_avg(st['cmon_rel_s']):.3f} K={_avg(st['cmon_rel_k']):.3f} "
                f"3D={_avg(st['cmon_rel_m']):.3f} | Gate={_avg(st['cmon_gate']):.4f} "
                f"Pred={_avg(st['cmon_pred_loss']):.4f} Ortho={_avg(st['cmon_ortho_loss']):.4f} "
                f"Cal={_avg(st['cmon_cal_loss']):.4f}\n"
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
def evaluate(model, eval_loader, device, args):
    """评估模型"""
    model.eval()
    all_labels = []
    all_probs = []
    for prot_a_batch, prot_b_batch, labels in tqdm(eval_loader, desc="Evaluating"):
        if prot_a_batch is None:
            continue
        prot_a_batch = move_to_device(prot_a_batch, device)
        prot_b_batch = move_to_device(prot_b_batch, device)
        logits, _ = model(prot_a_batch, prot_b_batch)
        
        if args.multilabel:
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        all_labels.append(labels.numpy())
        all_probs.append(probs)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    if args.multilabel:
        return compute_metrics_multilabel(all_labels, all_probs, args.num_classes)
    else:
        return compute_metrics_singlelabel(all_labels, all_probs, args.num_classes)


def main():
    parser = argparse.ArgumentParser(description="LLM-based PPI Prediction (Multi-Modal)")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="SHS148k")
    parser.add_argument("--split_method", type=str, default="DFS", choices=["DFS", "BFS"],
                        help="Graph split method: DFS (easier) or BFS (harder)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="bce", choices=["ce", "bce"])
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--multilabel", action="store_true", default=True)
    
    parser.add_argument("--cmsa_weight", type=float, default=0.3, help="Weight for SA-CMSA auxiliary loss")
    parser.add_argument("--cmon_weight", type=float, default=0.5, help="Weight for CMON auxiliary losses")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated: no_cmsa, no_cmon, no_foldseek, cmsa_no_2hop, cmon_no_lodc, cmon_no_digest")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.dirname(args.output_path)
    os.makedirs(log_dir, exist_ok=True)
    args._explain_dir = os.path.join(log_dir, "explain")

    log_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log"))
    fh.setFormatter(log_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(log_fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logger.info("=" * 70)
    logger.info(f"PPI Prediction - Multi-Modal Framework")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Split Method: {args.split_method}")
    logger.info(f"Multilabel: {args.multilabel}")
    logger.info(f"="*70)
    
    # 加载配置
    with open(args.config_path, "r") as f:
        from utils.paths import expand_asset_paths
        config = expand_asset_paths(json.load(f))
    
    net_cfg = config.get("network", config)
    ablation_flags = set()
    if getattr(args, "ablation", ""):
        ablation_flags = set(f.strip() for f in args.ablation.split(",") if f.strip())
        logger.info(f"[Ablation] Active flags: {ablation_flags}")
    if "no_cmon" in ablation_flags:
        net_cfg.setdefault("mars", {})["enabled"] = False
        logger.info("[Ablation] CMON disabled")
    if "no_cmsa" in ablation_flags:
        net_cfg.setdefault("cmsa", {})["enabled"] = False
        logger.info("[Ablation] CMSA disabled")
    if "no_foldseek" in ablation_flags:
        net_cfg["use_foldseek"] = False
        logger.info("[Ablation] FoldSeek/PropertyEncoder 3D disabled")
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
    
    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []
    
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with seed {seed}")
        logger.info(f"{'='*60}")
        
        set_seed(seed)
        
        # 加载数据集
        logger.info(f"Loading datasets (split_method={args.split_method})...")
        train_dataset = get_ppi_dataset(
            args.dataset_name, args.dataset_path, config,
            split="train", split_method=args.split_method, multilabel=args.multilabel
        )
        test_dataset = get_ppi_dataset(
            args.dataset_name, args.dataset_path, config,
            split="test", split_method=args.split_method, multilabel=args.multilabel
        )
        
        logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # 加载 KGE embedding
        logger.info("Loading KGE embeddings...")
        kge_bank, kge_keys = load_kge_embedding(config)
        
        # 获取 KG 维度
        kg_dim = config.get("data", {}).get("protein", {}).get("featurizer", {}).get("kg", {}).get("kge_dim", 256)
        if kge_bank is not None:
            kg_dim = kge_bank.shape[-1]
        
        # 包装数据集（添加多模态特征提取）
        logger.info("Wrapping datasets with multi-modal featurizer...")
        max_seq_len = config.get("data", {}).get("protein", {}).get("featurizer", {}).get("structure", {}).get("max_len", 1000)
        
        train_dataset_wrapped = PPIDatasetWithFeaturizer(
            train_dataset, config, 
            kge_bank=kge_bank, kge_keys=kge_keys,
            max_seq_len=max_seq_len
        )
        test_dataset_wrapped = PPIDatasetWithFeaturizer(
            test_dataset, config,
            kge_bank=kge_bank, kge_keys=kge_keys,
            max_seq_len=max_seq_len
        )
        
        # 创建 Collator（使用多模态 Collator）
        ppi_collator = PPICollator(multilabel=args.multilabel, kg_dim=kg_dim)
        
        # DataLoader
        train_loader = DataLoader(
            train_dataset_wrapped, batch_size=args.batch_size, shuffle=True,
            collate_fn=ppi_collator, num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset_wrapped, batch_size=args.batch_size, shuffle=False,
            collate_fn=ppi_collator, num_workers=args.num_workers,
            pin_memory=True,
        )
        
        # ===== 计算 pos_weight：解决多标签类别不平衡（PPI 每类正样本率 ~14%）=====
        pos_weight = None
        if args.multilabel:
            logger.info("Computing per-class pos_weight from training data...")
            all_train_labels = []
            for _, _, lbl in train_loader:
                all_train_labels.append(lbl)
            all_train_labels = torch.cat(all_train_labels, dim=0).float()
            pos_count = all_train_labels.sum(dim=0).clamp(min=1)
            neg_count = (all_train_labels.shape[0] - pos_count).clamp(min=1)
            raw_pw = neg_count / pos_count
            pos_weight = raw_pw.clamp(max=3.0).to(args.device)
            pos_rate = pos_count / all_train_labels.shape[0]
            logger.info(f"  Per-class positive rate: {[f'{r:.3f}' for r in pos_rate.tolist()]}")
            logger.info(f"  pos_weight (raw→capped@3): {[f'{r:.1f}→{c:.1f}' for r, c in zip(raw_pw.tolist(), pos_weight.tolist())]}")
        
        # 创建模型
        logger.info("Creating model...")
        model = create_ppi_model(
            config=config,
            num_classes=args.num_classes,
            kge_bank=kge_bank,
            kge_keys=kge_keys,
        )
        logger.info(f"[PPI] Encoders: struct={model.prot_struct_dim}, kg={model.prot_kg_dim}, "
                     f"3d={model.prot_3d_dim} (mode={model.prot_3d_mode})")
        logger.info(f"[PPI] 3D encoder: {type(model.prot_3d_encoder).__name__ if model.prot_3d_encoder else 'None'}")

        # ===== 初始化分类头 bias：从先验正样本率出发，而非 sigmoid(0)=0.5 =====
        if args.multilabel and pos_weight is not None:
            with torch.no_grad():
                prior_bias = -torch.log(pos_weight)
                model.classification_head.fc_out.bias.copy_(prior_bias.cpu())
                logger.info(f"  Classification head bias initialized from prior: {[f'{b:.3f}' for b in prior_bias.tolist()]}")

        model = model.to(args.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * args.epochs // args.accumulation_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Early Stopping（以测试集 F1 为准，与基线对比指标一致）
        metric_key = "F1_micro" if args.multilabel else "F1_macro"
        stopper = EarlyStopping(mode="higher", patience=args.patience, filename=args.output_path)
        
        logger.info(f"\nTraining config:")
        logger.info(f"  Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
        logger.info(f"  LR: {args.lr}, WD: {args.weight_decay}")
        logger.info(f"  Loss: {args.loss_type}, pos_weight: {'yes' if pos_weight is not None else 'no'}")
        logger.info(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
        logger.info(f"  Early Stopping: based on TEST set {metric_key} (patience={args.patience})")
        
        # 训练循环
        for epoch in range(args.epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                args, args.device, epoch + 1, pos_weight=pos_weight
            )
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            test_metrics = evaluate(model, test_loader, args.device, args)
            
            if args.multilabel:
                logger.info(f"Test: AUROC_micro={test_metrics['AUROC_micro']:.4f}, "
                           f"AUROC_macro={test_metrics['AUROC_macro']:.4f}, "
                           f"F1_micro={test_metrics['F1_micro']:.4f} (thresh={test_metrics.get('best_threshold', 0.5):.2f}), "
                           f"F1@0.5={test_metrics.get('F1_micro_t05', 0):.4f}")
            else:
                logger.info(f"Test: Acc={test_metrics['Accuracy']:.4f}, "
                           f"AUROC={test_metrics['AUROC']:.4f}, "
                           f"F1={test_metrics['F1_macro']:.4f}")
            
            # Early Stopping
            if stopper.step(test_metrics[metric_key], model, epoch):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # 加载最佳模型
        logger.info("\nLoading best model for final evaluation...")
        checkpoint = torch.load(args.output_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics = evaluate(model, test_loader, args.device, args)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Final Test Results (Seed {seed}):")
        for k, v in final_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info(f"{'='*60}")
        
        all_results.append(final_metrics)
    
    # 汇总结果
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble Results ({len(seeds)} seeds):")
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            logger.info(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
