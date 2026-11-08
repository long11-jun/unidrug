import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from datasets.dp_dataset import SUPPORTED_DP_DATASETS, Task
from utils import DPCollator, AverageMeter, EarlyStopping, ToDevice
from models.task_model.dp import DPModel

REGRESSION_DATASETS = {"ESOL", "Lipo", "FreeSolv", "qm7", "qm8", "qm9"}

def is_regression_task(dataset_name: str) -> bool:
    
    return dataset_name in REGRESSION_DATASETS

logger = logging.getLogger(__name__)

def grad_norm_l2(parameters) -> float:
    
    total = 0.0
    for p in parameters:
        if p is None or p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            return float("nan")
        total += g.float().pow(2).sum().item()
    return float(total ** 0.5)

def grad_norm_groups(model, group_patterns: Dict[str, List[str]]) -> Dict[str, float]:
    """
    group_patterns: {"llm":["llm","lora"], "struct":["struct","graph"], ...}
    """
    sums = {k: 0.0 for k in group_patterns.keys()}
    sums["other"] = 0.0
    for name, p in model.named_parameters():
        if p is None or p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            continue
        g2 = g.float().pow(2).sum().item()
        hit = False
        low = name.lower()
        for gname, pats in group_patterns.items():
            for pat in pats:
                if pat in low:
                    sums[gname] += g2
                    hit = True
                    break
            if hit:
                break
        if not hit:
            sums["other"] += g2
    return {k: float(v ** 0.5) for k, v in sums.items()}

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def setup_distributed(args):
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    args.distributed = args.world_size > 1
    if not args.distributed:
        return

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

def all_gather_concat(t: torch.Tensor) -> torch.Tensor:
    
    if not is_dist():
        return t
    world = get_world_size()
    device = t.device
    local_n = torch.tensor([t.size(0)], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(sizes, local_n)
    sizes = [int(x.item()) for x in sizes]
    max_n = max(sizes)
    if t.size(0) < max_n:
        pad = torch.zeros((max_n - t.size(0),) + t.shape[1:], device=device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t
    gathered = [torch.empty_like(t_pad) for _ in range(world)]
    dist.all_gather(gathered, t_pad)
    out = []
    for g, n in zip(gathered, sizes):
        out.append(g[:n].cpu())
    return torch.cat(out, dim=0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args, cfg):
    dataset = SUPPORTED_DP_DATASETS[args.dataset](
        args.dataset_path,
        cfg["data"],
        args.dataset_name,
        split_strategy=getattr(args, "split_strategy", "scaffold"),
    )
    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    test_dataset = dataset.index_select(dataset.test_index)

    collator = DPCollator(cfg["data"]["drug"])
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    if is_main_process():
        logger.info(f" Data: Train={len(train_dataset)} Val={len(val_dataset)} Test={len(test_dataset)} | world={get_world_size()} bs_per_gpu={args.batch_size}")
    return dataset, train_loader, val_loader, test_loader


def prepare_labels(raw_labels: torch.Tensor, is_regression: bool = False):
    """
    
    """
    if is_regression:
        mask = ~torch.isnan(raw_labels)
        targets = torch.where(mask, raw_labels, torch.zeros_like(raw_labels))
    else:
        mask = raw_labels.abs() > 0
        targets = torch.where(raw_labels > 0, torch.ones_like(raw_labels), torch.zeros_like(raw_labels))
    return targets, mask


def masked_bce_with_logits(logits, targets, mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = (loss * mask).sum() / mask.sum().clamp(min=1)
    return masked


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    metrics = {}
    roc_list = []

    if np.isnan(probs).any():
        nan_cnt = int(np.isnan(probs).sum())
        raise ValueError(
            f"[compute_metrics] probs contains NaN ({nan_cnt}/{probs.size}). "
            f"Please locate NaN source in model forward/training step."
        )
    
    for i in range(labels.shape[1]):
        y = labels[:, i]
        mask = np.abs(y) > 0
        if mask.sum() < 2:
            continue
        y_bin = ((y[mask] + 1) / 2).clip(0, 1)
        scores = probs[mask, i]
        
        if len(np.unique(y_bin)) < 2:
            continue
        roc_list.append(roc_auc_score(y_bin, scores))
    metrics["roc_auc"] = float(np.mean(roc_list)) if roc_list else 0.0
    metrics["roc_auc_std"] = float(np.std(roc_list)) if roc_list else 0.0
    metrics["roc_auc_n"] = int(len(roc_list))
    return metrics


def compute_regression_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    
    Args:
    
    Returns:
        metrics: {"rmse": float, "mae": float, "r2": float}
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    
    rmse_list, mae_list, r2_list = [], [], []
    
    for i in range(labels.shape[1]):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid_mask.sum() < 2:
            continue
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        
        if np.std(y_true_valid) > 1e-6:
            r2 = r2_score(y_true_valid, y_pred_valid)
        else:
            r2 = 0.0
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
    
    return {
        "rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "rmse_std": float(np.std(rmse_list)) if rmse_list else 0.0,
        "mae": float(np.mean(mae_list)) if mae_list else 0.0,
        "mae_std": float(np.std(mae_list)) if mae_list else 0.0,
        "r2": float(np.mean(r2_list)) if r2_list else 0.0,
        "r2_std": float(np.std(r2_list)) if r2_list else 0.0,
        "n_tasks": len(rmse_list),
    }


class EMA:
    """
    
    - shadow_weight = decay * shadow_weight + (1 - decay) * model_weight
    
    - Polyak averaging (1992)
    
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class DynamicLossScheduler:
    """
    
    """
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.history = {"train_auc": [], "test_auc": [], "gap": []}
        
        self.base_weights = {
            "main": 1.0,
            "align": 0.15,
            "kg_recon": 0.05,
        }
        
    def update_history(self, train_auc: float, test_auc: float):
        
        self.history["train_auc"].append(train_auc)
        self.history["test_auc"].append(test_auc)
        self.history["gap"].append(train_auc - test_auc)
        
    def get_weights(self, epoch: int) -> Dict[str, float]:
        
        return self.base_weights.copy()


def poly_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1.0, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    
    poly_term = epsilon * (1 - pt)
    
    return bce_loss + poly_term


def confidence_aware_focal_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    fusion_conf: torch.Tensor,
    gamma: float = 2.0,
    conf_boost: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    probs = torch.sigmoid(logits)
    # p_t calculation
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    
    conf_weight = 1.0 + conf_boost * (1.0 - fusion_conf.detach())
    conf_weight = conf_weight.unsqueeze(-1)  # (B, 1)
    
    return focal_weight * conf_weight * bce_loss


def asymmetric_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    gamma_pos: float = 0.0, 
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """
    SOTAAsymmetric Loss for Multi-Label Classification
    
    
    
    """
    num_tasks = logits.size(-1) if logits.dim() > 1 else 1
    is_single_task = (num_tasks <= 2)
    
    if is_single_task:
        logits = logits.clamp(-10, 10)
        min_prob = 0.001
    else:
        min_prob = 1e-8
    
    probs = torch.sigmoid(logits)
    
    # Positive samples
    pos_probs = probs
    pos_loss = targets * torch.log(pos_probs.clamp(min=min_prob))
    if gamma_pos > 0:
        pos_loss = pos_loss * ((1 - pos_probs) ** gamma_pos)
    
    # Negative samples with probability clipping
    neg_probs = 1 - probs
    if clip > 0:
        # Clip high-confidence negatives
        neg_probs = (neg_probs + clip).clamp(max=1)
    neg_loss = (1 - targets) * torch.log(neg_probs.clamp(min=min_prob))
    if gamma_neg > 0:
        neg_loss = neg_loss * (probs ** gamma_neg)
    
    loss = -(pos_loss + neg_loss)
    
    if is_single_task:
        loss = loss.clamp(max=5.0)
    
    return loss


def focal_loss_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
    
    Returns:
    """
    probs = torch.sigmoid(logits)
    
    # p_t = p if y=1, else 1-p
    p_t = torch.where(targets > 0.5, probs, 1 - probs)
    
    # alpha_t = alpha if y=1, else 1-alpha
    alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    # Apply focal weight and alpha
    focal_loss = alpha_t * focal_weight * bce_loss
    
    # Mask and average
    loss = (focal_loss * mask).sum() / (mask.sum() + 1e-6)
    
    return loss.clamp(max=2.0)


def balanced_finetuning_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor,
    tau: float = 0.5,
    lambda_s: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    
    
    
    
    Args:
    
    Returns:
    """
    probs = torch.sigmoid(logits)
    
    confidence = torch.where(targets > 0.5, probs, 1 - probs)
    confidence = confidence.clamp(0.1, 0.9)
    
    confidence_detached = confidence.detach()
    
    token_weight = confidence_detached ** tau
    
    # 3. Sample-level Weighting: s_b = 1 + lambda_s * (1 - min_confidence)
    masked_confidence = confidence_detached * mask + (1 - mask) * 1.0
    min_confidence_per_sample = masked_confidence.min(dim=1, keepdim=True)[0]  # (B, 1)
    
    sample_weight = 1.0 + lambda_s * (1.0 - min_confidence_per_sample)
    
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    bce_loss = bce_loss.clamp(max=5.0)
    
    weighted_loss = bce_loss * token_weight * sample_weight
    
    loss = (weighted_loss * mask).sum() / (mask.sum() + 1e-6)
    
    loss = loss.clamp(max=2.0)
    
    return loss


def kl_divergence_loss(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    
    """
    p = torch.sigmoid(p_logits)
    q = torch.sigmoid(q_logits)
    
    eps = 1e-6
    kl_pq = p * (torch.log(p + eps) - torch.log(q + eps)) + \
            (1 - p) * (torch.log(1 - p + eps) - torch.log(1 - q + eps))
    kl_qp = q * (torch.log(q + eps) - torch.log(p + eps)) + \
            (1 - q) * (torch.log(1 - q + eps) - torch.log(1 - p + eps))
    
    kl_loss = 0.5 * (kl_pq + kl_qp)
    
    return (kl_loss * mask).sum() / (mask.sum() + eps)


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    grad_scaler=None,
    use_amp: bool = False,
    amp_dtype=torch.float16,
    log_interval=10,
    label_smoothing=0.0,
    warm_scheduler=None,
    mixup_alpha=0.0,
    loss_scheduler=None,
    partition_weights_history=None,
    use_uncertainty_weighting=True,
    use_focal_loss=True,
    ema=None,
    pos_weight: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0,
    loss_type: str = "bft",
    conf_calib_weight: float = 0.05,
    preclip_cap: float = 0.0,
    log_grad_groups: bool = False,
    use_rdrop: bool = False,
    accumulation_steps: int = 1,
    explain_dir: Optional[str] = None,
    explain_max_samples: int = 200,
    is_regression: bool = False,
    normalizers: Optional[List] = None,
):
    
    model.train()
    
    _raw_model = model.module if hasattr(model, 'module') else model
    
    loss_meter = AverageMeter()
    main_loss_meter = AverageMeter()
    align_loss_meter = AverageMeter()
    
    all_probs = []
    all_labels = []
    total_batches = len(loader)
    
    epoch_partition_weights = []
    epoch_partition_modal_names = None
    
    uw_weights_log = []

    optimizer.zero_grad(set_to_none=True)

    if getattr(loader, "sampler", None) is not None and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for batch_idx, (drug, label) in enumerate(loader, 1):
        drug = ToDevice(drug, device)
        label = label.to(device).float()
        targets, mask = prepare_labels(label, is_regression=is_regression)
        mask = mask.float()
        with torch.no_grad():
            valid_frac = float(mask.mean().item())
            if is_regression:
                pos_rate = float(targets[mask > 0.5].mean().item()) if mask.sum() > 0 else 0.0
            else:
                pos_rate = float(((targets > 0.5).float() * mask).sum().item() / (mask.sum().item() + 1e-6))
        
        if label_smoothing > 0 and not is_regression:
            smooth = label_smoothing
            targets = targets * (1 - smooth) + 0.5 * smooth
        mixup_params = None
        if mixup_alpha > 0 and model.training:
            lam_struct = np.random.beta(mixup_alpha, mixup_alpha)
            lam_kg = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(label.size(0), device=device)
            mixup_params = {"lam_struct": lam_struct, "lam_kg": lam_kg, "perm": perm}
            targets = lam_struct * targets + (1 - lam_struct) * targets[perm]
            mask = lam_struct * mask + (1 - lam_struct) * mask[perm]


        with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
            output, _ = model(drug, mixup=mixup_params)
            logits = output["classification"]

            if not torch.isfinite(logits).all():
                bad = (~torch.isfinite(logits)).sum().item()
                safe = torch.nan_to_num(logits)
                logger.error(
                    f"[NaNGuard] Non-finite logits detected: {bad}/{logits.numel()} | "
                    f"min={safe.min().item():.4f} max={safe.max().item():.4f} | "
                    f"epoch={epoch} batch_idx={batch_idx}"
                )
                raise FloatingPointError("Non-finite logits detected (NaN/Inf).")
            
            logits2 = None
            if use_rdrop and model.training:
                output2, _ = model(drug, mixup=mixup_params)
                logits2 = output2["classification"]
            
            if output.get("partition_info") is not None:
                pi = output.get("partition_info")
                if isinstance(pi, dict) and pi.get("modal_weights") is not None:
                    modal_w = pi["modal_weights"].detach().cpu()
                    epoch_partition_weights.append(modal_w)
                if epoch_partition_modal_names is None:
                    names = output.get("partition_modal_names", None)
                    if names is not None and isinstance(names, (list, tuple)):
                        epoch_partition_modal_names = [str(x) for x in names]
            
            targets_smooth = targets
            
            # pos_weight[t] ≈ (#neg_t / #pos_t)
            pw = pos_weight

            sample_weights = output.get("sample_weights", None)  # (B,)
            if sample_weights is not None:
                sample_weights = sample_weights.unsqueeze(-1).expand_as(mask)
            
            lt = str(loss_type).lower()
            
            if lt == "mse":
                _mse = F.mse_loss(logits, targets, reduction="none")
                if sample_weights is not None:
                    _mse = _mse * sample_weights
                main_loss = (_mse * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "mae":
                _mae = F.l1_loss(logits, targets, reduction="none")
                if sample_weights is not None:
                    _mae = _mae * sample_weights
                main_loss = (_mae * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "huber":
                _huber = F.smooth_l1_loss(logits, targets, reduction="none", beta=1.0)
                if sample_weights is not None:
                    _huber = _huber * sample_weights
                main_loss = (_huber * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "bce":
                _bce = F.binary_cross_entropy_with_logits(
                    logits, targets_smooth, reduction="none", pos_weight=pw
                )
                _bce = _bce.clamp(max=5.0)
                if sample_weights is not None:
                    _bce = _bce * sample_weights
                main_loss = (_bce * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "focal":
                main_loss = focal_loss_binary(
                    logits, targets_smooth, mask,
                    gamma=2.0, alpha=0.25, pos_weight=pw,
                )
            elif lt == "asl":
                num_tasks = logits.size(1) if logits.dim() > 1 else 1
                if num_tasks > 100:
                    gamma_pos, gamma_neg, clip_val = 0.5, 1.5, 0.03
                elif num_tasks > 20:
                    gamma_pos, gamma_neg, clip_val = 1.0, 2.0, 0.05
                else:
                    gamma_pos, gamma_neg, clip_val = 1.0, 2.5, 0.05
                    
                _asl = asymmetric_loss(
                    logits, targets_smooth,
                    gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip_val,
                )
                if sample_weights is not None:
                    _asl = _asl * sample_weights
                main_loss = (_asl * mask).sum() / (mask.sum() + 1e-6)
            else:
                _bce = F.binary_cross_entropy_with_logits(
                    logits, targets_smooth, reduction="none", pos_weight=pw
                )
                _bce = _bce.clamp(max=5.0)
                if sample_weights is not None:
                    _bce = _bce * sample_weights
                main_loss = (_bce * mask).sum() / (mask.sum() + 1e-6)

            aux_losses = output.get("aux_losses", {})
            
            rdrop_loss = aux_losses.get("rdrop_loss", torch.tensor(0.0, device=device))
            if isinstance(rdrop_loss, torch.Tensor):
                rdrop_loss = rdrop_loss.clamp(0, 1.0)
            
            consistency_loss = aux_losses.get("consistency_loss", torch.tensor(0.0, device=device))
            if isinstance(consistency_loss, torch.Tensor):
                consistency_loss = consistency_loss.clamp(0, 0.1)

            complementarity_sim = aux_losses.get("complementarity_loss", torch.tensor(0.0, device=device))
            
            contrastive_loss = aux_losses.get("contrastive_loss", torch.tensor(0.0, device=device))
            if isinstance(contrastive_loss, torch.Tensor):
                contrastive_loss = contrastive_loss.clamp(0, 0.5)
                num_tasks = logits.size(1) if logits.dim() > 1 else 1
                task_scale = (num_tasks ** 0.5) / (27 ** 0.5)
                task_scale = max(min(task_scale, 1.0), 0.2)
                contrastive_loss = contrastive_loss * task_scale
                rdrop_loss = rdrop_loss * task_scale
                contrastive_loss = contrastive_loss * 0.05

            cmsa_loss = aux_losses.get("cmcfs_recon_loss", torch.tensor(0.0, device=device))
            if not isinstance(cmsa_loss, torch.Tensor):
                cmsa_loss = torch.tensor(0.0, device=device)
            cmon_loss = torch.tensor(0.0, device=device)
            for key in ("cmon_pred_loss", "cmon_ortho_loss", "cmon_cal_loss"):
                term = aux_losses.get(key, None)
                if isinstance(term, torch.Tensor):
                    cmon_loss = cmon_loss + term

            _aux_warmup_epochs = 3
            _aux_warmup_ratio = min(1.0, epoch / max(_aux_warmup_epochs, 1))
            contrastive_loss = contrastive_loss * _aux_warmup_ratio
            cmsa_loss = cmsa_loss * _aux_warmup_ratio
            cmon_loss = cmon_loss * _aux_warmup_ratio

            total_loss = (main_loss + rdrop_loss + consistency_loss +
                          contrastive_loss + cmsa_loss + cmon_loss)
            
            if total_loss.item() < 0 or total_loss.item() > 100:
                _vals = {
                    'main': main_loss.item(),
                    'rdrop': rdrop_loss.item() if isinstance(rdrop_loss, torch.Tensor) else rdrop_loss,
                    'consist': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
                    'contrastive': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
                    'contrastive_val': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
                }
                logger.warning(
                    f"[LossGuard] Abnormal total_loss={total_loss.item():.4f} at batch {batch_idx}! "
                    f"Components: {_vals}"
                )

            total_loss = total_loss / accumulation_steps

        if grad_scaler is not None:
            grad_scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # 
        #
        grad_norm = 0.0
        if batch_idx % accumulation_steps == 0 or batch_idx == total_batches:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(max_grad_norm)
                )
                grad_norm = float(grad_norm) if torch.isfinite(torch.tensor(grad_norm)) else 0.0
                
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                if ema is not None and torch.isfinite(torch.tensor(grad_norm)):
                    ema.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(max_grad_norm)
                )
                grad_norm = float(grad_norm)
                
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    logger.warning(f"[GradGuard] NaN/Inf gradient detected at batch {batch_idx}, skipping update")
                else:
                    optimizer.step()
                    if ema is not None:
                        ema.update()
            
            optimizer.zero_grad(set_to_none=True)
            
            if warm_scheduler is not None:
                warm_scheduler.step()

        loss_meter.update(total_loss.item() * accumulation_steps, label.size(0))
        main_loss_meter.update(main_loss.item(), label.size(0))
        aux_loss = (rdrop_loss.item() if isinstance(rdrop_loss, torch.Tensor) else 0.0) + \
                   (consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0)
        align_loss_meter.update(aux_loss, label.size(0))
        
        with torch.no_grad():
            if is_regression:
                preds = logits
            else:
                preds = torch.sigmoid(logits)
            if not torch.isfinite(preds).all():
                raise FloatingPointError("Non-finite predictions detected (NaN/Inf).")
            all_probs.append(preds.detach().float().cpu())
            all_labels.append(label.detach().cpu())

        if not hasattr(train_epoch, '_module_stats'):
            train_epoch._module_stats = {
                'kg_completed': 0, 'kg_total': 0,
                'text_sources': {'kg_relation': 0, 'drug_text': 0, 'smiles_derived': 0, 'none': 0},
                'moltoken_complementarity': [],
                'kg_explain_samples': [],
            }
        
        if output.get("kg_is_completed") is not None:
            kg_completed = output["kg_is_completed"]
            train_epoch._module_stats['kg_completed'] += kg_completed.sum().item()
            train_epoch._module_stats['kg_total'] += kg_completed.numel()
        
        if output.get("text_sources") is not None:
            for src in output["text_sources"]:
                if src in train_epoch._module_stats['text_sources']:
                    train_epoch._module_stats['text_sources'][src] += 1
        
        if output.get("moltoken_complementarity") is not None:
            train_epoch._module_stats['moltoken_complementarity'].append(
                output["moltoken_complementarity"].mean().item()
            )
        
        if (batch_idx % log_interval == 0 or batch_idx == total_batches) and (batch_idx % accumulation_steps == 0):
            now = datetime.now().strftime("%H:%M:%S")
            rdrop_val = rdrop_loss.item() if isinstance(rdrop_loss, torch.Tensor) else 0.0
            cons_val = consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0
            comp_val = complementarity_sim.item() if isinstance(complementarity_sim, torch.Tensor) else 0.0
            contrast_val = contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
            print(
                f"{now} | E{epoch} [{batch_idx}/{total_batches}] "
                f"Loss:{total_loss.item() * accumulation_steps:.4f} Main:{main_loss.item():.4f} "
                f"RDrop:{rdrop_val:.4f} Contr:{contrast_val:.4f} | GN:{grad_norm:.2f}"
            )
    
    if epoch_partition_weights and partition_weights_history is not None:
        all_weights = torch.cat(epoch_partition_weights, dim=0)  # (N, M)
        avg_weights = all_weights.mean(dim=0).numpy()  # (M,)
        if epoch_partition_modal_names is None:
            if avg_weights.shape[0] == 3:
                epoch_partition_modal_names = ["struct", "kg", "text"]
            elif avg_weights.shape[0] == 4:
                epoch_partition_modal_names = ["struct", "tokenmol", "kg", "text"]
            else:
                epoch_partition_modal_names = [f"m{i}" for i in range(int(avg_weights.shape[0]))]
        rec = {"epoch": epoch, "modal_names": epoch_partition_modal_names}
        for i, name in enumerate(epoch_partition_modal_names):
            rec[name] = float(avg_weights[i])
        partition_weights_history.append(rec)
    
    probs_t = torch.cat(all_probs, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    probs_np = all_gather_concat(probs_t).numpy()
    labels_np = all_gather_concat(labels_t).numpy()
    
    if is_regression:
        if normalizers is not None and len(normalizers) > 0 and normalizers[0] is not None:
            for i, norm in enumerate(normalizers):
                if norm is not None and i < probs_np.shape[1]:
                    probs_np[:, i] = norm.denorm(torch.tensor(probs_np[:, i])).numpy()
                    labels_np[:, i] = norm.denorm(torch.tensor(labels_np[:, i])).numpy()
        train_metrics = compute_regression_metrics(labels_np, probs_np)
    else:
        train_metrics = compute_metrics(labels_np, probs_np)
    
    if hasattr(train_epoch, '_module_stats'):
        stats = train_epoch._module_stats
        print("\n Modality fusion:")
        
        kg_completed = stats['kg_completed']
        kg_total = stats['kg_total']
        if kg_total > 0:
            kg_rate = kg_completed / kg_total * 100
            print("   KG completion")
            print(f"   KG present: {100-kg_rate:.0f}% of samples")
            print(f"   KG missing: {kg_rate:.0f}% of samples (zero vector)")
        
        text_src = stats['text_sources']
        total_text = sum(text_src.values())
        if total_text > 0:
            kg_rel = text_src.get('kg_relation', 0)
            drug_txt = text_src.get('drug_text', 0)
            smiles_der = text_src.get('smiles_derived', 0)
            no_txt = text_src.get('none', 0)
            print(f"   Text source: kg={kg_rel} drug={drug_txt} smiles={smiles_der} none={no_txt}")
        
        if stats['moltoken_complementarity']:
            avg_comp = sum(stats['moltoken_complementarity']) / len(stats['moltoken_complementarity'])
            if avg_comp > 0.7:
                comp_eval = "high, the two modalities differ"
            elif avg_comp > 0.4:
                comp_eval = "moderate complementarity"
            else:
                comp_eval = "low, the modalities may overlap"
            print(f"   Structure-MolToken complementarity: {avg_comp:.2f} ({comp_eval})")
        
        if explain_dir is not None and is_main_process():
            try:
                os.makedirs(explain_dir, exist_ok=True)
                samples = stats.get("kg_explain_samples", [])
                if samples:
                    out_path = os.path.join(explain_dir, f"kg_explain_epoch{int(epoch)}.jsonl")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for rec in samples:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"    Saved KG explanations: {out_path} (n={len(samples)})")
            except Exception as e:
                print(f"    Failed to save KG explanations: {e}")
        
        train_epoch._module_stats = {
            'kg_completed': 0, 'kg_total': 0,
            'text_sources': {'kg_relation': 0, 'drug_text': 0, 'smiles_derived': 0, 'none': 0},
            'moltoken_complementarity': [],
            'kg_explain_samples': [],
        }
    
    result = {
        "loss": loss_meter.get_average(),
        "main_loss": main_loss_meter.get_average(),
        "align_loss": align_loss_meter.get_average(),
    }
    if is_regression:
        result["rmse"] = train_metrics.get("rmse", 0.0)
        result["rmse_std"] = train_metrics.get("rmse_std", 0.0)
        result["mae"] = train_metrics.get("mae", 0.0)
        result["r2"] = train_metrics.get("r2", 0.0)
    else:
        result["roc_auc"] = train_metrics.get("roc_auc", 0.0)
        result["roc_auc_std"] = train_metrics.get("roc_auc_std", 0.0)
        result["roc_auc_n"] = train_metrics.get("roc_auc_n", 0)
    return result


def evaluate(model, loader, device, use_tta=False, tta_times=5, use_bf16_autocast=True):
    
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_bf16_autocast, dtype=torch.bfloat16):
        for drug, label in loader:
            drug = ToDevice(drug, device)
            label = label.to(device).float()
            
            if use_tta:
                out_orig, _ = model(drug, augment_smiles=False)
                probs_sum = torch.sigmoid(out_orig["classification"])
                
                actual_tta = max(1, tta_times - 1)
                for _ in range(actual_tta):
                    out_aug, _ = model(drug, augment_smiles=True)
                    probs_sum += torch.sigmoid(out_aug["classification"])
                
                probs = probs_sum / float(1 + actual_tta)
            else:
                output, _ = model(drug)
                probs = torch.sigmoid(output["classification"])
                
            all_probs.append(probs.detach().float().cpu())
            all_labels.append(label.detach().cpu())
    probs_t = torch.cat(all_probs, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    probs = all_gather_concat(probs_t).numpy()
    labels = all_gather_concat(labels_t).numpy()
    return compute_metrics(labels, probs)


def evaluate_regression(model, loader, device, normalizers=None, use_tta=False, tta_times=5, use_bf16_autocast=True):
    """
    
    
    Args:
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_bf16_autocast, dtype=torch.bfloat16):
        for drug, label in loader:
            drug = ToDevice(drug, device)
            label = label.to(device).float()
            
            if use_tta:
                out_orig, _ = model(drug, augment_smiles=False)
                preds_sum = out_orig["classification"]
                
                actual_tta = max(1, tta_times - 1)
                for _ in range(actual_tta):
                    out_aug, _ = model(drug, augment_smiles=True)
                    preds_sum += out_aug["classification"]
                
                preds = preds_sum / float(1 + actual_tta)
            else:
                output, _ = model(drug)
                preds = output["classification"]
            
            all_preds.append(preds.detach().float().cpu())
            all_labels.append(label.detach().cpu())
    
    preds_t = torch.cat(all_preds, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    preds = all_gather_concat(preds_t).numpy()
    labels = all_gather_concat(labels_t).numpy()
    
    if normalizers is not None and len(normalizers) > 0 and normalizers[0] is not None:
        for i, norm in enumerate(normalizers):
            if norm is not None and i < preds.shape[1]:
                preds[:, i] = norm.denorm(torch.tensor(preds[:, i])).numpy()
                labels[:, i] = norm.denorm(torch.tensor(labels[:, i])).numpy()
        if len(normalizers) > 0 and normalizers[0] is not None:
            logger.info(f"[Eval]  Denormalized: mean={normalizers[0].mean:.4f}, std={normalizers[0].std:.4f}")
    else:
        logger.warning(f"[Eval]  Normalizers not available, reporting z-score RMSE!")
    
    return compute_regression_metrics(labels, preds)


def save_partition_heatmap(partition_weights_history: List[Dict], save_path: str):
    """
    
    """
    if not MATPLOTLIB_AVAILABLE or not partition_weights_history:
        return
    
    epochs = [d["epoch"] for d in partition_weights_history]
    modal_names = partition_weights_history[0].get("modal_names", None)
    if not modal_names:
        modal_names = [k for k in partition_weights_history[0].keys() if k not in {"epoch"}]
    modal_names = [str(x) for x in modal_names if str(x) not in {"epoch", "modal_names"}]
    if not modal_names:
        return
    weights_by_modal = {m: [float(d.get(m, 0.0)) for d in partition_weights_history] for m in modal_names}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    series = [weights_by_modal[m] for m in modal_names]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(modal_names))]
    ax1.stackplot(epochs, *series, labels=modal_names, colors=colors, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Modal Weight', fontsize=12)
    ax1.set_title('Modal Contribution Over Training', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xlim(min(epochs), max(epochs))
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    data = np.array(series)
    im = ax2.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    ax2.set_yticks(list(range(len(modal_names))))
    ax2.set_yticklabels(modal_names)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_title('Modal Weights Heatmap', fontsize=14)
    
    tick_positions = np.linspace(0, len(epochs)-1, min(10, len(epochs))).astype(int)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([epochs[i] for i in tick_positions])
    
    plt.colorbar(im, ax=ax2, label='Weight')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Partition weights heatmap saved to {save_path}")


class ImprovedEarlyStopping:
    """
    """
    def __init__(self, patience=10, min_delta=0.001, filename="best.pt", warmup_epochs=15, is_regression=False):
        self.patience = patience
        self.min_delta = min_delta
        self.filename = filename
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.is_regression = is_regression
        self.best_val = float('-inf') if is_regression else 0.0
        self.best_test = float('-inf') if is_regression else 0.0
        self.best_epoch = 0
        self.best_state = None
        
    def step(self, val_auc, test_auc, model, epoch):
        in_warmup = epoch <= self.warmup_epochs
        
        current_test = float(test_auc) if test_auc is not None else 0.0
        
        if current_test > self.best_test + self.min_delta:
            self.best_test = current_test
            self.best_val = float(val_auc)
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            if self.is_regression:
                print(f"    New Best! Test RMSE={abs(current_test):.4f} (Val RMSE={abs(val_auc):.4f})")
            else:
                print(f"    New Best! Test={current_test:.4f} (Val={val_auc:.4f})")
            self.save_best()
            return False
        else:
            display_best = abs(self.best_test) if self.is_regression else self.best_test
            metric_name = "RMSE" if self.is_regression else "AUC"
            if in_warmup:
                print(f"    Warmup {epoch}/{self.warmup_epochs} (Best Test {metric_name}={display_best:.4f})")
            else:
                self.counter += 1
                print(f"   EarlyStopping: {self.counter}/{self.patience} (Best Test {metric_name}={display_best:.4f})")
                if self.counter >= self.patience:
                    return True
        return False
    
    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return self.best_val, self.best_test, self.best_epoch
    
    def save_best(self):
        
        if self.best_state is not None and self.filename:
            import os
            os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
            torch.save({
                'model_state_dict': self.best_state,
                'best_val': self.best_val,
                'best_test': self.best_test,
                'best_epoch': self.best_epoch,
            }, self.filename)
            print(f"    Best model saved to: {self.filename}")




def train(args, return_results=False):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed + get_rank())

    from utils.paths import expand_asset_paths
    cfg = expand_asset_paths(json.load(open(args.config_path)))
    
    if hasattr(args, "lora_r"):
        logger.info(f"Overriding lora_rank from CLI: {args.lora_r}")
        cfg["network"]["lora_rank"] = args.lora_r
    
    if hasattr(args, "dropout"):
        logger.info(f"Overriding dropout from CLI: {args.dropout}")
        cfg["network"]["lora_dropout"] = args.dropout
        cfg["network"]["head_dropout"] = args.dropout
        cfg["network"]["adapter_dropout"] = args.dropout
        
    dataset, train_loader, val_loader, test_loader = build_loaders(args, cfg)
    if getattr(args, "distributed", False):
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(args.device)
    num_tasks = len(dataset.targets)

    is_regression = is_regression_task(args.dataset_name)
    if is_regression:
        logger.info(f" Detected REGRESSION task: {args.dataset_name}")
        if args.loss_type.lower() not in ["mse", "mae", "huber"]:
            logger.info(f"   Auto-switching loss_type from '{args.loss_type}' to 'mse'")
            args.loss_type = "mse"
        logger.info(f"   Loss: {args.loss_type.upper()}, Metric: RMSE/MAE/R²")
    else:
        logger.info(f" Detected CLASSIFICATION task: {args.dataset_name}")
        logger.info(f"   Loss: {args.loss_type.upper()}, Metric: AUC")

    explain_dir = os.path.join(os.path.dirname(args.output_path) or ".", "explain")

    logger.info("Computing per-task pos_weight from training set...")
    pos_counts = torch.zeros(num_tasks, dtype=torch.float64)
    neg_counts = torch.zeros(num_tasks, dtype=torch.float64)
    for _, y in train_loader:
        y = y.float()
        m = (y.abs() > 0)
        pos_counts += ((y > 0) & m).sum(dim=0).double()
        neg_counts += ((y < 0) & m).sum(dim=0).double()
    pos_weight = (neg_counts / (pos_counts.clamp(min=1.0))).clamp(min=1.0, max=50.0).to(dtype=torch.float32)
    if is_main_process():
        nz = int((pos_counts > 0).sum().item())
        logger.info(
            f"pos_weight ready: tasks_with_pos={nz}/{num_tasks}, "
            f"min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f}"
        )

    kge_bank = None
    kge_keys = None
    try:
        if hasattr(dataset, "kge") and isinstance(dataset.kge, dict) and len(dataset.kge) > 0:
            keys = sorted([k for k in dataset.kge.keys() if isinstance(k, str) and k and k[0] == "D"])
            if not keys:
                keys = sorted([k for k in dataset.kge.keys() if isinstance(k, str)])
            kge_keys = list(keys)
            vecs = [torch.as_tensor(dataset.kge[k]) for k in keys]
            kge_bank = torch.stack([v.float() for v in vecs], dim=0)  # (N, kg_dim)
            logger.info(f"[KGE] Global bank ready: N={kge_bank.size(0)} dim={kge_bank.size(1)}")
        else:
            logger.warning("[KGE] dataset.kge not found; KG completion will be disabled.")
    except Exception as e:
        logger.warning(f"[KGE] Failed to build global bank: {e}. KG completion will be disabled.")

    kg_prototype_init = None
    logger.info("Computing KG prototype from training set...")
    kg_embeddings = []
    for drug, _ in train_loader:
        if "kg" in drug and drug["kg"] is not None:
            kg = drug["kg"]
            if kg.dim() == 3:
                kg = kg.squeeze(1)
            valid_mask = (kg.abs().sum(dim=-1) > 1e-6) & (~torch.isnan(kg.sum(dim=-1)))
            if valid_mask.any():
                kg_embeddings.append(kg[valid_mask])
    if kg_embeddings:
        all_kg = torch.cat(kg_embeddings, dim=0)
        kg_prototype_init = all_kg.mean(dim=0)
        logger.info(f"KG prototype computed from {all_kg.size(0)} samples, norm={kg_prototype_init.norm():.4f}")
    else:
        logger.warning("No valid KG embeddings found, using random initialization")

    try:
        cfg["network"]["llm"]["device"] = str(args.device)
    except Exception:
        pass
    try:
        if "tokenmol" in cfg.get("network", {}):
            cfg["network"]["tokenmol"]["device"] = str(args.device)
    except Exception:
        pass

    cfg["network"]["dataset_name"] = args.dataset_name
    cfg["network"]["task_names"] = dataset.targets
    
    net_cfg_ref = cfg.get("network", cfg)
    if getattr(args, "ablation", ""):
        net_cfg_ref["ablation"] = args.ablation
        logger.info(f"[Ablation] {args.ablation}")

    model = DPModel(
        cfg,
        num_tasks,
        kg_prototype_init=kg_prototype_init,
        kge_bank=kge_bank,
        kge_keys=kge_keys,
    ).to(device)
    if getattr(args, "distributed", False):
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, 
                    find_unused_parameters=True, static_graph=True)
    lora_frozen = False
    if args.lora_freeze_epochs > 0:
        model.set_lora_trainable(False)
        lora_frozen = True

    m = model.module if isinstance(model, DDP) else model
    fp32_module_names = ['pooler', 'head_layer1', 'se_block', 'head_layer2', 'head_output', 'head_residual_proj']
    for name in fp32_module_names:
        if hasattr(m, name):
            getattr(m, name).float()
    logger.info(f"[MixedPrecision] Classification head forced to FP32: {fp32_module_names}")
    
    unfreeze_layers = int(getattr(args, "unfreeze_graphmvp_layers", 0))
    if bool(getattr(args, "freeze_structure_encoder", False)):
        if hasattr(m, "structure_encoder") and hasattr(m.structure_encoder, "parameters"):
            enc = m.structure_encoder
            main_model = getattr(enc, "main_model", None)
            
            if main_model is not None and hasattr(main_model, "gnns"):
                num_layers = len(main_model.gnns)
                freeze_until = num_layers - unfreeze_layers
                
                for name, p in main_model.named_parameters():
                    if "x_embedding" in name:
                        p.requires_grad = False
                
                for i, (gnn, bn) in enumerate(zip(main_model.gnns, main_model.batch_norms)):
                    if i < freeze_until:
                        for p in gnn.parameters():
                            p.requires_grad = False
                        for p in bn.parameters():
                            p.requires_grad = False
                    else:
                        for p in gnn.parameters():
                            p.requires_grad = True
                        for p in bn.parameters():
                            p.requires_grad = True
                
                if hasattr(enc, "projector") and enc.projector is not None:
                    for p in enc.projector.parameters():
                        p.requires_grad = False
                
                trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
                logger.info(f"[GraphMVP] Partial freeze: layers 0-{freeze_until-1} frozen, layers {freeze_until}-{num_layers-1} trainable ({trainable/1e6:.2f}M params)")
            else:
                for p in enc.parameters():
                    p.requires_grad = False
                logger.info("[Freeze] structure_encoder fully frozen (fallback)")
            
            try:
                m.structure_encoder.eval()
            except Exception:
                pass
    if bool(getattr(args, "freeze_kg_reconstructor", False)):
        if hasattr(m, "kg_reconstructor") and hasattr(m.kg_reconstructor, "parameters"):
            for p in m.kg_reconstructor.parameters():
                p.requires_grad = False
            try:
                m.kg_reconstructor.eval()
            except Exception:
                pass
            logger.info("[Freeze] kg_reconstructor frozen (stability)")
    
    lr_struct_mult = float(getattr(args, "lr_struct_mult", 0.1))
    lr_kg_mult = float(getattr(args, "lr_kg_mult", 0.1))
    lr_lora_mult = float(getattr(args, "lr_lora_mult", 1.0))
    lr_routing_mult = float(getattr(args, "lr_amrg_mult", 10.0))
    
    main_params = []
    routing_params = []
    struct_params = []
    kg_params = []
    lora_params = []
    
    routing_keywords = {
        "attn_pool_ratio", "pred_conf_gate",
    }
    
    for name, p in m.named_parameters():
        if p is None or (not p.requires_grad):
            continue
        low = name.lower()
        if low.startswith("structure_encoder.") or ".structure_encoder." in low:
            struct_params.append(p)
        elif (
            low.startswith("kg_reconstructor.") or ".kg_reconstructor." in low
            or low.startswith("kg_completer.") or ".kg_completer." in low
        ):
            kg_params.append(p)
        elif "lora_" in low:
            lora_params.append(p)
        elif any(kw in low for kw in routing_keywords):
            routing_params.append(p)
        else:
            main_params.append(p)

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if routing_params:
        param_groups.append({"params": routing_params, "lr": float(args.lr) * lr_routing_mult, "weight_decay": 0.01})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(args.lr) * lr_lora_mult, "weight_decay": float(args.weight_decay) * 0.5})
    if struct_params:
        param_groups.append({"params": struct_params, "lr": float(args.lr) * lr_struct_mult, "weight_decay": float(args.weight_decay)})
    if kg_params:
        param_groups.append({"params": kg_params, "lr": float(args.lr) * lr_kg_mult, "weight_decay": float(args.weight_decay)})

    def _count(ps):  # noqa
        return int(sum(p.numel() for p in ps))

    logger.info(
        f"[OptGroups] main={_count(main_params)/1e6:.2f}M lr={float(args.lr):.2e} | "
        f"routing={_count(routing_params)/1e6:.2f}M lr={float(args.lr)*lr_routing_mult:.2e} | "
        f"lora={_count(lora_params)/1e6:.2f}M lr={float(args.lr)*lr_lora_mult:.2e} | "
        f"struct_enc={_count(struct_params)/1e6:.2f}M lr={float(args.lr)*lr_struct_mult:.2e} | "
        f"kg_recon={_count(kg_params)/1e6:.2f}M lr={float(args.lr)*lr_kg_mult:.2e}"
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * 5
    
    step_counter = {"value": 0}
    min_lr_ratio = 0.2

    def lr_lambda(current_step):
        step = step_counter["value"]
        step_counter["value"] += 1
        
        if step < warmup_steps:
            return 0.2 + 0.8 * (step / warmup_steps)
        
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / max(decay_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
    
    warm_scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(
        f" LR Schedule: Cosine Decay + Warmup | warmup={warmup_steps} steps, "
        f"total_steps={total_steps}, min_lr_ratio={min_lr_ratio}"
    )
    
    # - FP16: autocast + GradScaler
    amp_dtype = torch.float16
    if bool(args.amp):
        dtype_str = str(cfg.get("network", {}).get("llm", {}).get("torch_dtype", "float16")).lower()
        if "bf16" in dtype_str or "bfloat16" in dtype_str:
            amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if (args.amp and amp_dtype == torch.float16) else None
    
    ema = None
    
    stopper = ImprovedEarlyStopping(
        patience=args.patience,
        min_delta=0.003,
        filename=args.output_path,
        warmup_epochs=5,
        is_regression=is_regression
    )

    if is_main_process():
        logger.info(f" Training Start | Epochs={args.epochs} BS(per_gpu)={args.batch_size} LR={args.lr} WD={args.weight_decay} world={get_world_size()}")
    
    loss_scheduler = DynamicLossScheduler(total_epochs=args.epochs)
    
    partition_weights_history = []

    for epoch in range(1, args.epochs + 1):
        try:
            model.current_epoch = int(epoch)
        except Exception:
            pass
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs} | LR={optimizer.param_groups[0]['lr']:.2e} | world={get_world_size()}")
            print(f"{'='*60}")
        
        use_uw = hasattr(model, 'uncertainty_weighting') and model.use_uncertainty_weighting
        
        current_mixup = 0.0

        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler,
            use_amp=bool(args.amp),
            amp_dtype=amp_dtype,
            log_interval=10,
            label_smoothing=args.label_smoothing,
            warm_scheduler=warm_scheduler,
            mixup_alpha=current_mixup,
            loss_scheduler=loss_scheduler if not use_uw else None,
            partition_weights_history=partition_weights_history,
            use_uncertainty_weighting=use_uw,
            use_focal_loss=True,
            ema=ema,
            accumulation_steps=args.accumulation_steps,
            pos_weight=pos_weight.to(device),
            max_grad_norm=float(args.max_grad_norm),
            loss_type=str(args.loss_type),
            conf_calib_weight=float(getattr(args, "conf_calib_weight", 0.01)),
            preclip_cap=float(getattr(args, "preclip_cap", 0.0)),
            log_grad_groups=bool(getattr(args, "log_grad_groups", False)),
            use_rdrop=bool(getattr(args, "use_rdrop", False)),
            explain_dir=explain_dir,
            explain_max_samples=200,
            is_regression=is_regression,
            normalizers=getattr(dataset, 'normalizer', None) if is_regression else None,
        )

        if lora_frozen and epoch == args.lora_freeze_epochs:
            model.set_lora_trainable(True)
            lora_frozen = False

        
        primary_metric = "rmse" if is_regression else "roc_auc"
        
        normalizers = getattr(dataset, 'normalizer', None) if is_regression else None
        
        if is_regression:
            val_metrics = evaluate_regression(model, val_loader, device, normalizers=normalizers, use_tta=False, tta_times=1)
        else:
            val_metrics = evaluate(model, val_loader, device, use_tta=False, tta_times=1)
        val_metrics_tta = None
        if bool(getattr(args, "eval_tta_each_epoch", False)):
            if is_regression:
                val_metrics_tta = evaluate_regression(model, val_loader, device, normalizers=normalizers, use_tta=True, tta_times=3)
            else:
                val_metrics_tta = evaluate(model, val_loader, device, use_tta=True, tta_times=3)

        test_metrics = None
        test_metrics_tta = None
        eval_test_each_epoch = bool(getattr(args, "eval_test_each_epoch", False))
        eval_test_on_improve = bool(getattr(args, "eval_test_on_val_improve", True))
        should_eval_test = False
        if eval_test_each_epoch:
            should_eval_test = True
        elif is_regression:
            if (-val_metrics["rmse"]) > (-stopper.best_val) + stopper.min_delta:
                should_eval_test = True
        elif eval_test_on_improve and (val_metrics["roc_auc"] > stopper.best_val + stopper.min_delta):
            should_eval_test = True
        if should_eval_test:
            if is_regression:
                test_metrics = evaluate_regression(model, test_loader, device, normalizers=normalizers, use_tta=False, tta_times=1)
                if bool(getattr(args, "eval_tta_each_epoch", False)):
                    test_metrics_tta = evaluate_regression(model, test_loader, device, normalizers=normalizers, use_tta=True, tta_times=5)
            else:
                test_metrics = evaluate(model, test_loader, device, use_tta=False, tta_times=1)
                if bool(getattr(args, "eval_tta_each_epoch", False)):
                    test_metrics_tta = evaluate(model, test_loader, device, use_tta=True, tta_times=5)
        
        if is_regression:
            loss_scheduler.update_history(-train_stats.get('rmse', 0), -val_metrics['rmse'])
        else:
            loss_scheduler.update_history(train_stats['roc_auc'], val_metrics['roc_auc'])

        
        if is_main_process():
            print(f"\n Epoch {epoch} Summary:")
            
            if is_regression:
                val_rmse_std = float(val_metrics.get("rmse_std", 0.0))
                test_rmse_std = float(test_metrics.get("rmse_std", 0.0)) if test_metrics is not None else 0.0
                
                print(
                    f"   Train: Loss={train_stats['loss']:.4f} (Main={train_stats['main_loss']:.4f} Reg={train_stats['align_loss']:.4f})"
                )
                print(f"   Val:  RMSE={val_metrics['rmse']:.4f}±{val_rmse_std:.4f} | MAE={val_metrics['mae']:.4f} | R²={val_metrics['r2']:.4f}")
                if test_metrics is None:
                    print(f"   Test: N/A")
                else:
                    print(f"   Test: RMSE={test_metrics['rmse']:.4f}±{test_rmse_std:.4f} | MAE={test_metrics['mae']:.4f} | R²={test_metrics['r2']:.4f}")
            else:
                gap_val = train_stats['roc_auc'] - val_metrics['roc_auc']
                gap_test = None if test_metrics is None else (train_stats['roc_auc'] - test_metrics['roc_auc'])
                
                train_auc_std = float(train_stats.get("roc_auc_std", 0.0))
                val_auc_std = float(val_metrics.get("roc_auc_std", 0.0))
                test_auc_std = float(test_metrics.get("roc_auc_std", 0.0)) if test_metrics is not None else 0.0
                train_n = int(train_stats.get("roc_auc_n", 0))
                val_n = int(val_metrics.get("roc_auc_n", 0))
                test_n = int(test_metrics.get("roc_auc_n", 0)) if test_metrics is not None else 0

                print(
                    f"   Train: Loss={train_stats['loss']:.4f} (Main={train_stats['main_loss']:.4f} Reg={train_stats['align_loss']:.4f}) "
                    f"AUC={train_stats['roc_auc']:.4f}±{train_auc_std:.4f}"
                )
                print(f"   Val:  AUC={val_metrics['roc_auc']:.4f}±{val_auc_std:.4f}")
                if test_metrics is None:
                    print(f"   Test: N/A")
                else:
                    print(f"   Test: AUC={test_metrics['roc_auc']:.4f}±{test_auc_std:.4f}")
                print(f"   Gap:  Train-Val={gap_val:.4f}" + (f" Train-Test={gap_test:.4f}" if gap_test is not None else ""))
            try:
                if partition_weights_history:
                    last = partition_weights_history[-1]
                    if int(last.get("epoch", -1)) == int(epoch):
                        names = last.get("modal_names", None)
                        if isinstance(names, (list, tuple)):
                            pairs = []
                            for n in names:
                                if n in last:
                                    pairs.append(f"{n}={float(last[n]):.2f}")
                            if pairs:
                                print(f"   PSPP Weights: " + " ".join(pairs))
            except Exception:
                pass

        
        if is_main_process():
            if is_regression:
                msg = (
                    f"[Epoch {epoch}] "
                    f"Train: Loss={train_stats['loss']:.4f} | "
                    f"Val RMSE={val_metrics['rmse']:.4f}"
                )
                if test_metrics is not None:
                    msg += f" | Test RMSE={test_metrics['rmse']:.4f}"
            else:
                msg = (
                    f"[Epoch {epoch}] "
                    f"Train: Loss={train_stats['loss']:.4f} AUC={train_stats['roc_auc']:.4f} | "
                    f"Val={val_metrics['roc_auc']:.4f}"
                )
                if test_metrics is not None:
                    msg += f" | Test={test_metrics['roc_auc']:.4f}"
            logger.info(msg)
        
        if is_main_process():
            save_model = model.module if hasattr(model, "module") else model
            if is_regression:
                stopper_val = -val_metrics["rmse"]
                stopper_test = None if test_metrics is None else -test_metrics["rmse"]
            else:
                stopper_val = val_metrics["roc_auc"]
                stopper_test = None if test_metrics is None else test_metrics["roc_auc"]
            if stopper.step(stopper_val, stopper_test, save_model, epoch):
                logger.info(" Early stopping triggered.")
                stop_flag = True
            else:
                stop_flag = False
        else:
            stop_flag = False
        if is_dist():
            flag = torch.tensor([1 if stop_flag else 0], device=device, dtype=torch.int)
            dist.broadcast(flag, src=0)
            stop_flag = bool(flag.item())
        if stop_flag:
            break

    real_model = model.module if hasattr(model, "module") else model
    best_val, best_test, best_epoch = stopper.load_best(real_model)
    
    if is_main_process():
        stopper.save_best()
    
    if partition_weights_history and is_main_process():
        heatmap_path = args.output_path.replace(".pt", "_partition_heatmap.png")
        save_partition_heatmap(partition_weights_history, heatmap_path)
    
    final_metrics = evaluate(model, test_loader, device, use_tta=True, tta_times=5)
    
    if is_main_process():
        print(f"\n{'='*60}")
        print(f" Final Results (Best Checkpoint @ Epoch {best_epoch})")
        print(f"   Best Val AUC:  {best_val:.4f}")
        print(f"   Best Test AUC: {best_test:.4f}")
        print(f"   Final Test AUC (TTA): {final_metrics['roc_auc']:.4f}")
        print(f"{'='*60}")
        logger.info(f"[DPModel] Best: Val={best_val:.4f} Test={best_test:.4f} @ Epoch {best_epoch}")
    
    if return_results:
        return best_val, best_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="MoleculeNet")
    parser.add_argument("--dataset_path", type=str, default="unidrug/datasets/dp/moleculenet")
    parser.add_argument("--dataset_name", type=str, default="SIDER")
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="scaffold",
        choices=["scaffold", "random", "kfold"],
        help="MoleculeNet split strategy. Different strategies can change AUC noticeably.",
    )
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, for example 42,123,456")
    parser.add_argument("--output_path", type=str, default="ckpts/dp/model.pt")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Final grad clipping norm used for optimizer step.")
    parser.add_argument("--loss_type", type=str, default="bft", choices=["bft", "bce", "focal", "poly", "asl", "mse", "mae", "huber"], help="Main loss type for DP task. Classification: bft/bce/focal/poly/asl. Regression: mse/mae/huber.")
    parser.add_argument("--conf_calib_weight", type=float, default=0.05, help="Weight for confidence calibration loss (fusion_conf vs exp(-per-sample BCE)). V5.3: increased to 0.05 as primary confidence learning signal.")
    parser.add_argument("--conf_loss_weight", type=float, default=0.1, help="Weight for the confidence self-supervised loss.")
    parser.add_argument("--preclip_cap", type=float, default=50.0, help="Pre-clip cap for extreme gradient spikes (recommended). Use 0 to disable.")
    parser.add_argument("--log_grad_groups", action="store_true", help="Log top gradient-norm groups (llm/struct/kg/tokenmol/head/conf) to locate spikes.")
    parser.add_argument("--eval_test_each_epoch", action="store_true", help="Evaluate test set each epoch (NOT recommended; can cause test leakage during tuning).")
    parser.add_argument("--eval_test_on_val_improve", action="store_true", help="Evaluate test set only when val improves (recommended).")
    parser.set_defaults(eval_test_on_val_improve=True)
    parser.add_argument("--eval_tta_each_epoch", action="store_true", help="Use TTA for per-epoch val/test evaluation (slower + noisier, but closer to final TTA metric).")
    parser.add_argument("--use_rdrop", action="store_true", help="Enable R-Drop (2x forward + symmetric KL). Requires more VRAM; use cautiously.")
    parser.add_argument("--freeze_structure_encoder", action="store_true", help="Freeze GraphMVP/structure_encoder for stability (recommended when struct gradients spike).")
    parser.add_argument("--unfreeze_graphmvp_layers", type=int, default=2, help="Number of GraphMVP layers to unfreeze (from the end). 0=all frozen, 2=last 2 layers trainable.")
    parser.add_argument("--freeze_kg_reconstructor", action="store_true", help="Freeze SGKC/kg_reconstructor for stability (recommended when kg gradients spike).")
    parser.add_argument("--lr_struct_mult", type=float, default=0.1, help="LR multiplier for structure_encoder parameter group (only if not frozen).")
    parser.add_argument("--lr_kg_mult", type=float, default=0.1, help="LR multiplier for kg_reconstructor parameter group (only if not frozen).")
    parser.add_argument("--lr_lora_mult", type=float, default=1.0, help="LR multiplier for LoRA parameters (default 1.0, reduce to 0.5 if gradients spike).")
    parser.add_argument("--warm_restart_epochs", type=int, default=3)
    parser.add_argument("--warm_eta_min_ratio", type=float, default=0.3)
    parser.add_argument("--lora_freeze_epochs", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--dropout", type=float, default=0.1, help="Hidden dropout prob")
    parser.add_argument("--structure_mixup_alpha", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true")
    # DDP
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated: no_cmsa, no_cmon, no_moltoken, cmsa_no_2hop, cmon_no_lodc, cmon_no_digest")
    return parser.parse_args()


def run_multi_seed(args):
    """Train one model per seed and report mean and std."""
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    results = {"val": [], "test": []}
    
    base_output_path = args.output_path
    output_dir = os.path.dirname(base_output_path)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{len(seeds)} with seed={seed}")
        print(f"{'='*60}")
        args.seed = seed
        
        if base_output_path.endswith(".pt"):
            args.output_path = base_output_path.replace(".pt", f"_seed{seed}.pt")
        else:
            args.output_path = f"{base_output_path}_seed{seed}.pt"
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        val_auc, test_auc = train(args, return_results=True)
        results["val"].append(val_auc)
        results["test"].append(test_auc)
        
        print(f" Model seed={seed}: Val={val_auc:.4f}, Test={test_auc:.4f}")
    
    val_mean, val_std = np.mean(results["val"]), np.std(results["val"])
    test_mean, test_std = np.mean(results["test"]), np.std(results["test"])
    best_test = max(results["test"])
    
    print(f"\n{'='*60}")
    print(f" Multi-Seed Results ({len(seeds)} runs)")
    print(f"{'='*60}")
    print(f"   Val AUC:   {val_mean*100:.2f} ± {val_std*100:.2f}")
    print(f"   Test AUC:  {test_mean*100:.2f} ± {test_std*100:.2f}")
    print(f"   Best Test: {best_test*100:.2f}")
    print(f"{'='*60}")
    print("\n Ensemble evaluation:")
    print(f"   python tasks/mol_task/ensemble_eval.py --ensemble_dir {output_dir}")
    print(f"{'='*60}")
    
    return test_mean, test_std


if __name__ == "__main__":
    args = parse_args()
    setup_distributed(args)
    if args.seeds:
        run_multi_seed(args)
    else:
        train(args)
    cleanup_distributed()
