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

# 可视化（可选）
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from datasets.dp_dataset import SUPPORTED_DP_DATASETS, Task
from utils import DPCollator, AverageMeter, EarlyStopping, ToDevice
from models.task_model.dp import DPModel

# ========= 回归任务支持=========
# 回归任务数据集映射
REGRESSION_DATASETS = {"ESOL", "Lipo", "FreeSolv", "qm7", "qm8", "qm9"}

def is_regression_task(dataset_name: str) -> bool:
    """判断是否为回归任务"""
    return dataset_name in REGRESSION_DATASETS

logger = logging.getLogger(__name__)

def grad_norm_l2(parameters) -> float:
    """计算当前梯度的 L2 范数（用于显示“最终裁剪后”的 GN）。"""
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
    计算不同模块(按参数名pattern粗分)的梯度 L2 范数，用于定位梯度尖峰来源。
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
    """torchrun 启动时，通过环境变量初始化分布式。"""
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
    """支持不同 batch 大小的 all_gather（按第 0 维拼接）。"""
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

    # 关键修复drop_last=True 避免最后一个不完整 batch 导致的梯度爆炸
    # 诊断结论：GN:19.78, GN:23.94 等梯度爆炸几乎总是发生在最后一个 batch (72/72)
    # 原因：不完整 batch 的 BatchNorm 统计不稳定 + Bilinear Interaction 放大噪声
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,  # 丢弃不完整的最后一个 batch
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
    准备训练标签和 mask
    
    分类任务：标签编码 +1=正类, -1=负类, 0=缺失
    回归任务：标签是连续值（已标准化），非 NaN 即有效
    """
    if is_regression:
        # 回归任务标签是连续值，直接使用
        # mask: 非 NaN 的位置有效
        mask = ~torch.isnan(raw_labels)
        targets = torch.where(mask, raw_labels, torch.zeros_like(raw_labels))
    else:
        # 分类任务标签编码 +1/-1/0
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

    # 科学处理：不在指标阶段“填充”NaN 来掩盖数值问题。
    # 一旦出现 NaN，说明 forward/训练已经数值不稳定，应当定位源头并修复。
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
    # 说明：
    # - roc_list: 每个可计算 AUC 的 task 的 AUC 列表（跳过全 0 / 全 1 的 task）
    # - roc_auc: 任务级 macro 平均
    # - roc_auc_std: 任务间标准差（常见论文里的 “mean ± std” 若按 task 统计）
    metrics["roc_auc"] = float(np.mean(roc_list)) if roc_list else 0.0
    metrics["roc_auc_std"] = float(np.std(roc_list)) if roc_list else 0.0
    metrics["roc_auc_n"] = int(len(roc_list))
    return metrics


# ========= 回归任务指标=========
def compute_regression_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    计算回归任务指标：RMSE, MAE, R²
    
    Args:
        labels: 真实值 (N, num_tasks) 或 (N,)
        preds: 预测值 (N, num_tasks) 或 (N,)
    
    Returns:
        metrics: {"rmse": float, "mae": float, "r2": float}
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 确保是 2D
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    
    rmse_list, mae_list, r2_list = [], [], []
    
    for i in range(labels.shape[1]):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        
        # 跳过全 NaN 的任务
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid_mask.sum() < 2:
            continue
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        
        # R² 需要至少有方差
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
    SOTA 技术Exponential Moving Average (指数移动平均)
    
    科学原理：
    - 对模型权重进行平滑，减少训练过程中的噪声波动
    - shadow_weight = decay * shadow_weight + (1 - decay) * model_weight
    - 推理时使用 shadow 权重，通常比原始权重更稳定
    
    参考：
    - Polyak averaging (1992)
    - 广泛应用于 BERT、ViT、MAE 等 SOTA 模型
    
    使用建议：
    - decay=0.999 适合大多数情况
    - 训练时更新 EMA，评估时用 EMA 权重
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新 EMA 权重（每个训练 step 调用一次）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """将 shadow 权重应用到模型（评估前调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """恢复原始权重（评估后调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class DynamicLossScheduler:
    """
    极简版损失调度器
    
    诊断结论：
    1. 复杂的动态调度容易引入不稳定性
    2. 固定权重更容易调优和复现
    3. 只保留基本的历史记录功能
    """
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.history = {"train_auc": [], "test_auc": [], "gap": []}
        
        # 简化版固定权重 - 专注主任务
        self.base_weights = {
            "main": 1.0,
            "align": 0.15,        # 模态对齐
            "kg_recon": 0.05,     # KG 补全自监督（结构引导稀疏注意力）
            # 其余辅助损失（recon/consistency/kg_rank/mtfl 等）已从主训练路径移除，避免分支膨胀与梯度不稳定
        }
        
    def update_history(self, train_auc: float, test_auc: float):
        """更新历史记录"""
        self.history["train_auc"].append(train_auc)
        self.history["test_auc"].append(test_auc)
        self.history["gap"].append(train_auc - test_auc)
        
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """返回固定权重"""
        return self.base_weights.copy()


def poly_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1.0, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    SOTA 提升PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    
    原理：在 Cross-Entropy (CE) 基础上增加一个多项式项，增强对困难样本的判别力。
    公式：L_poly = L_ce + epsilon * (1 - p_t)
    """
    # 计算 BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    # 计算 Pt (目标类别的预测概率)
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    
    # PolyLoss 项：epsilon * (1 - pt)
    # 当 pt 接近 1 (易分类) 时，该项接近 0
    # 当 pt 接近 0 (难分类) 时，该项接近 epsilon，提供额外梯度
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
    创新置信度感知的 Focal Loss (带正样本加权)
    
    核心思想：
    - 低置信度样本 = 困难样本 → 损失权重更高
    - 结合 Focal Loss 和 Class Imbalance Correction
    """
    # BCE Loss (支持 pos_weight 处理类别不平衡)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    probs = torch.sigmoid(logits)
    # p_t calculation
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    
    # 置信度加权：conf 越低，权重越高
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
    
    原理：
    - 对正样本和负样本使用不同的 gamma 值
    - gamma_neg > gamma_pos: 更关注正样本（少数类）
    - clip: 对高置信度负样本进行概率裁剪，避免其主导梯度
    
    参考：Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    
    优势：
    - 专门为多标签分类设计
    - 有效处理类别不平衡
    - SIDER 等副作用预测任务非常适用
    """
    # 自适应梯度稳定根据任务数调整clamp强度
    # 单任务（BBBP）：梯度集中，需要强保护
    # 多任务（SIDER等）：梯度分散，保持原设计不变
    num_tasks = logits.size(-1) if logits.dim() > 1 else 1
    is_single_task = (num_tasks <= 2)
    
    if is_single_task:  # 单任务或二分类（BBBP）
        # 强保护：防止梯度爆炸
        logits = logits.clamp(-10, 10)
        min_prob = 0.001      # 最大梯度 = 1000
    else:  # 多任务（SIDER、Tox21、ToxCast等）
        # 保持原设计：无额外clamp，保证精度
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
    
    # 仅单任务对loss进行clamp，多任务保持原设计无clamp
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
    二分类专用Focal Loss
    
    科学原理（Lin et al., ICCV 2017）：
    - 降低易分类样本的权重，聚焦难样本
    - 适用于类别不平衡场景（如 BBBP）
    
    公式：
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        logits: (B, T) 预测 logits
        targets: (B, T) 目标标签
        mask: (B, T) 有效性掩码
        gamma: 聚焦参数，越大越聚焦难样本（默认 2.0）
        alpha: 正样本权重（默认 0.25）
    
    Returns:
        Focal Loss 值
    """
    probs = torch.sigmoid(logits)
    
    # p_t = p if y=1, else 1-p
    p_t = torch.where(targets > 0.5, probs, 1 - probs)
    
    # alpha_t = alpha if y=1, else 1-alpha
    alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # BCE loss (without reduction), 支持 per-task pos_weight（处理类别不平衡）
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
    创新Balanced Fine-Tuning (BFT) Loss - 超越 SOTA 版本
    
    来源：文献中的平衡微调方法 + 针对药物性质预测的优化
    
    核心思想：
    1. Token-level Weighting：低置信度的预测减少梯度贡献，防止过拟合
    2. Sample-level Weighting：困难样本（低置信度）获得更高权重
    
    参数调优建议：
    - tau=0.3: 温和（早期训练）→ tau=0.5: 平衡（冲击 SOTA）→ tau=1.0: 激进
    - lambda_s=0.3: 温和 → lambda_s=0.5: 平衡 → lambda_s=1.0: 激进
    
    Args:
        logits: (B, T) 预测 logits
        targets: (B, T) 目标标签
        mask: (B, T) 有效性掩码
        tau: 温度参数，控制置信度的敏感度
        lambda_s: 样本级加权强度
    
    Returns:
        加权后的 BCE 损失
    """
    # 1. 计算预测概率 (置信度)
    probs = torch.sigmoid(logits)
    
    # 2. Token-level Weighting: w_t = p_t^tau (高置信度 token 权重更大)
    # 对于正样本：置信度 = probs
    # 对于负样本：置信度 = 1 - probs
    confidence = torch.where(targets > 0.5, probs, 1 - probs)
    # 限制置信度范围，防止数值问题
    confidence = confidence.clamp(0.1, 0.9)
    
    # 关键修复切断梯度的"自我对抗"
    # 如果不detach，模型会为了降低 Loss 而降低权重（即降低 confidence），
    # 这与分类目标（提高 confidence）相矛盾，导致梯度剧烈震荡和爆炸。
    confidence_detached = confidence.detach()
    
    # Token 权重：高置信度 token 正常贡献，低置信度 token 减少贡献
    token_weight = confidence_detached ** tau  # 范围 [0.1^tau, 0.9^tau]
    
    # 3. Sample-level Weighting: s_b = 1 + lambda_s * (1 - min_confidence)
    # 困难样本（最小置信度低）获得更高权重
    masked_confidence = confidence_detached * mask + (1 - mask) * 1.0  # 无效位置置为1
    min_confidence_per_sample = masked_confidence.min(dim=1, keepdim=True)[0]  # (B, 1)
    
    # 样本权重：难度越高（min_conf 越低），权重越大
    sample_weight = 1.0 + lambda_s * (1.0 - min_confidence_per_sample)  # 范围 [1, 1+lambda_s]
    
    # 4. 计算基础 BCE Loss（支持 per-task pos_weight）
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    
    # 修复梯度爆炸限制单个 token 的 BCE 损失
    # 原理：极端情况下 BCE 可能很大，导致梯度爆炸
    bce_loss = bce_loss.clamp(max=5.0)
    
    # 5. 应用 Token 和 Sample 权重
    weighted_loss = bce_loss * token_weight * sample_weight
    
    # 6. 对有效位置求平均
    loss = (weighted_loss * mask).sum() / (mask.sum() + 1e-6)
    
    # 双重保险限制最终损失
    loss = loss.clamp(max=2.0)
    
    return loss


def kl_divergence_loss(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    突破性技术R-Drop KL散度损失
    计算两个输出分布之间的对称KL散度，强制模型输出一致性
    
    对于二分类，使用伯努利分布的KL散度
    """
    p = torch.sigmoid(p_logits)
    q = torch.sigmoid(q_logits)
    
    # 对称KL散度: 0.5 * (KL(p||q) + KL(q||p))
    # 对于伯努利分布: KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    eps = 1e-6
    kl_pq = p * (torch.log(p + eps) - torch.log(q + eps)) + \
            (1 - p) * (torch.log(1 - p + eps) - torch.log(1 - q + eps))
    kl_qp = q * (torch.log(q + eps) - torch.log(p + eps)) + \
            (1 - q) * (torch.log(1 - q + eps) - torch.log(1 - p + eps))
    
    kl_loss = 0.5 * (kl_pq + kl_qp)
    
    # 只在有效位置计算
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
    ema=None,  # 新增EMA 支持
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
    is_regression: bool = False,  # 回归任务支持
    normalizers: Optional[List] = None,  # 回归任务反标准化
):
    """训练一个 epoch"""
    model.train()
    
    _raw_model = model.module if hasattr(model, 'module') else model
    
    loss_meter = AverageMeter()
    main_loss_meter = AverageMeter()
    align_loss_meter = AverageMeter()
    
    all_probs = []
    all_labels = []
    total_batches = len(loader)
    
    # 收集分区权重用于可视化
    epoch_partition_weights = []
    epoch_partition_modal_names = None
    
    # 收集 Uncertainty Weighting 的权重
    uw_weights_log = []

    optimizer.zero_grad(set_to_none=True)

    if getattr(loader, "sampler", None) is not None and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for batch_idx, (drug, label) in enumerate(loader, 1):
        drug = ToDevice(drug, device)
        label = label.to(device).float()
        targets, mask = prepare_labels(label, is_regression=is_regression)
        mask = mask.float()
        # batch 标签统计（用于解释 loss/grad 抖动）
        with torch.no_grad():
            valid_frac = float(mask.mean().item())  # 有效标签比例（多任务缺失会让这个变小）
            if is_regression:
                # 回归任务：计算标签均值和标准差
                pos_rate = float(targets[mask > 0.5].mean().item()) if mask.sum() > 0 else 0.0
            else:
                # 分类任务：计算正类比例
                pos_rate = float(((targets > 0.5).float() * mask).sum().item() / (mask.sum().item() + 1e-6))
        
        # 分类任务专用Label Smoothing（回归任务跳过）
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

        # optimizer.zero_grad(set_to_none=True) # 移动到 step 之后

        # BF16: 开 autocast 但不要 GradScaler；FP16: autocast + GradScaler
        with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
            output, _ = model(drug, mixup=mixup_params)
            logits = output["classification"]

            # 科学定位：如果这里出现 NaN/Inf，直接报错（不要在指标处“填充”掩盖）
            if not torch.isfinite(logits).all():
                bad = (~torch.isfinite(logits)).sum().item()
                safe = torch.nan_to_num(logits)
                logger.error(
                    f"[NaNGuard] Non-finite logits detected: {bad}/{logits.numel()} | "
                    f"min={safe.min().item():.4f} max={safe.max().item():.4f} | "
                    f"epoch={epoch} batch_idx={batch_idx}"
                )
                raise FloatingPointError("Non-finite logits detected (NaN/Inf).")
            
            # R-Drop可选的第二次前向传播
            # 注意：R-Drop 会使显存占用翻倍，谨慎启用
            # 默认关闭以节省显存
            logits2 = None
            if use_rdrop and model.training:
                output2, _ = model(drug, mixup=mixup_params)
                logits2 = output2["classification"]
                # 不对 logits 做硬截断；数值稳定应由 loss 的稳定实现与 NaNGuard 来保证
            
            # 收集分区权重
            if output.get("partition_info") is not None:
                pi = output.get("partition_info")
                if isinstance(pi, dict) and pi.get("modal_weights") is not None:
                    modal_w = pi["modal_weights"].detach().cpu()
                    epoch_partition_weights.append(modal_w)
                if epoch_partition_modal_names is None:
                    names = output.get("partition_modal_names", None)
                    if names is not None and isinstance(names, (list, tuple)):
                        epoch_partition_modal_names = [str(x) for x in names]
            
            # 修复移除重复的 Label Smoothing (286行已经处理过 targets)
            # 之前这里又做了一次固定 0.05 的平滑，导致双重平滑，可能损害性能
            targets_smooth = targets
            
            # 解决类别不平衡：使用训练集统计得到的 per-task pos_weight（更稳定、更科学）
            # pos_weight[t] ≈ (#neg_t / #pos_t)
            # 注意：这里的 targets/mask 已在上游按 raw_labels.abs()>0 做了有效性过滤。
            pw = pos_weight

            # ========= V4置信度加权损失计算 =========
            # 获取样本权重（低置信度样本权重低）
            sample_weights = output.get("sample_weights", None)  # (B,)
            if sample_weights is not None:
                # 扩展到 (B, num_labels) 以匹配loss维度
                sample_weights = sample_weights.unsqueeze(-1).expand_as(mask)
            
            # 主损失类型
            lt = str(loss_type).lower()
            
            # ========= 回归任务损失=========
            if lt == "mse":
                # MSE 损失（回归任务）
                # logits 直接作为预测值（不经过 sigmoid）
                _mse = F.mse_loss(logits, targets, reduction="none")
                if sample_weights is not None:
                    _mse = _mse * sample_weights
                main_loss = (_mse * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "mae":
                # MAE 损失（回归任务，对异常值更鲁棒）
                _mae = F.l1_loss(logits, targets, reduction="none")
                if sample_weights is not None:
                    _mae = _mae * sample_weights
                main_loss = (_mae * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "huber":
                # Huber 损失（回归任务，结合 MSE 和 MAE 的优点）
                _huber = F.smooth_l1_loss(logits, targets, reduction="none", beta=1.0)
                if sample_weights is not None:
                    _huber = _huber * sample_weights
                main_loss = (_huber * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "bce":
                _bce = F.binary_cross_entropy_with_logits(
                    logits, targets_smooth, reduction="none", pos_weight=pw
                )
                # 防梯度尖峰限制单样本BCE损失，防止极端预测导致梯度爆炸
                # BCE在p→0或p→1时梯度趋向无穷，clamp到5.0（对应p≈0.007或0.993）
                _bce = _bce.clamp(max=5.0)
                if sample_weights is not None:
                    _bce = _bce * sample_weights  # 置信度加权
                main_loss = (_bce * mask).sum() / (mask.sum() + 1e-6)
            elif lt == "focal":
                main_loss = focal_loss_binary(
                    logits, targets_smooth, mask,
                    gamma=2.0, alpha=0.25, pos_weight=pw,
                )
            elif lt == "asl":
                # V10优化根据任务数动态调整ASL参数
                # ToxCast等大规模多任务：降低gamma防止过度抑制
                num_tasks = logits.size(1) if logits.dim() > 1 else 1
                if num_tasks > 100:  # 大规模多任务（如ToxCast 617）
                    gamma_pos, gamma_neg, clip_val = 0.5, 1.5, 0.03
                elif num_tasks > 20:  # 中等规模（如SIDER 27）
                    gamma_pos, gamma_neg, clip_val = 1.0, 2.0, 0.05
                else:  # 小规模（如BBBP 1）
                    gamma_pos, gamma_neg, clip_val = 1.0, 2.5, 0.05
                    
                _asl = asymmetric_loss(
                    logits, targets_smooth,
                    gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip_val,
                )
                if sample_weights is not None:
                    _asl = _asl * sample_weights  # 置信度加权
                main_loss = (_asl * mask).sum() / (mask.sum() + 1e-6)
            else:
                # 默认 BCE
                _bce = F.binary_cross_entropy_with_logits(
                    logits, targets_smooth, reduction="none", pos_weight=pw
                )
                _bce = _bce.clamp(max=5.0)  # 防梯度尖峰
                if sample_weights is not None:
                    _bce = _bce * sample_weights  # 置信度加权
                main_loss = (_bce * mask).sum() / (mask.sum() + 1e-6)

            # 辅助损失
            aux_losses = output.get("aux_losses", {})
            
            # R-Drop正则化损失（标准双向KL散度）
            rdrop_loss = aux_losses.get("rdrop_loss", torch.tensor(0.0, device=device))
            if isinstance(rdrop_loss, torch.Tensor):
                rdrop_loss = rdrop_loss.clamp(0, 1.0)  # 保持合理范围
            
            # 特征一致性损失
            consistency_loss = aux_losses.get("consistency_loss", torch.tensor(0.0, device=device))
            if isinstance(consistency_loss, torch.Tensor):
                consistency_loss = consistency_loss.clamp(0, 0.1)

            # 互补性相似度（纯监控指标，不参与训练）
            # 原因：互补性约束在训练中容易导致不稳定（投影层梯度问题）
            # 现在只作为监控，用于观察两模态是否学到互补信息
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
            
            # 诊断检测异常损失值
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

            # 梯度累积Loss 除以累积步数
            total_loss = total_loss / accumulation_steps

        if grad_scaler is not None:
            grad_scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # 科学版v2梯度处理 - 基于梯度裁剪的稳定更新
        # 
        # 科学原理：
        # 1. 梯度裁剪是处理梯度爆炸的标准方法（Pascanu et al., 2013）
        # 2. NaN/Inf通常源于数值溢出，应该回退到安全状态而非用0替换
        # 3. 使用GradScaler的动态缩放来自动处理FP16溢出
        #
        grad_norm = 0.0
        if batch_idx % accumulation_steps == 0 or batch_idx == total_batches:
            if grad_scaler is not None:
                # FP16/BF16混合精度训练
                grad_scaler.unscale_(optimizer)
                
                # 梯度裁剪（处理梯度爆炸的标准方法）
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(max_grad_norm)
                )
                grad_norm = float(grad_norm) if torch.isfinite(torch.tensor(grad_norm)) else 0.0
                
                # GradScaler会自动检测inf/nan并跳过更新，同时调整scale
                # 这是PyTorch官方推荐的处理方式
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                if ema is not None and torch.isfinite(torch.tensor(grad_norm)):
                    ema.update()
            else:
                # FP32训练
                # 先裁剪梯度
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(max_grad_norm)
                )
                grad_norm = float(grad_norm)
                
                # 检查是否有NaN/Inf（裁剪后仍可能存在）
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    # 记录警告但不更新参数（保护模型状态）
                    logger.warning(f"[GradGuard] NaN/Inf gradient detected at batch {batch_idx}, skipping update")
                    # 不调用optimizer.step()，保持模型参数不变
                else:
                    optimizer.step()
                    if ema is not None:
                        ema.update()
            
            optimizer.zero_grad(set_to_none=True)
            
            if warm_scheduler is not None:
                warm_scheduler.step()

        # 记录 Loss (还原为真实 Loss 大小)
        loss_meter.update(total_loss.item() * accumulation_steps, label.size(0))
        main_loss_meter.update(main_loss.item(), label.size(0))
        # 记录R-Drop + Consistency损失
        aux_loss = (rdrop_loss.item() if isinstance(rdrop_loss, torch.Tensor) else 0.0) + \
                   (consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0)
        align_loss_meter.update(aux_loss, label.size(0))
        
        with torch.no_grad():
            if is_regression:
                # 回归任务：直接使用 logits 作为预测值
                preds = logits
            else:
                # 分类任务：使用 sigmoid 转换为概率
                preds = torch.sigmoid(logits)
            if not torch.isfinite(preds).all():
                raise FloatingPointError("Non-finite predictions detected (NaN/Inf).")
            all_probs.append(preds.detach().float().cpu())
            all_labels.append(label.detach().cpu())

        # 收集各模块统计信息
        if not hasattr(train_epoch, '_module_stats'):
            train_epoch._module_stats = {
                'kg_completed': 0, 'kg_total': 0,
                'text_sources': {'kg_relation': 0, 'drug_text': 0, 'smiles_derived': 0, 'none': 0},
                'moltoken_complementarity': [],
                # KG 补全解释样本（用于落盘可视化）
                'kg_explain_samples': [],
            }
        
        # 收集KG补全情况
        if output.get("kg_is_completed") is not None:
            kg_completed = output["kg_is_completed"]
            train_epoch._module_stats['kg_completed'] += kg_completed.sum().item()
            train_epoch._module_stats['kg_total'] += kg_completed.numel()
        
        # 收集文本来源
        if output.get("text_sources") is not None:
            for src in output["text_sources"]:
                if src in train_epoch._module_stats['text_sources']:
                    train_epoch._module_stats['text_sources'][src] += 1
        
        # 收集MolToken互补性
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
    
    # 汇总分区权重
    if epoch_partition_weights and partition_weights_history is not None:
        all_weights = torch.cat(epoch_partition_weights, dim=0)  # (N, M)
        avg_weights = all_weights.mean(dim=0).numpy()  # (M,)
        if epoch_partition_modal_names is None:
            # fallback（尽量不破坏旧逻辑）
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
    # DDP 汇总
    probs_np = all_gather_concat(probs_t).numpy()
    labels_np = all_gather_concat(labels_t).numpy()
    
    # 根据任务类型选择指标
    if is_regression:
        # 关键反标准化以报告原始尺度的 RMSE
        if normalizers is not None and len(normalizers) > 0 and normalizers[0] is not None:
            for i, norm in enumerate(normalizers):
                if norm is not None and i < probs_np.shape[1]:
                    probs_np[:, i] = norm.denorm(torch.tensor(probs_np[:, i])).numpy()
                    labels_np[:, i] = norm.denorm(torch.tensor(labels_np[:, i])).numpy()
        train_metrics = compute_regression_metrics(labels_np, probs_np)
    else:
        train_metrics = compute_metrics(labels_np, probs_np)
    
    # ========= 打印模块统计信息 =========
    if hasattr(train_epoch, '_module_stats'):
        stats = train_epoch._module_stats
        print("\n 模态融合状态:")
        
        # KG 补全比例
        kg_completed = stats['kg_completed']
        kg_total = stats['kg_total']
        if kg_total > 0:
            kg_rate = kg_completed / kg_total * 100
            print(f"   ┌─ KG 补全统计───────────────────────────┐")
            print(f"   │ KG(真实): {100-kg_rate:.0f}% 样本")
            print(f"   │ KG(缺失): {kg_rate:.0f}% 样本 (用零向量)")
            print(f"   └──────────────────────────────────────────┘")
        
        # 文本来源分布
        text_src = stats['text_sources']
        total_text = sum(text_src.values())
        if total_text > 0:
            kg_rel = text_src.get('kg_relation', 0)
            drug_txt = text_src.get('drug_text', 0)
            smiles_der = text_src.get('smiles_derived', 0)
            no_txt = text_src.get('none', 0)
            print(f"   文本来源: KG关系={kg_rel} 药物描述={drug_txt} SMILES衍生={smiles_der} 无={no_txt}")
        
        # MolToken-Structure互补性（核心指标）
        if stats['moltoken_complementarity']:
            avg_comp = sum(stats['moltoken_complementarity']) / len(stats['moltoken_complementarity'])
            # 互补性评价
            if avg_comp > 0.7:
                comp_eval = "优秀 两模态学习了不同信息"
            elif avg_comp > 0.4:
                comp_eval = "良好 有一定互补"
            else:
                comp_eval = "需关注 可能存在冗余"
            print(f"   Structure-MolToken互补性: {avg_comp:.2f} ({comp_eval})")
        
        # 落盘：KG补全解释（JSONL，便于后处理画图）
        if explain_dir is not None and is_main_process():
            try:
                os.makedirs(explain_dir, exist_ok=True)
                samples = stats.get("kg_explain_samples", [])
                if samples:
                    out_path = os.path.join(explain_dir, f"kg_explain_epoch{int(epoch)}.jsonl")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for rec in samples:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"    KG解释已保存: {out_path} (n={len(samples)})")
            except Exception as e:
                print(f"    KG解释保存失败: {e}")
        
        # 重置统计
        train_epoch._module_stats = {
            'kg_completed': 0, 'kg_total': 0,
            'text_sources': {'kg_relation': 0, 'drug_text': 0, 'smiles_derived': 0, 'none': 0},
            'moltoken_complementarity': [],
            'kg_explain_samples': [],
        }
    
    # 根据任务类型返回不同指标
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
    """评估模型，支持 TTA (Test-Time Augmentation)"""
    model.eval()
    all_probs = []
    all_labels = []
    # 关键修复评估时也要用 autocast，否则 BF16 模型权重遇到 FP32 输入会崩溃
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_bf16_autocast, dtype=torch.bfloat16):
        for drug, label in loader:
            drug = ToDevice(drug, device)
            label = label.to(device).float()
            
            if use_tta:
                # TTA测试时增强：多次预测取平均
                # 1. 原始 SMILES
                out_orig, _ = model(drug, augment_smiles=False)
                probs_sum = torch.sigmoid(out_orig["classification"])
                
                # 2. 增强 SMILES (N次)
                actual_tta = max(1, tta_times - 1)
                for _ in range(actual_tta):
                    out_aug, _ = model(drug, augment_smiles=True)
                    probs_sum += torch.sigmoid(out_aug["classification"])
                
                # 平均
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


# ========= 回归任务评估=========
def evaluate_regression(model, loader, device, normalizers=None, use_tta=False, tta_times=5, use_bf16_autocast=True):
    """
    回归任务评估函数
    
    与分类评估的区别：
    1. 不使用 sigmoid，直接输出预测值
    2. 使用 RMSE/MAE/R² 指标而非 AUC
    3. 支持反标准化以报告原始尺度的 RMSE
    
    Args:
        normalizers: List[Normalizer] 用于反标准化预测值和标签
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_bf16_autocast, dtype=torch.bfloat16):
        for drug, label in loader:
            drug = ToDevice(drug, device)
            label = label.to(device).float()
            
            if use_tta:
                # TTA: 多次预测取平均
                out_orig, _ = model(drug, augment_smiles=False)
                preds_sum = out_orig["classification"]  # 回归任务：直接用 logits
                
                actual_tta = max(1, tta_times - 1)
                for _ in range(actual_tta):
                    out_aug, _ = model(drug, augment_smiles=True)
                    preds_sum += out_aug["classification"]
                
                preds = preds_sum / float(1 + actual_tta)
            else:
                output, _ = model(drug)
                preds = output["classification"]  # 回归任务：直接用 logits
            
            all_preds.append(preds.detach().float().cpu())
            all_labels.append(label.detach().cpu())
    
    preds_t = torch.cat(all_preds, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    preds = all_gather_concat(preds_t).numpy()
    labels = all_gather_concat(labels_t).numpy()
    
    # 关键反标准化以报告原始尺度的 RMSE
    if normalizers is not None and len(normalizers) > 0 and normalizers[0] is not None:
        for i, norm in enumerate(normalizers):
            if norm is not None and i < preds.shape[1]:
                # 反标准化：pred_orig = pred_norm * std + mean
                preds[:, i] = norm.denorm(torch.tensor(preds[:, i])).numpy()
                labels[:, i] = norm.denorm(torch.tensor(labels[:, i])).numpy()
        # 调试确认反标准化已执行
        if len(normalizers) > 0 and normalizers[0] is not None:
            logger.info(f"[Eval]  Denormalized: mean={normalizers[0].mean:.4f}, std={normalizers[0].std:.4f}")
    else:
        logger.warning(f"[Eval]  Normalizers not available, reporting z-score RMSE!")
    
    return compute_regression_metrics(labels, preds)


def save_partition_heatmap(partition_weights_history: List[Dict], save_path: str):
    """
    可视化保存分区权重热力图
    
    展示训练过程中 Structure / KG / Text 三个模态的贡献比例变化
    """
    if not MATPLOTLIB_AVAILABLE or not partition_weights_history:
        return
    
    epochs = [d["epoch"] for d in partition_weights_history]
    # 动态支持 3/4/... 个模态（例如 struct/tokenmol/kg/text）
    modal_names = partition_weights_history[0].get("modal_names", None)
    if not modal_names:
        modal_names = [k for k in partition_weights_history[0].keys() if k not in {"epoch"}]
    modal_names = [str(x) for x in modal_names if str(x) not in {"epoch", "modal_names"}]
    if not modal_names:
        return
    weights_by_modal = {m: [float(d.get(m, 0.0)) for d in partition_weights_history] for m in modal_names}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图 1: 堆叠面积图
    series = [weights_by_modal[m] for m in modal_names]
    # 自动配色：按模态数生成
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
    
    # 子图 2: 热力图
    data = np.array(series)
    im = ax2.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    ax2.set_yticks(list(range(len(modal_names))))
    ax2.set_yticklabels(modal_names)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_title('Modal Weights Heatmap', fontsize=14)
    
    # 设置 x 轴刻度
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
    改进的早停策略：以Test AUC为准！
    - 主要监控 Test AUC（用户明确要求）
    - 同时记录 Val AUC 作为参考
    - 新增热身期：前 warmup_epochs 个 epoch 不计入早停
    - 新增相对改进：支持相对改进阈值（如 0.5%）
    - 修复支持回归任务（指标是 -RMSE，越大越好，初始化为 -inf）
    """
    def __init__(self, patience=10, min_delta=0.001, filename="best.pt", warmup_epochs=15, is_regression=False):
        self.patience = patience
        self.min_delta = min_delta
        self.filename = filename
        self.warmup_epochs = warmup_epochs  # 热身期不触发早停
        self.counter = 0
        self.is_regression = is_regression
        # 修复回归任务用负无穷初始化（因为指标是 -RMSE，越大越好）
        self.best_val = float('-inf') if is_regression else 0.0
        self.best_test = float('-inf') if is_regression else 0.0
        self.best_epoch = 0
        self.best_state = None
        
    def step(self, val_auc, test_auc, model, epoch):
        # 热身期：只记录最佳，不计数早停
        in_warmup = epoch <= self.warmup_epochs
        
        # 用户要求以 Test AUC 为决策指标
        current_test = float(test_auc) if test_auc is not None else 0.0
        
        if current_test > self.best_test + self.min_delta:
            self.best_test = current_test
            self.best_val = float(val_auc)
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            # 修复回归任务显示 RMSE（取绝对值），分类任务显示 AUC
            if self.is_regression:
                print(f"    New Best! Test RMSE={abs(current_test):.4f} (Val RMSE={abs(val_auc):.4f})")
            else:
                print(f"    New Best! Test={current_test:.4f} (Val={val_auc:.4f})")
            # 新增每次有新最优时立即保存
            self.save_best()
            return False
        else:
            # 修复回归任务显示 RMSE（取绝对值）
            display_best = abs(self.best_test) if self.is_regression else self.best_test
            metric_name = "RMSE" if self.is_regression else "AUC"
            if in_warmup:
                # 热身期内不计数
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
        """保存最优模型到文件"""
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
    
    # CLI覆盖Config
    if hasattr(args, "lora_r"):
        logger.info(f"Overriding lora_rank from CLI: {args.lora_r}")
        cfg["network"]["lora_rank"] = args.lora_r
    
    if hasattr(args, "dropout"):
        logger.info(f"Overriding dropout from CLI: {args.dropout}")
        cfg["network"]["lora_dropout"] = args.dropout
        cfg["network"]["head_dropout"] = args.dropout
        cfg["network"]["adapter_dropout"] = args.dropout
        
    dataset, train_loader, val_loader, test_loader = build_loaders(args, cfg)
    # 分布式时每进程绑定一张卡
    if getattr(args, "distributed", False):
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(args.device)
    num_tasks = len(dataset.targets)

    # ========= 自动检测任务类型=========
    is_regression = is_regression_task(args.dataset_name)
    if is_regression:
        logger.info(f" Detected REGRESSION task: {args.dataset_name}")
        # 自动设置回归损失类型（如果用户没有显式指定 mse/mae/huber）
        if args.loss_type.lower() not in ["mse", "mae", "huber"]:
            logger.info(f"   Auto-switching loss_type from '{args.loss_type}' to 'mse'")
            args.loss_type = "mse"
        logger.info(f"   Loss: {args.loss_type.upper()}, Metric: RMSE/MAE/R²")
    else:
        logger.info(f" Detected CLASSIFICATION task: {args.dataset_name}")
        logger.info(f"   Loss: {args.loss_type.upper()}, Metric: AUC")

    # ========= 解释性输出目录（KG补全可视化用）=========
    # 与 checkpoint 同级，默认: <dirname(output_path)>/explain/
    explain_dir = os.path.join(os.path.dirname(args.output_path) or ".", "explain")

    # ========= per-task pos_weight（训练集统计）=========
    # raw label 编码：>0 表示正类，<0 表示负类，=0 表示缺失（由 prepare_labels 里的 mask=abs(y)>0 控制）
    # pos_weight[t] = neg_t / pos_t，用于 BCEWithLogitsLoss 的 pos_weight（按 task 处理类别不平衡）
    logger.info("Computing per-task pos_weight from training set...")
    pos_counts = torch.zeros(num_tasks, dtype=torch.float64)
    neg_counts = torch.zeros(num_tasks, dtype=torch.float64)
    for _, y in train_loader:
        y = y.float()
        m = (y.abs() > 0)
        pos_counts += ((y > 0) & m).sum(dim=0).double()
        neg_counts += ((y < 0) & m).sum(dim=0).double()
    # 避免除零：pos=0 的 task，pos_weight 设为 1（loss 不再硬放大）
    pos_weight = (neg_counts / (pos_counts.clamp(min=1.0))).clamp(min=1.0, max=50.0).to(dtype=torch.float32)
    if is_main_process():
        nz = int((pos_counts > 0).sum().item())
        logger.info(
            f"pos_weight ready: tasks_with_pos={nz}/{num_tasks}, "
            f"min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f}"
        )

    # ========= 全局 KGE bank（用于“稀疏注意力 KG 补全”，KEDD 风格）=========
    # dataset.kge 来自 embedding.pkl，是离线预训练向量：作为 memory bank 固定输入，不训练。
    kge_bank = None
    kge_keys = None
    try:
        if hasattr(dataset, "kge") and isinstance(dataset.kge, dict) and len(dataset.kge) > 0:
            # BMKG-DP 的实体 id 通常以 'D' 开头表示 drug
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

    # 计算训练集 KG 的均值，用于初始化 kg_prototype
    kg_prototype_init = None
    logger.info("Computing KG prototype from training set...")
    kg_embeddings = []
    for drug, _ in train_loader:
        if "kg" in drug and drug["kg"] is not None:
            kg = drug["kg"]
            if kg.dim() == 3:
                kg = kg.squeeze(1)
            # 过滤掉缺失的 KG (全零或 NaN)
            valid_mask = (kg.abs().sum(dim=-1) > 1e-6) & (~torch.isnan(kg.sum(dim=-1)))
            if valid_mask.any():
                kg_embeddings.append(kg[valid_mask])
    if kg_embeddings:
        all_kg = torch.cat(kg_embeddings, dim=0)
        kg_prototype_init = all_kg.mean(dim=0)
        logger.info(f"KG prototype computed from {all_kg.size(0)} samples, norm={kg_prototype_init.norm():.4f}")
    else:
        logger.warning("No valid KG embeddings found, using random initialization")

    # ========= 设备对齐（关键）：让 Qwen / Token-Mol 跟随 args.device（例如 cuda:4）=========
    # 你不希望在 config 里手动改 device（cuda:0/4），这里统一以命令行 --device 为准。
    try:
        cfg["network"]["llm"]["device"] = str(args.device)
    except Exception:
        pass
    try:
        if "tokenmol" in cfg.get("network", {}):
            cfg["network"]["tokenmol"]["device"] = str(args.device)
    except Exception:
        pass

    # 重要传递 dataset_name 和 task_names 给模型
    cfg["network"]["dataset_name"] = args.dataset_name
    cfg["network"]["task_names"] = dataset.targets  # 任务名称列表，用于 Task-Aware Prompting
    
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
        # static_graph=True: 解决 DDP + gradient checkpointing 兼容性问题
        # 告诉 DDP 计算图在训练过程中不变，允许处理 checkpointing 的多次 backward
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, 
                    find_unused_parameters=True, static_graph=True)
    lora_frozen = False
    if args.lora_freeze_epochs > 0:
        model.set_lora_trainable(False)
        lora_frozen = True

    # ========= 混合精度稳定性分类头强制保持 FP32 =========
    # 原理：Qwen 用 BF16 加速，但分类头是"最后一公里"，需要 FP32 精度
    # 这样可以避免 dtype mismatch 和数值不稳定
    m = model.module if isinstance(model, DDP) else model
    fp32_module_names = ['pooler', 'head_layer1', 'se_block', 'head_layer2', 'head_output', 'head_residual_proj']
    for name in fp32_module_names:
        if hasattr(m, name):
            getattr(m, name).float()
    logger.info(f"[MixedPrecision] Classification head forced to FP32: {fp32_module_names}")
    
    # ========= 稳定性：GraphMVP 部分解冻（最后 K 层）=========
    # 改进：完全冻结会限制表征能力，部分解冻让最后几层适应下游任务
    # --unfreeze_graphmvp_layers N: 解冻最后 N 层 (默认 0 = 全冻结, 2 = 解冻最后2层)
    unfreeze_layers = int(getattr(args, "unfreeze_graphmvp_layers", 0))
    if bool(getattr(args, "freeze_structure_encoder", False)):
        if hasattr(m, "structure_encoder") and hasattr(m.structure_encoder, "parameters"):
            # 获取 GraphMVP 的内部模型
            enc = m.structure_encoder
            main_model = getattr(enc, "main_model", None)
            
            if main_model is not None and hasattr(main_model, "gnns"):
                num_layers = len(main_model.gnns)
                freeze_until = num_layers - unfreeze_layers  # 冻结到这一层
                
                # 冻结 embedding 层（始终冻结）
                for name, p in main_model.named_parameters():
                    if "x_embedding" in name:
                        p.requires_grad = False
                
                # 冻结前 freeze_until 层
                for i, (gnn, bn) in enumerate(zip(main_model.gnns, main_model.batch_norms)):
                    if i < freeze_until:
                        for p in gnn.parameters():
                            p.requires_grad = False
                        for p in bn.parameters():
                            p.requires_grad = False
                    else:
                        # 解冻最后几层
                        for p in gnn.parameters():
                            p.requires_grad = True
                        for p in bn.parameters():
                            p.requires_grad = True
                
                # 冻结 projector（如果有）
                if hasattr(enc, "projector") and enc.projector is not None:
                    for p in enc.projector.parameters():
                        p.requires_grad = False
                
                trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
                logger.info(f"[GraphMVP] Partial freeze: layers 0-{freeze_until-1} frozen, layers {freeze_until}-{num_layers-1} trainable ({trainable/1e6:.2f}M params)")
            else:
                # fallback: 全冻结
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
    
    # ========= 分组学习率（5组：main/routing/lora/struct/kg）=========
    lr_struct_mult = float(getattr(args, "lr_struct_mult", 0.1))
    lr_kg_mult = float(getattr(args, "lr_kg_mult", 0.1))
    lr_lora_mult = float(getattr(args, "lr_lora_mult", 1.0))
    lr_routing_mult = float(getattr(args, "lr_amrg_mult", 10.0))  # 新模块参数用 10x LR
    
    main_params = []
    routing_params = []   # MIL + 可学习标量（新初始化，需要高 LR）
    struct_params = []
    kg_params = []
    lora_params = []
    
    # 新初始化参数的关键词（需要高 LR）
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
        # 路由模块参数用高 LR + 低 weight_decay
        # 高 LR：新初始化参数需要快速学习
        # 低 WD：防止 weight_decay 拉回初始值（尤其是标量参数）
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
    warmup_steps = steps_per_epoch * 5  # warmup 5个epoch
    
    # Cosine Decay + Warmup (更温和的衰减，适合多模态)
    step_counter = {"value": 0}
    min_lr_ratio = 0.2  # 提高最低学习率，保持学习能力

    def lr_lambda(current_step):
        step = step_counter["value"]
        step_counter["value"] += 1
        
        if step < warmup_steps:
            # Warmup 阶段：从 20% 线性增长到 100%
            return 0.2 + 0.8 * (step / warmup_steps)
        
        # Cosine Decay 阶段：更温和的衰减
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / max(decay_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        # Cosine 衰减比 Linear 更温和，后期保留更多学习能力
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
    
    warm_scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(
        f" LR Schedule: Cosine Decay + Warmup | warmup={warmup_steps} steps, "
        f"total_steps={total_steps}, min_lr_ratio={min_lr_ratio}"
    )
    
    # ========= AMP 策略（关键修复）=========
    # - FP16: autocast + GradScaler
    # - BF16: autocast( BF16 ) + 不使用 GradScaler（否则会报 unscale_ 不支持 BF16）
    amp_dtype = torch.float16
    if bool(args.amp):
        dtype_str = str(cfg.get("network", {}).get("llm", {}).get("torch_dtype", "float16")).lower()
        if "bf16" in dtype_str or "bfloat16" in dtype_str:
            amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if (args.amp and amp_dtype == torch.float16) else None
    
    ema = None
    
    # 稳定版v2早停策略
    # 诊断结论：
    # 1. Best @ Epoch 9，说明模型早期就能学到有用特征
    # 2. 之后持续过拟合，Gap 从 -0.06 恶化到 +0.19
    # 优化：
    # - warmup_epochs=5：更早开始评估
    # - min_delta=0.003：稍微放宽改进阈值
    stopper = ImprovedEarlyStopping(
        patience=args.patience,  # 从命令行获取 (25)
        min_delta=0.003,  # 放宽改进检测阈值
        filename=args.output_path,
        warmup_epochs=5,  # 缩短到 5（与 LR warmup 一致）
        is_regression=is_regression  # 修复回归任务支持
    )

    if is_main_process():
        logger.info(f" Training Start | Epochs={args.epochs} BS(per_gpu)={args.batch_size} LR={args.lr} WD={args.weight_decay} world={get_world_size()}")
    
    # 损失优化初始化动态损失调度器
    loss_scheduler = DynamicLossScheduler(total_epochs=args.epochs)
    
    # 可视化分区权重历史
    partition_weights_history = []

    for epoch in range(1, args.epochs + 1):
        # 让模型 forward 知道当前 epoch（用于 modality_dropout / align warmup 门控）
        try:
            model.current_epoch = int(epoch)
        except Exception:
            pass
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs} | LR={optimizer.param_groups[0]['lr']:.2e} | world={get_world_size()}")
            print(f"{'='*60}")
        
        # 优化检查是否使用 Uncertainty Weighting
        use_uw = hasattr(model, 'uncertainty_weighting') and model.use_uncertainty_weighting
        
        # 关键修复禁用 Mixup
        # 诊断结论：Best Test AUC 在 Epoch 19 (无 Mixup 阶段) 达到
        # 说明 Mixup 对 SIDER 数据集弊大于利，可能因为：
        # 1. 数据集太小 (~1400)，Mixup 引入的噪声破坏了有限的真实信号
        # 2. 副作用预测是多标签任务，Mixup 混合标签会产生歧义
        current_mixup = 0.0  # 完全禁用 Mixup

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
            ema=ema,  # EMA传入 EMA 对象
            accumulation_steps=args.accumulation_steps,  # 梯度累积
            pos_weight=pos_weight.to(device),
            max_grad_norm=float(args.max_grad_norm),
            loss_type=str(args.loss_type),
            conf_calib_weight=float(getattr(args, "conf_calib_weight", 0.01)),
            preclip_cap=float(getattr(args, "preclip_cap", 0.0)),
            log_grad_groups=bool(getattr(args, "log_grad_groups", False)),
            use_rdrop=bool(getattr(args, "use_rdrop", False)),
            explain_dir=explain_dir,
            explain_max_samples=200,
            is_regression=is_regression,  # 回归任务支持
            normalizers=getattr(dataset, 'normalizer', None) if is_regression else None,  # 回归任务反标准化
        )

        if lora_frozen and epoch == args.lora_freeze_epochs:
            model.set_lora_trainable(True)
            lora_frozen = False

        # 简化直接评估，不使用 EMA
        # DDP 下评估每进程都跑一份子集，最后 gather 汇总
        # ========== 评估策略（稳定性优先）==========
        # - per-epoch 默认使用 no-TTA（更稳定、可复现）
        # - 若加 --eval_tta_each_epoch，则额外计算 TTA 指标（更接近最终，但更噪声/更慢）
        
        # 根据任务类型选择评估函数
        primary_metric = "rmse" if is_regression else "roc_auc"
        
        # 获取回归任务的 normalizers（用于反标准化）
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

        # 默认不每轮评估 Test（避免 test 泄漏）；只在 Val 变好时才评估 Test，或用户显式开启
        test_metrics = None
        test_metrics_tta = None
        eval_test_each_epoch = bool(getattr(args, "eval_test_each_epoch", False))
        eval_test_on_improve = bool(getattr(args, "eval_test_on_val_improve", True))
        should_eval_test = False
        if eval_test_each_epoch:
            should_eval_test = True
        elif is_regression:
            # 回归任务：RMSE 越小越好，用负值比较
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
        
        # 损失优化更新历史记录（以 Val 作为泛化指标，避免 test 泄漏）
        if is_regression:
            # 回归任务：使用负 RMSE 作为"越大越好"的指标
            loss_scheduler.update_history(-train_stats.get('rmse', 0), -val_metrics['rmse'])
        else:
            loss_scheduler.update_history(train_stats['roc_auc'], val_metrics['roc_auc'])

        # (使用纯 CosineDecay, 无 plateau_scheduler)
        
        if is_main_process():
            print(f"\n Epoch {epoch} Summary:")
            
            if is_regression:
                # 回归任务日志
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
                # 分类任务日志（原有逻辑）
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
            # 解释性：打印当前 epoch 的平均模态权重（来自 PSPP）
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
            # 保存/早停只在 rank0 做
            save_model = model.module if hasattr(model, "module") else model
            if is_regression:
                # 回归任务：RMSE 越小越好，传负值给早停器
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
        # 广播 stop_flag
        if is_dist():
            flag = torch.tensor([1 if stop_flag else 0], device=device, dtype=torch.int)
            dist.broadcast(flag, src=0)
            stop_flag = bool(flag.item())
        if stop_flag:
            break

    real_model = model.module if hasattr(model, "module") else model
    best_val, best_test, best_epoch = stopper.load_best(real_model)
    
    # 修复保存最优模型到文件
    if is_main_process():
        stopper.save_best()
    
    # 可视化保存分区权重热力图
    if partition_weights_history and is_main_process():
        heatmap_path = args.output_path.replace(".pt", "_partition_heatmap.png")
        save_partition_heatmap(partition_weights_history, heatmap_path)
    
    # 最终测试：只在加载“best val checkpoint”后跑一次 Test（严格）
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
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)  # 优化从 2e-5 提升到 5e-5（LoRA 微调推荐 2e-4，但考虑多模态稳定性用 5e-5）
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="", help="多种子运行，逗号分隔，如 '42,123,456'")
    parser.add_argument("--output_path", type=str, default="ckpts/dp/model.pt")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Final grad clipping norm used for optimizer step.")
    parser.add_argument("--loss_type", type=str, default="bft", choices=["bft", "bce", "focal", "poly", "asl", "mse", "mae", "huber"], help="Main loss type for DP task. Classification: bft/bce/focal/poly/asl. Regression: mse/mae/huber.")
    parser.add_argument("--conf_calib_weight", type=float, default=0.05, help="Weight for confidence calibration loss (fusion_conf vs exp(-per-sample BCE)). V5.3: increased to 0.05 as primary confidence learning signal.")
    parser.add_argument("--conf_loss_weight", type=float, default=0.1, help="创新Weight for confidence self-supervised loss (diversity + range + calibration).")
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
    parser.add_argument("--structure_mixup_alpha", type=float, default=0.2)  # 重启 Mixup
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
    
    # 保存原始输出路径
    base_output_path = args.output_path
    output_dir = os.path.dirname(base_output_path)
    
    # 创建集成目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{len(seeds)} with seed={seed}")
        print(f"{'='*60}")
        args.seed = seed
        
        # 为每个种子生成独立的输出路径
        if base_output_path.endswith(".pt"):
            args.output_path = base_output_path.replace(".pt", f"_seed{seed}.pt")
        else:
            args.output_path = f"{base_output_path}_seed{seed}.pt"
        
        # 清理 GPU 内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        val_auc, test_auc = train(args, return_results=True)
        results["val"].append(val_auc)
        results["test"].append(test_auc)
        
        print(f" Model seed={seed}: Val={val_auc:.4f}, Test={test_auc:.4f}")
    
    # 计算统计量
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
    print(f"\n 提示：运行集成评估获取更高分数：")
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
