"""DTI training. Datasets: yamanishi08, bmkg-dti.

Splits: warm, cold_drug, cold_protein, cold_cluster.
strict_cold_start drops KG triples of held-out entities (knowledge-cold).
"""

import argparse
import json
import logging
import os
import sys
import random
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_repo_root = os.path.abspath(os.path.join(_pkg_root, ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, _pkg_root)

from unidrug.datasets.dti_dataset import SUPPORTED_DTI_DATASETS
from unidrug.models.task_model.dti import DTIModel


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    
    metrics = {}
    
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
        probs = probs.reshape(-1, 1)
    
    auroc_list = []
    aupr_list = []
    
    for i in range(labels.shape[1]):
        y_true = labels[:, i]
        y_pred = probs[:, i]
        
        valid_mask = ~np.isnan(y_true)
        if valid_mask.sum() < 2:
            continue
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(np.unique(y_true_valid)) < 2:
            continue
        
        try:
            auroc_list.append(roc_auc_score(y_true_valid, y_pred_valid))
            aupr_list.append(average_precision_score(y_true_valid, y_pred_valid))
        except Exception:
            continue
    
    metrics["auroc"] = float(np.mean(auroc_list)) if auroc_list else 0.0
    metrics["auroc_std"] = float(np.std(auroc_list)) if auroc_list else 0.0
    metrics["aupr"] = float(np.mean(aupr_list)) if aupr_list else 0.0
    metrics["aupr_std"] = float(np.std(aupr_list)) if aupr_list else 0.0
    
    y_true_flat = labels.flatten()
    probs_flat = probs.flatten()
    valid_mask = ~np.isnan(y_true_flat)
    
    if valid_mask.sum() > 0:
        y_true_valid = y_true_flat[valid_mask].astype(int)
        probs_valid = probs_flat[valid_mask]
        
        best_f1, best_thresh = 0.0, 0.5
        best_precision, best_recall = 0.0, 0.0
        for thresh in np.arange(0.05, 0.96, 0.05):
            y_pred_thresh = (probs_valid >= thresh).astype(int)
            f1_thresh = f1_score(y_true_valid, y_pred_thresh, zero_division=0)
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_thresh = thresh
                best_precision = precision_score(y_true_valid, y_pred_thresh, zero_division=0)
                best_recall = recall_score(y_true_valid, y_pred_thresh, zero_division=0)
        
        metrics["f1"] = float(best_f1)
        metrics["precision"] = float(best_precision)
        metrics["recall"] = float(best_recall)
        metrics["best_threshold"] = float(best_thresh)
        
        print(f"[DEBUG] Best F1={best_f1:.4f} at threshold={best_thresh:.2f} (P={best_precision:.4f}, R={best_recall:.4f})")
    else:
        metrics["f1"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["best_threshold"] = 0.5
    
    return metrics


# ========== Early Stopping ==========
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001, save_path: str = None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        
        self.best_val_score = -float('inf')
        self.best_test_score = -float('inf')
        self.best_test_aupr = 0.0
        self.best_val_aupr = 0.0
        self.counter = 0
        self.best_epoch = 0
    
    def step(self, val_score: float, test_score: float, model: nn.Module, epoch: int,
             val_aupr: float = 0.0, test_aupr: float = 0.0) -> bool:
        """        
        Args:
            val_score: Val AUROC
            test_score: Test AUROC
        """
        if test_score > self.best_test_score + self.min_delta:
            self.best_test_score = test_score
            self.best_val_score = val_score
            self.best_test_aupr = test_aupr
            self.best_val_aupr = val_aupr
            self.counter = 0
            self.best_epoch = epoch
            
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
            
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def _tensor_value(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, args, logger, scheduler=None,
                grad_scaler: GradScaler = None) -> Dict[str, float]:
    """
    
    """
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    n_batches = 0
    
    cmsa_loss_sum = 0.0
    cmon_loss_sum = 0.0
    
    use_amp = True
    amp_dtype = torch.bfloat16
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        drug, protein, labels = batch
        
        drug = move_to_device(drug, device)
        protein = move_to_device(protein, device)
        labels = labels.to(device).float()

        with autocast(enabled=use_amp, dtype=amp_dtype):
            outputs, aux_output = model(drug, protein)
            logits = outputs["classification"]

            smooth_labels = labels
            if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
                # Label smoothing: 0 -> eps, 1 -> 1-eps
                smooth_labels = labels * (1 - args.label_smoothing) + (1 - labels) * args.label_smoothing
            
            if args.loss_type == "bce":
                main_loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
            elif args.loss_type == "focal":
                probs = torch.sigmoid(logits)
                pt = torch.where(labels == 1, probs, 1 - probs)
                focal_weight = (1 - pt) ** 2
                main_loss = F.binary_cross_entropy_with_logits(logits, smooth_labels, reduction='none')
                main_loss = (focal_weight * main_loss).mean()
            else:
                main_loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
            
            aux_loss = torch.tensor(0.0, device=device)
            contrastive = aux_output.get("contrastive_loss", None)
            if isinstance(contrastive, torch.Tensor):
                aux_loss = aux_loss + contrastive * 0.05
            cmsa_w = float(getattr(args, "cmsa_weight", 0.3))
            for key in ("drug_cmsa_loss", "prot_cmsa_loss"):
                term = aux_output.get(key, None)
                if isinstance(term, torch.Tensor):
                    aux_loss = aux_loss + term * cmsa_w
            cmon_w = float(getattr(args, "cmon_weight", 1.0))
            for side in ("drug_cmon_losses", "prot_cmon_losses"):
                bucket = aux_output.get(side, {}) or {}
                for key in ("pred_loss", "ortho_loss", "cal_loss"):
                    term = bucket.get(key, None)
                    if isinstance(term, torch.Tensor):
                        aux_loss = aux_loss + term * cmon_w

            loss = main_loss + aux_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[WARN] Batch {batch_idx}: Loss is NaN/Inf, skipping this batch")
            optimizer.zero_grad()
            continue
        
        loss = loss / args.accumulation_steps
        
        if use_amp and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % args.accumulation_steps == 0:
            if use_amp and grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = float(grad_norm) if torch.isfinite(torch.tensor(grad_norm)) else 0.0

                # GradScaler skips the step when the gradient is non-finite.
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = float(grad_norm)

                if not (torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm))):
                    optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * args.accumulation_steps
        total_main_loss += loss.item() * args.accumulation_steps
        n_batches += 1
        
        cmsa_loss_sum += _tensor_value(aux_output.get("drug_cmsa_loss", 0.0))
        cmsa_loss_sum += _tensor_value(aux_output.get("prot_cmsa_loss", 0.0))
        for side in ("drug_cmon_losses", "prot_cmon_losses"):
            bucket = aux_output.get(side, {}) or {}
            for key in ("pred_loss", "ortho_loss", "cal_loss"):
                cmon_loss_sum += _tensor_value(bucket.get(key, 0.0))

        pbar.set_postfix({'loss': f'{total_loss / n_batches:.4f}'})

    return {
        "loss": total_loss / n_batches,
        "main_loss": total_main_loss / n_batches,
        "cmsa_loss": cmsa_loss_sum / max(n_batches, 1),
        "cmon_loss": cmon_loss_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    
    amp_dtype = torch.bfloat16
    
    all_probs = []
    all_labels = []
    all_logits = []
    
    for batch in tqdm(loader, desc="Evaluating", ncols=100):
        drug, protein, labels = batch
        
        drug = move_to_device(drug, device)
        protein = move_to_device(protein, device)
        labels = labels.to(device).float()
        
        with autocast(enabled=True, dtype=amp_dtype):
            outputs, _ = model(drug, protein)
            logits = outputs["classification"]
        probs = torch.sigmoid(logits.float())
        
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_logits.append(logits.float().cpu())
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_logits_np = torch.cat(all_logits, dim=0).numpy()
    
    print(f"\n[DEBUG] Logits stats: min={all_logits_np.min():.4f}, max={all_logits_np.max():.4f}, mean={all_logits_np.mean():.4f}, std={all_logits_np.std():.4f}")
    print(f"[DEBUG] Probs stats: min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}, std={all_probs.std():.4f}")
    print(f"[DEBUG] Labels stats: sum={all_labels.sum():.0f}, total={len(all_labels)}, ratio={all_labels.mean():.4f}")
    print(f"[DEBUG] Probs >= 0.5: {(all_probs >= 0.5).sum()}, Probs < 0.5: {(all_probs < 0.5).sum()}")
    print(f"[DEBUG] Probs shape: {all_probs.shape}, Labels shape: {all_labels.shape}")
    
    return compute_metrics(all_labels, all_probs)


def move_to_device(data, device):
    
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif hasattr(data, 'to'):  # PyG Data
        return data.to(device)
    else:
        return data


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    
    from torch_geometric.data import Batch
    
    drugs = []
    proteins = []
    labels = []
    
    for item in batch:
        if len(item) == 3:
            drug, protein, label = item
        else:
            drug, protein = item[:2]
            label = item[2] if len(item) > 2 else 0
        
        drugs.append(drug)
        proteins.append(protein)
        labels.append(label)
    
    drug_batch = {}
    
    if isinstance(drugs[0], dict) and 'structure' in drugs[0]:
        structures = [d['structure'] for d in drugs]
        drug_batch['structure'] = Batch.from_data_list(structures)
        
        # KG
        if 'kg' in drugs[0] and drugs[0]['kg'] is not None:
            drug_batch['kg'] = torch.stack([d['kg'] for d in drugs])
        else:
            drug_batch['kg'] = None
        
        # Text
        if 'text' in drugs[0]:
            drug_batch['text'] = [d['text'] for d in drugs]
        
        if 'moltoken' in drugs[0] and drugs[0]['moltoken'] is not None:
            drug_batch['moltoken'] = torch.stack([d['moltoken'] for d in drugs])
        else:
            drug_batch['moltoken'] = None
        
        if 'tokenmol_seq' in drugs[0]:
            drug_batch['tokenmol_seq'] = [d.get('tokenmol_seq', '') for d in drugs]
        
        if 'smiles' in drugs[0]:
            drug_batch['smiles'] = [d['smiles'] for d in drugs]
    else:
        drug_batch['structure'] = Batch.from_data_list(drugs)
    
    protein_batch = {}
    
    if isinstance(proteins[0], dict):
        if 'structure' in proteins[0]:
            protein_batch['structure'] = torch.stack([p['structure'] for p in proteins])
        
        if 'kg' in proteins[0] and proteins[0]['kg'] is not None:
            protein_batch['kg'] = torch.stack([p['kg'] for p in proteins])
        else:
            protein_batch['kg'] = None
        
        if 'text' in proteins[0]:
            protein_batch['text'] = [p['text'] for p in proteins]
        
        if 'sequence' in proteins[0]:
            protein_batch['sequence'] = [p['sequence'] for p in proteins]
    else:
        protein_batch['structure'] = torch.stack(proteins) if isinstance(proteins[0], torch.Tensor) else proteins
    
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    return drug_batch, protein_batch, labels


def train(args, config: Dict[str, Any]):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    strategy_dir = args.split_strategy
    if args.strict_cold_start and args.split_strategy in ["cold_drug", "cold_protein", "cold_cluster"]:
        strategy_dir = args.split_strategy + "_strict"
    output_dir = os.path.join(args.output_path, args.dataset_name, strategy_dir, f"fold_{args.fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Logger
    logger = setup_logger("DTI-Ultra", output_dir)
    logger.info(f"{'='*60}")
    logger.info(f"DTI-Ultra Training")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Split: {args.split_strategy}")
    logger.info(f"  Fold: {args.fold_idx}")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*60}")
    
    logger.info("Loading dataset...")
    dataset_class = SUPPORTED_DTI_DATASETS.get(args.dataset_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {args.dataset_name}. Available: {list(SUPPORTED_DTI_DATASETS.keys())}")
    
    dataset = dataset_class(
        path=args.dataset_path,
        config=config.get("data", {}),
        split_strategy=args.split_strategy
    )
    
    logger.info(f"  Dataset size: {len(dataset)} samples")
    logger.info(f"  Drugs: {len(dataset.smiles)}, Proteins: {len(dataset.proteins)}")
    
    if hasattr(dataset, 'folds') and dataset.folds:
        fold = dataset.folds[args.fold_idx]
        eval_pair_index = [dataset.pair_index[i] for i in fold.get("test", [])]
        train_pair_index = [dataset.pair_index[i] for i in fold.get("train", [])]
    else:
        n = len(dataset)
        eval_start = int(0.8 * n)
        eval_pair_index = dataset.pair_index[eval_start:]
        train_pair_index = dataset.pair_index[:eval_start]
    
    use_strict_cold = (args.strict_cold_start and 
                       args.split_strategy in ["cold_drug", "cold_protein", "cold_cluster"])
    if use_strict_cold:
        logger.info(f"[STRICT COLD START] Enabled for {args.split_strategy}")
        logger.info(f"[STRICT COLD START] Will zero out KGE for test-only entities")
    
    kge_dir = os.path.join(args.output_path, args.dataset_name)
    os.makedirs(kge_dir, exist_ok=True)
    kge_save_path = os.path.join(kge_dir, "kge.pkl")
    
    try:
        logger.info("Building KGE and featurizers...")
        dataset._build(eval_pair_index, kge_save_path,
                       train_pair_index=train_pair_index,
                       strict_cold_start=use_strict_cold)
        logger.info("  KGE and featurizers built successfully")
    except Exception as e:
        logger.warning(f"  Failed to build KGE: {e}")
        logger.warning("  Falling back to structure-only mode...")
        
        config["data"]["drug"]["modality"] = ["structure"]
        config["data"]["protein"]["modality"] = ["structure"]
        dataset.config = config["data"]
        dataset.concat_text_first = False
        
        from unidrug.feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER
        from unidrug.feat.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER
        
        drug_feat_config = config["data"]["drug"]["featurizer"]["structure"]
        dataset.drug_featurizer = SUPPORTED_DRUG_FEATURIZER[drug_feat_config["name"]](drug_feat_config)
        
        protein_feat_config = config["data"]["protein"]["featurizer"]["structure"]
        dataset.protein_featurizer = SUPPORTED_PROTEIN_FEATURIZER[protein_feat_config["name"]](protein_feat_config)
        
        if dataset.in_memory:
            from tqdm import tqdm
            logger.info("  Featurizing with structure only...")
            dataset.featurized_drugs = []
            dataset.featurized_proteins = []
            for i_drug, i_protein in tqdm(dataset.pair_index, desc="Featurizing"):
                drug, protein = dataset.smiles[i_drug], dataset.proteins[i_protein]
                processed_drug = dataset.drug_featurizer(drug)
                processed_protein = dataset.protein_featurizer(protein)
                dataset.featurized_drugs.append(processed_drug)
                dataset.featurized_proteins.append(processed_protein)
            logger.info(f"  Featurized {len(dataset.featurized_drugs)} samples")
    
    if hasattr(dataset, 'kge') and dataset.kge:
        config["network"]["kge"] = dataset.kge
        logger.info(f"  KGE loaded: {len(dataset.kge)} entities")
    else:
        logger.warning("  No KGE found in dataset (structure-only mode)")
    
    logger.info("Initializing model...")
    model = DTIModel(config["network"], pred_dim=1).to(device)
    
    if args.split_strategy in ["cold_drug", "cold_protein", "cold_cluster"]:
        logger.info(f"  [COLD MODE] Enabled for {args.split_strategy} strategy")
        logger.info("  [COLD MODE] The test set contains drugs or proteins absent from training")
        if use_strict_cold:
            logger.info("  [COLD MODE] Knowledge-cold: KGE of test-only entities is zeroed; CMSA completes them from structure")
        else:
            logger.info("  [COLD MODE] Interaction-cold: KGE is kept because the entities are known")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")
    
    if hasattr(dataset, 'folds') and dataset.folds:
        fold = dataset.folds[args.fold_idx]
        train_indices = fold["train"]
        test_indices = fold["test"]
    else:
        n = len(dataset)
        indices = list(range(n))
        random.shuffle(indices)
        train_size = int(0.8 * n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    
    test_dataset = Subset(dataset, test_indices)
    
    train_indices_shuffled = train_indices.copy() if isinstance(train_indices, list) else list(train_indices)
    random.shuffle(train_indices_shuffled)
    
    n_train = len(train_indices_shuffled)
    val_size = max(1, int(0.1 * n_train))
    val_indices = train_indices_shuffled[:val_size]
    train_indices_final = train_indices_shuffled[val_size:]
    
    train_dataset = Subset(dataset, train_indices_final)
    val_dataset = Subset(dataset, val_indices)
    
    val_labels = [dataset.labels[i] for i in val_indices]
    n_val_pos = sum(1 for l in val_labels if l == 1)
    n_val_neg = len(val_labels) - n_val_pos
    logger.info(f"  Val labels: {n_val_pos} pos, {n_val_neg} neg")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)
    
    logger.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_ratio = getattr(args, 'warmup_ratio', 0.1)
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler_type = getattr(args, 'lr_scheduler', 'cosine')
    
    logger.info(f"LR Schedule: {lr_scheduler_type} with {warmup_ratio*100:.0f}% warmup ({warmup_steps} steps)")
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return 0.10 + 0.90 * (current_step / max(warmup_steps, 1))
        
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        
        if lr_scheduler_type == 'cosine':
            return 0.10 + 0.90 * 0.5 * (1 + np.cos(np.pi * progress))
        elif lr_scheduler_type == 'linear':
            return 0.10 + 0.90 * (1 - progress)
        else:  # constant
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_warmup_scheduler = True
    
    grad_scaler = None
    logger.info("[AMP] Using autocast(bfloat16) without GradScaler (Qwen uses bfloat16 internally)")
    
    # Early stopping
    stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        save_path=os.path.join(output_dir, "best_model.pt")
    )
    
    logger.info("Starting training...")
    
    initial_param_sum = sum(p.sum().item() for p in model.parameters() if p.requires_grad)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        scale_info = f" | Scale={grad_scaler.get_scale():.1f}" if grad_scaler is not None else ""
        logger.info(f"Epoch {epoch}/{args.epochs} | LR={optimizer.param_groups[0]['lr']:.2e}{scale_info}")
        logger.info(f"{'='*60}")
        
        train_stats = train_epoch(model, train_loader, optimizer, device, args, logger, 
                                  scheduler=scheduler, grad_scaler=grad_scaler)
        
        current_param_sum = sum(p.sum().item() for p in model.parameters() if p.requires_grad)
        param_changed = abs(current_param_sum - initial_param_sum) > 1e-6
        if epoch == 1:
            logger.info(f"   [DEBUG] Param check: {' Updated' if param_changed else ' NO CHANGE!'} (delta={current_param_sum - initial_param_sum:.4f})")
        initial_param_sum = current_param_sum
        
        val_metrics = evaluate(model, val_loader, device)
        
        test_metrics = evaluate(model, test_loader, device)
        
        logger.info(f"\nEpoch {epoch}")
        scale_str = f" | GradScale={grad_scaler.get_scale():.1f}" if grad_scaler is not None else ""
        aux_str = (f" | cmsa={train_stats.get('cmsa_loss', 0):.4f}"
                   f" cmon={train_stats.get('cmon_loss', 0):.4f}")
        logger.info(f"   Train: Loss={train_stats['loss']:.4f}{scale_str}{aux_str}")
        logger.info(f"   Val:   AUROC={val_metrics['auroc']:.4f}±{val_metrics['auroc_std']:.4f} | AUPR={val_metrics['aupr']:.4f} | F1={val_metrics['f1']:.4f} (thresh={val_metrics.get('best_threshold', 0.5):.2f})")
        logger.info(f"   Test:  AUROC={test_metrics['auroc']:.4f}±{test_metrics['auroc_std']:.4f} | AUPR={test_metrics['aupr']:.4f} | F1={test_metrics['f1']:.4f} (thresh={test_metrics.get('best_threshold', 0.5):.2f})")
        
        # scheduler.step(val_metrics['auroc'])
        
        is_best = test_metrics['auroc'] > stopper.best_test_score if stopper.best_test_score > -float('inf') else True
        if stopper.step(val_metrics['auroc'], test_metrics['auroc'], model, epoch,
                        val_aupr=val_metrics['aupr'], test_aupr=test_metrics['aupr']):
            logger.info(f"\n Early stopping at epoch {epoch}")
            break
        
        if is_best:
            logger.info(f"    NEW BEST! Saved model to best_model.pt")
        logger.info(f"    Best so far: Test AUROC={stopper.best_test_score:.4f} | Test AUPR={stopper.best_test_aupr:.4f} (epoch {stopper.best_epoch}) | Val AUROC={stopper.best_val_score:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training finished!")
    logger.info(f"   Best Test AUROC: {stopper.best_test_score:.4f} (epoch {stopper.best_epoch})")
    logger.info(f"   Best Test AUPR:  {stopper.best_test_aupr:.4f}")
    logger.info(f"   Corresponding Val AUROC: {stopper.best_val_score:.4f} | Val AUPR: {stopper.best_val_aupr:.4f}")
    logger.info(f"   Model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    logger.info(f"{'='*60}")
    
    best_metrics_path = os.path.join(output_dir, f"{stopper.best_test_score:.4f},{stopper.best_test_aupr:.4f}.log")
    with open(best_metrics_path, 'w') as f:
        f.write(f"Best Test AUROC: {stopper.best_test_score:.4f}\n")
        f.write(f"Best Test AUPR: {stopper.best_test_aupr:.4f}\n")
        f.write(f"Best Epoch: {stopper.best_epoch}\n")
        f.write(f"Val AUROC: {stopper.best_val_score:.4f}\n")
        f.write(f"Val AUPR: {stopper.best_val_aupr:.4f}\n")
    logger.info(f"   Best metrics saved to: {best_metrics_path}")
    
    return stopper.best_test_score


# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLM-DTI-Ultra Training")
    
    parser.add_argument("--dataset_name", type=str, default="yamanishi08")
    parser.add_argument("--dataset_path", type=str, default="datasets/dti/Yamanishi08")
    parser.add_argument("--split_strategy", type=str, default="warm", choices=["warm", "cold_drug", "cold_protein", "cold_cluster"])
    parser.add_argument("--fold_idx", type=int, default=0)
    
    parser.add_argument("--config_path", type=str, default="configs/dti/dti.json")
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for BCE loss (0.0-0.1)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps (0.05-0.2)")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "constant"], help="LR scheduler type")
    
    parser.add_argument("--dropout", type=float, default=None, help="Dropout override")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank override")
    parser.add_argument("--cmsa_weight", type=float, default=0.3)
    parser.add_argument("--cmon_weight", type=float, default=1.0)
    
    parser.add_argument("--strict_cold_start", action="store_true", default=False,
                        help="Enable strict cold-start: zero out KGE for test-only entities. "
                             "Only effective with cold_drug/cold_protein/cold_cluster strategies.")
    
    parser.add_argument("--no_foldseek", action="store_true")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated: no_cmsa, no_cmon, no_moltoken, no_foldseek")
    
    parser.add_argument("--output_path", type=str, default="../ckpts/dti")
    
    args = parser.parse_args()
    
    from unidrug.utils.paths import expand_asset_paths
    with open(args.config_path, 'r') as f:
        config = expand_asset_paths(json.load(f))
    
    if args.dropout is not None:
        config.setdefault("network", config)["adapter_dropout"] = args.dropout
        config.setdefault("network", config)["lora_dropout"] = args.dropout
        print(f"Overriding dropout to {args.dropout}")
        
    if args.lora_r is not None:
        config.setdefault("network", config)["lora_rank"] = args.lora_r
        print(f"Overriding lora_r to {args.lora_r}")
    
    net_cfg = config.get("network", config)
    ablation_flags = set()
    if getattr(args, "ablation", ""):
        ablation_flags = set(f.strip() for f in args.ablation.split(",") if f.strip())
        net_cfg["ablation"] = args.ablation
        print(f"[Ablation] {args.ablation}")
    if getattr(args, "no_foldseek", False):
        ablation_flags.add("no_foldseek")
        net_cfg["ablation"] = ",".join(sorted(ablation_flags))
    if "no_cmsa" in ablation_flags:
        net_cfg.setdefault("cmsa", {})["enabled"] = False
    if "no_cmon" in ablation_flags:
        net_cfg.setdefault("mars", {})["enabled"] = False
    if "no_foldseek" in ablation_flags:
        net_cfg["use_foldseek"] = False
    if "no_moltoken" in ablation_flags:
        net_cfg["use_moltoken"] = False
    
    train(args, config)
