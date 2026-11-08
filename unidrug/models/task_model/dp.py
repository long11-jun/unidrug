"""
DPModel — Multimodal Drug Property Prediction

SA-CMSA (Sparse-Attention guided Cross-Modal Subspace Alignment):
  Stage 1: Multi-head sparse attention retrieval from kge_bank (43K entities)
  Stage 2: Cross-modal evidence (struct+mol) residual refinement

Data flow:
  Raw Features → SA-CMSA (complete missing KG) → Q-Former Adapters → Tokens
  → [modality markers + tokens + prompt] → Qwen-2.5 (LoRA) → Pooling
  → Prediction Head
"""
import json
import logging
import math
import os
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = AllChem = Descriptors = None
    RDKIT_AVAILABLE = False

from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


# =============================================================================
# 模块级优化1Stochastic Depth (Layer Drop) - 防止过拟合
# =============================================================================

# =============================================================================
# Feature Noise Injection - 训练时特征增强
# =============================================================================

class FeatureNoiseInjector(nn.Module):
    """
    修复3特征级噪声注入 - R-Drop兼容版
    
    关键修复：
    - 使用确定性噪声（基于输入的hash）
    - 确保R-Drop两次forward时噪声相同，避免差异放大导致梯度爆炸
    - 重要正确保存/恢复CPU和CUDA的随机状态，不影响Dropout
    
    原理：
    - 在特征空间添加微小高斯噪声
    - 迫使模型学习更鲁棒的表示
    """
    def __init__(self, noise_std: float = 0.1, adaptive: bool = True):
        super().__init__()
        self.noise_std = noise_std
        self.adaptive = adaptive
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.noise_std == 0.0:
            return x
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.adaptive:
            with torch.no_grad():
                feat_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                noise_scale = self.noise_std * feat_norm / (x.shape[-1] ** 0.5)
        else:
            noise_scale = self.noise_std
        
        with torch.no_grad():
            seed = int(x.sum().item() * 1e6) % (2**31)
        
        # 修复正确保存CPU和CUDA的随机状态
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if x.is_cuda else None
        
        # 创建独立的Generator，不影响全局状态
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * noise_scale
        
        # 恢复随机状态（确保Dropout不受影响）
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        
        return x + noise


# =============================================================================
# 模块级优化3Learnable Attention Pooling - 替代简单的last-token
# =============================================================================

class AttentionPooling(nn.Module):
    """
    科学池化可学习的注意力池化（稳定版）
    
    原理：
    - 学习一个query向量，与所有token做attention
    - 相比mean/last-token，可以自适应聚焦重要位置
    
    稳定性优化：
    - 小初始化query
    - 残差缩放防止梯度爆炸
    - 温度缩放attention
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 可学习的query（小初始化）
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 残差缩放因子（防止梯度爆炸）
        self.residual_scale = 0.5
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_dim)
            attention_mask: (B, seq_len) 1表示有效，0表示padding
        Returns:
            pooled: (B, hidden_dim)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        input_dtype = hidden_states.dtype
        
        # 关键修复统一dtype，处理BFloat16
        # 将输入转为float32进行attention计算，然后转回原dtype
        hidden_states_float = hidden_states.float()
        
        # 扩展query到batch维度，并转为float32
        query = self.query.expand(batch_size, -1, -1).float()  # (B, 1, dim)
        
        # 构建key_padding_mask（True表示忽略）
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (B, seq_len)
        
        # 注意力池化（在float32下计算）
        pooled, _ = self.attention(
            query=query,
            key=hidden_states_float,
            value=hidden_states_float,
            key_padding_mask=key_padding_mask,
        )  # (B, 1, dim)
        
        # 稳定性残差缩放 + LayerNorm
        pooled = self.layer_norm(self.dropout(pooled) * self.residual_scale + query)
        
        return pooled.squeeze(1)  # (B, dim) 返回float32，后续会统一处理


# =============================================================================
# Label Graph Network - 学习标签间关系
# =============================================================================

class LabelGraphNetwork(nn.Module):
    """
    科学多标签标签关系图网络（稳定版）
    
    原理：
    - SIDER的27个副作用标签之间存在医学相关性
    - 用简化的交互机制建模标签间关系
    
    稳定性优化：
    - 移除复杂的双层attention（梯度爆炸源）
    - 使用简单的标签-样本点积交互
    - 添加残差缩放防止梯度爆炸
    """
    def __init__(self, num_labels: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        # 标签embedding（小初始化）
        self.label_embedding = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.01)
        
        # 标签投影（降维后交互，更稳定）
        self.label_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 样本投影
        self.sample_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 最终分类（从投影空间）
        self.classifier = nn.Linear(hidden_dim // 4, 1)
        
        # 残差缩放（防止梯度爆炸）
        self.residual_scale = 0.1
        
        # 初始化
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, sample_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample_features: (B, hidden_dim)
        Returns:
            logits: (B, num_labels)
        """
        batch_size = sample_features.shape[0]
        sample_features = sample_features.float()
        
        # 修复4L2归一化label embedding，防止长期训练发散
        label_emb_normalized = F.normalize(self.label_embedding, p=2, dim=-1)
        
        # 1. 投影到低维空间
        sample_proj = self.sample_proj(sample_features)  # (B, dim//4)
        labels_proj = self.label_proj(label_emb_normalized)  # (L, dim//4)
        
        # 归一化投影特征
        sample_proj = F.normalize(sample_proj, p=2, dim=-1)
        labels_proj = F.normalize(labels_proj, p=2, dim=-1)
        
        # 2. 标签-样本交互（余弦相似度基础）
        sample_expanded = sample_proj.unsqueeze(1)  # (B, 1, dim//4)
        labels_expanded = labels_proj.unsqueeze(0)  # (1, L, dim//4)
        
        # 交互特征
        interaction = sample_expanded + self.residual_scale * labels_expanded  # (B, L, dim//4)
        
        # 3. 分类
        logits = self.classifier(interaction).squeeze(-1)  # (B, L)
        
        # 修复2clamp输出防止极端值
        logits = logits.clamp(-10.0, 10.0)
        
        return logits


# =============================================================================
# 残差预测头
# =============================================================================

class ResidualPredictionHead(nn.Module):
    """
    科学创新带残差连接的预测头
    
    原理：
    - Highway Network思想：让梯度有直接通路
    - 对于深层MLP，残差连接显著改善梯度流
    
    结构：
    input(2048) → [LayerNorm → Linear → GELU → Dropout] → h1(1024)
                                                           ↓
                   h1 + proj(input) → [LayerNorm → Linear → GELU → Dropout] → h2(512)
                                                                               ↓
                                                        h2 + proj(h1) → Linear → output(27)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.15):
        super().__init__()
        
        # 第一层：2048 → 1024
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        
        # 残差投影1：2048 → 1024
        self.res_proj1 = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # 第二层：1024 → 512
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.drop2 = nn.Dropout(dropout)
        
        # 残差投影2：1024 → 512
        self.res_proj2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        
        # 输出层：512 → output_dim
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)
        
        # 残差缩放因子
        self.res_scale = 0.3
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
        for m in [self.res_proj1, self.res_proj2]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层 + 残差
        h1 = self.drop1(F.gelu(self.fc1(self.ln1(x))))
        h1 = h1 + self.res_scale * self.res_proj1(x)  # 残差连接
        
        # 第二层 + 残差
        h2 = self.drop2(F.gelu(self.fc2(self.ln2(h1))))
        h2 = h2 + self.res_scale * self.res_proj2(h1)  # 残差连接
        
        # 输出
        return self.fc_out(h2)


# =============================================================================
# 模态对比学习损失
# =============================================================================

class ModalContrastiveLoss(nn.Module):
    """
    ECCL: Evidence-Calibrated Contrastive Learning.

    Standard InfoNCE treats every sample as an equally reliable anchor and
    negative.  When a modality is missing and has been completed by DEC, its
    representation is *synthesised* — using it as a strong positive/negative
    injects noisy gradient.

    ECCL fixes this by computing a per-sample **evidence quality** from DEC:
      - Available modality   →  quality = 1.0  (real observation)
      - Completed modality   →  quality = 1 − vacuity  (evidence-derived)
    For each modality pair (a, b), pair quality = min(q_a, q_b).
    The per-sample InfoNCE contribution is scaled by pair quality.

    This is task-agnostic and dataset-agnostic — it only requires DEC
    vacuity and completion flags, both of which are always available.
    """

    def __init__(self, hidden_dim: int = 2048, proj_dim: int = 256,
                 temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature
        self.proj_dim = proj_dim

        self.proj_struct = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.proj_kg = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.proj_moltoken = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self.struct_prototype = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.kg_prototype = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.moltoken_prototype = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.semantic_conf_temperature = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    def forward(
        self,
        struct_feat: torch.Tensor,
        kg_feat: torch.Tensor,
        moltoken_feat: torch.Tensor = None,
        return_semantic_conf: bool = False,
        vacuity: torch.Tensor = None,
        completion_flags: torch.Tensor = None,
    ):
        """
        Args:
            struct_feat / kg_feat / moltoken_feat: (B, hidden_dim)
            vacuity:          (B, 3) from DEC — per-modality epistemic uncertainty
            completion_flags: (B, 3) bool — True = modality was originally missing
        """
        z_struct = F.normalize(self.proj_struct(struct_feat), dim=-1)
        z_kg = F.normalize(self.proj_kg(kg_feat), dim=-1)

        # ECCL: per-modality evidence quality
        q = self._evidence_quality(vacuity, completion_flags)

        w_sk = torch.min(q[:, 0], q[:, 1]) if q is not None else None
        loss_sk = self._infonce(z_struct, z_kg, weights=w_sk)
        total_loss = loss_sk

        z_mol = None
        if moltoken_feat is not None:
            z_mol = F.normalize(self.proj_moltoken(moltoken_feat), dim=-1)
            w_sm = torch.min(q[:, 0], q[:, 2]) if q is not None else None
            w_km = torch.min(q[:, 1], q[:, 2]) if q is not None else None
            loss_sm = self._infonce(z_struct, z_mol, weights=w_sm)
            loss_km = self._infonce(z_kg, z_mol, weights=w_km)
            total_loss = (loss_sk + loss_sm + loss_km) / 3

        contrastive_loss = total_loss * 0.05

        # Collect ECCL diagnostics (detached, no grad)
        if q is not None:
            with torch.no_grad():
                self._last_diag = {
                    'eccl_q_struct': q[:, 0].mean().item(),
                    'eccl_q_kg': q[:, 1].mean().item(),
                    'eccl_q_mol': q[:, 2].mean().item(),
                    'eccl_w_sk': w_sk.mean().item() if w_sk is not None else 1.0,
                    'eccl_w_sm': w_sm.mean().item() if w_sm is not None else 1.0,
                    'eccl_w_km': w_km.mean().item() if w_km is not None else 1.0,
                }
        else:
            self._last_diag = {}

        if not return_semantic_conf:
            return contrastive_loss

        semantic_confs = self._compute_semantic_confidence(z_struct, z_kg, z_mol)
        return contrastive_loss, semantic_confs

    # ------------------------------------------------------------------
    @staticmethod
    def _evidence_quality(vacuity, completion_flags):
        """Per-modality quality: 1.0 if available, (1-vacuity) if completed."""
        if vacuity is None or completion_flags is None:
            return None
        return torch.where(
            completion_flags,
            (1.0 - vacuity).clamp(min=0.1),
            torch.ones_like(vacuity),
        )

    # ------------------------------------------------------------------
    def _compute_semantic_confidence(self, z_struct, z_kg, z_mol):
        struct_proto = F.normalize(self.struct_prototype, dim=-1)
        kg_proto = F.normalize(self.kg_prototype, dim=-1)
        temp = self.semantic_conf_temperature.clamp(min=0.5, max=2.0)

        struct_conf = torch.sigmoid((z_struct * struct_proto).sum(-1) / temp)
        kg_conf = torch.sigmoid((z_kg * kg_proto).sum(-1) / temp)

        mol_conf = None
        if z_mol is not None:
            mol_proto = F.normalize(self.moltoken_prototype, dim=-1)
            mol_conf = torch.sigmoid((z_mol * mol_proto).sum(-1) / temp)

        return {"struct": struct_conf, "kg": kg_conf, "moltoken": mol_conf}

    # ------------------------------------------------------------------
    def _infonce(self, z1: torch.Tensor, z2: torch.Tensor,
                 weights: torch.Tensor = None) -> torch.Tensor:
        """Evidence-weighted symmetric InfoNCE."""
        batch_size = z1.shape[0]
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        sim_matrix = sim_matrix.clamp(-50, 50)
        labels = torch.arange(batch_size, device=z1.device)

        if weights is not None:
            loss_12 = F.cross_entropy(sim_matrix, labels, reduction='none')
            loss_21 = F.cross_entropy(sim_matrix.T, labels, reduction='none')
            w = weights.detach()
            return (w * (loss_12 + loss_21) / 2).mean()

        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.T, labels)
        return (loss_12 + loss_21) / 2


# =============================================================================
# 互补性约束模块
# =============================================================================

class ModalityComplementarityLoss(nn.Module):
    """
    稳定版模态互补性监控（不参与训练loss）
    
    原理：
    - Structure (GraphMVP) 擅长：局部原子环境、3D几何、化学键
    - MolToken 擅长：序列模式、全局化学语义、构象柔性
    
    设计：
    - 计算原始特征的余弦相似度（直接在公共维度上）
    - 返回相似度作为监控指标
    - 不使用可学习投影（避免不稳定性）
    
    输出解读：
    - similarity ≈ 0：两模态正交，互补性极好
    - similarity ≈ 0.3-0.5：两模态有适度互补
    - similarity > 0.7：两模态高度相似，可能冗余
    """
    def __init__(self, struct_dim: int = 300, moltoken_dim: int = 768, 
                 common_dim: int = 128, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.common_dim = common_dim
        
        # 固定的投影层（用于维度对齐，不参与训练）
        self.struct_proj = nn.Linear(struct_dim, common_dim, bias=False)
        self.moltoken_proj = nn.Linear(moltoken_dim, common_dim, bias=False)
        
        # 初始化为正交投影（更稳定）
        nn.init.orthogonal_(self.struct_proj.weight)
        nn.init.orthogonal_(self.moltoken_proj.weight)
        
        # 冻结参数：互补性只作为监控，不参与训练
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, struct_feat: torch.Tensor, moltoken_feat: torch.Tensor) -> torch.Tensor:
        """
        返回互补性相似度（用于监控，不参与梯度）
        
        Args:
            struct_feat: (B, struct_dim)
            moltoken_feat: (B, moltoken_dim)
        Returns:
            similarity: 平均相似度（越低越互补）
        """
        if struct_feat is None or moltoken_feat is None:
            return torch.tensor(0.0)
        
        with torch.no_grad():
            # 投影到共同空间
            struct_common = self.struct_proj(struct_feat.float())
            moltoken_common = self.moltoken_proj(moltoken_feat.float())
            
            # L2归一化
            struct_norm = F.normalize(struct_common, p=2, dim=-1)
            moltoken_norm = F.normalize(moltoken_common, p=2, dim=-1)
            
            # 逐样本余弦相似度
            similarity = (struct_norm * moltoken_norm).sum(dim=-1)  # (B,)
            
            # 返回互补性指标（和epoch summary一致）
            # complementarity = 1 - |similarity|
            # 范围 [0, 1]：越高表示互补性越好
            complementarity = 1.0 - similarity.abs()
            return complementarity.mean()


# =============================================================================
# R-Drop正则化（真正有效的一致性约束）
# =============================================================================

class RDropRegularizer(nn.Module):
    """
    修复版R-Drop正则化 - 正确的KL散度实现
    
    科学原理：
    - 对同一个输入进行两次前向传播（dropout会产生不同结果）
    - 计算两次输出的双向KL散度
    - 鼓励模型对dropout不敏感，提高泛化能力
    
    修复问题：
    - 之前的实现值太小（0.0000）因为过度缩放
    - 现在使用标准的双向KL散度，alpha直接控制权重
    
    参考：R-Drop: Regularized Dropout for Neural Networks (NeurIPS 2021)
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha  # R-Drop权重，原论文推荐1.0
    
    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """
        计算R-Drop损失（稳定版KL散度）
        
        稳定性修复：
        - 概率clamp [0.02, 0.98]，避免极端概率
        - 单样本KL散度clamp，防止个别样本梯度爆炸
        """
        if logits1 is None or logits2 is None:
            return torch.tensor(0.0, device=logits1.device if logits1 is not None else 'cpu')
        
        # 转换为概率
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)
        
        # 稳定性clamp：[0.02, 0.98]
        eps = 0.02
        p1 = p1.clamp(eps, 1-eps)
        p2 = p2.clamp(eps, 1-eps)
        
        # 双向KL散度（对于Bernoulli分布）
        kl_12 = p1 * (p1.log() - p2.log()) + (1-p1) * ((1-p1).log() - (1-p2).log())
        kl_21 = p2 * (p2.log() - p1.log()) + (1-p2) * ((1-p2).log() - (1-p1).log())
        
        # 对称KL散度
        sym_kl = 0.5 * (kl_12 + kl_21)
        
        # 稳定性单样本KL clamp
        sym_kl = sym_kl.clamp(0.0, 1.0)
        
        # 平均并应用权重
        rdrop_loss = sym_kl.mean() * self.alpha
        
        return rdrop_loss


class FeatureConsistencyRegularizer(nn.Module):
    """
    特征一致性正则化
    
    在R-Drop的基础上，额外约束pooled特征的一致性
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        if feat1 is None or feat2 is None:
            return torch.tensor(0.0, device=feat1.device if feat1 is not None else 'cpu')
        
        # 余弦相似度损失
        f1 = F.normalize(feat1, p=2, dim=-1)
        f2 = F.normalize(feat2, p=2, dim=-1)
        cos_sim = (f1 * f2).sum(dim=-1)
        consistency_loss = (1.0 - cos_sim).mean()
        
        return consistency_loss.clamp(0.0, 2.0)


# =============================================================================
# EFR — Dual-source Per-dimension Uncertainty Complementary Fusion
# =============================================================================

class SparseAttentionCMSA(nn.Module):
    """
    SA-CMSA: Sparse-Attention guided Cross-Modal Subspace Alignment.

    Two-stage KG completion (no confidence/gate -- left for Innovation 2):
      Stage 1a: Structural retrieval
               struct -> sparse attn(bank) -> top-K -> hop1
      Stage 1b: Cross-modal 2-hop retrieval
               hop2_fuse(hop1, mol3D) -> hop2_query
               hop2_query -> sparse attn(FULL bank) -> top-K -> hop2
               novelty = hop2 - hop1
               gate = hop_gate([hop1, novelty])
               kg_retrieved = hop1 + gate * novelty
      Stage 2: Retrieval-conditioned generation
               z_s = struct_enc(struct), z_m = mol_enc(mol)
               z_r = retrieve_enc(kg_retrieved)    [no detach]
               route = route_net([z_s, z_m, z_r])  [per-dim sigmoid]
               residual = residual_dec(route * z_s + (1-route) * z_m)
               kg_pred = kg_retrieved + residual

    Design: hop2 uses hop1+mol3D for high-quality cross-modal retrieval.
    Gate input is [hop1, hop2-hop1] (base vs novelty) to prevent information
    asymmetry that causes gate saturation.
    """

    def __init__(
        self,
        modal_dims: list,
        kge_bank: torch.Tensor,
        sub_dim: int = 128,
        topk: int = 16,
        num_heads: int = 2,
        dropout: float = 0.1,
        align_lambda: float = 0.05,
        retrieve_lambda: float = 0.1,
        enable_mol_completion: bool = False,
        ablation_flags: set = None,
        **kwargs,
    ):
        super().__init__()
        struct_dim = modal_dims[0]
        self.kg_dim = modal_dims[1]
        self.mol_dim = modal_dims[2]
        self.sub_dim = sub_dim
        self.kg_idx = 1
        self.topk = topk
        self.num_heads = num_heads
        self.align_lambda = align_lambda
        self.retrieve_lambda = retrieve_lambda
        self.enable_mol_completion = enable_mol_completion
        self._ablation = ablation_flags or set()
        proj_dim = 256

        # ---- Stage 1a: Multi-head sparse attention retrieval (struct-based) ----
        self.register_buffer("kge_bank", kge_bank.float())
        head_dim = proj_dim // num_heads
        self.query_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(struct_dim),
                nn.Linear(struct_dim, 384), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(384, head_dim),
            ) for _ in range(num_heads)
        ])
        self.bank_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.kg_dim),
                nn.Linear(self.kg_dim, head_dim),
            ) for _ in range(num_heads)
        ])
        self.log_temperature = nn.Parameter(torch.tensor(math.log(50.0)))

        # ---- Stage 1b: Cross-modal 2-hop retrieval (hop1+mol3D -> full bank) ----
        self.hop2_fuse = nn.Sequential(
            nn.LayerNorm(self.kg_dim + self.mol_dim),
            nn.Linear(self.kg_dim + self.mol_dim, 384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, self.kg_dim),
        )
        self.hop2_query_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.kg_dim),
                nn.Linear(self.kg_dim, head_dim),
            ) for _ in range(num_heads)
        ])
        self.hop_gate = nn.Sequential(
            nn.Linear(self.kg_dim * 2, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.hop_gate[-2].weight, gain=0.1)
        nn.init.constant_(self.hop_gate[-2].bias, -2.0)

        # ---- Stage 2: Retrieval-conditioned cross-modal generation ----
        self.struct_enc = nn.Sequential(
            nn.Linear(struct_dim, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(),
            nn.Linear(sub_dim, sub_dim), nn.LayerNorm(sub_dim),
        )
        self.mol_enc = nn.Sequential(
            nn.Linear(self.mol_dim, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(),
            nn.Linear(sub_dim, sub_dim), nn.LayerNorm(sub_dim),
        )
        self.kg_enc = nn.Sequential(
            nn.Linear(self.kg_dim, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(),
            nn.Linear(sub_dim, sub_dim), nn.LayerNorm(sub_dim),
        )

        # kg_retrieved -> retrieve_enc -> z_r (retrieval-conditioned signal for route_net)
        self.retrieve_enc = nn.Sequential(
            nn.LayerNorm(self.kg_dim),
            nn.Linear(self.kg_dim, sub_dim),
        )

        # Retrieval-conditioned per-dimension routing: [z_s, z_m, z_r] -> sigmoid weights
        self.route_net = nn.Sequential(
            nn.Linear(sub_dim * 3, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(),
            nn.Linear(sub_dim, sub_dim), nn.Sigmoid(),
        )

        # Residual decoder (gain=0.3 -> small initial residuals)
        self.residual_dec = nn.Sequential(
            nn.Linear(sub_dim, sub_dim * 2), nn.LayerNorm(sub_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sub_dim * 2, self.kg_dim),
        )
        nn.init.xavier_uniform_(self.residual_dec[-1].weight, gain=0.3)
        nn.init.zeros_(self.residual_dec[-1].bias)

        # MolToken completion branch (optional)
        if self.enable_mol_completion:
            self.mol_residual_dec = nn.Sequential(
                nn.Linear(sub_dim, sub_dim * 2), nn.LayerNorm(sub_dim * 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(sub_dim * 2, self.mol_dim),
            )
            nn.init.xavier_uniform_(self.mol_residual_dec[-1].weight, gain=0.3)
            nn.init.zeros_(self.mol_residual_dec[-1].bias)
            self.mol_route_net = nn.Sequential(
                nn.Linear(sub_dim * 3, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(),
                nn.Linear(sub_dim, sub_dim), nn.Sigmoid(),
            )

        n_params = sum(p.numel() for p in self.parameters())
        abl_str = ",".join(sorted(self._ablation)) if self._ablation else "none"
        logger.info(
            f"[SA-CMSA] sub={sub_dim} topk={topk} heads={num_heads} 2hop=novelty_gate | "
            f"bank={kge_bank.shape[0]}x{kge_bank.shape[1]} | "
            f"struct({struct_dim})+mol({self.mol_dim}) -> kg({self.kg_dim}) | "
            f"mol_completion={enable_mol_completion} "
            f"align_l={align_lambda} retr_l={retrieve_lambda} "
            f"ablation=[{abl_str}] "
            f"params={n_params/1e3:.1f}K"
        )

    def _recon_loss(self, pred, tgt):
        return (
            F.mse_loss(pred, tgt)
            + 0.5 * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean())
        )

    def _sparse_retrieve(self, struct_feat):
        """Multi-head sparse attention retrieval from kge_bank.

        Each head does independent softmax over its own top-K, then
        per-head retrieved embeddings are averaged. Also returns all
        top-K embeddings for 2-hop cross-attention refinement.
        """
        bank = self.kge_bank
        temperature = self.log_temperature.exp().clamp(5.0, 200.0)
        B = struct_feat.size(0)

        head_retrieved = []
        all_topk_idx = []
        all_topk_attn = []
        all_topk_embs = []
        head_top1 = []
        head_ent = []

        for h in range(self.num_heads):
            q_h = F.normalize(self.query_heads[h](struct_feat), dim=-1)
            k_h = F.normalize(self.bank_heads[h](bank), dim=-1)
            scores_h = torch.matmul(q_h, k_h.T) * temperature
            tk_s, tk_i = torch.topk(scores_h, self.topk, dim=-1)

            attn_h = F.softmax(tk_s, dim=-1)
            embs_h = bank[tk_i.view(-1)].view(B, self.topk, self.kg_dim)
            retrieved_h = (attn_h.unsqueeze(-1) * embs_h).sum(dim=1)

            head_retrieved.append(retrieved_h)
            all_topk_idx.append(tk_i)
            all_topk_attn.append(attn_h)
            all_topk_embs.append(embs_h)

            head_top1.append(attn_h.max(dim=-1).values)
            ent_h = -(attn_h * (attn_h + 1e-8).log()).sum(dim=-1)
            head_ent.append(ent_h / math.log(self.topk))

        hop1 = torch.stack(head_retrieved, dim=0).mean(dim=0)
        topk_embs_cat = torch.cat(all_topk_embs, dim=1)  # (B, num_heads*K, kg_dim)

        merged_idx = torch.cat(all_topk_idx, dim=-1)
        merged_attn = torch.cat(all_topk_attn, dim=-1)
        top1_w = torch.stack(head_top1, dim=0).mean(dim=0)
        norm_ent = torch.stack(head_ent, dim=0).mean(dim=0)

        return hop1, topk_embs_cat, merged_attn, merged_idx, top1_w, norm_ent, temperature

    def _sparse_retrieve_hop2(self, cross_modal_query):
        """Hop 2: retrieve from FULL bank using cross-modal query (hop1 + mol3D).

        Returns: (hop2_embedding, merged_idx, merged_attn)
        """
        bank = self.kge_bank
        temperature = self.log_temperature.exp().clamp(5.0, 200.0)
        B = cross_modal_query.size(0)
        head_retrieved = []
        all_topk_idx, all_topk_attn = [], []
        for h in range(self.num_heads):
            q_h = F.normalize(self.hop2_query_heads[h](cross_modal_query), dim=-1)
            k_h = F.normalize(self.bank_heads[h](bank), dim=-1)
            scores_h = torch.matmul(q_h, k_h.T) * temperature
            tk_s, tk_i = torch.topk(scores_h, self.topk, dim=-1)
            attn_h = F.softmax(tk_s, dim=-1)
            embs_h = bank[tk_i.view(-1)].view(B, self.topk, self.kg_dim)
            retrieved_h = (attn_h.unsqueeze(-1) * embs_h).sum(dim=1)
            head_retrieved.append(retrieved_h)
            all_topk_idx.append(tk_i)
            all_topk_attn.append(attn_h)
        hop2 = torch.stack(head_retrieved, dim=0).mean(dim=0)
        merged_idx = torch.cat(all_topk_idx, dim=-1)
        merged_attn = torch.cat(all_topk_attn, dim=-1)
        return hop2, merged_idx, merged_attn

    def forward(self, feat_list, missing_flags):
        B = feat_list[0].size(0)
        device = feat_list[0].device

        raw_struct = torch.nan_to_num(feat_list[0].float(), nan=0.0)
        raw_kg = torch.nan_to_num(feat_list[self.kg_idx].float(), nan=0.0)
        raw_mol = torch.nan_to_num(feat_list[2].float(), nan=0.0)

        # ======== Stage 1a: Multi-head sparse attention (1-hop) ========
        hop1, topk_embs, attn, merged_idx, top1_w, norm_ent, temp = \
            self._sparse_retrieve(raw_struct.detach())

        # ======== Stage 1b: Cross-modal 2-hop retrieval (hop1+mol -> full bank) ========
        hop2_idx, hop2_attn = None, None
        if "cmsa_no_2hop" in self._ablation:
            kg_retrieved = hop1
            gate = torch.zeros(B, 1, device=device)
        else:
            hop2_query = self.hop2_fuse(torch.cat([hop1, raw_mol.detach()], dim=-1))
            hop2, hop2_idx, hop2_attn = self._sparse_retrieve_hop2(hop2_query)
            novelty = hop2 - hop1
            gate = self.hop_gate(torch.cat([hop1, novelty], dim=-1))
            kg_retrieved = hop1 + gate * novelty

        # ======== Stage 2: Retrieval-conditioned per-dimension routing ========
        z_s = self.struct_enc(raw_struct.detach())
        z_m = self.mol_enc(raw_mol.detach())
        z_r = self.retrieve_enc(kg_retrieved)

        if "cmsa_no_routing" in self._ablation:
            route = torch.full((B, self.sub_dim), 0.5, device=device)
        else:
            route = self.route_net(torch.cat([z_s, z_m, z_r], dim=-1))
        z_evidence = route * z_s + (1.0 - route) * z_m
        residual = self.residual_dec(z_evidence)

        kg_pred = kg_retrieved + residual

        # ======== Output selection ========
        kg_truly_missing = missing_flags[:, self.kg_idx]
        kg_out = torch.where(
            kg_truly_missing.unsqueeze(-1).expand_as(kg_pred),
            kg_pred,
            raw_kg,
        )

        mol_truly_missing = missing_flags[:, 2]
        if self.enable_mol_completion and mol_truly_missing.any():
            mol_route = self.mol_route_net(torch.cat([z_s, z_m, z_r], dim=-1))
            z_mol_evi = mol_route * z_s + (1.0 - mol_route) * z_m
            mol_pred = self.mol_residual_dec(z_mol_evi)
            mol_out = torch.where(
                mol_truly_missing.unsqueeze(-1).expand_as(mol_pred),
                mol_pred,
                raw_mol,
            )
        else:
            mol_out = raw_mol

        enhanced_feats = [raw_struct, kg_out, mol_out]
        completion_flags = missing_flags.clone()

        total_loss = torch.tensor(0.0, device=device)
        diag = {}
        kg_pred_for_gt = None

        if self.training:
            has_kg_gt = ~kg_truly_missing
            tgt_kg = raw_kg.detach()

            fuse_loss = torch.tensor(0.0, device=device)
            retrieve_loss = torch.tensor(0.0, device=device)
            align_s_loss = torch.tensor(0.0, device=device)
            align_m_loss = torch.tensor(0.0, device=device)

            if has_kg_gt.any():
                tgt_kg_gt = tgt_kg[has_kg_gt]

                # Fuse loss: use already-computed kg_pred (same dropout mask as LLM input)
                fuse_loss = self._recon_loss(kg_pred[has_kg_gt], tgt_kg_gt)
                kg_pred_for_gt = kg_pred[has_kg_gt]

                # Retrieve loss: pure retrieval (after 2-hop) vs true KG
                retrieve_loss = self._recon_loss(kg_retrieved[has_kg_gt], tgt_kg_gt)

                # Subspace alignment
                z_k = self.kg_enc(raw_kg.detach())
                z_k_sg = z_k[has_kg_gt].detach()
                align_s_loss = 1.0 - F.cosine_similarity(
                    z_s[has_kg_gt], z_k_sg, dim=-1).mean()
                align_m_loss = 1.0 - F.cosine_similarity(
                    z_m[has_kg_gt], z_k_sg, dim=-1).mean()

            # Safeguard A: attention entropy penalty
            entropy_penalty = F.relu(norm_ent.mean() - 0.85)

            # Evidence orthogonality
            ortho_loss = F.cosine_similarity(z_s, z_m, dim=-1).abs().mean()

            _align_w = 0.0 if "cmsa_no_align" in self._ablation else self.align_lambda
            total_loss = (
                fuse_loss
                + self.retrieve_lambda * retrieve_loss
                + _align_w * (align_s_loss + align_m_loss)
                + 0.01 * entropy_penalty
                + 0.005 * ortho_loss
            )

            with torch.no_grad():
                r = route.detach().mean(dim=0)
                g = gate.detach().mean().item()
                diag['recon_loss'] = fuse_loss.item()
                diag['retrieve_loss'] = retrieve_loss.item()
                diag['align_s'] = align_s_loss.item()
                diag['align_m'] = align_m_loss.item()
                diag['route_s'] = r.mean().item()
                diag['route_m'] = (1.0 - r).mean().item()
                diag['route_std'] = route.detach().mean(dim=-1).std().item()
                diag['kg_true_miss'] = kg_truly_missing.float().mean().item()
                diag['top1_weight'] = top1_w.mean().item()
                diag['norm_entropy'] = norm_ent.mean().item()
                diag['temperature'] = temp.item()
                diag['hop_gate'] = g
                diag['entropy_penalty'] = entropy_penalty.item()
                diag['ortho_loss'] = ortho_loss.item()
                diag['_topk_idx'] = merged_idx.detach().cpu()
                diag['_topk_attn'] = attn.detach().cpu()
                if hop2_idx is not None:
                    diag['_hop2_topk_idx'] = hop2_idx.detach().cpu()
                    diag['_hop2_topk_attn'] = hop2_attn.detach().cpu()
                diag['_hop_gate'] = gate.detach().cpu()
                diag['_route'] = route.detach().cpu()

        return enhanced_feats, completion_flags, total_loss, diag, kg_pred_for_gt


# =============================================================================
# CMON: Cross-Modal Orthogonal Decomposition Network
# =============================================================================

class CMONModule(nn.Module):
    """
    CMON: Cross-Modal Orthogonal Decomposition Network.

    独立于 SA-CMSA 的第二创新模块。将每个模态表示分解为共享分量(可由其他模态预测)
    与私有分量(独有信息)，用可靠性门控做选择性跨模态注入，并生成 Digest Token
    供 LLM 显式感知跨模态可靠性。

    四阶段:
      Stage 1 — 正交分解: p_i = p_shared_i + p_private_i, p_shared_i = Predictor_i(p_j, p_k)
      Stage 2 — 可靠性估计: confidence_i = ConfidenceHead(...), reliability_i = uniqueness_i × confidence_i
      Stage 3 — Token 增强: tokens_i += gate × enhance_proj(Σ reliability_j × p_private_j)
      Stage 4 — Digest Token: digest_i = DigestGenerator(...) 插入 LLM 输入
    """

    def __init__(self, llm_dim: int, n_modalities: int = 3, sub_dim: int = 64,
                 ablation_flags: set = None):
        super().__init__()
        self.n_modalities = n_modalities
        self.n_mod = n_modalities
        self.llm_dim = llm_dim
        self.sub_dim = sub_dim
        self._ablation = ablation_flags or set()

        self.shared_proj = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, sub_dim),
        )

        # Cross predictors: for modality i, predict p_shared_i from (p_j, p_k)
        self.cross_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * sub_dim, sub_dim),
                nn.GELU(),
                nn.Linear(sub_dim, sub_dim),
            )
            for _ in range(n_modalities)
        ])

        # Confidence head per modality: (p_i, p_shared_i, p_private_i) -> scalar (stability-calibrated)
        self.confidence_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * sub_dim, sub_dim),
                nn.GELU(),
                nn.Linear(sub_dim, 1),
            )
            for _ in range(n_modalities)
        ])
        for m in self.confidence_head:
            nn.init.xavier_uniform_(m[-1].weight, gain=0.1)
            nn.init.constant_(m[-1].bias, 0.0)

        self.enhance_proj = nn.Linear(sub_dim, llm_dim)
        nn.init.xavier_uniform_(self.enhance_proj.weight, gain=0.01)
        nn.init.zeros_(self.enhance_proj.bias)

        self.gate = nn.Parameter(torch.tensor(-1.5))
        self.max_gate = 0.3

        self.pred_loss_weight = 0.1
        self.ortho_loss_weight = 0.1
        self.cal_loss_weight = 0.1
        self._eps = 1e-6

        # DigestGenerator: (p_private, uniqueness, confidence, cross_reliable_privates) -> (B, 1, llm_dim)
        self.digest_self_proj = nn.Linear(sub_dim + 2, sub_dim)
        self.digest_cross_proj = nn.Linear(sub_dim, sub_dim)
        self.digest_fuse = nn.Sequential(
            nn.Linear(2 * sub_dim, sub_dim),
            nn.GELU(),
            nn.Linear(sub_dim, llm_dim),
        )
        nn.init.xavier_uniform_(self.digest_fuse[-1].weight, gain=0.01)
        nn.init.zeros_(self.digest_fuse[-1].bias)

        # Backward compat: load checkpoints that saved "reliability_head"
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        n_params = sum(p.numel() for p in self.parameters())
        abl_str = ",".join(sorted(self._ablation)) if self._ablation else "none"
        logger.info(
            f"[CMON] cross-modal orthogonal decomposition | "
            f"llm_dim={llm_dim}, sub_dim={sub_dim}, ablation=[{abl_str}], "
            f"params={n_params/1e3:.1f}K"
        )

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args):
        keys = list(state_dict.keys())
        for k in keys:
            if prefix + "reliability_head." in k:
                new_k = k.replace(prefix + "reliability_head.", prefix + "confidence_head.")
                state_dict[new_k] = state_dict.pop(k)

    def forward(
        self,
        mod_tokens_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict, Dict, torch.Tensor]:
        """
        Returns:
            refined_tokens: list of (B, N_i, D) per modality
            digest_tokens: list of 3 x (B, 1, D) for LLM input
            diag: dict of scalars for logging
            aux_losses: dict with 'pred_loss', 'ortho_loss', 'cal_loss'
            reliability_samples: (B, 3)
        """
        B = mod_tokens_list[0].size(0)
        device = mod_tokens_list[0].device
        dtype_orig = mod_tokens_list[0].dtype

        # Cast to float32 for stable decomposition
        projs = []
        for t in mod_tokens_list:
            mu = t.float().mean(dim=1)
            projs.append(self.shared_proj(mu))

        # Stage 1: Orthogonal decomposition
        p_shared_list = []
        for i in range(self.n_mod):
            j, k = (i + 1) % self.n_mod, (i + 2) % self.n_mod
            other_cat = torch.cat([projs[j], projs[k]], dim=-1)
            p_shared_list.append(self.cross_predictors[i](other_cat))

        p_private_list = [projs[i] - p_shared_list[i] for i in range(self.n_mod)]

        # Prediction loss
        pred_loss = torch.tensor(0.0, device=device)
        for i in range(self.n_mod):
            pred_loss = pred_loss + F.mse_loss(p_shared_list[i], projs[i])
        pred_loss = pred_loss / self.n_mod * self.pred_loss_weight

        # Orthogonality loss: p_shared_i ⊥ p_private_i
        ortho_loss = torch.tensor(0.0, device=device)
        for i in range(self.n_mod):
            a, b = p_shared_list[i], p_private_list[i]
            na = a.norm(dim=-1, keepdim=True).clamp(min=self._eps)
            nb = b.norm(dim=-1, keepdim=True).clamp(min=self._eps)
            cos_sq = ((a / na) * (b / nb)).sum(dim=-1).clamp(-1, 1).pow(2)
            ortho_loss = ortho_loss + cos_sq.mean()
        ortho_loss = ortho_loss / self.n_mod * self.ortho_loss_weight

        # Private ratio (uniqueness): ||p_private||^2 / (||p||^2 + eps)
        priv_ratios = []
        for i in range(self.n_mod):
            pn = (p_private_list[i] ** 2).sum(dim=-1).clamp(min=self._eps)
            qn = (projs[i] ** 2).sum(dim=-1).clamp(min=self._eps)
            priv_ratios.append((pn / qn).clamp(0, 1))
        priv_ratio_stack = torch.stack(priv_ratios, dim=-1)

        # ---- LODC: Leave-One-Out Decomposition Consistency ----
        _skip_lodc = "cmon_no_lodc" in self._ablation
        zero_vec = torch.zeros(B, self.sub_dim, device=device, dtype=projs[0].dtype)
        lodc_scores = []
        if not _skip_lodc:
            for i in range(self.n_mod):
                j, k = (i + 1) % self.n_mod, (i + 2) % self.n_mod
                mask_j_cat = torch.cat([zero_vec, projs[k]], dim=-1)
                mask_k_cat = torch.cat([projs[j], zero_vec], dim=-1)
                p_shared_mask_j = self.cross_predictors[i](mask_j_cat)
                p_shared_mask_k = self.cross_predictors[i](mask_k_cat)
                p_priv_mask_j = projs[i] - p_shared_mask_j
                p_priv_mask_k = projs[i] - p_shared_mask_k
                cos_j = F.cosine_similarity(p_private_list[i], p_priv_mask_j, dim=-1)
                cos_k = F.cosine_similarity(p_private_list[i], p_priv_mask_k, dim=-1)
                lodc_scores.append((0.5 * (cos_j + cos_k)).clamp(0.0, 1.0))
        else:
            lodc_scores = [torch.ones(B, device=device) for _ in range(self.n_mod)]
        lodc_stack = torch.stack(lodc_scores, dim=-1)

        # Stage 2: Confidence (LODC-calibrated) + reliability = uniqueness × confidence
        conf_logits = []
        confidences = []
        reliabilities = []
        cal_loss = torch.tensor(0.0, device=device)
        _rel_only_uniq = "cmon_rel_only_uniq" in self._ablation
        for i in range(self.n_mod):
            rel_in = torch.cat([projs[i], p_shared_list[i], p_private_list[i]], dim=-1)
            logit_i = self.confidence_head[i](rel_in).squeeze(-1)
            conf_logits.append(logit_i)
            confidence_i = torch.sigmoid(logit_i)
            confidences.append(confidence_i)
            if _rel_only_uniq:
                reliabilities.append(priv_ratios[i])
            else:
                reliabilities.append(priv_ratios[i] * confidence_i)
            if self.training and not _skip_lodc:
                target_logit = torch.logit(lodc_scores[i].detach().clamp(0.01, 0.99))
                cal_loss = cal_loss + F.mse_loss(logit_i, target_logit)
        if self.training and not _skip_lodc:
            cal_loss = cal_loss / self.n_mod * self.cal_loss_weight
        rel_stack = torch.stack(reliabilities, dim=-1)

        # Stage 3: Token enhancement (reliability-gated private injection)
        g = torch.sigmoid(self.gate) * self.max_gate
        refined_tokens = []
        if "cmon_no_enhance" in self._ablation:
            refined_tokens = list(mod_tokens_list)
        else:
            for i in range(self.n_mod):
                cross_signal = torch.zeros(B, self.sub_dim, device=device, dtype=projs[0].dtype)
                for j in range(self.n_mod):
                    if j != i:
                        cross_signal = cross_signal + reliabilities[j].unsqueeze(-1) * p_private_list[j]
                enhancement = self.enhance_proj(cross_signal)
                tokens_i = mod_tokens_list[i].float() + g * enhancement.unsqueeze(1)
                refined_tokens.append(tokens_i.to(dtype_orig))

        # Stage 4: Digest tokens (p_private, uniqueness, confidence) for LLM
        digest_tokens = []
        if "cmon_no_digest" in self._ablation:
            for _ in range(self.n_mod):
                digest_tokens.append(torch.zeros(B, 1, self.llm_dim, device=device, dtype=dtype_orig))
        else:
            for i in range(self.n_mod):
                cross_reliable = torch.zeros(B, self.sub_dim, device=device, dtype=projs[0].dtype)
                for j in range(self.n_mod):
                    if j != i:
                        cross_reliable = cross_reliable + reliabilities[j].unsqueeze(-1) * p_private_list[j]
                self_info = torch.cat([
                    p_private_list[i],
                    priv_ratio_stack[:, i:i + 1],
                    confidences[i].unsqueeze(-1),
                ], dim=-1)
                self_enc = self.digest_self_proj(self_info)
                cross_enc = self.digest_cross_proj(cross_reliable)
                digest = self.digest_fuse(torch.cat([self_enc, cross_enc], dim=-1))
                digest_tokens.append(digest.unsqueeze(1).to(dtype_orig))

        with torch.no_grad():
            ortho_cos = []
            for i in range(self.n_mod):
                a, b = p_shared_list[i], p_private_list[i]
                na = a.norm(dim=-1).clamp(min=self._eps)
                nb = b.norm(dim=-1).clamp(min=self._eps)
                cos_val = ((a * b).sum(dim=-1) / (na * nb)).abs().mean().item()
                ortho_cos.append(cos_val)
            diag = {
                'cmon_priv_ratio_s': priv_ratio_stack[:, 0].mean().item(),
                'cmon_priv_ratio_k': priv_ratio_stack[:, 1].mean().item(),
                'cmon_priv_ratio_m': priv_ratio_stack[:, 2].mean().item(),
                'cmon_lodc_s': lodc_stack[:, 0].mean().item(),
                'cmon_lodc_k': lodc_stack[:, 1].mean().item(),
                'cmon_lodc_m': lodc_stack[:, 2].mean().item(),
                'cmon_conf_s': confidences[0].mean().item(),
                'cmon_conf_k': confidences[1].mean().item(),
                'cmon_conf_m': confidences[2].mean().item(),
                'cmon_rel_s': reliabilities[0].mean().item(),
                'cmon_rel_k': reliabilities[1].mean().item(),
                'cmon_rel_m': reliabilities[2].mean().item(),
                'cmon_ortho_s': ortho_cos[0],
                'cmon_ortho_k': ortho_cos[1],
                'cmon_ortho_m': ortho_cos[2],
                'cmon_gate': g.item(),
                'cmon_pred_loss': pred_loss.item(),
                'cmon_ortho_loss': ortho_loss.item(),
                'cmon_cal_loss': cal_loss.item() if isinstance(cal_loss, torch.Tensor) else 0.0,
            }
        aux_losses = {'pred_loss': pred_loss, 'ortho_loss': ortho_loss, 'cal_loss': cal_loss}

        return refined_tokens, digest_tokens, diag, aux_losses, rel_stack


# =============================================================================
# Q-Former Adapter
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """稳定版Cross Attention Block with gradient stability"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # 稳定性残差缩放因子，避免残差累加导致梯度爆炸
        self.residual_scale = 0.5
    
    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm1(queries)
        k = v = self.norm2(context)
        # 确保dtype一致
        if q.dtype != k.dtype:
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        out, _ = self.cross_attn(q, k, v)
        # 稳定性缩放残差，避免累加过大
        queries = queries + out * self.residual_scale
        queries = queries + self.ffn(self.norm3(queries)) * self.residual_scale
        return queries


class AdapterTokenGenerator(nn.Module):
    """
    创新Q-Former style Adapter with Learnable Confidence
    
    核心创新：
    - 使用额外的 [CONF] query token 来学习模态置信度
    - 内部维度保持 3072（能被 num_heads=8 整除）
    - 置信度完全可学习，不依赖任何先验统计量
    - 重要每个模态有独立的 modality_id 来区分置信度
    
    科学依据：
    - 置信度应该反映"模态特征对当前预测任务的有用程度"
    - 这种有用程度应该由模型在训练中自动学习，而非人工设计
    """
    # 类级别计数器，用于给每个实例分配唯一 ID
    _instance_count = 0
    
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int = 32, 
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim  # 3072
        
        # 创新模态标识：每个 Adapter 实例有唯一 ID
        # 这样不同模态的置信度预测能够区分
        self.modality_id = AdapterTokenGenerator._instance_count
        AdapterTokenGenerator._instance_count += 1
        
        # 原有的 query tokens（用于生成模态表示）
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)
        
        # 创新置信度 query token（额外1个token，用于学习置信度）
        self.conf_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.TransformerEncoderLayer(
                    d_model=output_dim, nhead=num_heads, dim_feedforward=output_dim * 4,
                    batch_first=True, dropout=dropout, activation="gelu", norm_first=True,
                ),
                'cross_attn': CrossAttentionBlock(output_dim, num_heads, dropout),
            }))
        
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 创新模态特定的可学习 embedding
        # 每个模态有独立的 embedding，注入到置信度计算中
        # 这样不同模态即使特征相似，置信度也能分化
        self.modality_embedding = nn.Parameter(torch.randn(1, output_dim) * 0.02)
        
        # 置信度预测头：从 conf_query + modality_embedding 预测置信度
        # 简化设计：小网络 + 零初始化（初始输出 sigmoid(0)=0.5）
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Xavier 初始化（gain=0.1），让不同样本有不同初始置信度
        # 不能零初始化！零权重 → 所有输入映射到 0 → sigmoid(0)=0.5 → 永远不变
        nn.init.xavier_uniform_(self.confidence_head[3].weight, gain=0.1)
        nn.init.zeros_(self.confidence_head[3].bias)

    def forward(self, features: torch.Tensor, return_confidence: bool = False,
                return_global_token: bool = False):
        """
        Args:
            features: (B, input_dim) 或 (B, seq_len, input_dim)
            return_confidence: 是否返回置信度 (标量)
            return_global_token: 是否额外返回 conf token
        Returns:
            默认: tokens (B, num_tokens, output_dim)
            return_confidence: (tokens, confidence)  confidence is (B,)
            return_global_token: (tokens, global_token)  global_token is (B, D)
            两者都: (tokens, confidence, global_token)
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        param_dtype = self.input_norm.weight.dtype
        if features.dtype != param_dtype:
            features = features.to(param_dtype)
        
        features = self.input_proj(self.input_norm(features))
        batch = features.shape[0]
        
        # 拼接 query_tokens 和 conf_query
        content_queries = self.query_tokens.expand(batch, -1, -1)
        conf_query = self.conf_query.expand(batch, -1, -1)
        queries = torch.cat([content_queries, conf_query], dim=1)  # (B, num_tokens+1, dim)
        
        # 通过 Transformer 层处理
        for layer in self.layers:
            queries = layer['self_attn'](queries)
            queries = layer['cross_attn'](queries, features)
        
        # 分离 content tokens 和 conf token
        content_tokens = queries[:, :self.num_tokens, :]  # (B, num_tokens, dim)
        conf_token = queries[:, self.num_tokens:, :]      # (B, 1, dim)
        
        # 输出 tokens
        tokens = self.dropout(self.output_norm(content_tokens))
        
        # conf token plus the modality embedding
        global_token = conf_token.squeeze(1) + self.modality_embedding  # (B, D)
        
        if return_confidence and return_global_token:
            conf_logit = self.confidence_head(global_token).squeeze(-1)
            confidence = torch.sigmoid(conf_logit)
            return tokens, confidence, global_token
        elif return_global_token:
            return tokens, global_token
        elif return_confidence:
            conf_logit = self.confidence_head(global_token).squeeze(-1)
            confidence = torch.sigmoid(conf_logit)
            return tokens, confidence
        else:
            return tokens


# =============================================================================
# Prompt 构建器（优先级文本提取）
# =============================================================================

class PromptComposer:
    """
    多模态 Prompt 构建器
    
    设计原则：
    1. 任务无关的核心结构：适用于所有药物性质预测任务
    2. 任务特定的描述注入：通过 TASK_DESCRIPTIONS 提供上下文
    3. 简洁高效：减少 token 数量，聚焦预测
    
    文本优先级：
    1. KG 关系文本（最可信：来自知识图谱的结构化信息）
    2. 药物描述文本（drug.json）
    3. SMILES 衍生描述（化学结构信息）
    4. 无文本（最低优先级）
    """
    
    # ========= 多数据集任务描述（支持泛化）=========
    # 格式：{数据集名: (任务描述, 任务类型, 标签数)}
    # 任务类型：multi_label (多标签), binary (二分类), multi_class (多分类)
    TASK_CONFIGS = {
        # MoleculeNet 数据集
        "SIDER": ("side effect prediction across organ systems", "multi_label", 27),
        "Tox21": ("toxicity prediction across biological assays", "multi_label", 12),
        "ToxCast": ("toxicity prediction across high-throughput assays", "multi_label", 617),
        "BBBP": ("blood-brain barrier penetration prediction", "binary", 1),
        "BACE": ("beta-secretase 1 inhibitor prediction", "binary", 1),
        "ClinTox": ("clinical trial toxicity and FDA approval prediction", "multi_label", 2),
        "HIV": ("HIV replication inhibition prediction", "binary", 1),
        "MUV": ("maximum unbiased validation prediction", "multi_label", 17),
        
        # 其他常用数据集
        "PCBA": ("PubChem BioAssay prediction", "multi_label", 128),
        "ChEMBL": ("bioactivity prediction", "multi_label", None),  # 动态标签数
        "DrugBank": ("drug-target interaction prediction", "multi_label", None),
        
        # 默认配置
        "default": ("drug property prediction", "multi_label", None),
    }
    
    # 向后兼容的简化描述
    TASK_DESCRIPTIONS = {
        k: v[0] for k, v in TASK_CONFIGS.items() if k != "default"
    }
    
    # 化学官能团到文本的映射
    FUNCTIONAL_GROUP_DESCRIPTIONS = {
        "benzene": "aromatic ring",
        "hydroxyl": "alcohol group (-OH)",
        "carboxyl": "carboxylic acid (-COOH)",
        "amino": "amine group (-NH2)",
        "amide": "amide bond (-CONH-)",
        "ester": "ester linkage (-COO-)",
        "ether": "ether linkage (-O-)",
        "nitro": "nitro group (-NO2)",
        "halogen": "halogen substituent",
        "sulfur": "sulfur-containing group",
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.text_map = {}  # drug.json文本
        self.kg_text_map = {}  # KG关系文本
        self.dataset_name = config.get("dataset_name", "SIDER")
        
        # 加载药物描述文本
        drug_text_path = config.get("drug_text_path")
        if drug_text_path and os.path.exists(drug_text_path):
            try:
                with open(drug_text_path, 'r') as f:
                    raw = json.load(f)
                # 修复：drug.json 的 key 是 DrugBank ID，需要用 value['SMILES'] 作为 key
                for k, v in raw.items():
                    if isinstance(v, dict) and 'SMILES' in v:
                        # 新格式：key=DrugBank ID, value={'SMILES':..., 'text':...}
                        smiles = v.get('SMILES', '')
                        if smiles:
                            # 用 SMILES 作为 key，整个 value 作为值
                            canon = self._canonicalize_smiles(smiles)
                            if canon:
                                self.text_map[canon] = v
                            self.text_map[smiles] = v
                    else:
                        # 旧格式：key=SMILES, value=text
                        canon = self._canonicalize_smiles(k)
                        if canon:
                            self.text_map[canon] = v
                        self.text_map[k] = v
                logger.info(f"[PromptComposer] Loaded {len(self.text_map)} drug texts")
            except Exception as e:
                logger.warning(f"[PromptComposer] Failed to load drug texts: {e}")
        
        # 加载KG关系文本（如果提供）
        kg_text_path = config.get("kg_text_path")
        if kg_text_path and os.path.exists(kg_text_path):
            try:
                with open(kg_text_path, 'r') as f:
                    self.kg_text_map = json.load(f)
                logger.info(f"[PromptComposer] Loaded {len(self.kg_text_map)} KG relation texts")
            except Exception as e:
                logger.warning(f"[PromptComposer] Failed to load KG texts: {e}")
    
    @staticmethod
    def _canonicalize_smiles(smiles: str) -> str:
        if not RDKIT_AVAILABLE or not smiles:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True) if mol else smiles
        except:
            return smiles
    
    def _extract_smiles_description(self, smiles: str) -> str:
        """
        从SMILES提取丰富的药物化学描述
        
        重点：MolToken不擅长的显式药物化学特征
        - 药代动力学相关：LogP, TPSA, 氢键
        - 药物相似性：QED, Lipinski规则
        - 毒性预警：结构警报
        """
        if not RDKIT_AVAILABLE or not smiles:
            return ""
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            
            features = []
            
            # ========= 基础结构 =========
            num_atoms = mol.GetNumAtoms()
            num_heavy = mol.GetNumHeavyAtoms()
            features.append(f"{num_heavy} heavy atoms")
            
            # 环系统
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            num_aromatic = sum(1 for ring in ring_info.AtomRings() 
                              if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
            if num_rings > 0:
                features.append(f"{num_rings} rings ({num_aromatic} aromatic)")
            
            # ========= 药代动力学相关（MolToken弱） =========
            # 分子量
            mw = Descriptors.MolWt(mol)
            features.append(f"MW={mw:.0f}")
            
            # LogP（脂溶性，影响吸收）
            logp = Descriptors.MolLogP(mol)
            logp_desc = "lipophilic" if logp > 3 else ("hydrophilic" if logp < 1 else "moderate lipophilicity")
            features.append(f"LogP={logp:.1f} ({logp_desc})")
            
            # TPSA（极性表面积，影响膜渗透）
            tpsa = Descriptors.TPSA(mol)
            if tpsa > 140:
                features.append(f"TPSA={tpsa:.0f} (poor membrane permeability)")
            elif tpsa < 60:
                features.append(f"TPSA={tpsa:.0f} (good CNS penetration potential)")
            else:
                features.append(f"TPSA={tpsa:.0f}")
            
            # 氢键供体/受体
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            features.append(f"HBD={hbd}, HBA={hba}")
            
            # 可旋转键（影响口服生物利用度）
            rotatable = Descriptors.NumRotatableBonds(mol)
            if rotatable > 10:
                features.append(f"{rotatable} rotatable bonds (high flexibility)")
            else:
                features.append(f"{rotatable} rotatable bonds")
            
            # ========= 药物相似性（MolToken无） =========
            # Lipinski规则判断
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            if lipinski_violations == 0:
                features.append("Lipinski: drug-like")
            elif lipinski_violations == 1:
                features.append("Lipinski: 1 violation")
            else:
                features.append(f"Lipinski: {lipinski_violations} violations (poor oral availability)")
            
            # ========= 官能团（毒性/活性相关） =========
            functional_groups = []
            
            # 检测关键官能团
            smarts_patterns = {
                "nitro": "[N+](=O)[O-]",
                "amine": "[NX3;H2,H1;!$(NC=O)]",
                "carboxylic_acid": "[CX3](=O)[OX2H1]",
                "amide": "[NX3][CX3](=[OX1])",
                "ester": "[CX3](=O)[OX2][#6]",
                "aldehyde": "[CX3H1](=O)",
                "ketone": "[CX3](=O)[#6]",
                "alcohol": "[OX2H]",
                "ether": "[OD2]([#6])[#6]",
                "halogen": "[F,Cl,Br,I]",
                "sulfide": "[#16X2]",
                "sulfonamide": "[$([SX4](=[OX1])(=[OX1])([NX3])[#6])]",
            }
            
            for name, smarts in smarts_patterns.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        matches = len(mol.GetSubstructMatches(pattern))
                        if matches > 0:
                            functional_groups.append(name.replace("_", " "))
                except:
                    pass
            
            if functional_groups:
                features.append(f"Functional groups: {', '.join(functional_groups[:5])}")
            
            # ========= 毒性预警结构（MolToken无） =========
            toxicity_alerts = []
            
            # 一些常见的毒性警报结构
            toxic_smarts = {
                "reactive_aldehyde": "[CX3H1](=O)[#6]",
                "michael_acceptor": "[CX3]=[CX3][CX3](=O)",
                "epoxide": "C1OC1",
                "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
            }
            
            for name, smarts in toxic_smarts.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        toxicity_alerts.append(name.replace("_", " "))
                except:
                    pass
            
            if toxicity_alerts:
                features.append(f"Structural alerts: {', '.join(toxicity_alerts)}")
            
            # 组合描述
            return "Molecular properties: " + "; ".join(features)
            
        except Exception as e:
            return ""
    
    def get_text_with_priority(self, smiles: str, drug_text: str = None,
                               kg_relations: List[str] = None) -> Tuple[str, str]:
        """
        按优先级获取文本
        
        返回：(文本内容, 来源标签)
        来源标签：kg_relation, drug_text, smiles_derived, none
        """
        MAX_TEXT_LEN = 800  # 文本长度限制（配合512 prompt）
        MAX_KG_RELATIONS = 8  # KG关系数量
        
        # 优先级1：KG关系文本
        if kg_relations and len(kg_relations) > 0:
            kg_text = "; ".join(kg_relations[:MAX_KG_RELATIONS])
            return kg_text[:MAX_TEXT_LEN], "kg_relation"
        
        # 检查KG文本缓存
        canon_smi = self._canonicalize_smiles(smiles)
        if canon_smi in self.kg_text_map:
            return str(self.kg_text_map[canon_smi])[:MAX_TEXT_LEN], "kg_relation"
        if smiles in self.kg_text_map:
            return str(self.kg_text_map[smiles])[:MAX_TEXT_LEN], "kg_relation"
        
        # 优先级2：药物描述文本
        if drug_text:
            return str(drug_text)[:MAX_TEXT_LEN], "drug_text"
        
        # 辅助函数：从 text_map 值中提取文本
        def _extract_text(v):
            if isinstance(v, dict):
                return v.get('text', str(v))  # 新格式：{'SMILES':..., 'text':...}
            return str(v)  # 旧格式：直接是文本
        
        if canon_smi in self.text_map:
            return _extract_text(self.text_map[canon_smi])[:MAX_TEXT_LEN], "drug_text"
        if smiles in self.text_map:
            return _extract_text(self.text_map[smiles])[:MAX_TEXT_LEN], "drug_text"
        
        # 优先级3：SMILES衍生描述
        smiles_desc = self._extract_smiles_description(smiles)
        if smiles_desc:
            return smiles_desc[:MAX_TEXT_LEN], "smiles_derived"
        
        # 优先级4：无文本
        return "", "none"
    
    @staticmethod
    def _vacuity_to_confidence(vac):
        """Convert scalar vacuity to discrete confidence label."""
        if vac is None:
            return "high"
        if vac < 0.5:
            return "high"
        if vac < 0.8:
            return "moderate"
        return "low"

    def build_prompt(self, smiles: str, drug_text: str = None,
                     kg_relations: List[str] = None,
                     has_kg: bool = True,
                     has_moltoken: bool = True,
                     vacuity_kg: float = None) -> Tuple[str, str]:
        """
        Dynamic prompt with per-modality confidence labels and reasoning guidance.

        Returns: (prompt_text, text_source)
        """
        parts = []

        # ── 1. Task description ──
        task_cfg = self.TASK_CONFIGS.get(self.dataset_name, self.TASK_CONFIGS["default"])
        task_desc, task_type, num_labels = task_cfg
        if task_type == "binary":
            task_str = f"Predict whether this molecule exhibits {task_desc}."
        else:
            label_info = f" across {num_labels} targets" if num_labels else ""
            task_str = f"Predict {task_desc}{label_info}."
        parts.append(task_str)

        # ── 2. Drug information ──
        text, source = self.get_text_with_priority(smiles, drug_text, kg_relations)
        if text:
            parts.append(f"Drug: {text[:250]}")

        # ── 3. Modality confidence assessment ──
        parts.append("Input reliability:")
        parts.append("[Structure] directly computed from molecular graph — high confidence.")

        if has_moltoken:
            parts.append("[MolToken] derived from 3D torsion angles — high confidence.")
        else:
            parts.append("[MolToken] cross-modal completion — moderate confidence, treat with caution.")

        if has_kg:
            parts.append("[KG] directly retrieved from knowledge graph — high confidence.")
        else:
            conf = self._vacuity_to_confidence(vacuity_kg)
            if conf == "high":
                parts.append("[KG] cross-modal completion with strong evidence — moderate-high confidence.")
            elif conf == "moderate":
                parts.append("[KG] cross-modal completion with partial evidence — moderate confidence.")
            else:
                parts.append("[KG] cross-modal completion with limited evidence — low confidence, treat as auxiliary.")

        # ── 4. Reasoning guidance ──
        any_completed = (not has_kg) or (not has_moltoken)
        if not any_completed:
            parts.append("All modalities are directly observed; integrate them with equal weight.")
        else:
            inferred = []
            if not has_kg:
                inferred.append("KG")
            if not has_moltoken:
                inferred.append("MolToken")
            inferred_str = " and ".join(inferred)
            parts.append(
                f"{inferred_str} features were inferred, not observed. "
                "Rely primarily on directly observed modalities for prediction; "
                "use inferred features as supporting evidence only."
            )

        # ── 5. Prediction anchor ──
        parts.append("Prediction:")

        return " ".join(parts), source
    
    def get_prompt_example(self, dataset_name: str = None) -> str:
        """
        获取指定数据集的prompt示例（用于调试和文档）
        """
        ds = dataset_name or self.dataset_name
        example_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        example_text = "Aspirin is a nonsteroidal anti-inflammatory drug with analgesic and antipyretic properties."
        
        prompt, _ = self.build_prompt(
            smiles=example_smiles,
            drug_text=example_text,
            has_kg=True
        )
        return f"示例Prompt ({ds}):\n{prompt}\n\n预估token数: ~{len(prompt.split())*1.3:.0f}"


# =============================================================================
# 主模型
# =============================================================================

class DPModel(nn.Module):
    """
    DPModel — Multimodal Drug Property Prediction.

    Architecture:
      Three modalities: Structure (2D graph), KG (knowledge graph),
      MolToken (3D torsion-aware)
           |
      SA-CMSA: Sparse-Attention Cross-Modal Subspace Alignment
        -- Stage 1: struct -> sparse attn(kge_bank) -> kg_retrieved
        -- Stage 2: struct+mol+retrieved -> route -> residual_dec -> residual
        -- Output: kg_retrieved + scale * residual
           |
      Q-Former Adapters -> local tokens per modality (LLM-dim)
           |
      [start][tokens][end] per modality + Dynamic Prompt
           |
      Qwen-2.5-3B (LoRA)
           |
      Attention Pooling -> Prediction Head -> Logits

    Core innovation -- CMSA: Alignment-driven KG completion with dense
      subspace alignment losses. completion_flags tracks which KG is
      synthesized vs real, for downstream confidence modules.
    """
    
    def __init__(
        self,
        config: Dict,
        pred_dim: int,
        kg_prototype_init: torch.Tensor = None,
        struct_stats: Dict = None,
        kge_bank: torch.Tensor = None,
        kge_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.config = config
        self.pred_dim = pred_dim
        net_cfg = config["network"]

        ablation_str = net_cfg.get("ablation", "")
        self.ablation_flags = set(a.strip() for a in ablation_str.split(",") if a.strip())
        if "no_cmsa" in self.ablation_flags:
            self.ablation_flags.add("no_sacmsa")
        if self.ablation_flags:
            logger.info(f"[DPModel] ablation: {self.ablation_flags}")

        if "no_moltoken" in self.ablation_flags:
            net_cfg["use_moltoken"] = False
        if "no_sacmsa" in self.ablation_flags:
            net_cfg.setdefault("cmsa", {})["enabled"] = False
            logger.info("[Ablation] CMSA disabled")
        if "no_cmon" in self.ablation_flags:
            net_cfg.setdefault("mars", {})["enabled"] = False
            logger.info("[Ablation] CMON disabled")

        # ========= 编码器 =========
        self.structure_encoder = self._build_encoder(net_cfg["drug"]["structure"])
        struct_dim = self.structure_encoder.output_dim
        self.struct_dim = struct_dim
        self.llm_encoder = self._build_llm(net_cfg["llm"])
        self.llm_dim = getattr(self.llm_encoder.model.config, "hidden_size", 2048)
        self.kg_dim = net_cfg["drug"]["kg"]["embedding_dim"]
        
        # ========= MolToken编码器（第四模态）=========
        self.use_moltoken = net_cfg.get("use_moltoken", False)
        self.moltoken_encoder = None
        self.moltoken_adapter = None
        self.moltoken_dim = 768
        
        if self.use_moltoken:
            moltoken_cfg = net_cfg.get("moltoken", {})
            if moltoken_cfg.get("model_name"):
                try:
                    from models.llm_encoder import TokenMolEncoder
                    self.moltoken_encoder = TokenMolEncoder(moltoken_cfg)
                    self.moltoken_dim = self.moltoken_encoder.output_dim
                    for p in self.moltoken_encoder.model.parameters():
                            p.requires_grad = False
                    self.moltoken_encoder.model.eval()
                    logger.info(f"[DPModel] MolToken encoder enabled: dim={self.moltoken_dim}")
                except Exception as e:
                    logger.warning(f"[DPModel] MolToken init failed: {e}")
                    self.use_moltoken = False
        
        # ========= KG 补全实体索引映射（用于可视化解释）=========
        # kge_bank 的第 i 行对应 kge_keys[i]（通常是 BMKG-DP 实体 id）
        self.kge_keys = kge_keys
        
        # CMSA
        cmsa_cfg = net_cfg.get("cmsa", net_cfg.get("dec", {}))
        self.use_cmcfs = cmsa_cfg.get("enabled", True)
        self.cmcfs = None
        if self.use_cmcfs and kge_bank is not None:
            self.cmcfs = SparseAttentionCMSA(
                modal_dims=[struct_dim, self.kg_dim, self.moltoken_dim],
                kge_bank=kge_bank,
                sub_dim=cmsa_cfg.get("sub_dim", 128),
                topk=cmsa_cfg.get("topk", 16),
                num_heads=cmsa_cfg.get("num_heads", 2),
                dropout=cmsa_cfg.get("dropout", 0.1),
                align_lambda=cmsa_cfg.get("align_lambda", 0.05),
                retrieve_lambda=cmsa_cfg.get("retrieve_lambda", 0.1),
                enable_mol_completion=cmsa_cfg.get("enable_mol_completion", False),
                ablation_flags=self.ablation_flags,
            )
        else:
            self.use_cmcfs = False
            if kge_bank is None:
                logger.warning("[DPModel] SA-CMSA disabled: kge_bank is None")
            else:
                logger.info("[DPModel] SA-CMSA disabled by config")

        # CMON
        mars_cfg = net_cfg.get("mars", {})
        self.use_mars = mars_cfg.get("enabled", False)
        self.mars = None
        if self.use_mars:
            self.mars = CMONModule(
                llm_dim=self.llm_dim,
                n_modalities=3,
                sub_dim=mars_cfg.get("sub_dim", 64),
                ablation_flags=self.ablation_flags,
            )

        # ========= Adapter Token配置 =========
        # 总token预算：~400 = 64(struct) + 48(kg) + 56(moltoken) + ~230(prompt)
        # 设计原则：
        # 1. Structure是核心模态，给最多token (64)
        # 2. MolToken提供互补语义信息 (56)
        # 3. KG提供知识图谱信息 (48)
        # 4. 剩余~230 token给Prompt文本
        # 优化增加Adapter的dropout到0.25，减少过拟合
        adapter_dropout = net_cfg.get("adapter_dropout", 0.25)
        num_struct_tokens = net_cfg.get("num_struct_tokens", 64)   # 64 tokens
        num_kg_tokens = net_cfg.get("num_kg_tokens", 48)           # 48 tokens
        num_moltoken_tokens = net_cfg.get("num_moltoken_tokens", 56)  # 56 tokens
        
        # Q-Former style Adapter
        self.struct_adapter = AdapterTokenGenerator(
            struct_dim, self.llm_dim, num_tokens=num_struct_tokens,
            num_layers=4, dropout=adapter_dropout,
        )
        self.kg_adapter = AdapterTokenGenerator(
            self.kg_dim, self.llm_dim, num_tokens=num_kg_tokens,
            num_layers=4, dropout=adapter_dropout,
        )
        if self.use_moltoken:
            self.moltoken_adapter = AdapterTokenGenerator(
                self.moltoken_dim, self.llm_dim, num_tokens=num_moltoken_tokens,
                num_layers=4, dropout=adapter_dropout,
            )
        
        # ========= 模态文本标记 =========
        self.modality_markers = {
            "struct_start": "<STRUCT_START>",
            "struct_end": "<STRUCT_END>",
            "kg_start": "<KG_START>",
            "kg_end": "<KG_END>",
            "moltoken_start": "<CHEM_START>",
            "moltoken_end": "<CHEM_END>",
        }
        base_markers = ["struct_start", "struct_end", "kg_start", "kg_end"]
        for marker_name in base_markers:
            setattr(self, f"{marker_name}_residual",
                    nn.Parameter(torch.randn(1, 1, self.llm_dim) * 0.02))
        if self.use_moltoken:
            for marker_name in ["moltoken_start", "moltoken_end"]:
                setattr(self, f"{marker_name}_residual",
                        nn.Parameter(torch.randn(1, 1, self.llm_dim) * 0.02))

        # ========= Prompt 构建器 =========
        self.prompt_builder = PromptComposer({
            "drug_text_path": net_cfg.get("drug_text_path"),
            "kg_text_path": net_cfg.get("kg_text_path"),
            "dataset_name": net_cfg.get("dataset_name", "SIDER"),
        })
        # Prompt文本的最大长度
        # Token预算：~400总 - 174模态token = ~230 prompt token
        # 但tokenizer可能会分词，所以给256以确保安全
        self.prompt_max_length = net_cfg.get("prompt_max_length", 256)
        
        # 记录总token预算（用于调试）；CMON 开启时 +3 digest tokens
        num_markers = 9 if self.use_mars else 6
        total_modal_tokens = num_struct_tokens + num_kg_tokens + num_moltoken_tokens + num_markers
        logger.info(
            f"[DPModel] Token budget: "
            f"struct={num_struct_tokens} kg={num_kg_tokens} moltoken={num_moltoken_tokens} "
            f"markers={num_markers} prompt_max={self.prompt_max_length} "
            f"total≈{total_modal_tokens + self.prompt_max_length}"
        )

        # ========= 预测头（残差版）=========
        head_dropout = net_cfg.get("head_dropout", 0.30)
        # 带残差连接的预测头: 2048 → 1024 → 512 → pred_dim
        hidden_dim = 1024
        
        self.head = ResidualPredictionHead(
            input_dim=self.llm_dim,
            hidden_dim=hidden_dim,
            output_dim=pred_dim,
            dropout=head_dropout * 0.6,
        )
        logger.info("[DPModel] Residual Prediction Head enabled")
        
        # ========= 正则化模块 =========
        # R-Drop 正则化
        self.rdrop_reg = RDropRegularizer(alpha=0.5)
        self.feat_consistency_reg = FeatureConsistencyRegularizer()
        self.use_rdrop = net_cfg.get("use_consistent_feature", True)
        
        # MolToken-Structure互补性损失（使用可学习投影）
        if self.use_moltoken:
            self.complementarity_loss = ModalityComplementarityLoss(
                struct_dim=struct_dim,
                moltoken_dim=self.moltoken_dim,
                common_dim=128,
                margin=0.3,
            )
        
        # 模态对比学习
        self.use_contrastive = net_cfg.get("use_contrastive", True)
        if self.use_contrastive:
            self.contrastive_loss = ModalContrastiveLoss(
                hidden_dim=self.llm_dim,
                proj_dim=256,
                temperature=0.15,  # 防梯度尖峰从0.07提高到0.15
            )
            logger.info("[DPModel] Modal Contrastive Learning enabled (temp=0.15)")
        
        # ========= 模块级优化突破0.7的关键创新 =========
        
        # 优化1特征噪声注入 - 提升泛化能力
        self.use_feature_noise = net_cfg.get("use_feature_noise", True)
        if self.use_feature_noise:
            noise_std = net_cfg.get("feature_noise_std", 0.05)
            self.struct_noise = FeatureNoiseInjector(noise_std=noise_std, adaptive=True)
            self.kg_noise = FeatureNoiseInjector(noise_std=noise_std, adaptive=True)
            self.moltoken_noise = FeatureNoiseInjector(noise_std=noise_std * 0.5, adaptive=True)
            logger.info(f"[DPModel] Feature noise injection enabled: std={noise_std}")
        
        # 优化2Attention Pooling - 替代简单的last-token
        self.use_attention_pooling = net_cfg.get("use_attention_pooling", True)
        if self.use_attention_pooling:
            self.attention_pooler = AttentionPooling(
                hidden_dim=self.llm_dim,
                num_heads=8,
                dropout=net_cfg.get("pooling_dropout", 0.15),
            )
            # 可学习的 Attention vs Last-token 融合比例
            # sigmoid(0.85) ≈ 0.70，与旧版初始值一致，但现在可以被模型自己调整
            self.attn_pool_ratio = nn.Parameter(torch.tensor(0.85))
            logger.info("[DPModel] Attention pooling enabled (learnable ratio, init≈0.70)")
        
        self.use_label_graph = net_cfg.get("use_label_graph", True)
        if self.use_label_graph:
            self.label_graph = LabelGraphNetwork(
                num_labels=pred_dim,
                hidden_dim=self.llm_dim,
                dropout=net_cfg.get("label_graph_dropout", 0.25),
            )
            # 融合权重：head预测 vs label_graph预测
            self.label_graph_weight = nn.Parameter(torch.tensor(0.3))
            logger.info(f"[DPModel] Label Graph Network enabled: {pred_dim} labels")

        # ========= LoRA =========
        if net_cfg.get("use_lora", True):
            self._enable_lora(net_cfg)

        logger.info(
            f"[DPModel] Three-modality system + SA-CMSA | "
            f"struct={num_struct_tokens} kg={num_kg_tokens} "
            f"moltoken={num_moltoken_tokens if self.use_moltoken else 'disabled'}"
        )


    def _build_encoder(self, cfg: Dict):
        from models import SUPPORTED_MOL_ENCODER
        if "config_path" in cfg:
            with open(cfg["config_path"], "r") as f:
                from ...utils.paths import expand_asset_paths
                enc_cfg = expand_asset_paths(json.load(f))
            enc_cfg["name"] = cfg["name"]
        else:
            enc_cfg = cfg
        
        encoder = SUPPORTED_MOL_ENCODER[enc_cfg["name"]](enc_cfg)
        
        # Partial fine-tune: unfreeze last N GNN layers (config-driven)
        n_unfreeze = self.config.get("network", {}).get("unfreeze_graphmvp_layers", 2)
        for param in encoder.parameters():
            param.requires_grad = False

        gnn = getattr(encoder, 'main_model', encoder)
        num_layers = getattr(gnn, 'num_layer', 5)
        unfreeze_from = max(num_layers - n_unfreeze, 0)

        if hasattr(gnn, 'gnns'):
            for i in range(unfreeze_from, num_layers):
                for p in gnn.gnns[i].parameters():
                    p.requires_grad = True
        if hasattr(gnn, 'batch_norms'):
            for i in range(unfreeze_from, num_layers):
                for p in gnn.batch_norms[i].parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())
        logger.info(
            f"[DPModel] GraphMVP: freeze layers 0-{unfreeze_from-1}, "
            f"unfreeze layers {unfreeze_from}-{num_layers-1} | "
            f"trainable={trainable/1e3:.1f}K / {total/1e3:.1f}K"
        )
        return encoder

    def _build_llm(self, cfg: Dict):
        from models import SUPPORTED_LLM_ENCODER
        return SUPPORTED_LLM_ENCODER[cfg["name"]](cfg)

    def _enable_lora(self, cfg: Dict):
        lora_rank = cfg.get("lora_rank", 16)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.llm_encoder.model = get_peft_model(self.llm_encoder.model, lora_config)
        
        # 显存优化启用梯度检查点（用计算换显存）
        # 注意：DDP 多卡模式下禁用 checkpointing（有 inplace 操作兼容性问题）
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            logger.info("[DPModel] Gradient checkpointing DISABLED (DDP multi-GPU mode)")
        elif hasattr(self.llm_encoder.model, 'gradient_checkpointing_enable'):
            self.llm_encoder.model.gradient_checkpointing_enable()
            logger.info("[DPModel] Gradient checkpointing enabled (saves ~50% activation memory)")
        
        trainable = sum(p.numel() for p in self.llm_encoder.model.parameters() if p.requires_grad)
        logger.info(f"[DPModel] LoRA enabled (r={lora_rank}) Trainable={trainable/1e6:.2f}M")

    def set_lora_trainable(self, trainable: bool):
        model = getattr(self.llm_encoder, "model", None)
        if model is None:
            return
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = trainable

    def _encode_structure(self, drug_batch: Dict, device: torch.device):
        data = drug_batch.get("structure", drug_batch)
        if hasattr(self.structure_encoder, "encode_structure"):
            feats = self.structure_encoder.encode_structure(data)
        elif hasattr(self.structure_encoder, "encode_mol"):
            feats = self.structure_encoder.encode_mol(data)
        else:
            feats = self.structure_encoder(data)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        smiles = []
        if isinstance(drug_batch, dict) and "smiles" in drug_batch:
            raw = drug_batch["smiles"]
            if isinstance(raw, (list, tuple)):
                smiles = [str(s) for s in raw]
            elif isinstance(raw, str):
                smiles = [raw] * feats.size(0)
        if not smiles:
            smiles = [""] * feats.size(0)
            
        return feats.to(device), smiles

    def _prepare_kg(self, drug_batch: Dict, batch_size: int, device: torch.device):
        """
        准备 KG 特征 + 候选集（用于 KG 检索注意力）
        
        Returns:
            kg_feat:        (B, kg_dim) KG embedding
            is_missing:     (B,) bool   缺失标记
            kg_candidates:  (B, K, kg_dim) 候选 KG embedding (或 None)
            candidate_mask: (B, K) bool  候选有效标记 (或 None)
            candidate_sims: (B, K) float 候选相似度 (或 None)
        """
        kg_feat = drug_batch.get("kg")
        if kg_feat is None:
            return (torch.zeros(batch_size, self.kg_dim, device=device),
                    torch.ones(batch_size, dtype=torch.bool, device=device),
                    None, None, None)
        
        if kg_feat.dim() == 3:
            kg_feat = kg_feat.squeeze(1)
        kg_feat = kg_feat.to(device)
        is_missing = (kg_feat.abs().sum(dim=-1) < 1e-6) | torch.isnan(kg_feat.sum(dim=-1))
        kg_feat = torch.nan_to_num(kg_feat, nan=0.0)
        
        # 获取候选集（由 KGEFeaturizer 指纹预筛选生成）
        kg_candidates = drug_batch.get("kg_candidates", None)
        candidate_mask = drug_batch.get("kg_candidate_mask", None)
        candidate_sims = drug_batch.get("kg_candidate_sims", None)
        
        if kg_candidates is not None:
            kg_candidates = kg_candidates.to(device).float()
            candidate_mask = candidate_mask.to(device).bool() if candidate_mask is not None else None
            candidate_sims = candidate_sims.to(device).float() if candidate_sims is not None else None
        
        return kg_feat, is_missing, kg_candidates, candidate_mask, candidate_sims

    def _encode_moltoken(self, drug_batch: Dict, smiles_list: List[str], device: torch.device):
        """编码MolToken模态"""
        if not self.use_moltoken or self.moltoken_encoder is None:
            return None
        
        cached_seq = drug_batch.get("tokenmol_seq", None) if isinstance(drug_batch, dict) else None
        
        token_lists = []
        if cached_seq is not None and isinstance(cached_seq, list) and len(cached_seq) == len(smiles_list):
            for seq in cached_seq:
                token_lists.append(str(seq).split())
        else:
            for smi in smiles_list:
                token_lists.append(list(smi) if smi else ["[UNK]"])
        
        try:
            with torch.no_grad():
                moltoken_feat, _ = self.moltoken_encoder.encode_token_lists(token_lists, return_lm_loss=False)
            return moltoken_feat.to(device)
        except Exception as e:
            logger.warning(f"[MolToken] Encoding failed: {e}")
            return None

    def _get_marker_embedding(self, marker_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        """获取模态标记的嵌入（文本嵌入 + 可学习残差）"""
        tokenizer = self.llm_encoder.tokenizer
        embed_layer = self.llm_encoder.model.get_input_embeddings()
        
        # 获取文本标记
        marker_text = self.modality_markers.get(marker_name, "[UNK]")
        marker_ids = tokenizer(marker_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        # 获取文本嵌入
        text_embed = embed_layer(marker_ids)  # (1, seq_len, dim)
        # 平均池化成单个token
        text_embed = text_embed.mean(dim=1, keepdim=True)  # (1, 1, dim)
        
        # 获取对应的可学习残差
        residual_name = marker_name + "_residual"
        if hasattr(self, residual_name):
            residual = getattr(self, residual_name)
            text_embed = text_embed + residual.to(text_embed.dtype)
        
        # 扩展到batch
        return text_embed.expand(batch_size, -1, -1)

    def _get_decoder_layers(self):
        """安全获取 LLM decoder layers, 兼容 PEFT 包装和原始模型."""
        llm = self.llm_encoder.model
        # 尝试多种层级路径 (PEFT 会多包一层 .model)
        candidates = [
            lambda: llm.model.model.layers,   # PEFT: PeftModel → CausalLM → Model → layers
            lambda: llm.model.layers,          # 原始: CausalLM → Model → layers
            lambda: llm.base_model.model.model.layers,  # PEFT v2 路径
        ]
        for fn in candidates:
            try:
                layers = fn()
                if layers is not None and len(layers) > 0:
                    return layers
            except (AttributeError, TypeError):
                continue
        raise AttributeError(
            f"无法找到 decoder layers. llm type={type(llm).__name__}. "
            f"请检查模型结构: {[n for n, _ in llm.named_children()]}"
        )

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """验证并修复tensor的有效性"""
        if tensor is None:
            return None
        
        # 检查NaN和Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"[{name}] Found NaN/Inf, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 检查数值范围（防止极端值）
        max_val = tensor.abs().max()
        if max_val > 1e4:
            logger.warning(f"[{name}] Large values detected (max={max_val:.2f}), clipping")
            tensor = tensor.clamp(-1e4, 1e4)
        
        return tensor

    def _forward_once(self, drug_batch: Dict,
                      struct_tokens: torch.Tensor, kg_tokens: torch.Tensor,
                      moltoken_tokens: torch.Tensor,
                      smiles_list: List[str],
                      confidences: Dict, device: torch.device,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LLM 前向传播: 起止符标记 + tokens → LLM → logits"""
        batch_size = struct_tokens.size(0)
        
        embed_layer = self.llm_encoder.model.get_input_embeddings()
        target_dtype = embed_layer.weight.dtype
        
        struct_start_embed = self._get_marker_embedding("struct_start", batch_size, device)
        struct_end_embed = self._get_marker_embedding("struct_end", batch_size, device)
        kg_start_embed = self._get_marker_embedding("kg_start", batch_size, device)
        kg_end_embed = self._get_marker_embedding("kg_end", batch_size, device)
        moltoken_start_embed = self._get_marker_embedding("moltoken_start", batch_size, device)
        moltoken_end_embed = self._get_marker_embedding("moltoken_end", batch_size, device)

        digest_tokens = confidences.get("cmon_digests", None)
        if digest_tokens is not None and len(digest_tokens) == 3:
            struct_seq = torch.cat([
                struct_start_embed.to(target_dtype),
                digest_tokens[0].to(target_dtype),
                struct_tokens.to(target_dtype),
                struct_end_embed.to(target_dtype),
            ], dim=1)
            kg_seq = torch.cat([
                kg_start_embed.to(target_dtype),
                digest_tokens[1].to(target_dtype),
                kg_tokens.to(target_dtype),
                kg_end_embed.to(target_dtype),
            ], dim=1)
            moltoken_seq = torch.cat([
                moltoken_start_embed.to(target_dtype),
                digest_tokens[2].to(target_dtype),
                moltoken_tokens.to(target_dtype),
                moltoken_end_embed.to(target_dtype),
            ], dim=1)
        else:
            struct_seq = torch.cat([
                struct_start_embed.to(target_dtype),
                struct_tokens.to(target_dtype),
                struct_end_embed.to(target_dtype),
            ], dim=1)
            kg_seq = torch.cat([
                kg_start_embed.to(target_dtype),
                kg_tokens.to(target_dtype),
                kg_end_embed.to(target_dtype),
            ], dim=1)
            moltoken_seq = torch.cat([
                moltoken_start_embed.to(target_dtype),
                moltoken_tokens.to(target_dtype),
                moltoken_end_embed.to(target_dtype),
            ], dim=1)

        prefix = torch.cat([struct_seq, kg_seq, moltoken_seq], dim=1)
        prefix = self._validate_tensor(prefix, "prefix")
        
        fusion_offset = 0
        struct_end_pos = fusion_offset + struct_seq.size(1)
        kg_end_pos = struct_end_pos + kg_seq.size(1)
        mol_end_pos = kg_end_pos + moltoken_seq.size(1)
        
        if prefix.shape[-1] != self.llm_dim:
            raise ValueError(f"Prefix dimension mismatch: {prefix.shape[-1]} vs {self.llm_dim}")
        
        # ========= 构建 Prompt =========
        batch_texts = drug_batch.get("text", None)
        if isinstance(batch_texts, list):
            batch_texts = [str(x) for x in batch_texts]
        else:
            batch_texts = None
        
        kg_relations = drug_batch.get("kg_relations", None)
        
        prompts = []
        text_sources = []
        dec_vac_prompt = confidences.get("dec_vacuity", None)
        for i, smi in enumerate(smiles_list):
            text = batch_texts[i] if batch_texts and i < len(batch_texts) else None
            kg_rel = kg_relations[i] if kg_relations and i < len(kg_relations) else None
            vac_kg_i = dec_vac_prompt[i, 1].item() if dec_vac_prompt is not None else None

            kg_completed_flags = confidences.get("kg_is_completed", [False]*batch_size)
            mol_completed_flags = confidences.get("mol_is_completed", [False]*batch_size)
            prompt, source = self.prompt_builder.build_prompt(
                smi, text, kg_relations=[kg_rel] if kg_rel else None,
                has_kg=not (kg_completed_flags[i] if isinstance(kg_completed_flags, list) else kg_completed_flags[i].item() if hasattr(kg_completed_flags[i], 'item') else kg_completed_flags[i]),
                has_moltoken=not (mol_completed_flags[i] if isinstance(mol_completed_flags, list) else mol_completed_flags[i].item() if hasattr(mol_completed_flags[i], 'item') else mol_completed_flags[i]),
                vacuity_kg=vac_kg_i,
            )
            prompts.append(prompt)
            text_sources.append(source)
        
        # ========= Tokenize + LLM =========
        tokenizer = self.llm_encoder.tokenizer
        tokenized = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.prompt_max_length,
        ).to(device)
        
        text_embeds = embed_layer(tokenized.input_ids)
        text_embeds = self._validate_tensor(text_embeds, "text_embeds")
        
        inputs_embeds = torch.cat([prefix.to(target_dtype), text_embeds.to(target_dtype)], dim=1)
        inputs_embeds = self._validate_tensor(inputs_embeds, "inputs_embeds")
        
        prefix_mask = torch.ones(batch_size, prefix.size(1), device=device, dtype=tokenized.attention_mask.dtype)
        attention_mask_2d = torch.cat([prefix_mask, tokenized.attention_mask], dim=1)

        _capture_attn = getattr(self, "_capture_digest_attn", False)
        outputs = self.llm_encoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_2d,
            output_hidden_states=True,
            output_attentions=_capture_attn,
        )

        if _capture_attn and hasattr(outputs, "attentions") and outputs.attentions is not None:
            has_digest = (digest_tokens is not None and len(digest_tokens) == 3)
            if has_digest:
                s_len = struct_seq.size(1)
                k_len = kg_seq.size(1)
                m_len = moltoken_seq.size(1)
                digest_positions = [1, s_len + 1, s_len + k_len + 1]
                attn_data = []
                for layer_attn in outputs.attentions:
                    avg_heads = layer_attn.mean(dim=1)
                    attn_data.append(avg_heads[:, :, digest_positions].detach().cpu())
                self._last_digest_attn = {
                    "positions": digest_positions,
                    "prefix_len": prefix.size(1),
                    "struct_seq_len": s_len,
                    "kg_seq_len": k_len,
                    "mol_seq_len": m_len,
                    "attn_to_digest": [a.numpy() for a in attn_data],
                }
        
        last_hidden = outputs.hidden_states[-1]
        
        # ========= Pooling =========
        last_pos = attention_mask_2d.sum(dim=1).clamp(min=1).to(torch.long) - 1
        batch_idx = torch.arange(batch_size, device=device)
        last_token_hidden = last_hidden[batch_idx, last_pos, :]
        
        if self.use_attention_pooling:
            attn_pooled = self.attention_pooler(last_hidden, attention_mask_2d)
            ratio = torch.sigmoid(self.attn_pool_ratio)
            pooled = (ratio * attn_pooled + (1.0 - ratio) * last_token_hidden).float()
        else:
            second_last_pos = (last_pos - 1).clamp(min=0)
            second_last_hidden = last_hidden[batch_idx, second_last_pos, :]
            pooled = (0.8 * last_token_hidden + 0.2 * second_last_hidden).float()

        # ========= 预测头 =========
        head_logits = self.head(pooled)
        
        if self.use_label_graph:
            label_graph_logits = self.label_graph(pooled)
            w = torch.sigmoid(self.label_graph_weight)
            logits = (1 - w) * head_logits + w * label_graph_logits
        else:
            logits = head_logits
        
        logits = logits.clamp(-10.0, 10.0)
        
        return logits, pooled, text_sources

    def forward(self, drug_batch: Dict, mixup: Optional[Dict] = None, augment_smiles: bool = False):
        device = next(self.parameters()).device
        
        # ========= 1. 原始特征编码 =========
        struct_feat, smiles_list = self._encode_structure(drug_batch, device)
        batch_size = struct_feat.size(0)
        kg_feat_raw, kg_mask, kg_candidates, candidate_mask, candidate_sims = \
            self._prepare_kg(drug_batch, batch_size, device)
        moltoken_feat = self._encode_moltoken(drug_batch, smiles_list, device)
        
        # ========= 2. Mixup on raw features（可选）=========
        if mixup is not None:
            perm = mixup["perm"]
            lam = mixup.get("lam", 0.5)
            struct_feat = lam * struct_feat + (1 - lam) * struct_feat[perm]
            if moltoken_feat is not None:
                moltoken_feat = lam * moltoken_feat + (1 - lam) * moltoken_feat[perm]
        
        # ========= 3. 特征噪声注入 =========
        if self.training and self.use_feature_noise:
            struct_feat = self.struct_noise(struct_feat)
            if moltoken_feat is not None:
                moltoken_feat = self.moltoken_noise(moltoken_feat)
        
        # ========= 3.5 CMSA: Cross-Modal Subspace Alignment =========
        kg_feat = kg_feat_raw
        cmcfs_recon_loss = torch.tensor(0.0, device=device)
        cfs_diag = {}
        mol_feat_for_cmcfs = moltoken_feat if moltoken_feat is not None else torch.zeros(
            batch_size, self.moltoken_dim, device=device
        )
        mol_missing = (moltoken_feat is None)
        missing_flags = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        missing_flags[:, 1] = kg_mask
        if mol_missing:
            missing_flags[:, 2] = True

        completion_flags = missing_flags.clone()
        dec_vacuity = None
        if self.cmcfs is not None:
            enhanced_feats, completion_flags, cmcfs_recon_loss, cfs_diag, _ = self.cmcfs(
                [struct_feat, kg_feat, mol_feat_for_cmcfs],
                missing_flags,
            )
            struct_feat, kg_feat, mol_feat_for_cmcfs = enhanced_feats

        if mixup is not None:
            kg_feat = lam * kg_feat + (1 - lam) * kg_feat[perm]

        # ========= 4. Adapter → Tokens =========
        struct_feat = self._validate_tensor(struct_feat, "struct_feat")
        struct_tokens = self.struct_adapter(struct_feat)
        struct_tokens = self._validate_tensor(struct_tokens, "struct_tokens")

        has_moltoken = False
        moltoken_tokens = None
        mol_was_completed = mol_missing and self.cmcfs is not None and getattr(self.cmcfs, 'enable_mol_completion', False)
        if (not mol_missing or mol_was_completed) and self.moltoken_adapter is not None:
            mol_feat_for_cmcfs = self._validate_tensor(mol_feat_for_cmcfs, "moltoken_feat")
            has_moltoken = True
            moltoken_tokens = self.moltoken_adapter(mol_feat_for_cmcfs)
            moltoken_tokens = self._validate_tensor(moltoken_tokens, "moltoken_tokens")

        if not has_moltoken or moltoken_tokens is None:
            has_moltoken = True
            moltoken_tokens = torch.zeros(
                batch_size, self.moltoken_adapter.num_tokens, self.llm_dim, device=device
            )

        # ========= 5. KG Adapter =========
        if self.training and self.use_feature_noise:
            kg_feat = self.kg_noise(kg_feat)
        kg_feat = self._validate_tensor(kg_feat, "kg_feat")
        kg_tokens = self.kg_adapter(kg_feat)
        kg_tokens = self._validate_tensor(kg_tokens, "kg_tokens")

        # ========= 6. 互补性监控 & 对比学习 (在 dropout 之前) =========
        struct_repr = struct_tokens.mean(dim=1)
        kg_repr = kg_tokens.mean(dim=1)
        moltoken_repr = moltoken_tokens.mean(dim=1)

        confidences = {
            "kg_is_completed": kg_mask.tolist(),
            "mol_is_completed": completion_flags[:, 2].tolist(),
            "kg_mask": kg_mask,
            "completion_flags": completion_flags,
            "dec_vacuity": dec_vacuity,
        }

        with torch.no_grad():
            struct_norm = F.normalize(struct_repr.float(), dim=-1)
            moltoken_norm_v = F.normalize(moltoken_repr.float(), dim=-1)
            cos_sim = (struct_norm * moltoken_norm_v).sum(dim=-1)
            confidences["moltoken_complementarity"] = (1.0 - cos_sim.abs()).mean()

        if self.training and hasattr(self, 'use_contrastive') and self.use_contrastive and hasattr(self, 'contrastive_loss'):
            # ECCL: pass vacuity unless ablation flag disables it
            _eccl_on = not self.config.get("network", self.config).get("_eccl_disabled", False)
            contrastive_l = self.contrastive_loss(
                struct_repr.float(), kg_repr.float(), moltoken_repr.float(),
                vacuity=dec_vacuity if _eccl_on else None,
                completion_flags=completion_flags if _eccl_on else None,
            )
            confidences["contrastive_loss"] = contrastive_l
            confidences["eccl_diag"] = getattr(self.contrastive_loss, '_last_diag', {})

        # ========= 6.5 CMON: Cross-modal orthogonal decomposition (before dropout) =========
        if self.use_mars and self.mars is not None:
            [struct_tokens, kg_tokens, moltoken_tokens], digest_tokens, mars_diag, mars_aux_dict, rel_samples = self.mars(
                [struct_tokens, kg_tokens, moltoken_tokens],
            )
            with torch.no_grad():
                real_mask = ~kg_mask.bool()
                if real_mask.any():
                    mars_diag['cmon_rel_k_real'] = rel_samples[real_mask, 1].mean().item()
                if (~real_mask).any():
                    mars_diag['cmon_rel_k_comp'] = rel_samples[~real_mask, 1].mean().item()
            confidences["mars_diag"] = mars_diag
            confidences["cmon_digests"] = digest_tokens
            if self.training:
                confidences["cmon_losses"] = mars_aux_dict

        # ========= 7. Sample-level Modality Dropout (~20% of samples lose one modality) =========
        if self.training:
            _B = struct_tokens.size(0)
            r = torch.rand(_B, device=device)
            mol_drop = (r < 0.08).unsqueeze(-1).unsqueeze(-1)
            kg_drop = ((r >= 0.08) & (r < 0.15)).unsqueeze(-1).unsqueeze(-1)
            str_drop = ((r >= 0.15) & (r < 0.20)).unsqueeze(-1).unsqueeze(-1)
            moltoken_tokens = moltoken_tokens * (~mol_drop).float()
            kg_tokens = kg_tokens * (~kg_drop).float()
            struct_tokens = struct_tokens * (~str_drop).float()

        # ========= 8. LLM 前向传播 =========
        logits1, pooled1, text_sources = self._forward_once(
            drug_batch, struct_tokens, kg_tokens, moltoken_tokens,
            smiles_list, confidences, device,
        )
        
        # 9. R-Drop
        logits2, pooled2 = None, None
        rdrop_loss = torch.tensor(0.0, device=device)
        feat_consistency_loss = torch.tensor(0.0, device=device)

        if self.training and self.use_rdrop:
            logits2, pooled2, _ = self._forward_once(
                drug_batch, struct_tokens, kg_tokens, moltoken_tokens,
                smiles_list, confidences, device,
            )
            rdrop_loss = self.rdrop_reg(logits1, logits2)
            feat_consistency_loss = self.feat_consistency_reg(pooled1, pooled2)

        # 10. 互补性损失
        complementarity_loss = torch.tensor(0.0, device=device)
        if self.training and self.use_moltoken and moltoken_feat is not None:
            complementarity_loss = self.complementarity_loss(struct_feat, moltoken_feat)

        # 11. 辅助损失
        aux_losses = {}
        if self.training:
            aux_losses["rdrop_loss"] = rdrop_loss.clamp(max=1.0)
            aux_losses["consistency_loss"] = feat_consistency_loss.clamp(max=0.5) * 0.1
            if self.use_moltoken:
                aux_losses["complementarity_loss"] = complementarity_loss
            if "contrastive_loss" in confidences:
                aux_losses["contrastive_loss"] = confidences["contrastive_loss"]
            if self.cmcfs is not None:
                aux_losses["cmcfs_recon_loss"] = cmcfs_recon_loss
            if "cmon_losses" in confidences:
                cmon_l = confidences["cmon_losses"]
                aux_losses["cmon_pred_loss"] = cmon_l.get("pred_loss", torch.tensor(0.0, device=device))
                aux_losses["cmon_ortho_loss"] = cmon_l.get("ortho_loss", torch.tensor(0.0, device=device))
                aux_losses["cmon_cal_loss"] = cmon_l.get("cal_loss", torch.tensor(0.0, device=device))

        sample_weights = torch.ones(batch_size, device=device)

        return {
            "classification": logits1.float(),
            "pred_proba": torch.sigmoid(logits1.float()),
            "aux_losses": aux_losses,
            "sample_weights": sample_weights,
            "text_sources": text_sources,
            "kg_is_completed": kg_mask,
            "cfs_diag": cfs_diag,
            "moltoken_complementarity": confidences.get("moltoken_complementarity", None),
            "smiles_list": smiles_list,
            "completion_flags": completion_flags,
            "dec_vacuity": dec_vacuity,
            "eccl_diag": confidences.get("eccl_diag", {}),
            "cmef_diag": confidences.get("cmef_diag", {}),
            "mars_diag": confidences.get("mars_diag", {}),
        }, None


# 保持向后兼容
DPModel.loss_weights = {"main": 1.0, "consistency": 0.1, "complementarity": 0.05}
