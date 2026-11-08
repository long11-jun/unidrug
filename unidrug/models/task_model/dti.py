"""Drug-target interaction model.

Drug side: GraphMVP, BMKG, TokenMol. Protein side: sequence CNN, BMKG, 3D properties.
SA-CMSA completes missing KG embeddings. CMON adds one digest token per modality.
Both sides are packed with text and passed through Qwen2.5-3B with LoRA.
"""

import json
import logging
import math
import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, TaskType, get_peft_model

from .. import SUPPORTED_MOL_ENCODER
from ..llm_encoder import SUPPORTED_LLM_ENCODER
from .dp import (
    SparseAttentionCMSA, CMONModule,
)

logger = logging.getLogger(__name__)


class ModalContrastiveLoss(nn.Module):
    """InfoNCE between modalities of the same sample."""
    def __init__(self, hidden_dim: int, proj_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        # 投影到对比空间
        self.struct_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.kg_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.mol_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        
    def forward(self, struct_tokens: torch.Tensor, kg_tokens: torch.Tensor, 
                mol_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算跨模态对比损失
        
        Args:
            struct_tokens: [B, num_tokens, dim]
            kg_tokens: [B, num_tokens, dim]
            mol_tokens: [B, num_tokens, dim]
        
        Returns:
            loss: 对比损失
        """
        # 全局池化
        struct_repr = struct_tokens.mean(dim=1)  # [B, dim]
        kg_repr = kg_tokens.mean(dim=1)
        mol_repr = mol_tokens.mean(dim=1)
        
        # 投影到对比空间
        struct_z = F.normalize(self.struct_proj(struct_repr), dim=-1)
        kg_z = F.normalize(self.kg_proj(kg_repr), dim=-1)
        mol_z = F.normalize(self.mol_proj(mol_repr), dim=-1)
        
        # InfoNCE Loss: struct-kg, struct-mol, kg-mol
        loss = 0.0
        
        # struct vs kg
        logits_sk = torch.matmul(struct_z, kg_z.T) / self.temperature
        labels = torch.arange(struct_z.size(0), device=struct_z.device)
        loss += F.cross_entropy(logits_sk, labels)
        loss += F.cross_entropy(logits_sk.T, labels)
        
        # struct vs mol
        logits_sm = torch.matmul(struct_z, mol_z.T) / self.temperature
        loss += F.cross_entropy(logits_sm, labels)
        loss += F.cross_entropy(logits_sm.T, labels)
        
        # kg vs mol
        logits_km = torch.matmul(kg_z, mol_z.T) / self.temperature
        loss += F.cross_entropy(logits_km, labels)
        loss += F.cross_entropy(logits_km.T, labels)
        
        return loss / 6.0  # 平均6个方向


# ============================================================
# DTI Prompt Composer（每个样本独立 Prompt）
# ============================================================
class DTIPromptComposer:
    """
    DTI 任务的文本提取与 Prompt 构建
    
    功能：
    1. 从 BMKG 加载药物/蛋白质文本
    2. 缺失时从 SMILES/序列生成描述
    3. 为每个样本构建独立的 Prompt（包含置信度信息）
    """
    
    def __init__(self, kg_path: str = None):
        self.drug_text_map = {}
        self.protein_text_map = {}
        
        if kg_path:
            self._load_bmkg_texts(kg_path)
    
    def _load_bmkg_texts(self, kg_path: str):
        """从 BMKG 加载药物和蛋白质文本"""
        # 药物文本
        drug_file = os.path.join(kg_path, "drug.json")
        if os.path.exists(drug_file):
            try:
                with open(drug_file, 'r') as f:
                    drugs = json.load(f)
                for drug_id, v in drugs.items():
                    if isinstance(v, dict):
                        text = v.get('text', '')
                        smiles = v.get('SMILES', '')
                        if text and text != 'text':
                            self.drug_text_map[drug_id] = text
                            if smiles:
                                self.drug_text_map[smiles] = text
                                # 规范化 SMILES
                                try:
                                    from rdkit import Chem
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        canon = Chem.MolToSmiles(mol, canonical=True)
                                        self.drug_text_map[canon] = text
                                except:
                                    pass
                logger.info(f"[DTIPromptComposer] Loaded {len(self.drug_text_map)} drug texts")
            except Exception as e:
                logger.warning(f"[DTIPromptComposer] Failed to load drug.json: {e}")
        
        # 蛋白质文本
        protein_file = os.path.join(kg_path, "protein.json")
        if os.path.exists(protein_file):
            try:
                with open(protein_file, 'r') as f:
                    proteins = json.load(f)
                for prot_id, v in proteins.items():
                    if isinstance(v, dict):
                        text = v.get('text', '')
                        seq = v.get('sequence', '')
                        if text and text != 'text':
                            self.protein_text_map[prot_id] = text
                            if seq:
                                self.protein_text_map[seq] = text
                                if len(seq) > 100:
                                    self.protein_text_map[seq[:100]] = text
                logger.info(f"[DTIPromptComposer] Loaded {len(self.protein_text_map)} protein texts")
            except Exception as e:
                logger.warning(f"[DTIPromptComposer] Failed to load protein.json: {e}")
    
    def get_drug_text(self, smiles: str, drug_id: str = None) -> str:
        """获取药物文本，缺失时从 SMILES 生成"""
        # 从缓存获取
        if drug_id and drug_id in self.drug_text_map:
            return self.drug_text_map[drug_id]
        if smiles and smiles in self.drug_text_map:
            return self.drug_text_map[smiles]
        
        # 尝试规范化 SMILES
        if smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canon = Chem.MolToSmiles(mol, canonical=True)
                    if canon in self.drug_text_map:
                        return self.drug_text_map[canon]
            except:
                pass
        
        # 从 SMILES 生成描述
        return self._generate_drug_text(smiles)
    
    def get_protein_text(self, sequence: str, prot_id: str = None) -> str:
        """获取蛋白质文本，缺失时从序列生成"""
        if prot_id and prot_id in self.protein_text_map:
            return self.protein_text_map[prot_id]
        if sequence and sequence in self.protein_text_map:
            return self.protein_text_map[sequence]
        if sequence and len(sequence) > 100 and sequence[:100] in self.protein_text_map:
            return self.protein_text_map[sequence[:100]]
        
        return self._generate_protein_text(sequence)
    
    def _generate_drug_text(self, smiles: str) -> str:
        """从 SMILES 生成药物描述"""
        if not smiles:
            return "A small molecule drug compound."
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Small molecule (SMILES: {smiles[:50]}...)"
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rings = Lipinski.RingCount(mol)
            
            desc = f"Small molecule (MW={mw:.0f}, LogP={logp:.1f}, {rings} rings, HBD={hbd}, HBA={hba})"
            
            # Lipinski
            violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            if violations == 0:
                desc += ", drug-like"
            return desc
        except:
            return f"Small molecule compound"
    
    def _generate_protein_text(self, sequence: str) -> str:
        """从序列生成蛋白质描述"""
        if not sequence:
            return "A protein target."
        
        length = len(sequence)
        # 氨基酸组成
        charged = sum(sequence.count(aa) for aa in 'RKDE')
        hydrophobic = sum(sequence.count(aa) for aa in 'AVLIMFW')
        
        desc = f"Protein target ({length} residues, {charged/length*100:.0f}% charged, {hydrophobic/length*100:.0f}% hydrophobic)"
        
        if length < 150:
            desc += ", short peptide"
        elif length < 400:
            desc += ", single-domain"
        else:
            desc += ", multi-domain"
        
        return desc
    
    def build_prompt(self, smiles: str, sequence: str,
                     drug_text: str = None, protein_text: str = None,
                     has_drug_kg: bool = True, has_moltoken: bool = True,
                     has_prot_kg: bool = True, has_prot3d: bool = True) -> str:
        """
        构建 DTI 动态提示词（与 DP 风格对齐，含模态置信度评估）
        
        LLM 通过文本提示词了解每个模态的可靠性，
        与 CMON Digest Token 的数值置信度形成互补。
        """
        if not drug_text:
            drug_text = self.get_drug_text(smiles)
        if not protein_text:
            protein_text = self.get_protein_text(sequence)
        
        drug_text = drug_text[:200] + "..." if len(drug_text) > 200 else drug_text
        protein_text = protein_text[:200] + "..." if len(protein_text) > 200 else protein_text
        
        parts = []
        
        # 1. 任务描述
        parts.append("Predict whether this drug interacts with the target protein.")
        
        # 2. 药物/蛋白质信息
        parts.append(f"Drug: {drug_text}")
        parts.append(f"Target: {protein_text}")
        
        # 3. 药物模态置信度
        parts.append("Drug input reliability:")
        parts.append("[Structure] computed from molecular graph — high confidence.")
        if has_moltoken:
            parts.append("[MolToken] derived from 3D torsion angles — high confidence.")
        else:
            parts.append("[MolToken] cross-modal completion — moderate confidence.")
        if has_drug_kg:
            parts.append("[KG] retrieved from knowledge graph — high confidence.")
        else:
            parts.append("[KG] cross-modal completion — moderate confidence.")
        
        # 4. 蛋白质模态置信度
        parts.append("Target input reliability:")
        parts.append("[Structure] computed from protein sequence features — high confidence.")
        if has_prot3d:
            parts.append("[Prot3D] physicochemical property encoding — high confidence.")
        else:
            parts.append("[Prot3D] cross-modal completion — moderate confidence.")
        if has_prot_kg:
            parts.append("[KG] retrieved from knowledge graph — high confidence.")
        else:
            parts.append("[KG] cross-modal completion — moderate confidence.")
        
        # 5. 推理指导
        drug_completed = (not has_drug_kg) or (not has_moltoken)
        prot_completed = (not has_prot_kg) or (not has_prot3d)
        if not drug_completed and not prot_completed:
            parts.append("All modalities are directly observed; integrate with equal weight.")
        else:
            inferred = []
            if not has_drug_kg:
                inferred.append("Drug-KG")
            if not has_moltoken:
                inferred.append("Drug-MolToken")
            if not has_prot_kg:
                inferred.append("Target-KG")
            if not has_prot3d:
                inferred.append("Target-Prot3D")
            inferred_str = ", ".join(inferred)
            parts.append(
                f"{inferred_str} were inferred, not observed. "
                "Rely primarily on observed modalities; use inferred features as supporting evidence."
            )
        
        # 6. 预测锚点
        parts.append("Prediction:")
        
        return " ".join(parts)


# ============================================================
# Cross Attention Block（与 DP 完全一致）
# ============================================================
class CrossAttentionBlock(nn.Module):
    """Cross Attention Block with gradient stability"""
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
        self.residual_scale = 0.5
    
    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm1(queries)
        k = v = self.norm2(context)
        if q.dtype != k.dtype:
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        out, _ = self.cross_attn(q, k, v)
        queries = queries + out * self.residual_scale
        queries = queries + self.ffn(self.norm3(queries)) * self.residual_scale
        return queries


# ============================================================
# Q-Former Style Adapter（与 DP 完全一致）
# ============================================================
class AdapterTokenGenerator(nn.Module):
    """
    Q-Former style Adapter with Learnable Confidence（与 DP 完全一致）
    
    核心设计：
    - Query tokens 通过 self-attention 学习模态内部关系
    - Cross-attention 将原始特征注入到 query tokens
    - 额外的 conf_query 学习模态置信度
    - 模态特定 embedding 区分不同模态的置信度
    """
    _instance_count = 0
    
    def __init__(self, input_dim: int, llm_embed_dim: int, num_tokens: int = 32, 
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = llm_embed_dim
        
        # 模态标识
        self.modality_id = AdapterTokenGenerator._instance_count
        AdapterTokenGenerator._instance_count += 1
        
        # Query tokens恢复使用较小的初始化（与0.9377成功版本一致）
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, llm_embed_dim) * 0.02)
        
        # 置信度 query token
        self.conf_query = nn.Parameter(torch.randn(1, 1, llm_embed_dim) * 0.02)
        
        # 模态特定 embedding
        self.modality_embedding = nn.Parameter(torch.randn(1, llm_embed_dim) * 0.02)
        
        # 输入投影
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, llm_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(llm_embed_dim),
        )
        
        # Transformer 层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.TransformerEncoderLayer(
                    d_model=llm_embed_dim, nhead=num_heads, 
                    dim_feedforward=llm_embed_dim * 4,
                    batch_first=True, dropout=dropout, 
                    activation="gelu", norm_first=True,
                ),
                'cross_attn': CrossAttentionBlock(llm_embed_dim, num_heads, dropout),
            }))
        
        self.output_norm = nn.LayerNorm(llm_embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(llm_embed_dim),
            nn.Linear(llm_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        nn.init.xavier_uniform_(self.confidence_head[4].weight, gain=0.1)
        nn.init.zeros_(self.confidence_head[4].bias)
        
        # 置信度 EMA
        self.register_buffer('running_confidence', torch.tensor(0.5))
        self.conf_momentum = 0.1
        
    def forward(self, features: torch.Tensor, return_confidence: bool = False):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        param_dtype = self.input_norm.weight.dtype
        if features.dtype != param_dtype:
            features = features.to(param_dtype)
        
        # 与 DP 一致不做额外裁剪
        features = self.input_proj(self.input_norm(features))
        batch = features.shape[0]
        
        # 拼接 query_tokens 和 conf_query
        content_queries = self.query_tokens.expand(batch, -1, -1)
        conf_query = self.conf_query.expand(batch, -1, -1)
        queries = torch.cat([content_queries, conf_query], dim=1)
        
        # Transformer 层
        for layer in self.layers:
            queries = layer['self_attn'](queries)
            queries = layer['cross_attn'](queries, features)
        
        # 分离
        content_tokens = queries[:, :self.num_tokens, :]
        conf_token = queries[:, self.num_tokens:, :]
        
        # 与 DP 一致输出归一化，不做额外裁剪
        tokens = self.dropout(self.output_norm(content_tokens))
        
        if return_confidence:
            conf_input = conf_token.squeeze(1) + self.modality_embedding
            conf_logit = self.confidence_head(conf_input).squeeze(-1)
            # 与 DP 完全一致直接 sigmoid，通过损失函数软约束
            instant_confidence = torch.sigmoid(conf_logit)
            
            if self.training:
                with torch.no_grad():
                    batch_mean = instant_confidence.mean()
                    self.running_confidence = (1 - self.conf_momentum) * self.running_confidence + self.conf_momentum * batch_mean
                confidence = instant_confidence
            else:
                # 与 DP 完全一致70% 瞬时 + 30% 历史
                confidence = 0.7 * instant_confidence + 0.3 * self.running_confidence
            
            return tokens, confidence
        else:
            return tokens


# ============================================================
# Attention Pooling（与 DP 完全一致）
# ============================================================
class AttentionPooling(nn.Module):
    """可学习的注意力池化"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 恢复使用较小的初始化（与成功版本一致）
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = 0.5
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states_float = hidden_states.float()
        query = self.query.expand(batch_size, -1, -1).float()
        
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        pooled, _ = self.attention(query, hidden_states_float, hidden_states_float, key_padding_mask=key_padding_mask)
        pooled = self.layer_norm(self.dropout(pooled) * self.residual_scale + query)
        return pooled.squeeze(1)


# ============================================================
# Residual Prediction Head
# ============================================================
class ResidualPredictionHead(nn.Module):
    """带残差连接的预测头"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.res_proj1 = nn.Linear(input_dim, hidden_dim, bias=False)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.drop2 = nn.Dropout(dropout)
        self.res_proj2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)
        self.res_scale = 0.3
        self._init_weights()
        
    def _init_weights(self):
        # SOTA突破增大 fc_out 的 gain，让 logits 有更大的初始范围
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
        # 关键修复fc_out 用更大的 gain，避免初始 logits 全负
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)
        nn.init.zeros_(self.fc_out.bias)
        for m in [self.res_proj1, self.res_proj2]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 DP 完全一致先 dropout 再加残差
        h1 = self.drop1(F.gelu(self.fc1(self.ln1(x))))
        h1 = h1 + self.res_scale * self.res_proj1(x)  # 残差连接
        
        h2 = self.drop2(F.gelu(self.fc2(self.ln2(h1))))
        h2 = h2 + self.res_scale * self.res_proj2(h1)  # 残差连接
        
        return self.fc_out(h2)


# ============================================================
# Deep MCNN
# ============================================================
class DeepMCNN(nn.Module):
    """
    深度多尺度卷积网络（优化版）
    
    - 5个 kernel sizes: [3, 5, 7, 9, 11]
    - 3层卷积深度（每个分支）
    - 残差连接
    - 注意力池化
    """
    def __init__(self, config):
        super().__init__()
        self.output_dim = config.get("output_size", 256)
        vocab_size = config.get("vocab_size", 26)
        embed_dim = config.get("embedding_num", 128)
        hidden_dim = config.get("hidden_size", 128)
        dropout = config.get("dropout", 0.1)
        
        # 氨基酸嵌入
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 5个多尺度卷积分支（3层深度）
        kernel_sizes = [3, 5, 7, 9, 11]
        self.conv_branches = nn.ModuleList()
        
        for ks in kernel_sizes:
            branch = nn.Sequential(
                # Layer 1
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Layer 2
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Layer 3
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv_branches.append(branch)
        
        # 残差投影（embed_dim -> hidden_dim）
        self.residual_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        
        concat_dim = hidden_dim * len(kernel_sizes)
        
        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 嵌入
        x = self.embed(x)  # [B, L, D]
        x = x.permute(0, 2, 1)  # [B, D, L]
        
        # 残差
        residual = self.residual_proj(x)  # [B, hidden_dim, L]
        
        # 多尺度并行
        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(x) + residual  # 残差连接
            branch_outputs.append(out)
        
        # 拼接
        multi_scale = torch.cat(branch_outputs, dim=1)  # [B, concat_dim, L]
        multi_scale = multi_scale.permute(0, 2, 1)  # [B, L, concat_dim]
        
        # 注意力池化
        attn = F.softmax(self.attention(multi_scale), dim=1)
        pooled = (multi_scale * attn).sum(dim=1)  # [B, concat_dim]
        
        return self.output_proj(pooled)
    
    def encode_protein(self, x):
        return self.forward(x)


# ============================================================
# Protein Physicochemical Property Encoder（蛋白质理化性质编码器）
# 替代 FoldSeek 3Di 近似映射（无需联网/3D结构预测）
# ============================================================
class ProteinPropertyEncoder(nn.Module):
    """
    蛋白质理化性质编码器 — 基于氨基酸理化性质的结构相关特征
    
    替代 FoldSeek 3Di 近似映射的理由
    近似 3Di 只是氨基酸 → 3Di 字符的一对一映射，与 DeepMCNN 完全冗余。
    本编码器从序列中提取 9 维理化性质向量，与 3D 结构高度相关。
    
    9 维理化性质：
    1. 疏水性 (Kyte-Doolittle) → buried/exposed
    2. 电荷 (pH 7) → 表面静电
    3. 极性 → 氢键网络
    4. 体积 (van der Waals) → packing 密度
    5. 柔性 (B-factor) → 结构动力学
    6. 螺旋倾向 (Chou-Fasman) → α-helix
    7. 折叠倾向 (Chou-Fasman) → β-sheet
    8. 转角倾向 (Chou-Fasman) → turn/coil
    9. 埋藏倾向 (Janin) → 溶剂可及性
    
    架构与 DeepMCNN 对称：多尺度 CNN + 残差 + 注意力池化
    """
    
    AA_PROPERTIES = {
        'A': [0.700, 0.500, 0.000, 0.267, 0.360, 0.740, 0.300, 0.350, 0.380],
        'R': [0.000, 1.000, 1.000, 0.733, 0.530, 0.650, 0.330, 0.470, 0.010],
        'N': [0.111, 0.500, 1.000, 0.467, 0.460, 0.520, 0.350, 0.560, 0.120],
        'D': [0.111, 0.000, 1.000, 0.433, 0.510, 0.500, 0.300, 0.590, 0.150],
        'C': [0.778, 0.500, 0.000, 0.400, 0.350, 0.440, 0.380, 0.480, 0.500],
        'Q': [0.111, 0.500, 1.000, 0.567, 0.490, 0.670, 0.320, 0.480, 0.070],
        'E': [0.111, 0.000, 1.000, 0.567, 0.500, 0.760, 0.260, 0.430, 0.180],
        'G': [0.456, 0.500, 0.000, 0.000, 0.540, 0.360, 0.370, 0.640, 0.360],
        'H': [0.144, 0.700, 1.000, 0.600, 0.320, 0.560, 0.340, 0.480, 0.170],
        'I': [1.000, 0.500, 0.000, 0.533, 0.460, 0.580, 0.490, 0.330, 0.600],
        'L': [0.922, 0.500, 0.000, 0.533, 0.400, 0.740, 0.320, 0.320, 0.450],
        'K': [0.067, 1.000, 1.000, 0.567, 0.470, 0.630, 0.300, 0.500, 0.030],
        'M': [0.711, 0.500, 0.000, 0.600, 0.300, 0.700, 0.320, 0.380, 0.400],
        'F': [0.811, 0.500, 0.000, 0.733, 0.310, 0.580, 0.450, 0.360, 0.500],
        'P': [0.322, 0.500, 0.000, 0.367, 0.510, 0.340, 0.280, 0.760, 0.180],
        'S': [0.411, 0.500, 1.000, 0.267, 0.510, 0.500, 0.360, 0.520, 0.220],
        'T': [0.422, 0.500, 1.000, 0.367, 0.440, 0.500, 0.400, 0.470, 0.230],
        'W': [0.400, 0.500, 0.000, 1.000, 0.310, 0.610, 0.440, 0.370, 0.410],
        'Y': [0.356, 0.500, 1.000, 0.800, 0.420, 0.410, 0.520, 0.410, 0.150],
        'V': [0.967, 0.500, 0.000, 0.433, 0.390, 0.550, 0.520, 0.330, 0.540],
    }
    N_PROPS = 9
    
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        hidden_dim = config.get("hidden_dim", 128)
        self.output_dim = config.get("output_dim", 256)
        self.max_length = config.get("max_length", 1500)
        dropout = config.get("dropout", 0.1)
        
        prop_table = torch.zeros(21, self.N_PROPS)
        aa_order = 'ARNDCQEGHILKMFPSTWYV'
        self._aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
        for i, aa in enumerate(aa_order):
            if aa in self.AA_PROPERTIES:
                prop_table[i] = torch.tensor(self.AA_PROPERTIES[aa])
        prop_table[20] = prop_table[:20].mean(dim=0)
        self.register_buffer('prop_table', prop_table)
        
        self.input_proj = nn.Sequential(
            nn.Linear(self.N_PROPS, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        
        kernel_sizes = [3, 5, 7, 9, 11]
        self.conv_branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.conv_branches.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, ks, padding=ks // 2),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, ks, padding=ks // 2),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, ks, padding=ks // 2),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            ))
        
        self.residual_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        concat_dim = hidden_dim * len(kernel_sizes)
        
        self.attention = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, self.output_dim),
            nn.LayerNorm(self.output_dim), nn.GELU(), nn.Dropout(dropout)
        )
        logger.info(f"[ProteinProperty] {self.N_PROPS}D→hidden={hidden_dim}→out={self.output_dim}")
    
    def _seq_to_props(self, sequences, device):
        batch_props = []
        max_len = max((min(len(s), self.max_length) for s in sequences), default=1)
        max_len = max(max_len, 1)
        for seq in sequences:
            seq = seq[:self.max_length]
            indices = [self._aa_to_idx.get(aa.upper(), 20) for aa in seq] or [20]
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            props = self.prop_table[idx_t]
            if props.size(0) < max_len:
                props = torch.cat([props, torch.zeros(max_len - props.size(0), self.N_PROPS, device=device)])
            batch_props.append(props)
        return torch.stack(batch_props)
    
    def forward(self, sequences, device=None):
        if device is None:
            device = self.prop_table.device
        x = self._seq_to_props(sequences, device)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        residual = self.residual_proj(x)
        branch_outputs = [branch(x) + residual for branch in self.conv_branches]
        multi_scale = torch.cat(branch_outputs, dim=1).permute(0, 2, 1)
        attn = F.softmax(self.attention(multi_scale), dim=1)
        pooled = (multi_scale * attn).sum(dim=1)
        return self.output_proj(pooled)


# ============================================================
# FoldSeek 3Di Encoder（需要真实3D结构，保留供未来使用）
# ============================================================
class FoldSeek3DiEncoder(nn.Module):
    """
    FoldSeek 3Di 结构字母表编码器
    
    将蛋白质 3D 结构的 FoldSeek 3Di 序列（20 字母 + padding）编码为特征向量。
    架构与 DeepMCNN 对称：多尺度 CNN + 注意力池化。
    
    FoldSeek 3Di 字母表：
    - 20 个状态描述残基与最近邻残基的几何关系
    - 字母表: A C D E F G H I K L M N P Q R S T V W Y (大写)
    - 加上 padding (#) 和 unknown (X) 共 22 个
    
    与 MolToken 的对称性：
    - MolToken: 药物 3D 扭转角 → 离散 token 序列 → embedding + encoder
    - FoldSeek:  蛋白质 3D 结构 → 3Di 字母序列 → embedding + encoder
    """
    # 3Di 字母表（FoldSeek 标准 20 + padding + unknown）
    ALPHABET = "#ACDEFGHIKLMNPQRSTVWYX"  # 22 个字符
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        
        vocab_size = len(self.ALPHABET)  # 22
        embed_dim = config.get("embedding_dim", 128)
        hidden_dim = config.get("hidden_dim", 128)
        self.output_dim = config.get("output_dim", 256)
        self.max_length = config.get("max_length", 1500)
        dropout = config.get("dropout", 0.1)
        
        # 构建字符 → 索引映射
        self.char2idx = {c: i for i, c in enumerate(self.ALPHABET)}
        
        # 3Di 字母嵌入
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多尺度卷积分支（与 DeepMCNN 对称）
        kernel_sizes = [3, 5, 7, 9, 11]
        self.conv_branches = nn.ModuleList()
        
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv_branches.append(branch)
        
        # 残差投影
        self.residual_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        
        concat_dim = hidden_dim * len(kernel_sizes)
        
        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        logger.info(
            f"[FoldSeek3Di] Encoder initialized: vocab={vocab_size}, "
            f"embed={embed_dim}, hidden={hidden_dim}, output={self.output_dim}, "
            f"max_len={self.max_length}, kernels={kernel_sizes}"
        )
    
    def tokenize(self, seq_3di: str) -> torch.LongTensor:
        """将 3Di 字符串转为 token indices"""
        seq_3di = seq_3di[:self.max_length].upper()
        indices = [self.char2idx.get(c, self.char2idx['X']) for c in seq_3di]
        return torch.tensor(indices, dtype=torch.long)
    
    def tokenize_batch(self, seqs: list, device=None) -> torch.LongTensor:
        """批量 tokenize 并 padding"""
        tokens_list = [self.tokenize(s) for s in seqs]
        max_len = max(len(t) for t in tokens_list)
        padded = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
        for i, t in enumerate(tokens_list):
            padded[i, :len(t)] = t
        if device is not None:
            padded = padded.to(device)
        return padded
    
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) token indices
        Returns:
            (B, output_dim) 特征向量
        """
        # 嵌入
        h = self.embed(x)  # (B, L, embed_dim)
        h = h.permute(0, 2, 1)  # (B, embed_dim, L)
        
        # 残差
        residual = self.residual_proj(h)  # (B, hidden_dim, L)
        
        # 多尺度并行
        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(h) + residual
            branch_outputs.append(out)
        
        # 拼接
        multi_scale = torch.cat(branch_outputs, dim=1)  # (B, concat_dim, L)
        multi_scale = multi_scale.permute(0, 2, 1)  # (B, L, concat_dim)
        
        # 注意力池化
        attn = F.softmax(self.attention(multi_scale), dim=1)
        pooled = (multi_scale * attn).sum(dim=1)  # (B, concat_dim)
        
        return self.output_proj(pooled)
    
    def encode_3di(self, x: torch.LongTensor) -> torch.Tensor:
        """别名，方便调用"""
        return self.forward(x)


# ============================================================
# DTIModel 主模型
# ============================================================
class DTIModel(nn.Module):
    """
    DTI 预测模型 - SA-CMSA + CMON + LLM 推理
    
    独立模态流架构（Drug 3模态 + Protein 3模态 = 6 独立通道）：
    Drug:    1. Struct(GraphMVP)→40tok  2. KG(BMKG)→28tok  3. MolToken(3D)→32tok
    Protein: 4. Struct(DeepMCNN)→40tok  5. KG(BMKG)→28tok  6. Prot3D(Property/FoldSeek)→32tok
    Text:    Drug_Text + Prot_Text → Prompt
    SA-CMSA and CMON each have one drug instance and one protein instance.
    """
    
    def __init__(self, config: Dict[str, Any], pred_dim: int = 1, kge_bank: torch.Tensor = None):
        super().__init__()
        self.config = config
        self.pred_dim = pred_dim
        
        net_cfg = config.get("network", config)
        
        # ========== 1. Prompt Composer ==========
        kg_path = config.get("kg_path", net_cfg.get(
            "kg_path", os.path.join(os.environ.get("UNIDRUG_ASSET_ROOT", "/data-nxs/songlongjun/KEDD_temporal-master"), "assets/kg/BMKG")))
        self.prompt_composer = DTIPromptComposer(kg_path)
        
        # ========== 2. Encoders ==========
        self._init_encoders(config)
        
        # ========== 3. Adapters (6 independent) ==========
        self._init_adapters(config)
        
        # ========== 4. LLM ==========
        self._init_llm(config)
        
        # ========== 5. CMSA ==========
        self._init_sacmsa(config, kge_bank)
        
        # ========== 6. CMON (Cross-Modal Orthogonal Decomposition) ==========
        self._init_cmon(config)
        
        # ========== 7. Modality Markers ==========
        self._init_modality_markers()
        
        # ========== 8. Prediction Heads ==========
        self._init_heads(config, pred_dim)
        
        # ========== 9. LoRA ==========
        if config.get("use_lora", True):
            self._enable_lora(config)
        
        # ========== 10. Stability Config ==========
        stability_cfg = net_cfg.get("stability", {})
        self.logits_clamp_abs = stability_cfg.get("logits_clamp_abs", 10.0)
        
        # ========== 10.5. Feature Noise ==========
        self.use_feature_noise = net_cfg.get("use_feature_noise", False)
        self.feature_noise_std = net_cfg.get("feature_noise_std", 0.02)
        if self.use_feature_noise:
            logger.info(f"[Feature Noise] Enabled: std={self.feature_noise_std}")
        
        # ========== 10.6. KGE Dropout (strict cold start 训练必需) ==========
        # 训练时随机将部分实体的 KGE 置零并标记为 missing，
        # 迫使 SA-CMSA 补全路径被训练，且 LLM 也能见到补全的 KGE
        self.kge_dropout_rate = net_cfg.get("kge_dropout_rate", 0.0)
        if self.kge_dropout_rate > 0:
            logger.info(f"[KGE Dropout] Enabled: rate={self.kge_dropout_rate}")
        
        # ========== 11. Freeze ==========
        self._freeze_pretrained()
        
        self._log_model_info()
    
    def _init_encoders(self, config):
        """初始化编码器"""
        drug_cfg = config.get("drug", config.get("network", {}).get("drug", {}))
        prot_cfg = config.get("protein", config.get("network", {}).get("protein", {}))
        
        # Drug Structure (GraphMVP)
        struct_cfg = drug_cfg.get("structure", {})
        if "config_path" in struct_cfg:
            with open(struct_cfg["config_path"], "r") as f:
                from ...utils.paths import expand_asset_paths
                enc_cfg = expand_asset_paths(json.load(f))
            self.drug_structure_encoder = SUPPORTED_MOL_ENCODER[struct_cfg["name"]](enc_cfg)
            if "init_checkpoint" in enc_cfg:
                ckpt = torch.load(enc_cfg["init_checkpoint"], map_location="cpu", weights_only=False)
                self.drug_structure_encoder.load_state_dict(ckpt, strict=False)
        else:
            self.drug_structure_encoder = SUPPORTED_MOL_ENCODER[struct_cfg.get("name", "gin")](struct_cfg)
        self.drug_struct_dim = self.drug_structure_encoder.output_dim
        
        # Protein Structure (DeepMCNN)
        mcnn_cfg = {
            "vocab_size": 26,
            "embedding_num": 128,
            "hidden_size": 128,
            "output_size": 256,
            "dropout": 0.1
        }
        self.protein_structure_encoder = DeepMCNN(mcnn_cfg)
        self.prot_struct_dim = self.protein_structure_encoder.output_dim
        
        # KG dims
        self.drug_kg_dim = drug_cfg.get("kg", {}).get("embedding_dim", 256)
        self.prot_kg_dim = prot_cfg.get("kg", {}).get("embedding_dim", 256)
        
        # MolToken (Drug 3D)
        net_cfg = config.get("network", config)
        self.use_moltoken = net_cfg.get("use_moltoken", False)
        self.moltoken_encoder = None
        self.moltoken_dim = 768
        
        if self.use_moltoken:
            try:
                from ..llm_encoder import TokenMolEncoder
                self.moltoken_encoder = TokenMolEncoder(net_cfg.get("moltoken", {}))
                self.moltoken_dim = self.moltoken_encoder.output_dim
                for p in self.moltoken_encoder.model.parameters():
                    p.requires_grad = False
                self.moltoken_encoder.model.eval()
                logger.info(f"[MolToken] Enabled: dim={self.moltoken_dim}")
            except Exception as e:
                logger.warning(f"[MolToken] Init failed: {e}")
                self.use_moltoken = False
        
        # Protein 3D 模态编码器
        # 优先使用 ProteinPropertyEncoder（理化性质，无需联网）
        # 如果有真实 FoldSeek 3Di 数据，可切换到 FoldSeek3DiEncoder
        self.use_foldseek = net_cfg.get("use_foldseek", False)
        self.prot_3d_mode = net_cfg.get("prot_3d_mode", "property")  # "property" or "foldseek"
        self.prot_3d_encoder = None
        self.foldseek_dim = 256  # 保持兼容字段名
        
        if self.prot_3d_mode == "foldseek" and self.use_foldseek:
            # 真实 FoldSeek 3Di（需要预计算的 3Di 缓存）
            foldseek_cfg = net_cfg.get("foldseek", {})
            self.prot_3d_encoder = FoldSeek3DiEncoder(foldseek_cfg)
            self.foldseek_dim = self.prot_3d_encoder.output_dim
            logger.info(f"[Prot3D] FoldSeek3Di mode: dim={self.foldseek_dim}")
        else:
            # 默认：理化性质编码器（无需联网、无需3D结构）
            prop_cfg = net_cfg.get("foldseek", {})  # 复用 foldseek 配置的 dim 参数
            self.prot_3d_encoder = ProteinPropertyEncoder(prop_cfg)
            self.foldseek_dim = self.prot_3d_encoder.output_dim
            self.prot_3d_mode = "property"  # 确保设置
            logger.info(f"[Prot3D] PhysicochemicalProperty mode: 9D→dim={self.foldseek_dim}")
        
        logger.info(f"[Encoders] Drug struct={self.drug_struct_dim}, Prot struct={self.prot_struct_dim}")
        logger.info(f"[Encoders] MolToken={'ON' if self.use_moltoken else 'OFF'}, Prot3D={self.prot_3d_mode}")
    
    def _init_transformer_weights(self, transformer: nn.Module, scale: float = 1.0):
        """重新初始化 TransformerEncoder（使用PyTorch默认初始化）"""
        # 修复不做特殊初始化，保持PyTorch默认（xavier_uniform, gain=1.0）
        # 之前 scale=1.5 可能导致初期特征不平衡
        pass
    
    def _init_sacmsa(self, config, kge_bank):
        """初始化 SA-CMSA: Sparse-Attention Cross-Modal Subspace Alignment
        
        Drug 侧: [Structure(300d), KG(256d), MolToken(768d)] + kge_bank → 补全缺失 KG
        Protein 侧: [Structure(256d), KG(256d), Prot3D(256d)] + kge_bank → 补全缺失 KG
        """
        net_cfg = config.get("network", config)
        cmsa_cfg = net_cfg.get("cmsa", net_cfg.get("dec", {}))
        use_cmsa = cmsa_cfg.get("enabled", True)
        
        self.drug_cmsa = None
        self.prot_cmsa = None
        self.use_cmsa = False
        
        if use_cmsa and kge_bank is not None:
            self.use_cmsa = True
            self.drug_cmsa = SparseAttentionCMSA(
                modal_dims=[self.drug_struct_dim, self.drug_kg_dim, self.moltoken_dim],
                kge_bank=kge_bank,
                sub_dim=cmsa_cfg.get("sub_dim", 128),
                topk=cmsa_cfg.get("topk", 16),
                num_heads=cmsa_cfg.get("num_heads", 2),
                dropout=cmsa_cfg.get("dropout", 0.1),
                align_lambda=cmsa_cfg.get("align_lambda", 0.05),
                retrieve_lambda=cmsa_cfg.get("retrieve_lambda", 0.1),
                enable_mol_completion=cmsa_cfg.get("enable_mol_completion", False),
            )
            self.prot_cmsa = SparseAttentionCMSA(
                modal_dims=[self.prot_struct_dim, self.prot_kg_dim, self.foldseek_dim],
                kge_bank=kge_bank,
                sub_dim=cmsa_cfg.get("sub_dim", 128),
                topk=cmsa_cfg.get("topk", 16),
                num_heads=cmsa_cfg.get("num_heads", 2),
                dropout=cmsa_cfg.get("dropout", 0.1),
                align_lambda=cmsa_cfg.get("align_lambda", 0.05),
                retrieve_lambda=cmsa_cfg.get("retrieve_lambda", 0.1),
                enable_mol_completion=False,
            )
            logger.info(
                f"[SA-CMSA] Drug: struct({self.drug_struct_dim})+mol({self.moltoken_dim})→kg({self.drug_kg_dim})")
            logger.info(
                f"[SA-CMSA] Prot: struct({self.prot_struct_dim})+3d({self.foldseek_dim})→kg({self.prot_kg_dim})")
            logger.info(
                f"[SA-CMSA] Shared kge_bank: {kge_bank.shape[0]}×{kge_bank.shape[1]}")
        else:
            if kge_bank is None:
                logger.warning("[SA-CMSA] Disabled: kge_bank is None")
            else:
                logger.info("[SA-CMSA] Disabled by config")
    
    def _init_adapters(self, config):
        """初始化 6 个独立 Adapter（Drug×3 + Protein×3）
        
        Token 预算（总计 200 modal + 6 digest = 206）：
            Drug:    struct 40 + KG 28 + MolToken 32 = 100
            Protein: struct 40 + KG 28 + Prot3D 32 = 100
        """
        net_cfg = config.get("network", config)
        adapter_dropout = net_cfg.get("adapter_dropout", 0.25)
        adapter_layers = 4
        
        # LLM embed dim 暂用 2048（Qwen-3B），_init_llm 后会更新
        self.llm_embed_dim = 2048
        
        # ===== Drug 侧 3 个独立 Adapter =====
        # Drug Structure: GraphMVP → 40 tokens
        self.drug_struct_adapter = AdapterTokenGenerator(
            input_dim=self.drug_struct_dim,
            llm_embed_dim=self.llm_embed_dim,
            num_tokens=40,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )
        # Drug KG: BMKG → 28 tokens
        self.drug_kg_adapter = AdapterTokenGenerator(
            input_dim=self.drug_kg_dim,
            llm_embed_dim=self.llm_embed_dim,
            num_tokens=28,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )
        # Drug MolToken: 3D torsion → 32 tokens
        self.drug_mol_adapter = AdapterTokenGenerator(
            input_dim=self.moltoken_dim if self.use_moltoken else self.drug_struct_dim,
            llm_embed_dim=self.llm_embed_dim,
            num_tokens=32,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )
        
        # ===== Protein 侧 3 个独立 Adapter =====
        # Protein Structure: DeepMCNN → 40 tokens
        self.prot_struct_adapter = AdapterTokenGenerator(
            input_dim=self.prot_struct_dim,
            llm_embed_dim=self.llm_embed_dim,
            num_tokens=40,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )
        # Protein KG: BMKG → 28 tokens
        self.prot_kg_adapter = AdapterTokenGenerator(
            input_dim=self.prot_kg_dim,
            llm_embed_dim=self.llm_embed_dim,
            num_tokens=28,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )
        # Protein 3D (ProteinProperty / FoldSeek): → 32 tokens
        self.prot_3d_adapter = AdapterTokenGenerator(
            input_dim=self.foldseek_dim,  # = prot_3d_encoder.output_dim
            llm_embed_dim=self.llm_embed_dim,
            num_tokens=32,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )
        
        logger.info(f"[Adapters] 6 independent streams:")
        logger.info(f"  Drug:    struct({self.drug_struct_dim})→40 + KG({self.drug_kg_dim})→28 + MolToken→32 = 100 tokens")
        logger.info(f"  Protein: struct({self.prot_struct_dim})→40 + KG({self.prot_kg_dim})→28 + Prot3D({self.prot_3d_mode})→32 = 100 tokens")
        logger.info(f"  Total: 200 modal tokens + 6 digest tokens (CMON)")
    
    def _init_llm(self, config):
        """初始化 LLM"""
        net_cfg = config.get("network", config)
        llm_cfg = net_cfg.get("llm", {})
        
        self.llm_encoder = SUPPORTED_LLM_ENCODER[llm_cfg.get("name", "qwen")](llm_cfg)
        self.llm_embed_dim = self.llm_encoder.model.config.hidden_size
        
        # 更新 Adapter 的 output_dim（如果 LLM dim 不是 2048）
        # 这里假设已经正确初始化
        
        # Pooling (与 DP 完全一致)
        net_cfg = config.get("network", config)
        self.use_attention_pooling = net_cfg.get("use_attention_pooling", True)
        self.attention_pooler = AttentionPooling(
            hidden_dim=self.llm_embed_dim,
            num_heads=8,
            dropout=net_cfg.get("pooling_dropout", 0.15),
        )
        self.attn_pool_ratio = nn.Parameter(torch.tensor(0.85))

    def _init_cmon(self, config):
        """初始化 CMON: Cross-Modal Orthogonal Decomposition Network
        
        Drug CMON: 3 模态 (struct, kg, moltoken) 正交分解 + digest tokens
        Prot CMON: 3 模态 (struct, kg, prot3d) 正交分解 + digest tokens
        """
        net_cfg = config.get("network", config)
        mars_cfg = net_cfg.get("mars", {})
        self.use_cmon = mars_cfg.get("enabled", False)
        
        self.drug_cmon = None
        self.prot_cmon = None
        
        if self.use_cmon:
            sub_dim = mars_cfg.get("sub_dim", 64)
            self.drug_cmon = CMONModule(
                llm_dim=self.llm_embed_dim,
                n_modalities=3,
                sub_dim=sub_dim,
            )
            self.prot_cmon = CMONModule(
                llm_dim=self.llm_embed_dim,
                n_modalities=3,
                sub_dim=sub_dim,
            )
            logger.info(f"[CMON] Drug + Prot dual CMON enabled: sub_dim={sub_dim}")
        else:
            logger.info(f"[CMON] Disabled")
    
    def _init_modality_markers(self):
        """Drug and protein modality boundary tokens."""
        self.modality_markers = {
            # Drug 侧
            "d_struct_start": "[d_struct_start]",
            "d_struct_end": "[d_struct_end]",
            "d_kg_start": "[d_kg_start]",
            "d_kg_end": "[d_kg_end]",
            "d_mol_start": "[d_mol_start]",
            "d_mol_end": "[d_mol_end]",
            # Protein 侧
            "p_struct_start": "[p_struct_start]",
            "p_struct_end": "[p_struct_end]",
            "p_kg_start": "[p_kg_start]",
            "p_kg_end": "[p_kg_end]",
            "p_3d_start": "[p_3d_start]",
            "p_3d_end": "[p_3d_end]",
        }
        
        # 可学习残差（12 个标记各自独立）
        for key in self.modality_markers:
            residual_name = key + "_residual"
            setattr(self, residual_name, nn.Parameter(torch.randn(1, 1, self.llm_embed_dim) * 0.02))
    
    def _get_marker_embedding(self, marker_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        """获取模态标记嵌入（与 DP 完全一致：文本嵌入 + 可学习残差）"""
        tokenizer = self.llm_encoder.tokenizer
        embed_layer = self.llm_encoder.model.get_input_embeddings()
        
        # 获取文本标记
        marker_text = self.modality_markers.get(marker_name, "[UNK]")
        marker_ids = tokenizer(marker_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        # 获取文本嵌入
        text_embed = embed_layer(marker_ids)  # (1, seq_len, dim)
        # 平均池化成单个 token
        text_embed = text_embed.mean(dim=1, keepdim=True)  # (1, 1, dim)
        
        # 获取对应的可学习残差
        residual_name = marker_name + "_residual"
        if hasattr(self, residual_name):
            residual = getattr(self, residual_name)
            text_embed = text_embed + residual.to(text_embed.dtype)
        
        # 扩展到 batch
        return text_embed.expand(batch_size, -1, -1)
    
    def _init_heads(self, config, pred_dim):
        """初始化预测头"""
        self.classification_head = ResidualPredictionHead(
            self.llm_embed_dim, 1024, pred_dim, dropout=0.1
        )
        self.regression_head = ResidualPredictionHead(
            self.llm_embed_dim, 1024, 1, dropout=0.1
        )
        self.final_confidence_head = nn.Sequential(
            nn.Linear(self.llm_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 对比损失模块（Drug 侧 + Protein 侧独立）
        self.drug_contrastive_loss = ModalContrastiveLoss(
            hidden_dim=self.llm_embed_dim, proj_dim=256, temperature=0.15
        )
        self.prot_contrastive_loss = ModalContrastiveLoss(
            hidden_dim=self.llm_embed_dim, proj_dim=256, temperature=0.15
        )
    
    def _enable_lora(self, cfg):
        """启用 LoRA"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.llm_encoder.model = get_peft_model(self.llm_encoder.model, lora_config)
        self.llm_encoder.model.enable_input_require_grads()
        self.llm_encoder.model.gradient_checkpointing_enable()
        
        trainable = sum(p.numel() for p in self.llm_encoder.model.parameters() if p.requires_grad)
        logger.info(f"[LoRA] Trainable: {trainable/1e6:.2f}M")
    
    def _freeze_pretrained(self):
        """冻结预训练参数"""
        # LLM（保留 LoRA）
        for name, param in self.llm_encoder.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        
        # GraphMVP（冻结前3层，解冻后2层）
        enc = self.drug_structure_encoder
        if hasattr(enc, "main_model") and hasattr(enc.main_model, "gnns"):
            for p in enc.parameters():
                p.requires_grad = False
            for i in range(3, 5):
                for p in enc.main_model.gnns[i].parameters():
                    p.requires_grad = True
                for p in enc.main_model.batch_norms[i].parameters():
                    p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"[Freeze] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def _log_model_info(self):
        logger.info("=" * 60)
        logger.info("DTIModel (SA-CMSA + CMON + 6 Modality Streams + LLM)")
        logger.info(f"  Drug Struct:   GraphMVP({self.drug_struct_dim}) → 40 tokens")
        logger.info(f"  Drug KG:       BMKG({self.drug_kg_dim}) + SA-CMSA[struct,moltoken] → 28 tokens")
        logger.info(f"  Drug MolToken: {'ON' if self.use_moltoken else 'OFF'} → 32 tokens")
        logger.info(f"  Prot Struct:   DeepMCNN({self.prot_struct_dim}) → 40 tokens")
        logger.info(f"  Prot KG:       BMKG({self.prot_kg_dim}) + SA-CMSA[struct,prot3d] → 28 tokens")
        logger.info(f"  Prot 3D:       {self.prot_3d_mode} → 32 tokens")
        logger.info(f"  KG Completion: SA-CMSA {'ON' if self.use_cmsa else 'OFF'}")
        logger.info(f"  CMON:          {'ON' if self.use_cmon else 'OFF'} (orthogonal decomposition + digest tokens)")
        logger.info(f"  Text:          Drug_Text + Prot_Text → Prompt")
        logger.info(f"  Total:         200 modal + {6 if self.use_cmon else 0} digest tokens")
        logger.info("=" * 60)
    
    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """验证并修复tensor的有效性（与 DP 完全一致）"""
        if tensor is None:
            return None
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"[{name}] Found NaN/Inf, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = tensor.abs().max()
        if max_val > 1e4:
            logger.warning(f"[{name}] Large values detected (max={max_val:.2f}), clipping")
            tensor = tensor.clamp(-1e4, 1e4)
        return tensor

    # ========== Forward ==========
    def forward(self, drug: Dict, protein: Dict) -> Tuple[Dict, Optional[Dict]]:
        batch_size = self._get_batch_size(drug)
        device = self._get_device(drug)
        
        # ===== 1. 原始特征编码 =====
        h_drug_struct, smiles_list = self._encode_drug_structure(drug, device)
        h_prot_struct = self._encode_protein_structure(protein)
        h_mol = self._encode_moltoken(drug, smiles_list, batch_size, device)
        h_prot_3di = self._encode_foldseek(protein, batch_size, device)
        
        # ===== 1.5 特征噪声增强（训练时）=====
        noise_std = self.feature_noise_std if self.training and self.use_feature_noise else 0.0
        if noise_std > 0:
            h_drug_struct = h_drug_struct + torch.randn_like(h_drug_struct) * noise_std
            h_prot_struct = h_prot_struct + torch.randn_like(h_prot_struct) * noise_std
            if h_mol is not None:
                h_mol = h_mol + torch.randn_like(h_mol) * noise_std
            if h_prot_3di is not None:
                h_prot_3di = h_prot_3di + torch.randn_like(h_prot_3di) * noise_std
        
        # ===== 2. SA-CMSA: KG 补全 =====
        h_mol_safe = h_mol if h_mol is not None else torch.zeros(batch_size, self.moltoken_dim, device=device)
        h_prot_3di_safe = h_prot_3di if h_prot_3di is not None else torch.zeros(batch_size, self.foldseek_dim, device=device)
        drug_mol_missing = (h_mol is None)
        prot_3d_missing = (h_prot_3di is None)
        
        # Drug KG
        h_drug_kg = self._get_kg_feat(drug, self.drug_kg_dim, device)
        drug_kg_missing = torch.all(h_drug_kg == 0, dim=1)
        
        # Prot KG
        h_prot_kg = self._get_kg_feat(protein, self.prot_kg_dim, device)
        prot_kg_missing = torch.all(h_prot_kg == 0, dim=1)
        
        # KGE Dropout: 训练时随机 mask 部分有效 KGE → 强制 SA-CMSA 补全路径被训练
        if self.training and self.kge_dropout_rate > 0:
            drug_has_kg = ~drug_kg_missing
            if drug_has_kg.any():
                drop_mask = torch.rand(batch_size, device=device) < self.kge_dropout_rate
                drug_drop = drug_has_kg & drop_mask
                if drug_drop.any():
                    h_drug_kg = h_drug_kg.clone()
                    h_drug_kg[drug_drop] = 0.0
                    drug_kg_missing = drug_kg_missing | drug_drop
            
            prot_has_kg = ~prot_kg_missing
            if prot_has_kg.any():
                drop_mask = torch.rand(batch_size, device=device) < self.kge_dropout_rate
                prot_drop = prot_has_kg & drop_mask
                if prot_drop.any():
                    h_prot_kg = h_prot_kg.clone()
                    h_prot_kg[prot_drop] = 0.0
                    prot_kg_missing = prot_kg_missing | prot_drop
        
        drug_cmsa_loss = torch.tensor(0.0, device=device)
        prot_cmsa_loss = torch.tensor(0.0, device=device)
        drug_cmsa_diag = {}
        prot_cmsa_diag = {}
        drug_kg_completed = drug_kg_missing.clone()
        prot_kg_completed = prot_kg_missing.clone()
        
        if self.use_cmsa and self.drug_cmsa is not None:
            drug_missing_flags = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
            drug_missing_flags[:, 1] = drug_kg_missing
            if drug_mol_missing:
                drug_missing_flags[:, 2] = True
            
            drug_feats, drug_comp_flags, drug_cmsa_loss, drug_cmsa_diag, _ = self.drug_cmsa(
                [h_drug_struct, h_drug_kg, h_mol_safe], drug_missing_flags)
            h_drug_struct, h_drug_kg, h_mol_safe = drug_feats
            drug_kg_completed = drug_comp_flags[:, 1]
        
        if self.use_cmsa and self.prot_cmsa is not None:
            prot_missing_flags = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
            prot_missing_flags[:, 1] = prot_kg_missing
            if prot_3d_missing:
                prot_missing_flags[:, 2] = True
            
            prot_feats, prot_comp_flags, prot_cmsa_loss, prot_cmsa_diag, _ = self.prot_cmsa(
                [h_prot_struct, h_prot_kg, h_prot_3di_safe], prot_missing_flags)
            h_prot_struct, h_prot_kg, h_prot_3di_safe = prot_feats
            prot_kg_completed = prot_comp_flags[:, 1]
        
        # ===== 3. Adapter → Tokens（与 DP 一致：validate + noise + adapter）=====
        # KG 特征噪声在 SA-CMSA 之后注入（与 DP 一致）
        h_drug_struct = self._validate_tensor(h_drug_struct, "drug_struct_feat")
        d_struct_tokens = self.drug_struct_adapter(h_drug_struct, return_confidence=False)
        d_struct_tokens = self._validate_tensor(d_struct_tokens, "d_struct_tokens")

        h_prot_struct = self._validate_tensor(h_prot_struct, "prot_struct_feat")
        p_struct_tokens = self.prot_struct_adapter(h_prot_struct, return_confidence=False)
        p_struct_tokens = self._validate_tensor(p_struct_tokens, "p_struct_tokens")

        if noise_std > 0:
            h_drug_kg = h_drug_kg + torch.randn_like(h_drug_kg) * noise_std
            h_prot_kg = h_prot_kg + torch.randn_like(h_prot_kg) * noise_std
        h_drug_kg = self._validate_tensor(h_drug_kg, "drug_kg_feat")
        d_kg_tokens = self.drug_kg_adapter(h_drug_kg, return_confidence=False)
        d_kg_tokens = self._validate_tensor(d_kg_tokens, "d_kg_tokens")

        h_prot_kg = self._validate_tensor(h_prot_kg, "prot_kg_feat")
        p_kg_tokens = self.prot_kg_adapter(h_prot_kg, return_confidence=False)
        p_kg_tokens = self._validate_tensor(p_kg_tokens, "p_kg_tokens")

        h_mol_safe = self._validate_tensor(h_mol_safe, "moltoken_feat")
        d_mol_tokens = self.drug_mol_adapter(h_mol_safe, return_confidence=False)
        d_mol_tokens = self._validate_tensor(d_mol_tokens, "d_mol_tokens")

        h_prot_3di_safe = self._validate_tensor(h_prot_3di_safe, "prot_3d_feat")
        p_3d_tokens = self.prot_3d_adapter(h_prot_3di_safe, return_confidence=False)
        p_3d_tokens = self._validate_tensor(p_3d_tokens, "p_3d_tokens")

        # ===== 5. 对比损失（与 DP 一致：在 CMON 之前计算，使用 PRE-CMON tokens）=====
        contrastive_loss = torch.tensor(0.0, device=device)
        if self.training:
            drug_cl = self.drug_contrastive_loss(d_struct_tokens, d_kg_tokens, d_mol_tokens)
            prot_cl = self.prot_contrastive_loss(p_struct_tokens, p_kg_tokens, p_3d_tokens)
            contrastive_loss = (drug_cl + prot_cl) / 2.0

        # ===== 6. CMON: 正交分解 + Digest Tokens（与 DP 一致：在对比损失之后）=====
        drug_cmon_diag = {}
        prot_cmon_diag = {}
        drug_cmon_losses = {}
        prot_cmon_losses = {}
        drug_digests = None
        prot_digests = None
        drug_rel_samples = None
        prot_rel_samples = None
        
        if self.use_cmon and self.drug_cmon is not None:
            [d_struct_tokens, d_kg_tokens, d_mol_tokens], drug_digests, drug_cmon_diag, drug_cmon_losses, drug_rel_samples = \
                self.drug_cmon([d_struct_tokens, d_kg_tokens, d_mol_tokens])
            with torch.no_grad():
                d_real_mask = ~drug_kg_missing
                if d_real_mask.any():
                    drug_cmon_diag['cmon_rel_k_real'] = drug_rel_samples[d_real_mask, 1].mean().item()
                if (~d_real_mask).any():
                    drug_cmon_diag['cmon_rel_k_comp'] = drug_rel_samples[~d_real_mask, 1].mean().item()
        
        if self.use_cmon and self.prot_cmon is not None:
            [p_struct_tokens, p_kg_tokens, p_3d_tokens], prot_digests, prot_cmon_diag, prot_cmon_losses, prot_rel_samples = \
                self.prot_cmon([p_struct_tokens, p_kg_tokens, p_3d_tokens])
            with torch.no_grad():
                p_real_mask = ~prot_kg_missing
                if p_real_mask.any():
                    prot_cmon_diag['cmon_rel_k_real'] = prot_rel_samples[p_real_mask, 1].mean().item()
                if (~p_real_mask).any():
                    prot_cmon_diag['cmon_rel_k_comp'] = prot_rel_samples[~p_real_mask, 1].mean().item()
        
        # ===== 7. 样本级模态 Dropout（与 DP 完全一致：每样本最多丢一个模态）=====
        if self.training:
            _B = d_struct_tokens.size(0)
            # Drug 侧：样本级互斥 dropout（与 DP 一致）
            r_d = torch.rand(_B, device=device)
            d_mol_drop = (r_d < 0.08).unsqueeze(-1).unsqueeze(-1)
            d_kg_drop = ((r_d >= 0.08) & (r_d < 0.15)).unsqueeze(-1).unsqueeze(-1)
            d_str_drop = ((r_d >= 0.15) & (r_d < 0.20)).unsqueeze(-1).unsqueeze(-1)
            d_mol_tokens = d_mol_tokens * (~d_mol_drop).float()
            d_kg_tokens = d_kg_tokens * (~d_kg_drop).float()
            d_struct_tokens = d_struct_tokens * (~d_str_drop).float()
            # Protein 侧：样本级互斥 dropout（与 DP 一致）
            r_p = torch.rand(_B, device=device)
            p_3d_drop = (r_p < 0.08).unsqueeze(-1).unsqueeze(-1)
            p_kg_drop = ((r_p >= 0.08) & (r_p < 0.15)).unsqueeze(-1).unsqueeze(-1)
            p_str_drop = ((r_p >= 0.15) & (r_p < 0.20)).unsqueeze(-1).unsqueeze(-1)
            p_3d_tokens = p_3d_tokens * (~p_3d_drop).float()
            p_kg_tokens = p_kg_tokens * (~p_kg_drop).float()
            p_struct_tokens = p_struct_tokens * (~p_str_drop).float()
        
        # ===== 8. LLM 前向 =====
        llm_output = self._llm_forward(
            d_struct_tokens, d_kg_tokens, d_mol_tokens,
            p_struct_tokens, p_kg_tokens, p_3d_tokens,
            drug_digests, prot_digests,
            drug, protein, smiles_list, batch_size, device,
            drug_kg_completed=drug_kg_completed,
            drug_mol_missing=drug_mol_missing,
            prot_kg_completed=prot_kg_completed,
            prot_3d_missing=prot_3d_missing,
        )
        
        # ===== 9. 预测（与 DP 一致：pooled → head → clamp）=====
        classification_logits = self.classification_head(llm_output)
        regression_output = self.regression_head(llm_output)
        confidence = self.final_confidence_head(llm_output)
        
        logits_clamp = getattr(self, 'logits_clamp_abs', 10.0)
        classification_logits = torch.clamp(classification_logits, -logits_clamp, logits_clamp)
        
        prediction = {
            "classification": classification_logits,
            "regression": regression_output,
            "confidence": confidence
        }
        
        # ===== 10. 辅助损失汇总 =====
        aux_output = {
            "drug_kg_completion_ratio": drug_kg_completed.float().mean().item(),
            "prot_kg_completion_ratio": prot_kg_completed.float().mean().item(),
            "contrastive_loss": contrastive_loss,
            "drug_cmsa_loss": drug_cmsa_loss,
            "prot_cmsa_loss": prot_cmsa_loss,
            "drug_cmsa_diag": drug_cmsa_diag,
            "prot_cmsa_diag": prot_cmsa_diag,
            "drug_cmon_diag": drug_cmon_diag,
            "prot_cmon_diag": prot_cmon_diag,
        }
        if self.training:
            aux_output["drug_cmon_losses"] = drug_cmon_losses
            aux_output["prot_cmon_losses"] = prot_cmon_losses
        
        return prediction, aux_output
    
    def _get_kg_feat(self, data, dim, device):
        """提取 KG embedding，缺失时返回零向量"""
        batch_size = None
        if isinstance(data, dict):
            if 'kg' in data and data['kg'] is not None:
                return data['kg'].to(device)
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break
                if hasattr(v, 'batch'):
                    batch_size = v.batch.max().item() + 1
                    break
        if batch_size is None:
            batch_size = 1
        return torch.zeros(batch_size, dim, device=device)
    
    def _llm_forward(self, d_struct_tokens, d_kg_tokens, d_mol_tokens,
                     p_struct_tokens, p_kg_tokens, p_3d_tokens,
                     drug_digests, prot_digests,
                     drug, protein, smiles_list, batch_size, device,
                     drug_kg_completed=None, drug_mol_missing=False,
                     prot_kg_completed=None, prot_3d_missing=False):
        """
        LLM 前向传播
        
        每个模态序列 = [marker_start] [digest_token(if CMON)] [adapter_tokens] [marker_end]
        
        Returns:
            pooled: (B, dim) 池化后的特征
        """
        target_dtype = next(self.llm_encoder.model.parameters()).dtype
        
        # ===== 1. 模态标记（12 个）=====
        ds_start = self._get_marker_embedding("d_struct_start", batch_size, device)
        ds_end = self._get_marker_embedding("d_struct_end", batch_size, device)
        dk_start = self._get_marker_embedding("d_kg_start", batch_size, device)
        dk_end = self._get_marker_embedding("d_kg_end", batch_size, device)
        dm_start = self._get_marker_embedding("d_mol_start", batch_size, device)
        dm_end = self._get_marker_embedding("d_mol_end", batch_size, device)
        ps_start = self._get_marker_embedding("p_struct_start", batch_size, device)
        ps_end = self._get_marker_embedding("p_struct_end", batch_size, device)
        pk_start = self._get_marker_embedding("p_kg_start", batch_size, device)
        pk_end = self._get_marker_embedding("p_kg_end", batch_size, device)
        pf_start = self._get_marker_embedding("p_3d_start", batch_size, device)
        pf_end = self._get_marker_embedding("p_3d_end", batch_size, device)
        
        # ===== 2. 构建 6 个模态序列（含 CMON digest tokens）=====
        if drug_digests is not None and len(drug_digests) == 3:
            d_struct_seq = torch.cat([ds_start, drug_digests[0].to(target_dtype), d_struct_tokens, ds_end], dim=1).to(target_dtype)
            d_kg_seq = torch.cat([dk_start, drug_digests[1].to(target_dtype), d_kg_tokens, dk_end], dim=1).to(target_dtype)
            d_mol_seq = torch.cat([dm_start, drug_digests[2].to(target_dtype), d_mol_tokens, dm_end], dim=1).to(target_dtype)
        else:
            d_struct_seq = torch.cat([ds_start, d_struct_tokens, ds_end], dim=1).to(target_dtype)
            d_kg_seq = torch.cat([dk_start, d_kg_tokens, dk_end], dim=1).to(target_dtype)
            d_mol_seq = torch.cat([dm_start, d_mol_tokens, dm_end], dim=1).to(target_dtype)
        
        if prot_digests is not None and len(prot_digests) == 3:
            p_struct_seq = torch.cat([ps_start, prot_digests[0].to(target_dtype), p_struct_tokens, ps_end], dim=1).to(target_dtype)
            p_kg_seq = torch.cat([pk_start, prot_digests[1].to(target_dtype), p_kg_tokens, pk_end], dim=1).to(target_dtype)
            p_3d_seq = torch.cat([pf_start, prot_digests[2].to(target_dtype), p_3d_tokens, pf_end], dim=1).to(target_dtype)
        else:
            p_struct_seq = torch.cat([ps_start, p_struct_tokens, ps_end], dim=1).to(target_dtype)
            p_kg_seq = torch.cat([pk_start, p_kg_tokens, pk_end], dim=1).to(target_dtype)
            p_3d_seq = torch.cat([pf_start, p_3d_tokens, pf_end], dim=1).to(target_dtype)
        
        prefix = torch.cat([
            d_struct_seq, d_kg_seq, d_mol_seq,
            p_struct_seq, p_kg_seq, p_3d_seq,
        ], dim=1)
        prefix = self._validate_tensor(prefix, "prefix")
        
        if prefix.shape[-1] != self.llm_embed_dim:
            raise ValueError(f"Prefix dimension mismatch: {prefix.shape[-1]} vs {self.llm_embed_dim}")
        
        # ===== 3. Prompt（含模态置信度评估）=====
        prompts = []
        sequences = self._get_protein_sequences(protein, batch_size)
        
        for i in range(batch_size):
            smi = smiles_list[i] if i < len(smiles_list) else ""
            seq = sequences[i] if i < len(sequences) else ""
            has_drug_kg_i = True
            if drug_kg_completed is not None:
                has_drug_kg_i = not bool(drug_kg_completed[i].item()) if drug_kg_completed.numel() > 0 else True
            has_prot_kg_i = True
            if prot_kg_completed is not None:
                has_prot_kg_i = not bool(prot_kg_completed[i].item()) if prot_kg_completed.numel() > 0 else True
            prompt = self.prompt_composer.build_prompt(
                smiles=smi, sequence=seq,
                has_drug_kg=has_drug_kg_i,
                has_moltoken=not drug_mol_missing,
                has_prot_kg=has_prot_kg_i,
                has_prot3d=not prot_3d_missing,
            )
            prompts.append(prompt)
        
        text_inputs = self.llm_encoder.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        text_embeds = self.llm_encoder.model.get_input_embeddings()(text_inputs.input_ids).to(target_dtype)
        text_embeds = self._validate_tensor(text_embeds, "text_embeds")
        
        # ===== 4. 拼接 prefix + text =====
        combined = torch.cat([prefix.to(target_dtype), text_embeds.to(target_dtype)], dim=1)
        combined = self._validate_tensor(combined, "inputs_embeds")
        prefix_mask = torch.ones(batch_size, prefix.size(1), device=device, dtype=text_inputs.attention_mask.dtype)
        attention_mask_2d = torch.cat([prefix_mask, text_inputs.attention_mask], dim=1)
        
        # ===== 5. LLM 前向 =====
        outputs = self.llm_encoder.model(
            inputs_embeds=combined,
            attention_mask=attention_mask_2d,
            output_hidden_states=True,
            use_cache=False
        )
        last_hidden = outputs.hidden_states[-1]
        
        # ===== 6. 池化（与 DP 完全一致：AttentionPooling + learnable ratio + last_token）=====
        last_pos = attention_mask_2d.sum(dim=1).clamp(min=1).to(torch.long) - 1
        batch_indices = torch.arange(batch_size, device=device)
        last_token = last_hidden[batch_indices, last_pos, :]
        
        if self.use_attention_pooling:
            attn_pooled = self.attention_pooler(last_hidden, attention_mask_2d)
            ratio = torch.sigmoid(self.attn_pool_ratio)
            pooled = (ratio * attn_pooled + (1.0 - ratio) * last_token).float()
        else:
            second_last_pos = (last_pos - 1).clamp(min=0)
            second_last_hidden = last_hidden[batch_indices, second_last_pos, :]
            pooled = (0.8 * last_token + 0.2 * second_last_hidden).float()
        
        return pooled
    
    # ========== Helpers ==========
    def _get_batch_size(self, drug):
        if isinstance(drug, dict):
            if 'kg' in drug and drug['kg'] is not None:
                return drug['kg'].shape[0]
            if 'structure' in drug and hasattr(drug['structure'], 'batch'):
                return drug['structure'].batch.max().item() + 1
        return 1
    
    def _get_device(self, drug):
        if isinstance(drug, dict):
            if 'kg' in drug and drug['kg'] is not None:
                return drug['kg'].device
            if 'structure' in drug and hasattr(drug['structure'], 'x'):
                return drug['structure'].x.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _encode_drug_structure(self, drug, device):
        data = drug.get("structure", drug) if isinstance(drug, dict) else drug
        if hasattr(self.drug_structure_encoder, "encode_structure"):
            h = self.drug_structure_encoder.encode_structure(data)
        elif hasattr(self.drug_structure_encoder, "encode_mol"):
            h = self.drug_structure_encoder.encode_mol(data)
        else:
            h = self.drug_structure_encoder(data)
        if isinstance(h, tuple):
            h = h[0]
        h = torch.nan_to_num(h, nan=0.0)
        
        smiles = []
        if isinstance(drug, dict) and "smiles" in drug:
            raw = drug["smiles"]
            if isinstance(raw, list):
                smiles = [str(s) for s in raw]
            elif isinstance(raw, str):
                smiles = [raw] * h.size(0)
        if not smiles:
            smiles = [""] * h.size(0)
        
        return h.to(device), smiles
    
    def _encode_protein_structure(self, protein):
        if isinstance(protein, dict) and 'structure' in protein:
            return self.protein_structure_encoder.encode_protein(protein['structure'])
        return self.protein_structure_encoder.encode_protein(protein)
    
    # _encode_kg removed: replaced by _get_kg_feat + SA-CMSA direct calls in forward()
    
    def _encode_moltoken(self, drug, smiles_list, batch_size, device):
        if not self.use_moltoken or self.moltoken_encoder is None:
            return torch.zeros(batch_size, self.moltoken_dim, device=device)
        
        cached_seq = drug.get("tokenmol_seq", None) if isinstance(drug, dict) else None
        
        token_lists = []
        if cached_seq and isinstance(cached_seq, list) and len(cached_seq) == batch_size:
            for seq in cached_seq:
                if seq and isinstance(seq, str):
                    token_lists.append(seq.split())
                else:
                    smi = smiles_list[len(token_lists)] if len(token_lists) < len(smiles_list) else ""
                    token_lists.append(list(smi) if smi else ["[UNK]"])
        else:
            for smi in smiles_list:
                token_lists.append(list(smi) if smi else ["[UNK]"])
        
        try:
            with torch.no_grad():
                feat, _ = self.moltoken_encoder.encode_token_lists(token_lists, return_lm_loss=False)
            return feat.to(device)
        except:
            return torch.zeros(batch_size, self.moltoken_dim, device=device)
    
    def _encode_foldseek(self, protein, batch_size, device):
        """编码蛋白质 3D 模态（ProteinProperty 或 FoldSeek）"""
        if self.prot_3d_encoder is None:
            return None
        
        # === 理化性质模式（默认）===
        if self.prot_3d_mode == "property":
            sequences = self._get_protein_sequences(protein, batch_size)
            try:
                return self.prot_3d_encoder(sequences, device)
            except Exception as e:
                logger.warning(f"[Prot3D-Property] Failed: {e}")
                return torch.zeros(batch_size, self.foldseek_dim, device=device)
        
        # === FoldSeek 模式（需要真实 3Di 缓存）===
        if not self.use_foldseek:
            return None
        
        # 从 protein dict 中获取预计算的 3Di 序列
        seq_3di = protein.get("foldseek_3di", None) if isinstance(protein, dict) else None
        
        if seq_3di is None:
            # 无 3Di 数据，返回零向量
            return torch.zeros(batch_size, self.foldseek_dim, device=device)
        
        # 处理 list 或 单个 str
        if isinstance(seq_3di, str):
            seq_3di = [seq_3di] * batch_size
        elif isinstance(seq_3di, list):
            # 确保长度匹配
            if len(seq_3di) != batch_size:
                logger.warning(f"[FoldSeek] 3Di list length {len(seq_3di)} != batch_size {batch_size}")
                seq_3di = seq_3di[:batch_size] + [""] * max(0, batch_size - len(seq_3di))
        
        # 替换空字符串为 padding
        seq_3di = [s if s else "#" * 10 for s in seq_3di]
        
        try:
            tokens = self.prot_3d_encoder.tokenize_batch(seq_3di, device=device)
            return self.prot_3d_encoder(tokens)
        except Exception as e:
            logger.warning(f"[FoldSeek] Encoding failed: {e}")
            return torch.zeros(batch_size, self.foldseek_dim, device=device)
    
    def _get_protein_sequences(self, protein, batch_size):
        if isinstance(protein, dict) and 'sequence' in protein:
            seqs = protein['sequence']
            if isinstance(seqs, list):
                return seqs
            elif isinstance(seqs, str):
                return [seqs] * batch_size
        return [""] * batch_size


# ========== 注册 ==========
SUPPORTED_DTI_NETWORKS = {
    "dti": DTIModel,
}
