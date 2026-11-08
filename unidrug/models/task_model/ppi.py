"""
PPIModel - 与 DTI/DP 架构对齐的蛋白质-蛋白质相互作用预测模型

数据流（与 DTI 蛋白质支路一致）：
    Raw Features → Feature Noise → SA-CMSA (KG补全) → Q-Former Adapters → Tokens
    → 对比损失(PRE-CMON) → CMON(正交分解+digest) → 样本级模态Dropout
    → [modality markers + digest + tokens + prompt] → Qwen (LoRA) → Attention Pooling
    → ResidualPredictionHead → 7-class logits

SA-CMSA and CMON each run separately on protein A and protein B.
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)

# 从 dp 导入共享组件
from models.task_model.dp import (
    SparseAttentionCMSA,
    CMONModule,
    AdapterTokenGenerator,
    AttentionPooling,
    ResidualPredictionHead,
)
# DTI 的 ModalContrastiveLoss 接受 (B,N,D) tokens，与 PPI 适配
from models.task_model.dti import (
    DeepMCNN,
    ProteinPropertyEncoder,
    FoldSeek3DiEncoder,
    ModalContrastiveLoss,
)


# ============================================================
# PPI Prompt Composer（与 DTI 风格对齐：含模态置信度）
# ============================================================
class PPIPromptComposer:
    """
    PPI 任务 Prompt 构建器，与 DTI build_prompt 对齐：
    任务描述 + Protein A/B 描述 + 各蛋白模态置信度 + 推理指导 + 预测锚点
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.protein_text_map = {}
        dataset_text_path = config.get("dataset_text_path")
        if dataset_text_path and os.path.exists(dataset_text_path):
            self._load_text_file(dataset_text_path, "dataset")
        kg_path = config.get("kg_path")
        if kg_path:
            protein_json = os.path.join(kg_path, "protein.json")
            if os.path.exists(protein_json):
                self._load_text_file(protein_json, "kg")
    
    def _load_text_file(self, path: str, source: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            count = 0
            for key, value in data.items():
                if key in self.protein_text_map:
                    continue
                if isinstance(value, dict):
                    text = value.get("text", value.get("description", ""))
                    seq = value.get("sequence", "")
                    if text and text != "text":
                        self.protein_text_map[key] = text
                        if seq:
                            self.protein_text_map[seq] = text
                            if len(seq) > 100:
                                self.protein_text_map[seq[:100]] = text
                        count += 1
                elif isinstance(value, str) and value:
                    self.protein_text_map[key] = value
                    count += 1
            logger.info(f"[PPIPromptComposer] Loaded {count} texts from {source}: {path}")
        except Exception as e:
            logger.warning(f"[PPIPromptComposer] Failed to load {path}: {e}")
    
    def get_protein_text(self, sequence: str, prot_id: str = None) -> str:
        if prot_id and prot_id in self.protein_text_map:
            return self.protein_text_map[prot_id]
        if sequence and sequence in self.protein_text_map:
            return self.protein_text_map[sequence]
        if sequence and len(sequence) > 100 and sequence[:100] in self.protein_text_map:
            return self.protein_text_map[sequence[:100]]
        return self._generate_protein_text(sequence)
    
    def _generate_protein_text(self, sequence: str) -> str:
        if not sequence:
            return "A protein target."
        length = len(sequence)
        charged = sum(sequence.count(aa) for aa in "RKDE")
        hydrophobic = sum(sequence.count(aa) for aa in "AVLIMFW")
        desc = f"Protein ({length} residues, {charged/length*100:.0f}% charged, {hydrophobic/length*100:.0f}% hydrophobic)"
        if length < 150:
            desc += ", short peptide"
        elif length < 400:
            desc += ", single-domain"
        else:
            desc += ", multi-domain"
        return desc
    
    def build_prompt(
        self,
        seq_a: str,
        seq_b: str,
        text_a: str = None,
        text_b: str = None,
        has_a_kg: bool = True,
        has_a_3d: bool = True,
        has_b_kg: bool = True,
        has_b_3d: bool = True,
    ) -> str:
        """构建与 DTI 风格一致的 PPI 提示（含模态置信度与推理指导）"""
        if not text_a:
            text_a = self.get_protein_text(seq_a)
        if not text_b:
            text_b = self.get_protein_text(seq_b)
        text_a = text_a[:200] + "..." if len(text_a) > 200 else text_a
        text_b = text_b[:200] + "..." if len(text_b) > 200 else text_b
        
        parts = []
        parts.append("Predict protein-protein interaction type using multimodal evidence.")
        parts.append(f"Protein A: {text_a}")
        parts.append(f"Protein B: {text_b}")
        parts.append("Protein A input reliability:")
        parts.append("[Structure] computed from sequence features — high confidence.")
        if has_a_3d:
            parts.append("[Prot3D] physicochemical property encoding — high confidence.")
        else:
            parts.append("[Prot3D] cross-modal completion — moderate confidence.")
        if has_a_kg:
            parts.append("[KG] retrieved from knowledge graph — high confidence.")
        else:
            parts.append("[KG] cross-modal completion — moderate confidence.")
        parts.append("Protein B input reliability:")
        parts.append("[Structure] computed from sequence features — high confidence.")
        if has_b_3d:
            parts.append("[Prot3D] physicochemical property encoding — high confidence.")
        else:
            parts.append("[Prot3D] cross-modal completion — moderate confidence.")
        if has_b_kg:
            parts.append("[KG] retrieved from knowledge graph — high confidence.")
        else:
            parts.append("[KG] cross-modal completion — moderate confidence.")
        a_inferred = (not has_a_kg) or (not has_a_3d)
        b_inferred = (not has_b_kg) or (not has_b_3d)
        if not a_inferred and not b_inferred:
            parts.append("All modalities are directly observed; integrate with equal weight.")
        else:
            inferred = []
            if not has_a_kg:
                inferred.append("A-KG")
            if not has_a_3d:
                inferred.append("A-Prot3D")
            if not has_b_kg:
                inferred.append("B-KG")
            if not has_b_3d:
                inferred.append("B-Prot3D")
            parts.append(
                f"{', '.join(inferred)} were inferred, not observed. "
                "Rely primarily on observed modalities; use inferred features as supporting evidence."
            )
        parts.append("Prediction:")
        return " ".join(parts)


# ============================================================
# PPIModel 主模型
# ============================================================
class PPIModel(nn.Module):
    """
    PPI 预测模型 — 双蛋白质独立 SA-CMSA + CMON + 共享 Adapters，与 DTI 对齐。
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int = 7,
        kge_bank: torch.Tensor = None,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pred_dim = num_classes
        net_cfg = config.get("network", config)
        
        # 消融
        ablation_str = net_cfg.get("ablation", "")
        self.ablation_flags = set(a.strip() for a in ablation_str.split(",") if a.strip())
        if "no_foldseek" in self.ablation_flags:
            net_cfg["use_foldseek"] = False
        if "no_cmon" in self.ablation_flags:
            net_cfg.setdefault("mars", {})["enabled"] = False
        if "no_cmsa" in self.ablation_flags:
            net_cfg.setdefault("cmsa", {})["enabled"] = False

        # 1. Encoders
        self._init_encoders(config)
        # 2. LLM (need llm_dim for adapters / pooler / head)
        self._init_llm(config)
        # 3. Adapters (3 shared: struct, kg, prot3d)
        self._init_adapters(config)
        # 4. SA-CMSA
        self._init_sacmsa(config, kge_bank)
        # 5. CMON
        self._init_cmon(config)
        # 6. Modality markers (12: a_struct_start/end, a_kg_start/end, a_3d_start/end, b_*)
        self._init_modality_markers()
        # 7. Contrastive loss (one shared)
        self.contrastive_loss = ModalContrastiveLoss(
            hidden_dim=self.llm_dim, proj_dim=256, temperature=0.15
        )
        # 8. Classification head
        head_dropout = net_cfg.get("head_dropout", 0.15)
        self.classification_head = ResidualPredictionHead(
            self.llm_dim, self.llm_dim // 2, num_classes, dropout=head_dropout
        )
        # 9. LoRA
        if net_cfg.get("use_lora", True):
            self._enable_lora(net_cfg)
        # 10. Freeze
        self._freeze_pretrained()
        # 11. Feature noise
        self.use_feature_noise = net_cfg.get("use_feature_noise", False)
        self.feature_noise_std = net_cfg.get("feature_noise_std", 0.02)
        self.logits_clamp_abs = net_cfg.get("stability", {}).get("logits_clamp_abs", 10.0)

        self._log_model_info()

    def _init_encoders(self, config):
        net_cfg = config.get("network", config)
        prot_cfg = net_cfg.get("protein", {})
        mcnn_cfg = prot_cfg.get("structure", {
            "vocab_size": 26, "embedding_num": 128, "hidden_size": 128,
            "output_size": 256, "dropout": 0.1,
        })
        self.protein_structure_encoder = DeepMCNN(mcnn_cfg)
        self.prot_struct_dim = self.protein_structure_encoder.output_dim
        self.prot_kg_dim = net_cfg.get("kg_dim", prot_cfg.get("kg", {}).get("embedding_dim", 256))
        self.use_foldseek = net_cfg.get("use_foldseek", False)
        self.prot_3d_mode = net_cfg.get("prot_3d_mode", "property")
        self.prot_3d_encoder = None
        self.prot_3d_dim = 0
        if "no_foldseek" not in self.ablation_flags:
            if self.prot_3d_mode == "foldseek" and self.use_foldseek:
                fs_cfg = net_cfg.get("foldseek", {})
                self.prot_3d_encoder = FoldSeek3DiEncoder(fs_cfg)
                self.prot_3d_dim = self.prot_3d_encoder.output_dim
            else:
                prop_cfg = net_cfg.get("foldseek", {})
                self.prot_3d_encoder = ProteinPropertyEncoder(prop_cfg)
                self.prot_3d_dim = self.prot_3d_encoder.output_dim
                self.prot_3d_mode = "property"
        logger.info(f"[PPI] Encoders: struct={self.prot_struct_dim}, kg={self.prot_kg_dim}, 3d={self.prot_3d_dim}")

    def _init_llm(self, config):
        from models.llm_encoder import SUPPORTED_LLM_ENCODER
        net_cfg = config.get("network", config)
        llm_cfg = net_cfg.get("llm", {})
        self.llm_encoder = SUPPORTED_LLM_ENCODER[llm_cfg.get("name", "qwen")](llm_cfg)
        self.llm_dim = getattr(self.llm_encoder.model.config, "hidden_size", 2048)
        self.use_attention_pooling = net_cfg.get("use_attention_pooling", True)
        self.attention_pooler = AttentionPooling(
            hidden_dim=self.llm_dim,
            num_heads=8,
            dropout=net_cfg.get("pooling_dropout", 0.15),
        )
        self.attn_pool_ratio = nn.Parameter(torch.tensor(0.85))
        kg_path = net_cfg.get("kg_path", "")
        self.prompt_composer = PPIPromptComposer({
            "dataset_text_path": net_cfg.get("protein_text_path"),
            "kg_path": kg_path,
        })
        self.prompt_max_length = net_cfg.get("prompt_max_length", 256)

    def _init_adapters(self, config):
        net_cfg = config.get("network", config)
        adapter_dropout = net_cfg.get("adapter_dropout", 0.25)
        num_struct = net_cfg.get("num_struct_tokens", 40)
        num_kg = net_cfg.get("num_kg_tokens", 28)
        num_3d = net_cfg.get("num_3d_tokens", 32)
        self.struct_adapter = AdapterTokenGenerator(
            self.prot_struct_dim, self.llm_dim, num_tokens=num_struct, num_layers=4, dropout=adapter_dropout
        )
        self.kg_adapter = AdapterTokenGenerator(
            self.prot_kg_dim, self.llm_dim, num_tokens=num_kg, num_layers=4, dropout=adapter_dropout
        )
        self.prot_3d_adapter = None
        if self.prot_3d_encoder is not None:
            self.prot_3d_adapter = AdapterTokenGenerator(
                self.prot_3d_dim, self.llm_dim, num_tokens=num_3d, num_layers=4, dropout=adapter_dropout
            )
        logger.info(f"[PPI] Adapters: struct→{num_struct}, kg→{num_kg}, 3d→{num_3d}")

    def _init_sacmsa(self, config, kge_bank):
        net_cfg = config.get("network", config)
        cmsa_cfg = net_cfg.get("cmsa", net_cfg.get("dec", {}))
        use_cmsa = cmsa_cfg.get("enabled", True) and kge_bank is not None
        self.use_cmsa = use_cmsa
        self.prot_cmsa = None
        if use_cmsa:
            prot_3d_dim = self.prot_3d_dim if self.prot_3d_dim else self.prot_struct_dim
            self.prot_cmsa = SparseAttentionCMSA(
                modal_dims=[self.prot_struct_dim, self.prot_kg_dim, prot_3d_dim],
                kge_bank=kge_bank,
                sub_dim=cmsa_cfg.get("sub_dim", 128),
                topk=cmsa_cfg.get("topk", 16),
                num_heads=cmsa_cfg.get("num_heads", 2),
                dropout=cmsa_cfg.get("dropout", 0.1),
                align_lambda=cmsa_cfg.get("align_lambda", 0.05),
                retrieve_lambda=cmsa_cfg.get("retrieve_lambda", 0.1),
                enable_mol_completion=False,
                ablation_flags=self.ablation_flags,
            )
            logger.info(f"[PPI] SA-CMSA enabled (shared): kge_bank {kge_bank.shape[0]}x{kge_bank.shape[1]}")
        else:
            logger.info("[PPI] SA-CMSA disabled")

    def _init_cmon(self, config):
        net_cfg = config.get("network", config)
        mars_cfg = net_cfg.get("mars", {})
        self.use_cmon = mars_cfg.get("enabled", False)
        self.prot_cmon = None
        if self.use_cmon:
            self.prot_cmon = CMONModule(
                llm_dim=self.llm_dim, n_modalities=3, sub_dim=mars_cfg.get("sub_dim", 64),
                ablation_flags=self.ablation_flags,
            )
            logger.info("[PPI] CMON enabled (shared)")
        else:
            logger.info("[PPI] CMON disabled")

    def _init_modality_markers(self):
        self.modality_markers = {
            "a_struct_start": "[a_struct_start]",
            "a_struct_end": "[a_struct_end]",
            "a_kg_start": "[a_kg_start]",
            "a_kg_end": "[a_kg_end]",
            "a_3d_start": "[a_3d_start]",
            "a_3d_end": "[a_3d_end]",
            "b_struct_start": "[b_struct_start]",
            "b_struct_end": "[b_struct_end]",
            "b_kg_start": "[b_kg_start]",
            "b_kg_end": "[b_kg_end]",
            "b_3d_start": "[b_3d_start]",
            "b_3d_end": "[b_3d_end]",
        }
        for key in self.modality_markers:
            setattr(self, key + "_residual", nn.Parameter(torch.randn(1, 1, self.llm_dim) * 0.02))

    def _enable_lora(self, net_cfg):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=net_cfg.get("lora_rank", 32),
            lora_alpha=net_cfg.get("lora_alpha", 64),
            lora_dropout=net_cfg.get("lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.llm_encoder.model = get_peft_model(self.llm_encoder.model, lora_config)
        self.llm_encoder.model.enable_input_require_grads()
        if hasattr(self.llm_encoder.model, "gradient_checkpointing_enable"):
            self.llm_encoder.model.gradient_checkpointing_enable()
        logger.info(f"[PPI] LoRA r={net_cfg.get('lora_rank', 32)}")
    
    def _freeze_pretrained(self):
        for name, param in self.llm_encoder.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[PPI] Freeze: trainable {trainable:,} / {total:,}")

    def _log_model_info(self):
        logger.info(
            f"[PPIModel] Classes={self.num_classes}, SA-CMSA={self.use_cmsa}, CMON={self.use_cmon}"
        )

    def _get_marker_embedding(self, marker_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        tokenizer = self.llm_encoder.tokenizer
        embed_layer = self.llm_encoder.model.get_input_embeddings()
        if hasattr(self.llm_encoder.model, "base_model"):
            embed_layer = self.llm_encoder.model.base_model.model.get_input_embeddings()
        text = self.modality_markers.get(marker_name, "[UNK]")
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            token_embed = embed_layer(ids).mean(dim=1, keepdim=True)
        token_embed = token_embed.expand(batch_size, -1, -1).clone()
        res_name = marker_name + "_residual"
        if hasattr(self, res_name):
            token_embed = token_embed + getattr(self, res_name).to(token_embed.dtype)
        return token_embed

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor is None:
            return None
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"[{name}] NaN/Inf, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = tensor.abs().max()
        if max_val > 1e4:
            logger.warning(f"[{name}] Large values max={max_val:.2f}, clipping")
            tensor = tensor.clamp(-1e4, 1e4)
        return tensor

    def _get_kg_feat(self, data: Dict, dim: int, device: torch.device) -> torch.Tensor:
        if isinstance(data, dict) and data.get("kg") is not None:
            kg = data["kg"].to(device).float()
            if kg.dim() == 3:
                kg = kg.squeeze(1)
            return kg
        batch_size = 1
        for v in data.values() if isinstance(data, dict) else []:
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
        return torch.zeros(batch_size, dim, device=device)

    def _encode_protein_structure(self, prot_batch: Dict, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
        struct_input = prot_batch.get("structure", prot_batch.get("input_ids"))
        if struct_input is None:
            raise ValueError("Protein batch must contain 'structure' or 'input_ids'")
        struct_input = struct_input.to(device)
        if hasattr(self.protein_structure_encoder, "encode_protein"):
            struct_feat = self.protein_structure_encoder.encode_protein(struct_input)
        else:
            struct_feat = self.protein_structure_encoder(struct_input)
        batch_size = struct_feat.size(0)
        sequences = prot_batch.get("sequences", prot_batch.get("sequence", [""] * batch_size))
        if isinstance(sequences, str):
            sequences = [sequences] * batch_size
        return struct_feat, sequences

    def _encode_protein_3d(self, prot_batch: Dict, sequences: List[str], device: torch.device) -> Optional[torch.Tensor]:
        if self.prot_3d_encoder is None:
            return None
        if self.prot_3d_mode == "property":
            return self.prot_3d_encoder(sequences, device=device)
        if self.prot_3d_mode == "foldseek":
            seq_3di = prot_batch.get("foldseek_3di", prot_batch.get("3di"))
            if seq_3di is None:
                return torch.zeros(len(sequences), self.prot_3d_dim, device=device)
            if isinstance(seq_3di, str):
                seq_3di = [seq_3di] * len(sequences)
            seq_3di = [s if s else "#" * 10 for s in seq_3di]
            tokens = self.prot_3d_encoder.tokenize_batch(seq_3di, device=device)
            return self.prot_3d_encoder(tokens)
        return None

    def forward(
        self,
        prot_a_batch: Dict,
        prot_b_batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        device = next(self.parameters()).device
        target_dtype = next(self.llm_encoder.model.parameters()).dtype

        # ----- 1. Encode both proteins -----
        struct_a, seq_a = self._encode_protein_structure(prot_a_batch, device)
        batch_size = struct_a.size(0)
        struct_b, seq_b = self._encode_protein_structure(prot_b_batch, device)
        prot_3d_a = self._encode_protein_3d(prot_a_batch, seq_a, device)
        prot_3d_b = self._encode_protein_3d(prot_b_batch, seq_b, device)

        # ----- 2. Feature noise -----
        noise_std = self.feature_noise_std if self.training and self.use_feature_noise else 0.0
        if noise_std > 0:
            struct_a = struct_a + torch.randn_like(struct_a, device=device) * noise_std
            struct_b = struct_b + torch.randn_like(struct_b, device=device) * noise_std
            if prot_3d_a is not None:
                prot_3d_a = prot_3d_a + torch.randn_like(prot_3d_a, device=device) * noise_std
            if prot_3d_b is not None:
                prot_3d_b = prot_3d_b + torch.randn_like(prot_3d_b, device=device) * noise_std

        # ----- 3. KG + SA-CMSA -----
        kg_a = self._get_kg_feat(prot_a_batch, self.prot_kg_dim, device)
        kg_b = self._get_kg_feat(prot_b_batch, self.prot_kg_dim, device)
        a_kg_missing = torch.all(kg_a == 0, dim=1)
        b_kg_missing = torch.all(kg_b == 0, dim=1)
        prot_3d_dim = self.prot_3d_dim if self.prot_3d_dim else self.prot_struct_dim
        zero_3d = torch.zeros(batch_size, prot_3d_dim, device=device)
        a_3d_safe = prot_3d_a if prot_3d_a is not None else zero_3d
        b_3d_safe = prot_3d_b if prot_3d_b is not None else zero_3d

        a_cmsa_loss = torch.tensor(0.0, device=device)
        b_cmsa_loss = torch.tensor(0.0, device=device)
        a_cmsa_diag = {}
        b_cmsa_diag = {}
        a_kg_completed = a_kg_missing.clone()
        b_kg_completed = b_kg_missing.clone()

        if self.use_cmsa and self.prot_cmsa is not None:
            a_missing = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
            a_missing[:, 1] = a_kg_missing
            if prot_3d_a is None:
                a_missing[:, 2] = True
            feats_a, comp_a, a_cmsa_loss, a_cmsa_diag, _ = self.prot_cmsa(
                [struct_a, kg_a, a_3d_safe], a_missing
            )
            struct_a, kg_a, a_3d_safe = feats_a
            a_kg_completed = comp_a[:, 1]

            b_missing = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
            b_missing[:, 1] = b_kg_missing
            if prot_3d_b is None:
                b_missing[:, 2] = True
            feats_b, comp_b, b_cmsa_loss, b_cmsa_diag, _ = self.prot_cmsa(
                [struct_b, kg_b, b_3d_safe], b_missing
            )
            struct_b, kg_b, b_3d_safe = feats_b
            b_kg_completed = comp_b[:, 1]

        # ----- 4. Adapters -> tokens (与 DTI 对齐：validate → adapter → validate) -----
        struct_a = self._validate_tensor(struct_a, "struct_a")
        a_struct_tok = self.struct_adapter(struct_a, return_confidence=False)
        a_struct_tok = self._validate_tensor(a_struct_tok, "a_struct_tok")

        struct_b = self._validate_tensor(struct_b, "struct_b")
        b_struct_tok = self.struct_adapter(struct_b, return_confidence=False)
        b_struct_tok = self._validate_tensor(b_struct_tok, "b_struct_tok")

        if noise_std > 0:
            kg_a = kg_a + torch.randn_like(kg_a) * noise_std
            kg_b = kg_b + torch.randn_like(kg_b) * noise_std
        kg_a = self._validate_tensor(kg_a, "kg_a")
        a_kg_tok = self.kg_adapter(kg_a, return_confidence=False)
        a_kg_tok = self._validate_tensor(a_kg_tok, "a_kg_tok")

        kg_b = self._validate_tensor(kg_b, "kg_b")
        b_kg_tok = self.kg_adapter(kg_b, return_confidence=False)
        b_kg_tok = self._validate_tensor(b_kg_tok, "b_kg_tok")

        a_3d_safe = self._validate_tensor(a_3d_safe, "a_3d")
        b_3d_safe = self._validate_tensor(b_3d_safe, "b_3d")
        a_3d_tok = b_3d_tok = None
        if self.prot_3d_adapter is not None:
            a_3d_tok = self.prot_3d_adapter(a_3d_safe, return_confidence=False)
            b_3d_tok = self.prot_3d_adapter(b_3d_safe, return_confidence=False)
            a_3d_tok = self._validate_tensor(a_3d_tok, "a_3d_tok")
            b_3d_tok = self._validate_tensor(b_3d_tok, "b_3d_tok")
        else:
            a_3d_tok = torch.zeros(batch_size, 1, self.llm_dim, device=device)
            b_3d_tok = torch.zeros(batch_size, 1, self.llm_dim, device=device)

        # ----- 5. Contrastive loss (pre-CMON) -----
        contrastive_loss = torch.tensor(0.0, device=device)
        if self.training:
            cl_a = self.contrastive_loss(a_struct_tok, a_kg_tok, a_3d_tok)
            cl_b = self.contrastive_loss(b_struct_tok, b_kg_tok, b_3d_tok)
            contrastive_loss = (cl_a + cl_b) / 2.0

        # ----- 6. CMON -----
        a_digests = b_digests = None
        a_cmon_diag = b_cmon_diag = {}
        a_cmon_losses = b_cmon_losses = {}
        a_rel = b_rel = None
        if self.use_cmon and self.prot_cmon is not None:
            (a_tok_list, a_digests, a_cmon_diag, a_cmon_losses, a_rel) = self.prot_cmon(
                [a_struct_tok, a_kg_tok, a_3d_tok]
            )
            a_struct_tok, a_kg_tok, a_3d_tok = a_tok_list
            (b_tok_list, b_digests, b_cmon_diag, b_cmon_losses, b_rel) = self.prot_cmon(
                [b_struct_tok, b_kg_tok, b_3d_tok]
            )
            b_struct_tok, b_kg_tok, b_3d_tok = b_tok_list
            with torch.no_grad():
                if (~a_kg_missing).any():
                    a_cmon_diag["cmon_rel_k_real"] = a_rel[~a_kg_missing, 1].mean().item()
                if a_kg_missing.any():
                    a_cmon_diag["cmon_rel_k_comp"] = a_rel[a_kg_missing, 1].mean().item()
                if (~b_kg_missing).any():
                    b_cmon_diag["cmon_rel_k_real"] = b_rel[~b_kg_missing, 1].mean().item()
                if b_kg_missing.any():
                    b_cmon_diag["cmon_rel_k_comp"] = b_rel[b_kg_missing, 1].mean().item()

        # ----- 7. Sample-level modality dropout -----
        if self.training:
            r_a = torch.rand(batch_size, device=device)
            a_3d_drop = (r_a < 0.08).unsqueeze(-1).unsqueeze(-1)
            a_kg_drop = ((r_a >= 0.08) & (r_a < 0.15)).unsqueeze(-1).unsqueeze(-1)
            a_str_drop = ((r_a >= 0.15) & (r_a < 0.20)).unsqueeze(-1).unsqueeze(-1)
            a_struct_tok = a_struct_tok * (~a_str_drop).float()
            a_kg_tok = a_kg_tok * (~a_kg_drop).float()
            a_3d_tok = a_3d_tok * (~a_3d_drop).float()
            r_b = torch.rand(batch_size, device=device)
            b_3d_drop = (r_b < 0.08).unsqueeze(-1).unsqueeze(-1)
            b_kg_drop = ((r_b >= 0.08) & (r_b < 0.15)).unsqueeze(-1).unsqueeze(-1)
            b_str_drop = ((r_b >= 0.15) & (r_b < 0.20)).unsqueeze(-1).unsqueeze(-1)
            b_struct_tok = b_struct_tok * (~b_str_drop).float()
            b_kg_tok = b_kg_tok * (~b_kg_drop).float()
            b_3d_tok = b_3d_tok * (~b_3d_drop).float()

        # ----- 8. LLM forward -----
        as_start = self._get_marker_embedding("a_struct_start", batch_size, device).to(target_dtype)
        as_end = self._get_marker_embedding("a_struct_end", batch_size, device).to(target_dtype)
        ak_start = self._get_marker_embedding("a_kg_start", batch_size, device).to(target_dtype)
        ak_end = self._get_marker_embedding("a_kg_end", batch_size, device).to(target_dtype)
        ad_start = self._get_marker_embedding("a_3d_start", batch_size, device).to(target_dtype)
        ad_end = self._get_marker_embedding("a_3d_end", batch_size, device).to(target_dtype)
        bs_start = self._get_marker_embedding("b_struct_start", batch_size, device).to(target_dtype)
        bs_end = self._get_marker_embedding("b_struct_end", batch_size, device).to(target_dtype)
        bk_start = self._get_marker_embedding("b_kg_start", batch_size, device).to(target_dtype)
        bk_end = self._get_marker_embedding("b_kg_end", batch_size, device).to(target_dtype)
        bd_start = self._get_marker_embedding("b_3d_start", batch_size, device).to(target_dtype)
        bd_end = self._get_marker_embedding("b_3d_end", batch_size, device).to(target_dtype)

        if a_digests is not None and len(a_digests) == 3:
            a_struct_seq = torch.cat([as_start, a_digests[0].to(target_dtype), a_struct_tok.to(target_dtype), as_end], dim=1)
            a_kg_seq = torch.cat([ak_start, a_digests[1].to(target_dtype), a_kg_tok.to(target_dtype), ak_end], dim=1)
            a_3d_seq = torch.cat([ad_start, a_digests[2].to(target_dtype), a_3d_tok.to(target_dtype), ad_end], dim=1)
        else:
            a_struct_seq = torch.cat([as_start, a_struct_tok.to(target_dtype), as_end], dim=1)
            a_kg_seq = torch.cat([ak_start, a_kg_tok.to(target_dtype), ak_end], dim=1)
            a_3d_seq = torch.cat([ad_start, a_3d_tok.to(target_dtype), ad_end], dim=1)
        if b_digests is not None and len(b_digests) == 3:
            b_struct_seq = torch.cat([bs_start, b_digests[0].to(target_dtype), b_struct_tok.to(target_dtype), bs_end], dim=1)
            b_kg_seq = torch.cat([bk_start, b_digests[1].to(target_dtype), b_kg_tok.to(target_dtype), bk_end], dim=1)
            b_3d_seq = torch.cat([bd_start, b_digests[2].to(target_dtype), b_3d_tok.to(target_dtype), bd_end], dim=1)
        else:
            b_struct_seq = torch.cat([bs_start, b_struct_tok.to(target_dtype), bs_end], dim=1)
            b_kg_seq = torch.cat([bk_start, b_kg_tok.to(target_dtype), bk_end], dim=1)
            b_3d_seq = torch.cat([bd_start, b_3d_tok.to(target_dtype), bd_end], dim=1)

        prefix = torch.cat([a_struct_seq, a_kg_seq, a_3d_seq, b_struct_seq, b_kg_seq, b_3d_seq], dim=1)
        prefix = self._validate_tensor(prefix, "prefix")

        prompts = []
        for i in range(batch_size):
            has_a_kg = not bool(a_kg_completed[i].item()) if a_kg_completed.numel() > 0 else True
            has_b_kg = not bool(b_kg_completed[i].item()) if b_kg_completed.numel() > 0 else True
            has_a_3d = prot_3d_a is not None
            has_b_3d = prot_3d_b is not None
            p = self.prompt_composer.build_prompt(
                seq_a[i] if i < len(seq_a) else "",
                seq_b[i] if i < len(seq_b) else "",
                has_a_kg=has_a_kg,
                has_a_3d=has_a_3d,
                has_b_kg=has_b_kg,
                has_b_3d=has_b_3d,
            )
            prompts.append(p)
        tokenizer = self.llm_encoder.tokenizer
        text_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.prompt_max_length).to(device)
        text_embeds = self.llm_encoder.model.get_input_embeddings()(text_inputs.input_ids).to(target_dtype)
        text_embeds = self._validate_tensor(text_embeds, "text_embeds")
        combined = torch.cat([prefix, text_embeds], dim=1)
        combined = self._validate_tensor(combined, "inputs_embeds")
        prefix_mask = torch.ones(batch_size, prefix.size(1), device=device, dtype=text_inputs.attention_mask.dtype)
        attention_mask_2d = torch.cat([prefix_mask, text_inputs.attention_mask], dim=1)

        outputs = self.llm_encoder.model(
            inputs_embeds=combined,
            attention_mask=attention_mask_2d,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]

        # ----- 9. Pooling -----
        last_pos = attention_mask_2d.sum(dim=1).clamp(min=1).long() - 1
        batch_indices = torch.arange(batch_size, device=device)
        last_token = last_hidden[batch_indices, last_pos, :]
        if self.use_attention_pooling:
            attn_pooled = self.attention_pooler(last_hidden, attention_mask_2d)
            ratio = torch.sigmoid(self.attn_pool_ratio)
            pooled = (ratio * attn_pooled + (1.0 - ratio) * last_token).float()
        else:
            pooled = last_token.float()

        # ----- 10. Classification -----
        logits = self.classification_head(pooled)
        logits = torch.clamp(logits, -self.logits_clamp_abs, self.logits_clamp_abs)

        aux_output = {
            "contrastive_loss": contrastive_loss,
            "a_cmsa_loss": a_cmsa_loss,
            "b_cmsa_loss": b_cmsa_loss,
            "a_cmsa_diag": a_cmsa_diag,
            "b_cmsa_diag": b_cmsa_diag,
            "a_cmon_diag": a_cmon_diag,
            "b_cmon_diag": b_cmon_diag,
            "a_kg_completion_ratio": a_kg_completed.float().mean().item(),
            "b_kg_completion_ratio": b_kg_completed.float().mean().item(),
        }
        if self.training:
            aux_output["a_cmon_losses"] = a_cmon_losses
            aux_output["b_cmon_losses"] = b_cmon_losses

        return logits, aux_output


def create_ppi_model(
    config: Dict,
    num_classes: int = 7,
    kge_bank: torch.Tensor = None,
    kge_keys: List[str] = None,
) -> PPIModel:
    """创建 PPI 模型，kge_bank 为 tensor 时直接传入 SA-CMSA"""
    if kge_bank is not None and not isinstance(kge_bank, torch.Tensor) and kge_keys is not None:
        import numpy as np
        if isinstance(kge_bank, dict):
            keys = sorted(kge_bank.keys())
            kge_bank = torch.tensor(np.stack([kge_bank[k] for k in keys]), dtype=torch.float32)
        elif isinstance(kge_bank, np.ndarray):
            kge_bank = torch.tensor(kge_bank, dtype=torch.float32)
    if kge_bank is not None and isinstance(kge_bank, torch.Tensor):
        logger.info(f"[create_ppi_model] KGE bank: {kge_bank.shape[0]}x{kge_bank.shape[1]}")
    return PPIModel(config=config, num_classes=num_classes, kge_bank=kge_bank)
