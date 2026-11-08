import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)

from models.task_model.dp import (
    SparseAttentionCMSA,
    CMONModule,
    AdapterTokenGenerator,
    AttentionPooling,
    ResidualPredictionHead,
)
from models.task_model.dti import (
    ModalContrastiveLoss,
)


class DDIPromptComposer:
    def __init__(self, config: Dict):
        self.config = config
        self.drug_text_map = {}
        self.scenario = config.get("scenario", "S1")
        
        drug_text_path = config.get("drug_text_path")
        if drug_text_path and os.path.exists(drug_text_path):
            self._load_text_file(drug_text_path)

        kg_path = config.get("kg_path")
        if kg_path:
            drug_json = os.path.join(kg_path, "drug.json")
            if os.path.exists(drug_json):
                self._load_text_file(drug_json)

    def _load_text_file(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            count = 0
            for key, value in data.items():
                if key in self.drug_text_map:
                    continue
                if isinstance(value, dict):
                    text = value.get("text", value.get("description", ""))
                    smiles = value.get("SMILES", "")
                    if text and text != "text":
                        self.drug_text_map[key] = text
                        if smiles:
                            self.drug_text_map[smiles] = text
                            try:
                                from rdkit import Chem
                                mol = Chem.MolFromSmiles(smiles)
                                if mol:
                                    canon = Chem.MolToSmiles(mol, canonical=True)
                                    self.drug_text_map[canon] = text
                            except Exception:
                                pass
                        count += 1
                elif isinstance(value, str) and value:
                    self.drug_text_map[key] = value
                    count += 1
            logger.info(f"[DDIPromptComposer] Loaded {count} drug texts from {path}")
        except Exception as e:
            logger.warning(f"[DDIPromptComposer] Failed to load {path}: {e}")

    def get_drug_text(self, smiles: str, drug_id: str = None) -> str:
        if drug_id and drug_id in self.drug_text_map:
            return self.drug_text_map[drug_id]
        if smiles and smiles in self.drug_text_map:
            return self.drug_text_map[smiles]
        if smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canon = Chem.MolToSmiles(mol, canonical=True)
                    if canon in self.drug_text_map:
                        return self.drug_text_map[canon]
            except Exception:
                pass
        return self._generate_drug_text(smiles)

    def _generate_drug_text(self, smiles: str) -> str:
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
            violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            if violations == 0:
                desc += ", drug-like"
            return desc
        except Exception:
            return "Small molecule compound"

    def build_prompt(
        self,
        smiles_a: str,
        smiles_b: str,
        has_a_kg: bool = True,
        has_a_mol: bool = True,
        has_b_kg: bool = True,
        has_b_mol: bool = True,
    ) -> str:
        text_a = self.get_drug_text(smiles_a)
        text_b = self.get_drug_text(smiles_b)
        text_a = text_a[:200] + "..." if len(text_a) > 200 else text_a
        text_b = text_b[:200] + "..." if len(text_b) > 200 else text_b

        parts = []
        
        parts.append("Predict drug-drug interaction type using multimodal evidence.")

        parts.append(f"Drug A: {text_a}")
        parts.append(f"Drug B: {text_b}")

        parts.append("Drug A input reliability:")
        parts.append("[Structure] computed from molecular graph — high confidence.")
        if has_a_mol:
            parts.append("[MolToken] derived from 3D torsion angles — high confidence.")
        else:
            parts.append("[MolToken] cross-modal completion — moderate confidence.")
        if has_a_kg:
            parts.append("[KG] retrieved from knowledge graph — high confidence.")
        else:
            parts.append("[KG] cross-modal completion — moderate confidence.")

        parts.append("Drug B input reliability:")
        parts.append("[Structure] computed from molecular graph — high confidence.")
        if has_b_mol:
            parts.append("[MolToken] derived from 3D torsion angles — high confidence.")
        else:
            parts.append("[MolToken] cross-modal completion — moderate confidence.")
        if has_b_kg:
            parts.append("[KG] retrieved from knowledge graph — high confidence.")
        else:
            parts.append("[KG] cross-modal completion — moderate confidence.")

        a_inferred = (not has_a_kg) or (not has_a_mol)
        b_inferred = (not has_b_kg) or (not has_b_mol)
        if not a_inferred and not b_inferred:
            parts.append("All modalities are directly observed; integrate with equal weight.")
        else:
            inferred = []
            if not has_a_kg: inferred.append("A-KG")
            if not has_a_mol: inferred.append("A-MolToken")
            if not has_b_kg: inferred.append("B-KG")
            if not has_b_mol: inferred.append("B-MolToken")
            parts.append(
                f"{', '.join(inferred)} were inferred, not observed. "
                "Rely primarily on observed modalities; use inferred features as supporting evidence."
            )

        parts.append("Prediction:")
        return " ".join(parts)


class DDIModel(nn.Module):
    """Two drug branches, each with SA-CMSA and CMON, then a shared adapter."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int = 4,
        kge_bank: torch.Tensor = None,
        kge_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pred_dim = num_classes
        net_cfg = config.get("network", config)
        
        ablation_str = net_cfg.get("ablation", "")
        self.ablation_flags = set(a.strip() for a in ablation_str.split(",") if a.strip())
        if self.ablation_flags:
            logger.info(f"[DDIModel] Ablation flags: {self.ablation_flags}")
        if "no_cmon" in self.ablation_flags:
            net_cfg.setdefault("mars", {})["enabled"] = False
        if "no_cmsa" in self.ablation_flags:
            net_cfg.setdefault("cmsa", {})["enabled"] = False

        self._init_encoders(config)
        self._init_llm(config)
        self._init_adapters(config)
        self._init_sacmsa(config, kge_bank)
        self._init_cmon(config)
        self._init_modality_markers()

        self.contrastive_loss = ModalContrastiveLoss(
            hidden_dim=self.llm_dim, proj_dim=256, temperature=0.15
        )

        head_dropout = net_cfg.get("head_dropout", 0.15)
        self.classification_head = ResidualPredictionHead(
            self.llm_dim, self.llm_dim // 2, num_classes, dropout=head_dropout
        )

        if net_cfg.get("use_lora", True):
            self._enable_lora(net_cfg)
        self._freeze_pretrained()

        self.use_feature_noise = net_cfg.get("use_feature_noise", False)
        self.feature_noise_std = net_cfg.get("feature_noise_std", 0.02)
        self.logits_clamp_abs = net_cfg.get("stability", {}).get("logits_clamp_abs", 10.0)

        self._log_model_info()

    def _init_encoders(self, config):
        net_cfg = config.get("network", config)
        drug_cfg = net_cfg.get("drug", {})

        from models import SUPPORTED_MOL_ENCODER
        struct_cfg = drug_cfg.get("structure", {})
        if "config_path" in struct_cfg:
            import json as _json
            with open(struct_cfg["config_path"], "r") as f:
                from ...utils.paths import expand_asset_paths
                enc_cfg = expand_asset_paths(_json.load(f))
            self.structure_encoder = SUPPORTED_MOL_ENCODER[struct_cfg["name"]](enc_cfg)
            if "init_checkpoint" in enc_cfg:
                ckpt = torch.load(enc_cfg["init_checkpoint"], map_location="cpu", weights_only=False)
                self.structure_encoder.load_state_dict(ckpt, strict=False)
        else:
            self.structure_encoder = SUPPORTED_MOL_ENCODER[struct_cfg.get("name", "gin")](struct_cfg)
        self.struct_dim = self.structure_encoder.output_dim

        self.kg_dim = drug_cfg.get("kg", {}).get("embedding_dim", 256)

        self.use_moltoken = net_cfg.get("use_moltoken", False)
        if "no_moltoken" in self.ablation_flags:
            self.use_moltoken = False
        self.moltoken_encoder = None
        self.moltoken_dim = 768

        if self.use_moltoken:
            try:
                from models.llm_encoder import TokenMolEncoder
                self.moltoken_encoder = TokenMolEncoder(net_cfg.get("moltoken", {}))
                self.moltoken_dim = self.moltoken_encoder.output_dim
                for p in self.moltoken_encoder.model.parameters():
                    p.requires_grad = False
                self.moltoken_encoder.model.eval()
                logger.info(f"[DDI] MolToken enabled: dim={self.moltoken_dim}")
            except Exception as e:
                logger.warning(f"[DDI] MolToken init failed: {e}")
                self.use_moltoken = False

        logger.info(f"[DDI] Encoders: struct={self.struct_dim}, kg={self.kg_dim}, "
                     f"moltoken={'ON' if self.use_moltoken else 'OFF'}({self.moltoken_dim})")

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

        kg_path = net_cfg.get("kg_path", net_cfg.get("drug", {}).get("kg", {}).get(
            "kg_path", os.path.join(os.environ.get("UNIDRUG_ASSET_ROOT", "/data-nxs/songlongjun/KEDD_temporal-master"), "assets/kg/BMKG")))
        self.prompt_composer = DDIPromptComposer({
            "drug_text_path": net_cfg.get("drug_text_path"),
            "kg_path": kg_path,
            "scenario": net_cfg.get("scenario", "S1"),
        })
        self.prompt_max_length = net_cfg.get("prompt_max_length", 256)

    def _init_adapters(self, config):
        net_cfg = config.get("network", config)
        adapter_dropout = net_cfg.get("adapter_dropout", 0.25)
        num_struct = net_cfg.get("num_struct_tokens", 40)
        num_kg = net_cfg.get("num_kg_tokens", 28)
        num_mol = net_cfg.get("num_moltoken_tokens", 32)

        self.struct_adapter = AdapterTokenGenerator(
            self.struct_dim, self.llm_dim, num_tokens=num_struct, num_layers=4, dropout=adapter_dropout
        )
        self.kg_adapter = AdapterTokenGenerator(
            self.kg_dim, self.llm_dim, num_tokens=num_kg, num_layers=4, dropout=adapter_dropout
        )
        self.mol_adapter = None
        if self.use_moltoken:
            self.mol_adapter = AdapterTokenGenerator(
                self.moltoken_dim, self.llm_dim, num_tokens=num_mol, num_layers=4, dropout=adapter_dropout
            )
        logger.info(f"[DDI] Adapters (shared): struct→{num_struct}, kg→{num_kg}, mol→{num_mol}")

    def _init_sacmsa(self, config, kge_bank):
        net_cfg = config.get("network", config)
        cmsa_cfg = net_cfg.get("cmsa", net_cfg.get("dec", {}))
        use_cmsa = cmsa_cfg.get("enabled", True) and kge_bank is not None
        self.use_cmsa = use_cmsa
        self.drug_cmsa = None
        if use_cmsa:
            mol_dim = self.moltoken_dim if self.use_moltoken else self.struct_dim
            self.drug_cmsa = SparseAttentionCMSA(
                modal_dims=[self.struct_dim, self.kg_dim, mol_dim],
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
            logger.info(f"[DDI] SA-CMSA enabled (shared): kge_bank {kge_bank.shape[0]}x{kge_bank.shape[1]}")
        else:
            logger.info("[DDI] SA-CMSA disabled")

    def _init_cmon(self, config):
        net_cfg = config.get("network", config)
        mars_cfg = net_cfg.get("mars", {})
        self.use_cmon = mars_cfg.get("enabled", False)
        self.drug_cmon = None
        if self.use_cmon:
            self.drug_cmon = CMONModule(
                llm_dim=self.llm_dim, n_modalities=3, sub_dim=mars_cfg.get("sub_dim", 64),
                ablation_flags=self.ablation_flags,
            )
            logger.info("[DDI] CMON enabled (shared)")
        else:
            logger.info("[DDI] CMON disabled")

    def _init_modality_markers(self):
        self.modality_markers = {
            "a_struct_start": "[a_struct_start]", "a_struct_end": "[a_struct_end]",
            "a_kg_start": "[a_kg_start]", "a_kg_end": "[a_kg_end]",
            "a_mol_start": "[a_mol_start]", "a_mol_end": "[a_mol_end]",
            "b_struct_start": "[b_struct_start]", "b_struct_end": "[b_struct_end]",
            "b_kg_start": "[b_kg_start]", "b_kg_end": "[b_kg_end]",
            "b_mol_start": "[b_mol_start]", "b_mol_end": "[b_mol_end]",
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
        logger.info(f"[DDI] LoRA r={net_cfg.get('lora_rank', 32)}")

    def _freeze_pretrained(self):
        for name, param in self.llm_encoder.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        enc = self.structure_encoder
        if hasattr(enc, "main_model") and hasattr(enc.main_model, "gnns"):
            for p in enc.parameters():
                p.requires_grad = False
            for i in range(3, 5):
                if i < len(enc.main_model.gnns):
                    for p in enc.main_model.gnns[i].parameters():
                        p.requires_grad = True
                    for p in enc.main_model.batch_norms[i].parameters():
                        p.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[DDI] Freeze: trainable {trainable:,} / {total:,}")

    def _log_model_info(self):
        logger.info(
            f"[DDIModel] Classes={self.num_classes}, SA-CMSA={self.use_cmsa}, CMON={self.use_cmon}"
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
            tensor = tensor.clamp(-1e4, 1e4)
        return tensor

    def _get_kg_feat(self, data: Dict, dim: int, device: torch.device) -> torch.Tensor:
        if isinstance(data, dict) and data.get("kg") is not None:
            kg = data["kg"].to(device).float()
            if kg.dim() == 3:
                kg = kg.squeeze(1)
            return kg
        batch_size = 1
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break
                if hasattr(v, "batch"):
                    batch_size = v.batch.max().item() + 1
                    break
        return torch.zeros(batch_size, dim, device=device)

    def _encode_drug_structure(self, drug_batch: Dict, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
        data = drug_batch.get("structure", drug_batch) if isinstance(drug_batch, dict) else drug_batch
        if hasattr(self.structure_encoder, "encode_structure"):
            h = self.structure_encoder.encode_structure(data)
        elif hasattr(self.structure_encoder, "encode_mol"):
            h = self.structure_encoder.encode_mol(data)
        else:
            h = self.structure_encoder(data)
        if isinstance(h, tuple):
            h = h[0]
        h = torch.nan_to_num(h, nan=0.0)

        smiles = []
        if isinstance(drug_batch, dict) and "smiles" in drug_batch:
            raw = drug_batch["smiles"]
            if isinstance(raw, list):
                smiles = [str(s) for s in raw]
            elif isinstance(raw, str):
                smiles = [raw] * h.size(0)
        if not smiles:
            smiles = [""] * h.size(0)
        return h.to(device), smiles

    def _encode_moltoken(self, drug_batch: Dict, smiles_list: List[str], batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_moltoken or self.moltoken_encoder is None:
            return None
        try:
            cached_seq = drug_batch.get("tokenmol_seq", None) if isinstance(drug_batch, dict) else None
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
            with torch.no_grad():
                feat, _ = self.moltoken_encoder.encode_token_lists(token_lists, return_lm_loss=False)
            return feat.to(device)
        except Exception:
            return None

    def forward(
        self,
        drug_a_batch: Dict,
        drug_b_batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        device = next(self.parameters()).device
        target_dtype = next(self.llm_encoder.model.parameters()).dtype

        # 1. Encode both drugs
        struct_a, smiles_a = self._encode_drug_structure(drug_a_batch, device)
        batch_size = struct_a.size(0)
        struct_b, smiles_b = self._encode_drug_structure(drug_b_batch, device)
        mol_a = self._encode_moltoken(drug_a_batch, smiles_a, batch_size, device)
        mol_b = self._encode_moltoken(drug_b_batch, smiles_b, batch_size, device)

        # 2. Feature noise
        noise_std = self.feature_noise_std if self.training and self.use_feature_noise else 0.0
        if noise_std > 0:
            struct_a = struct_a + torch.randn_like(struct_a) * noise_std
            struct_b = struct_b + torch.randn_like(struct_b) * noise_std
            if mol_a is not None:
                mol_a = mol_a + torch.randn_like(mol_a) * noise_std
            if mol_b is not None:
                mol_b = mol_b + torch.randn_like(mol_b) * noise_std

        # 3. KG + SA-CMSA
        kg_a = self._get_kg_feat(drug_a_batch, self.kg_dim, device)
        kg_b = self._get_kg_feat(drug_b_batch, self.kg_dim, device)
        a_kg_missing = torch.all(kg_a == 0, dim=1)
        b_kg_missing = torch.all(kg_b == 0, dim=1)

        mol_dim = self.moltoken_dim if self.use_moltoken else self.struct_dim
        zero_mol = torch.zeros(batch_size, mol_dim, device=device)
        a_mol_safe = mol_a if mol_a is not None else zero_mol
        b_mol_safe = mol_b if mol_b is not None else zero_mol

        a_cmsa_loss = torch.tensor(0.0, device=device)
        b_cmsa_loss = torch.tensor(0.0, device=device)
        a_cmsa_diag, b_cmsa_diag = {}, {}
        a_kg_completed = a_kg_missing.clone()
        b_kg_completed = b_kg_missing.clone()

        if self.use_cmsa and self.drug_cmsa is not None:
            a_missing = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
            a_missing[:, 1] = a_kg_missing
            if mol_a is None:
                a_missing[:, 2] = True
            feats_a, comp_a, a_cmsa_loss, a_cmsa_diag, _ = self.drug_cmsa(
                [struct_a, kg_a, a_mol_safe], a_missing
            )
            struct_a, kg_a, a_mol_safe = feats_a
            a_kg_completed = comp_a[:, 1]

            b_missing = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
            b_missing[:, 1] = b_kg_missing
            if mol_b is None:
                b_missing[:, 2] = True
            feats_b, comp_b, b_cmsa_loss, b_cmsa_diag, _ = self.drug_cmsa(
                [struct_b, kg_b, b_mol_safe], b_missing
            )
            struct_b, kg_b, b_mol_safe = feats_b
            b_kg_completed = comp_b[:, 1]

        # 4. Adapters → tokens
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

        a_mol_safe = self._validate_tensor(a_mol_safe, "a_mol")
        b_mol_safe = self._validate_tensor(b_mol_safe, "b_mol")
        if self.mol_adapter is not None:
            a_mol_tok = self.mol_adapter(a_mol_safe, return_confidence=False)
            b_mol_tok = self.mol_adapter(b_mol_safe, return_confidence=False)
            a_mol_tok = self._validate_tensor(a_mol_tok, "a_mol_tok")
            b_mol_tok = self._validate_tensor(b_mol_tok, "b_mol_tok")
        else:
            a_mol_tok = torch.zeros(batch_size, 1, self.llm_dim, device=device)
            b_mol_tok = torch.zeros(batch_size, 1, self.llm_dim, device=device)

        # 5. Contrastive loss (pre-CMON)
        contrastive_loss = torch.tensor(0.0, device=device)
        if self.training:
            cl_a = self.contrastive_loss(a_struct_tok, a_kg_tok, a_mol_tok)
            cl_b = self.contrastive_loss(b_struct_tok, b_kg_tok, b_mol_tok)
            contrastive_loss = (cl_a + cl_b) / 2.0

        # 6. CMON
        a_digests = b_digests = None
        a_cmon_diag, b_cmon_diag = {}, {}
        a_cmon_losses, b_cmon_losses = {}, {}
        a_rel = b_rel = None
        if self.use_cmon and self.drug_cmon is not None:
            (a_tok_list, a_digests, a_cmon_diag, a_cmon_losses, a_rel) = self.drug_cmon(
                [a_struct_tok, a_kg_tok, a_mol_tok]
            )
            a_struct_tok, a_kg_tok, a_mol_tok = a_tok_list
            (b_tok_list, b_digests, b_cmon_diag, b_cmon_losses, b_rel) = self.drug_cmon(
                [b_struct_tok, b_kg_tok, b_mol_tok]
            )
            b_struct_tok, b_kg_tok, b_mol_tok = b_tok_list
            with torch.no_grad():
                if (~a_kg_missing).any():
                    a_cmon_diag["cmon_rel_k_real"] = a_rel[~a_kg_missing, 1].mean().item()
                if a_kg_missing.any():
                    a_cmon_diag["cmon_rel_k_comp"] = a_rel[a_kg_missing, 1].mean().item()
                if (~b_kg_missing).any():
                    b_cmon_diag["cmon_rel_k_real"] = b_rel[~b_kg_missing, 1].mean().item()
                if b_kg_missing.any():
                    b_cmon_diag["cmon_rel_k_comp"] = b_rel[b_kg_missing, 1].mean().item()

        # 7. Sample-level modality dropout
        if self.training:
            r_a = torch.rand(batch_size, device=device)
            a_mol_drop = (r_a < 0.08).unsqueeze(-1).unsqueeze(-1)
            a_kg_drop = ((r_a >= 0.08) & (r_a < 0.15)).unsqueeze(-1).unsqueeze(-1)
            a_str_drop = ((r_a >= 0.15) & (r_a < 0.20)).unsqueeze(-1).unsqueeze(-1)
            a_struct_tok = a_struct_tok * (~a_str_drop).float()
            a_kg_tok = a_kg_tok * (~a_kg_drop).float()
            a_mol_tok = a_mol_tok * (~a_mol_drop).float()

            r_b = torch.rand(batch_size, device=device)
            b_mol_drop = (r_b < 0.08).unsqueeze(-1).unsqueeze(-1)
            b_kg_drop = ((r_b >= 0.08) & (r_b < 0.15)).unsqueeze(-1).unsqueeze(-1)
            b_str_drop = ((r_b >= 0.15) & (r_b < 0.20)).unsqueeze(-1).unsqueeze(-1)
            b_struct_tok = b_struct_tok * (~b_str_drop).float()
            b_kg_tok = b_kg_tok * (~b_kg_drop).float()
            b_mol_tok = b_mol_tok * (~b_mol_drop).float()

        # 8. LLM forward
        as_s = self._get_marker_embedding("a_struct_start", batch_size, device).to(target_dtype)
        as_e = self._get_marker_embedding("a_struct_end", batch_size, device).to(target_dtype)
        ak_s = self._get_marker_embedding("a_kg_start", batch_size, device).to(target_dtype)
        ak_e = self._get_marker_embedding("a_kg_end", batch_size, device).to(target_dtype)
        am_s = self._get_marker_embedding("a_mol_start", batch_size, device).to(target_dtype)
        am_e = self._get_marker_embedding("a_mol_end", batch_size, device).to(target_dtype)
        bs_s = self._get_marker_embedding("b_struct_start", batch_size, device).to(target_dtype)
        bs_e = self._get_marker_embedding("b_struct_end", batch_size, device).to(target_dtype)
        bk_s = self._get_marker_embedding("b_kg_start", batch_size, device).to(target_dtype)
        bk_e = self._get_marker_embedding("b_kg_end", batch_size, device).to(target_dtype)
        bm_s = self._get_marker_embedding("b_mol_start", batch_size, device).to(target_dtype)
        bm_e = self._get_marker_embedding("b_mol_end", batch_size, device).to(target_dtype)

        def _build_seq(s_start, s_end, tok, digest):
            if digest is not None:
                return torch.cat([s_start, digest.to(target_dtype), tok.to(target_dtype), s_end], dim=1)
            return torch.cat([s_start, tok.to(target_dtype), s_end], dim=1)

        a_struct_seq = _build_seq(as_s, as_e, a_struct_tok, a_digests[0] if a_digests else None)
        a_kg_seq = _build_seq(ak_s, ak_e, a_kg_tok, a_digests[1] if a_digests else None)
        a_mol_seq = _build_seq(am_s, am_e, a_mol_tok, a_digests[2] if a_digests else None)
        b_struct_seq = _build_seq(bs_s, bs_e, b_struct_tok, b_digests[0] if b_digests else None)
        b_kg_seq = _build_seq(bk_s, bk_e, b_kg_tok, b_digests[1] if b_digests else None)
        b_mol_seq = _build_seq(bm_s, bm_e, b_mol_tok, b_digests[2] if b_digests else None)

        prefix = torch.cat([a_struct_seq, a_kg_seq, a_mol_seq, b_struct_seq, b_kg_seq, b_mol_seq], dim=1)
        prefix = self._validate_tensor(prefix, "prefix")

        prompts = []
        for i in range(batch_size):
            has_a_kg = not bool(a_kg_completed[i].item()) if a_kg_completed.numel() > 0 else True
            has_b_kg = not bool(b_kg_completed[i].item()) if b_kg_completed.numel() > 0 else True
            has_a_mol = mol_a is not None
            has_b_mol = mol_b is not None
            p = self.prompt_composer.build_prompt(
                smiles_a[i] if i < len(smiles_a) else "",
                smiles_b[i] if i < len(smiles_b) else "",
                has_a_kg=has_a_kg, has_a_mol=has_a_mol,
                has_b_kg=has_b_kg, has_b_mol=has_b_mol,
            )
            prompts.append(p)
        
        tokenizer = self.llm_encoder.tokenizer
        text_inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.prompt_max_length).to(device)
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

        # 9. Pooling
        last_pos = attention_mask_2d.sum(dim=1).clamp(min=1).long() - 1
        batch_indices = torch.arange(batch_size, device=device)
        last_token = last_hidden[batch_indices, last_pos, :]
        if self.use_attention_pooling:
            attn_pooled = self.attention_pooler(last_hidden, attention_mask_2d)
            ratio = torch.sigmoid(self.attn_pool_ratio)
            pooled = (ratio * attn_pooled + (1.0 - ratio) * last_token).float()
        else:
            pooled = last_token.float()

        # 10. Classification
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


def create_ddi_model(
    config: Dict,
    num_classes: int = 4,
    kge_bank: torch.Tensor = None,
    kge_keys: Optional[List[str]] = None,
        **kwargs,
) -> DDIModel:
    import numpy as np
    if kge_bank is not None and not isinstance(kge_bank, torch.Tensor):
        if isinstance(kge_bank, dict):
            keys = sorted(kge_bank.keys())
            kge_bank = torch.tensor(np.stack([kge_bank[k] for k in keys]), dtype=torch.float32)
        elif isinstance(kge_bank, np.ndarray):
            kge_bank = torch.tensor(kge_bank, dtype=torch.float32)
    if kge_bank is not None and isinstance(kge_bank, torch.Tensor):
        logger.info(f"[create_ddi_model] KGE bank: {kge_bank.shape[0]}x{kge_bank.shape[1]}")
    return DDIModel(config=config, num_classes=num_classes, kge_bank=kge_bank, kge_keys=kge_keys, **kwargs)
