"""
Token-Mol 预训练模型加载器

支持：
1. 自定义 CSV 词表（Token-Mol 原生格式）
2. 标准 HuggingFace tokenizer（如果目录下有 tokenizer.json）
3. 与 DPModel 的 prefix embedding 注入方式兼容
"""
import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from ..base_models import TextEncoder
from .tokenmol_tokenizer import TokenMolTokenizer

logger = logging.getLogger(__name__)


class TokenMolEncoder(TextEncoder):
    """
    Token-Mol 预训练模型加载器（decoder-only, GPT2LMHeadModel）。

    配置参数：
    - model_name: 模型目录路径（包含 config.json, pytorch_model.bin）
    - vocab_file: CSV 词表路径（可选，不指定则尝试 AutoTokenizer）
    - device: 目标设备
    - max_length: 最大序列长度
    - torch_dtype: 数据类型（float16/float32/bfloat16）
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("[TokenMolEncoder] `model_name` is required (path to model directory).")

        self.max_length = int(config.get("max_length", 200))  # Token-Mol 默认 n_ctx=200
        self.device = config.get("device", "cuda:0")
        self.trust_remote_code = bool(config.get("trust_remote_code", False))

        # dtype 映射
        dtype_str = config.get("torch_dtype", "float32")
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(dtype_str).lower(), torch.float32)

        # ========== 加载 Tokenizer ==========
        vocab_file = config.get("vocab_file")
        if vocab_file and os.path.exists(vocab_file):
            # 使用自定义 CSV 词表
            logger.info(f"[TokenMolEncoder] Loading custom vocab from: {vocab_file}")
            self.tokenizer = TokenMolTokenizer(vocab_file)
        else:
            # 尝试标准 AutoTokenizer
            try:
                logger.info(f"[TokenMolEncoder] Trying AutoTokenizer from: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_fast=True,
                )
            except Exception as e:
                # 最后尝试在 model_name 目录下找 vocab.csv / vocab.txt
                vocab_candidates = [
                    os.path.join(self.model_name, "vocab.csv"),
                    os.path.join(self.model_name, "torsion_voc_property.csv"),
                    os.path.join(self.model_name, "vocab.txt"),
                ]
                found = None
                for vf in vocab_candidates:
                    if os.path.exists(vf):
                        found = vf
                        break
                if found:
                    logger.info(f"[TokenMolEncoder] Found vocab file: {found}")
                    self.tokenizer = TokenMolTokenizer(found)
                else:
                    raise ValueError(
                        f"[TokenMolEncoder] Cannot load tokenizer. "
                        f"Please provide `vocab_file` in config or place vocab.csv in {self.model_name}. "
                        f"Original error: {e}"
                    )
        
        # 确保 pad_token 存在
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token'):
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # ========== 加载模型 ==========
        logger.info(f"[TokenMolEncoder] Loading model from: {self.model_name}")
        
        # 检查是否有 pytorch_model.bin
        model_bin = os.path.join(self.model_name, "pytorch_model.bin")
        config_json = os.path.join(self.model_name, "config.json")
        
        if os.path.exists(model_bin) and os.path.exists(config_json):
            # 从本地文件加载
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            raise FileNotFoundError(
                f"[TokenMolEncoder] Model files not found. "
                f"Expected: {model_bin} and {config_json}"
            )

        # 移动到目标设备
        if self.device != "cpu" and torch.cuda.is_available():
            try:
                self.model = self.model.to(self.device)
                logger.info(f"[TokenMolEncoder] Model loaded on {self.device}")
            except torch.cuda.OutOfMemoryError:
                logger.warning("[TokenMolEncoder] CUDA OOM, falling back to CPU")
                self.device = "cpu"
                self.model = self.model.to("cpu")
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")
            logger.info("[TokenMolEncoder] Model loaded on CPU")

        # 获取隐藏层维度
        self.output_dim = getattr(self.model.config, "hidden_size", 768)
        self.n_embd = self.output_dim
        self.vocab_size = getattr(self.model.config, "vocab_size", len(self.tokenizer))
        
        logger.info(
            f"[TokenMolEncoder] Loaded: hidden_size={self.output_dim}, "
            f"vocab_size={self.vocab_size}, max_length={self.max_length}"
        )

    def encode_token_lists(
        self,
        token_lists,
        return_lm_loss: bool = False,
    ):
        """
        输入：token_lists: List[List[str]]
        输出：
        - pooled: (B, hidden)
        - lm_loss: Optional[Tensor]
        """
        # 编码成 ids
        input_ids = []
        for tokens in token_lists:
            ids = self.tokenizer.encode_tokens(
                tokens,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
            )
            input_ids.append(ids)

        # padding
        max_len = max(len(x) for x in input_ids) if input_ids else 1
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        padded = []
        attn = []
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_id] * pad_len)
            attn.append([1] * len(ids) + [0] * pad_len)
        # 重要：不要使用 self.device（它来自 config，可能与外部 model.to(...) 后的真实设备不一致）
        # 统一跟随当前 Token-Mol 模型权重所在设备，避免 cuda:0 vs cuda:4 报错
        dev = next(self.model.parameters()).device
        input_ids = torch.tensor(padded, dtype=torch.long, device=dev)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=dev)

        labels = None
        if return_lm_loss:
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, -100)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        last = out.hidden_states[-1]
        pooled = (last * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled, (out.loss if return_lm_loss else None)

    def embed_only_token_lists(self, token_lists):
        """
        轻量模式：只用 Token-Mol 的 tokenizer + input embedding(wte) 做表征，不跑 Transformer。

        适用场景：
        - 你希望保留 Token-Mol 词表对齐（SMILES/3D token 的离散语义）
        - 但不希望引入第二个 Transformer backbone 的计算/不对称复杂度

        返回：
        - pooled: (B, hidden_size) 由输入 embedding 按 attention_mask 做 mean pooling
        - lm_loss: None（embed-only 不支持 next-token 自监督）
        """
        # 编码成 ids
        input_ids = []
        for tokens in token_lists:
            ids = self.tokenizer.encode_tokens(
                tokens,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
            )
            input_ids.append(ids)

        # padding
        max_len = max(len(x) for x in input_ids) if input_ids else 1
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        padded = []
        attn = []
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_id] * pad_len)
            attn.append([1] * len(ids) + [0] * pad_len)
        # 同 encode_token_lists：跟随模型权重设备
        dev = next(self.model.parameters()).device
        input_ids = torch.tensor(padded, dtype=torch.long, device=dev)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=dev)

        # wte embedding
        emb_layer = self.model.get_input_embeddings()
        # input_ids 与 embedding 权重必须在同一设备
        x = emb_layer(input_ids)  # (B, L, D)
        mask_f = attention_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return pooled, None

    def forward(self, text_inputs):
        """
        兼容接口：返回最后层 pooled 表示。
        DPModel 通常不走这里，而是直接调用 self.model(..., output_hidden_states=True)。
        """
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        
        inputs = self.tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # 跟随模型权重设备（避免外部 .to(...) 后不一致）
        dev = next(self.model.parameters()).device
        inputs = inputs.to(dev)
        
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            last = out.hidden_states[-1]
            mask = inputs.get("attention_mask", None)
            if mask is None:
                pooled = last.mean(dim=1)
            else:
                pooled = (last * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled

    def get_input_embeddings(self):
        """获取输入 embedding 层（用于 prefix 注入）"""
        return self.model.get_input_embeddings()

    def encode_text(self, text):
        """
        实现抽象方法：编码文本（SMILES 或 token 序列）
        返回 pooled 表示 (batch, hidden_size)
        """
        return self.forward(text)
