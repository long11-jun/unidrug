from .qwen_encoder import QwenEncoder
from .tokenmol_encoder import TokenMolEncoder

SUPPORTED_LLM_ENCODER = {
    "qwen": QwenEncoder,
    "tokenmol": TokenMolEncoder,
}
