from .mol_encoder import *
from .protein_encoder import *
from .knowledge_encoder import *
from .text_encoder import *
from .multimodal_encoder import *
from .llm_encoder import *

SUPPORTED_MOL_ENCODER = {
    "drug_cnn": DrugCNN,
    "tgsa": GINTGSA,
    "graphcl": GraphCL,
    "graphmvp": GraphMVP,
    "molclr": MolCLR,
    "mgnn": MGNN,
    "bert": MolBERT,
    "biomedgpt": BioMedGPT,
    "kv-plm": KVPLM,
    "momu": MoMu,
    "molfm": MolFM
}

SUPPORTED_PROTEIN_ENCODER = {
    "prot_cnn": ProtCNN,
    "cnn_gru": CNNGRU,
    "mcnn": MCNN,
    "pipr": CNNPIPR,
    "prottrans": ProtTrans
}

SUPPORTED_TEXT_ENCODER = {
    "base_transformer": BaseTransformers,
    "biomedgpt": BioMedGPT,
    "kv-plm": KVPLM,
    "molfm": MolFM,
    "momu": MoMu,
}

SUPPORTED_KNOWLEDGE_ENCODER = {
    "TransE": TransE,
    "gin": GIN
}

SUPPORTED_LLM_ENCODER = {
    "qwen": QwenEncoder,
    "tokenmol": TokenMolEncoder,
}