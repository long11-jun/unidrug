import copy
import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch

from feat.base_featurizer import BaseFeaturizer
from feat.kg_featurizer import SUPPORTED_KG_FEATURIZER
from feat.text_featurizer import SUPPORTED_TEXT_FEATURIZER
from utils import ToDevice

from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ProteinIndexFeaturizer(BaseFeaturizer):
    VOCAB_PROTEIN = {
        "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
        "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
        "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
        "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
        "Z": 25
    }

    def __init__(self, config):
        super(ProteinIndexFeaturizer, self).__init__()
        self.max_length = config["max_len"]

    def __call__(self, data):
        temp = [self.VOCAB_PROTEIN[s] for s in data]
        if len(temp) < self.max_length:
            temp = np.pad(temp, (0, self.max_length - len(temp)))
        else:
            temp = temp[:self.max_length]
        return torch.LongTensor(temp)


class ProteinOneHotFeaturizer(BaseFeaturizer):
    amino_char = [
        '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
        'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z'
    ]

    def __init__(self, config):
        super(ProteinOneHotFeaturizer, self).__init__()
        self.max_length = config["max_len"]
        self.enc = OneHotEncoder().fit(np.array(self.amino_char).reshape(-1, 1))

    def __call__(self, data):
        temp = [i if i in self.amino_char else '?' for i in data]
        if len(temp) < self.max_length:
            temp = temp + ['?'] * (self.max_length - len(temp))
        else:
            temp = temp[:self.max_length]
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)


class Protein2VecFeaturizer(BaseFeaturizer):
    AMINO_VEC = {
        "A": [-0.17691335, -0.19057421, 0.045527875, -0.175985, 1.1090639, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "C": [-0.31572455, 0.38517416, 0.17325026, 0.3164464, 1.1512344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "B": [0.037789278, -0.1989614, -0.844488, -0.8851388, 0.57501, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "E": [-0.06940994, -0.34011552, -0.17767446, 0.251, 1.0661993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "D": [0.00600859, -0.1902303, -0.049640052, 0.15067418, 1.0812483, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "G": [-0.07281224, 0.01804472, 0.22983849, -0.045492448, 1.1139168, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "F": [0.2315121, -0.01626652, 0.25592703, 0.2703909, 1.0793934, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "I": [0.15077977, -0.1881559, 0.33855876, 0.39121667, 1.0793937, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "H": [0.019046513, -0.023256639, -0.06749539, 0.16737276, 1.0796973, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "K": [0.22048187, -0.34703028, 0.20346786, 0.65077996, 1.0620389, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "J": [0.06847394, 0.015362699, -0.7120714, -1.054779, 0.49967504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "M": [0.06302169, -0.10206237, 0.18976009, 0.115588315, 1.0927621, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "L": [0.0075188675, -0.17002057, 0.08902198, 0.066686414, 1.0804346, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "O": [-0.042549122, 0.11453196, 0.3218399, -0.96280265, 0.42855614, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "N": [0.41597384, -0.22671205, 0.31179032, 0.45883527, 1.0529875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "Q": [0.25189143, -0.40238172, -0.046555642, 0.22140719, 1.0362468, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "P": [0.017954966, -0.09864355, 0.028460773, -0.12924117, 1.0974121, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "S": [0.17177454, -0.16769698, 0.27776834, 0.10357749, 1.0800852, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "R": [-0.15621762, -0.19172126, -0.209409, 0.026799612, 1.0879921, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "U": [0.00069698587, -0.40677646, 0.045045465, 0.875985, 0.93636376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "T": [0.054446213, -0.16771607, 0.22424258, -0.01337227, 1.0967118, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "W": [0.25281385, 0.12420933, 0.0132171605, 0.09199735, 1.0842415, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "V": [-0.09511698, -0.11654304, 0.1440215, -0.0022315443, 1.1064949, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Y": [0.27962074, -0.051454283, 0.114876375, 0.3550331, 1.0615551, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "X": [0.5566999, -2.5784554, -4.580289, -0.46196952, 1.4881511, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "Z": [-0.020066334, -0.116254225, -0.69591016, -1.2875729, 0.6376922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "?": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    def __init__(self, config):
        super(Protein2VecFeaturizer, self).__init__()
        self.max_length = config["max_length"]

    def __call__(self, data):
        temp = [self.AMINO_VEC[i] if i in self.AMINO_VEC else [0.0] * 13 for i in data]
        if len(temp) < self.max_length:
            for i in range(self.max_length - len(temp)):
                temp.append([0.0] * 13)
        else:
            temp = temp[:self.max_length]
        return torch.tensor(temp)


class ProteinTransformerTokFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(ProteinTransformerTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)

    def __call__(self, data):
        result = self.tokenizer(data, max_length=self.max_length, padding=True, truncation=True)
        return result


class ProteinTransformerEncFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(ProteinTransformerEncFeaturizer, self).__init__()
        self.device = config["device"]
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], model_max_length=self.max_length)
        self.encoder = AutoModel.from_pretrained(config["model_name_or_path"])
        if "init_ckpt" in config:
            ckpt = torch.load(open(config["init_ckpt"], "rb"))
            if "param_key" in config:
                ckpt = ckpt[config["param_key"]]
            self.encoder.load_state_dict(ckpt)
        self.encoder = self.encoder.to(self.device)

    def __call__(self, data):
        with torch.no_grad():
            data = " ".join(list(data))
            data = self.tokenizer(data, truncation=True, padding=True, return_tensors='pt')
            data = ToDevice(data, self.device)
            h = self.encoder(**data)['last_hidden_state'].squeeze()
            h = h[1:-1].mean(dim=0)
            return h


class ProteinCTDFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(ProteinCTDFeaturizer, self).__init__()
        self.return_type = config['return_type']
        self.feats = [
            '_PolarizabilityC1', '_PolarizabilityC2', '_PolarizabilityC3', '_SolventAccessibilityC1',
            '_SolventAccessibilityC2', '_SolventAccessibilityC3', '_SecondaryStrC1', '_SecondaryStrC2',
            '_SecondaryStrC3', '_ChargeC1', '_ChargeC2', '_ChargeC3', '_PolarityC1', '_PolarityC2', '_PolarityC3',
            '_NormalizedVDWVC1', '_NormalizedVDWVC2', '_NormalizedVDWVC3', '_HydrophobicityC1',
            '_HydrophobicityC2', '_HydrophobicityC3', '_PolarizabilityT12', '_PolarizabilityT13',
            '_PolarizabilityT23', '_SolventAccessibilityT12', '_SolventAccessibilityT13',
            '_SolventAccessibilityT23', '_SecondaryStrT12', '_SecondaryStrT13', '_SecondaryStrT23', '_ChargeT12',
            '_ChargeT13', '_ChargeT23', '_PolarityT12', '_PolarityT13', '_PolarityT23', '_NormalizedVDWVT12',
            '_NormalizedVDWVT13', '_NormalizedVDWVT23', '_HydrophobicityT12', '_HydrophobicityT13',
            '_HydrophobicityT23', '_PolarizabilityD1001', '_PolarizabilityD1025', '_PolarizabilityD1050',
            '_PolarizabilityD1075', '_PolarizabilityD1100', '_PolarizabilityD2001', '_PolarizabilityD2025',
            '_PolarizabilityD2050', '_PolarizabilityD2075', '_PolarizabilityD2100', '_PolarizabilityD3001',
            '_PolarizabilityD3025', '_PolarizabilityD3050', '_PolarizabilityD3075', '_PolarizabilityD3100',
            '_SolventAccessibilityD1001', '_SolventAccessibilityD1025', '_SolventAccessibilityD1050',
            '_SolventAccessibilityD1075', '_SolventAccessibilityD1100', '_SolventAccessibilityD2001',
            '_SolventAccessibilityD2025', '_SolventAccessibilityD2050', '_SolventAccessibilityD2075',
            '_SolventAccessibilityD2100', '_SolventAccessibilityD3001', '_SolventAccessibilityD3025',
            '_SolventAccessibilityD3050', '_SolventAccessibilityD3075', '_SolventAccessibilityD3100',
            '_SecondaryStrD1001', '_SecondaryStrD1025', '_SecondaryStrD1050', '_SecondaryStrD1075',
            '_SecondaryStrD1100', '_SecondaryStrD2001', '_SecondaryStrD2025', '_SecondaryStrD2050',
            '_SecondaryStrD2075', '_SecondaryStrD2100', '_SecondaryStrD3001', '_SecondaryStrD3025',
            '_SecondaryStrD3050', '_SecondaryStrD3075', '_SecondaryStrD3100', '_ChargeD1001', '_ChargeD1025',
            '_ChargeD1050', '_ChargeD1075', '_ChargeD1100', '_ChargeD2001', '_ChargeD2025', '_ChargeD2050',
            '_ChargeD2075', '_ChargeD2100', '_ChargeD3001', '_ChargeD3025', '_ChargeD3050', '_ChargeD3075',
            '_ChargeD3100', '_PolarityD1001', '_PolarityD1025', '_PolarityD1050', '_PolarityD1075',
            '_PolarityD1100', '_PolarityD2001', '_PolarityD2025', '_PolarityD2050', '_PolarityD2075',
            '_PolarityD2100', '_PolarityD3001', '_PolarityD3025', '_PolarityD3050', '_PolarityD3075',
            '_PolarityD3100', '_NormalizedVDWVD1001', '_NormalizedVDWVD1025', '_NormalizedVDWVD1050',
            '_NormalizedVDWVD1075', '_NormalizedVDWVD1100', '_NormalizedVDWVD2001', '_NormalizedVDWVD2025',
            '_NormalizedVDWVD2050', '_NormalizedVDWVD2075', '_NormalizedVDWVD2100', '_NormalizedVDWVD3001',
            '_NormalizedVDWVD3025', '_NormalizedVDWVD3050', '_NormalizedVDWVD3075', '_NormalizedVDWVD3100',
            '_HydrophobicityD1001', '_HydrophobicityD1025', '_HydrophobicityD1050', '_HydrophobicityD1075',
            '_HydrophobicityD1100', '_HydrophobicityD2001', '_HydrophobicityD2025', '_HydrophobicityD2050',
            '_HydrophobicityD2075', '_HydrophobicityD2100', '_HydrophobicityD3001', '_HydrophobicityD3025',
            '_HydrophobicityD3050', '_HydrophobicityD3075', '_HydrophobicityD3100'
        ]

    def __call__(self, data):
        from PyBioMed.PyProtein import CTD
        ctd = CTD.CalculateCTD(data)
        ctd = [ctd[feat] for feat in self.feats]
        if self.return_type == 'pt':
            return torch.Tensor(ctd)
        elif self.return_type == 'np':
            return np.array(ctd)
        else:
            return ctd


SUPPORTED_SINGLE_MODAL_PROTEIN_FEATURIZER = {
    "index": ProteinIndexFeaturizer,
    "OneHot": ProteinOneHotFeaturizer,
    "protein2vec": Protein2VecFeaturizer,
    "transformertok": ProteinTransformerTokFeaturizer,
    "transformerenc": ProteinTransformerEncFeaturizer,
    'CTD': ProteinCTDFeaturizer
}


class ProteinMultiModalFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(ProteinMultiModalFeaturizer, self).__init__()
        self.modality = config["modality"]
        self.featurizers = {}
        if "structure" in config["modality"]:
            conf = config["featurizer"]["structure"]
            self.featurizers["structure"] = SUPPORTED_SINGLE_MODAL_PROTEIN_FEATURIZER[conf["name"]](conf)
        if "kg" in config["modality"]:
            conf = config["featurizer"]["kg"]
            self.featurizers["kg"] = SUPPORTED_KG_FEATURIZER[conf["name"]](conf)
        if "text" in config["modality"]:
            conf = config["featurizer"]["text"]
            self.featurizers["text"] = SUPPORTED_TEXT_FEATURIZER[conf["name"]](conf)

    def set_protein2kgid_dict(self, protein2kgid):
        self.featurizers["kg"].set_transform(protein2kgid)

    def set_protein2text_dict(self, protein2text):
        self.featurizers["text"].set_transform(protein2text)

    def __call__(self, data, skip=[]):
        feat = {}
        for modality in self.featurizers.keys():
            if modality not in skip:
                feat[modality] = self.featurizers[modality](data)
        return feat

    def __getitem__(self, index):
        if index not in self.modality:
            logger.error("%s is not a valid modality!" % (index))
            return None
        return self.featurizers[index]

SUPPORTED_PROTEIN_FEATURIZER = copy.deepcopy(SUPPORTED_SINGLE_MODAL_PROTEIN_FEATURIZER)
SUPPORTED_PROTEIN_FEATURIZER["MultiModal"] = ProteinMultiModalFeaturizer