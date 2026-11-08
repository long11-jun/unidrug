from abc import ABC, abstractmethod

import numpy as np
import torch

from feat.base_featurizer import BaseFeaturizer
from utils.kg_utils import SUPPORTED_KG, embed

import logging
logger = logging.getLogger(__name__)



class KGFeaturizer(BaseFeaturizer, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO：self.kg is no use
        # self.kg = SUPPORTED_KG[self.config["kg_name"]](self.config["kg_path"])
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


class KGIDFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config["embed_dim"]
        # TODO: hard code
        self.max_index = 49111

    # data: SMILES
    def __call__(self, data):
        if self.transform is not None:
            # 使用 get 避免 KeyError
            index = self.transform.get(data, None)
            if index == -1 or index is None:
                index = self.max_index
            return index
        else:
            return None
          
class KGNeighbourFeaturizer(KGFeaturizer):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config["embed_dim"]
        # TODO: hard code
        self.max_index = 49111

    # data: SMILES list
    def __call__(self, data):
        index_list = []
        if self.transform is not None:
            # 使用 get 避免 KeyError
            index_list = self.transform.get(data, [])
            if len(index_list) > 0 and index_list[0] != -1:
                return index_list
            else:
                return [self.max_index]
        else:
            return [self.max_index]


class KGEFeaturizer(KGFeaturizer):
    """
    KGE Embedding Featurizer — 精确 ID 映射
    
    对单个 SMILES 返回对应的 KG embedding:
    - 精确匹配成功 → 返回 KG embedding
    - 精确匹配失败 → 返回零向量（由 CMSA 补全）
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.kge = config["kge"]
        self.embed_dim = config["embed_dim"]
    
    def __call__(self, data):
        entity_id = None
        if self.transform is not None:
            entity_id = self.transform.get(data, None)
        
        if entity_id is not None and entity_id in self.kge:
            return torch.FloatTensor(self.kge[entity_id])
        
        return torch.zeros(self.embed_dim)


SUPPORTED_KG_FEATURIZER = {
    "id": KGIDFeaturizer,
    "KGE": KGEFeaturizer,
    "neighbour": KGNeighbourFeaturizer
}
