from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class KnowledgeEncoder(nn.Module, ABC):
    def __init__(self):
        super(KnowledgeEncoder, self).__init__()

    @abstractmethod
    def encode_knowledge(self, kg):
        raise NotImplementedError