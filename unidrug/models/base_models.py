from abc import ABC, abstractmethod
import torch.nn as nn

class MolEncoder(nn.Module, ABC):
    def __init__(self):
        super(MolEncoder, self).__init__()

    @abstractmethod
    def encode_mol(self, mol):
        raise NotImplementedError

class ProteinEncoder(nn.Module, ABC):
    def __init__(self):
        super(ProteinEncoder, self).__init__()

    @abstractmethod
    def encode_protein(self, prot):
        raise NotImplementedError

class TextEncoder(nn.Module, ABC):
    def __init__(self):
        super(TextEncoder, self).__init__()

    @abstractmethod
    def encode_text(self, text):
        raise NotImplementedError