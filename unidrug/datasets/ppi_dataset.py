"""
LLM-based PPI 数据集加载器

支持数据集：
- SHS148k: 5188 蛋白质，369042 条相互作用（7分类）
- SHS27k: 较小的子集

数据格式：
- sequences.tsv: 蛋白质ID -> 序列
- interactions.txt: 蛋白质对相互作用记录

7 类 PPI 类型:
- reaction (0): 反应
- binding (1): 结合
- ptmod (2): 翻译后修饰
- activation (3): 激活
- inhibition (4): 抑制
- catalysis (5): 催化
- expression (6): 表达调控
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# PPI 类别映射
PPI_CLASS_MAP = {
    'reaction': 0,
    'binding': 1,
    'ptmod': 2,
    'activation': 3,
    'inhibition': 4,
    'catalysis': 5,
    'expression': 6,
}


class PPIDataset(Dataset):
    """
    LLM-based PPI 数据集
    支持多标签分类（一对蛋白质可能有多种相互作用类型）
    """
    
    def __init__(
        self,
        data_path: str,
        config: dict,
        split: str = "train",  # train, test
        dataset_name: str = "SHS148k",  # SHS148k, SHS27k
        split_method: str = "DFS",  # DFS 或 BFS（与 KEDD 基线对齐）
        max_samples: int = None,
        multilabel: bool = True,  # 是否多标签
    ):
        """
        Args:
            data_path: 数据集根目录
            config: 配置字典
            split: train, test
            dataset_name: SHS148k 或 SHS27k
            split_method: DFS 或 BFS（图划分方式）
                - DFS: 深度优先划分，测试集包含更多训练中见过的蛋白质（较简单）
                - BFS: 广度优先划分，测试集包含更多未见蛋白质（更难，更符合实际场景）
            max_samples: 最大样本数
            multilabel: 是否多标签（True: 返回7维标签，False: 返回单标签）
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.split_method = split_method.upper()  # DFS 或 BFS
        self.max_samples = max_samples
        self.multilabel = multilabel
        self.num_classes = 7
        
        # 加载数据
        self._load_data()
        
        # 划分数据集
        self._split_data()
        
        logger.info(f"[PPIDataset] Loaded {len(self)} samples for {dataset_name}/{split} ({self.split_method})")
    
    def _load_data(self):
        """加载蛋白质序列和相互作用数据"""
        
        base_dir = os.path.join(self.data_path, self.dataset_name)
        
        # 1. 加载蛋白质序列
        seq_file = os.path.join(base_dir, "sequences.tsv")
        self.proteins = []
        self.protname2id = {}
        
        with open(seq_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    prot_name = parts[0]
                    sequence = parts[1]
                    self.protname2id[prot_name] = i
                    self.proteins.append(sequence)
        
        logger.info(f"Loaded {len(self.proteins)} proteins")
        
        # 2. 加载相互作用
        interaction_file = os.path.join(base_dir, "interactions.txt")
        ppi_dict = {}  # (prot1_id, prot2_id) -> [label_vector]
        
        with open(interaction_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 跳过标题行
                    continue
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                prot1_name = parts[0]
                prot2_name = parts[1]
                interaction_type = parts[2].lower()
                
                if prot1_name not in self.protname2id or prot2_name not in self.protname2id:
                    continue
                
                prot1_id = self.protname2id[prot1_name]
                prot2_id = self.protname2id[prot2_name]
                
                # 无向：统一顺序
                if prot1_id > prot2_id:
                    prot1_id, prot2_id = prot2_id, prot1_id
                
                key = (prot1_id, prot2_id)
                if key not in ppi_dict:
                    ppi_dict[key] = [0] * self.num_classes
                
                if interaction_type in PPI_CLASS_MAP:
                    class_idx = PPI_CLASS_MAP[interaction_type]
                    ppi_dict[key][class_idx] = 1
        
        # 转换为列表
        self.all_pair_index = []
        self.all_labels = []
        
        for (prot1_id, prot2_id), label_vector in ppi_dict.items():
            self.all_pair_index.append([prot1_id, prot2_id])
            if self.multilabel:
                self.all_labels.append(label_vector)
            else:
                # 单标签：取第一个非零类
                single_label = 0
                for i, v in enumerate(label_vector):
                    if v == 1:
                        single_label = i
                        break
                self.all_labels.append(single_label)
        
        logger.info(f"Loaded {len(self.all_pair_index)} protein pairs")
        
        # 统计类别分布
        if self.multilabel:
            label_counts = np.sum(self.all_labels, axis=0)
            logger.info(f"Label distribution (multilabel): {dict(zip(PPI_CLASS_MAP.keys(), label_counts))}")
        else:
            from collections import Counter
            label_counts = Counter(self.all_labels)
            logger.info(f"Label distribution (single): {label_counts}")
    
    def _split_data(self):
        """
        划分训练/测试集
        
        支持 DFS 和 BFS 两种划分方式（与 KEDD 基线对齐）：
        - DFS: split_dfs.json
        - BFS: split_bfs.json
        """
        # 选择划分文件
        split_filename = f"split_{self.split_method.lower()}.json"
        split_file = os.path.join(self.data_path, self.dataset_name, split_filename)
        
        if os.path.exists(split_file):
            # 使用预定义的划分
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            
            if self.split == "train":
                indices = split_info.get("train", list(range(int(len(self.all_pair_index) * 0.8))))
            else:
                indices = split_info.get("test", list(range(int(len(self.all_pair_index) * 0.8), len(self.all_pair_index))))
            
            logger.info(f"Using {self.split_method} split from {split_file}")
        else:
            # 回退到随机划分（80% train, 20% test）
            logger.warning(f"Split file {split_file} not found, using random split")
            np.random.seed(42)
            n_total = len(self.all_pair_index)
            perm = np.random.permutation(n_total)
            split_idx = int(n_total * 0.8)
            
            if self.split == "train":
                indices = perm[:split_idx].tolist()
            else:
                indices = perm[split_idx:].tolist()
        
        # 限制样本数
        if self.max_samples is not None and len(indices) > self.max_samples:
            indices = indices[:self.max_samples]
        
        self.pair_index = [self.all_pair_index[i] for i in indices]
        self.labels = [self.all_labels[i] for i in indices]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        返回：
            seq_a: Protein A 的序列
            seq_b: Protein B 的序列
            label: PPI 类别（多标签为 7 维向量，单标签为整数）
        """
        prot_a_id, prot_b_id = self.pair_index[idx]
        
        return {
            "seq_a": self.proteins[prot_a_id],
            "seq_b": self.proteins[prot_b_id],
            "label": self.labels[idx],
            "prot_a_id": prot_a_id,
            "prot_b_id": prot_b_id,
        }
    
    def get_protein_sequence(self, prot_id: int) -> str:
        """根据 ID 获取蛋白质序列"""
        return self.proteins[prot_id]


class PPICollator:
    """
    PPI 数据 Collator - 用于原始数据集（只有序列）
    """
    
    def __init__(self, config: dict, multilabel: bool = True):
        self.config = config
        self.multilabel = multilabel
    
    def __call__(self, batch: List[Dict]):
        """
        Args:
            batch: List of dicts with keys: seq_a, seq_b, label
        
        Returns:
            prot_a_batch: dict with sequences
            prot_b_batch: dict with sequences
            labels: Tensor
        """
        seq_a_list = [item["seq_a"] for item in batch]
        seq_b_list = [item["seq_b"] for item in batch]
        labels = [item["label"] for item in batch]
        
        prot_a_batch = {"sequences": seq_a_list}
        prot_b_batch = {"sequences": seq_b_list}
        
        if self.multilabel:
            labels_tensor = torch.tensor(labels, dtype=torch.float)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return prot_a_batch, prot_b_batch, labels_tensor


class PPIDatasetWithFeaturizer(Dataset):
    """
    PPI 数据集包装器 - 支持多模态特征提取（与 DDI 对称）
    
    将蛋白质序列转换为多模态特征：
    - structure: 氨基酸索引序列（用于 DeepMCNN）
    - kg: KGE 嵌入（可选）
    - sequences: 原始序列（用于 Prompt）
    """
    
    VOCAB_PROTEIN = {
        "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
        "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
        "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
        "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
        "Z": 25
    }
    
    def __init__(
        self,
        base_dataset: PPIDataset,
        config: dict,
        kge_bank: torch.Tensor = None,
        kge_keys: List[str] = None,
        max_seq_len: int = 1000,
    ):
        """
        Args:
            base_dataset: 原始 PPI 数据集
            config: 配置字典
            kge_bank: KGE 嵌入矩阵 [N, dim]
            kge_keys: KGE 实体 ID 列表
            max_seq_len: 最大序列长度
        """
        self.base_dataset = base_dataset
        self.config = config
        self.max_seq_len = max_seq_len
        
        # KGE 映射
        self.kge_bank = kge_bank
        self.kge_key2idx = {}
        if kge_keys is not None:
            self.kge_key2idx = {k: i for i, k in enumerate(kge_keys)}
        
        # 蛋白质 ID -> KGE ID 映射（如有）
        self.prot2kgid = {}
        
        logger.info(f"[PPIDatasetWithFeaturizer] Wrapped {len(base_dataset)} samples")
        logger.info(f"[PPIDatasetWithFeaturizer] KGE bank: {kge_bank.shape if kge_bank is not None else None}")
    
    def set_prot2kgid_mapping(self, mapping: Dict[str, str]):
        """设置蛋白质 -> KGE ID 映射"""
        self.prot2kgid = mapping
        logger.info(f"Set protein->KG mapping: {len(mapping)} proteins")
    
    def _seq_to_indices(self, seq: str) -> torch.Tensor:
        """将蛋白质序列转换为氨基酸索引（用于 DeepMCNN）"""
        indices = []
        for aa in seq[:self.max_seq_len]:
            indices.append(self.VOCAB_PROTEIN.get(aa.upper(), 24))  # 24 = X (unknown)
        
        # Padding
        if len(indices) < self.max_seq_len:
            indices.extend([0] * (self.max_seq_len - len(indices)))
        
        return torch.LongTensor(indices)
    
    def _get_kg_embedding(self, prot_id: int, sequence: str) -> torch.Tensor:
        """获取蛋白质的 KGE 嵌入"""
        if self.kge_bank is None:
            return None
        
        # 尝试多种 ID 格式
        possible_ids = [
            f"P{prot_id}",
            str(prot_id),
            sequence[:20],  # 用序列前缀作为 ID
        ]
        
        for pid in possible_ids:
            if pid in self.kge_key2idx:
                idx = self.kge_key2idx[pid]
                return self.kge_bank[idx]
        
        # 未找到，返回 None（模型会使用 KG 补全）
        return None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        返回多模态特征：
        - feat_a: dict with structure, kg, sequences
        - feat_b: dict with structure, kg, sequences
        - label: 标签
        """
        item = self.base_dataset[idx]
        seq_a = item["seq_a"]
        seq_b = item["seq_b"]
        prot_a_id = item.get("prot_a_id", idx)
        prot_b_id = item.get("prot_b_id", idx)
        label = item["label"]
        
        # 特征化蛋白质 A
        feat_a = {
            "structure": self._seq_to_indices(seq_a),
            "sequences": seq_a,
        }
        kg_a = self._get_kg_embedding(prot_a_id, seq_a)
        if kg_a is not None:
            feat_a["kg"] = kg_a
        
        # 特征化蛋白质 B
        feat_b = {
            "structure": self._seq_to_indices(seq_b),
            "sequences": seq_b,
        }
        kg_b = self._get_kg_embedding(prot_b_id, seq_b)
        if kg_b is not None:
            feat_b["kg"] = kg_b
        
        return feat_a, feat_b, label


class PPICollator:
    """
    PPI 数据整理器 - 用于已特征化的数据集（与 DDI 对称）
    """
    
    def __init__(self, multilabel: bool = True, kg_dim: int = 256):
        self.multilabel = multilabel
        self.kg_dim = kg_dim
    
    def __call__(self, batch):
        """
        整理一个 batch
        
        Args:
            batch: List of (feat_a, feat_b, label)
        
        Returns:
            prot_a_batch: dict with structure, kg, sequences
            prot_b_batch: dict with structure, kg, sequences
            labels: Tensor
        """
        feat_a_list = []
        feat_b_list = []
        labels = []
        
        for feat_a, feat_b, label in batch:
            if feat_a is not None and feat_b is not None:
                feat_a_list.append(feat_a)
                feat_b_list.append(feat_b)
                labels.append(label)
        
        if len(feat_a_list) == 0:
            return None, None, None
        
        # 整理蛋白质 A
        prot_a_batch = self._collate_proteins(feat_a_list)
        
        # 整理蛋白质 B
        prot_b_batch = self._collate_proteins(feat_b_list)
        
        # 标签
        if self.multilabel:
            labels_tensor = torch.tensor(labels, dtype=torch.float)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return prot_a_batch, prot_b_batch, labels_tensor
    
    def _collate_proteins(self, feat_list: List[Dict]) -> Dict:
        """整理蛋白质特征列表"""
        batch = {
            "structure": torch.stack([f["structure"] for f in feat_list]),
            "sequences": [f["sequences"] for f in feat_list],
        }
        
        # KG（可选）
        kg_list = []
        for f in feat_list:
            if "kg" in f and f["kg"] is not None:
                kg_list.append(f["kg"])
            else:
                kg_list.append(torch.zeros(self.kg_dim))
        
        batch["kg"] = torch.stack(kg_list)
        
        return batch


# 支持的 PPI 数据集映射
SUPPORTED_PPI_DATASETS = {
    "SHS148k": PPIDataset,
    "SHS27k": PPIDataset,
    "STRING": PPIDataset,
}


def get_ppi_dataset(
    dataset_name: str,
    data_path: str,
    config: dict,
    split: str = "train",
    **kwargs
) -> Dataset:
    """
    获取 PPI 数据集
    
    Args:
        dataset_name: 数据集名称 (SHS148k, SHS27k)
        data_path: 数据路径
        config: 配置
        split: train/test
    
    Returns:
        Dataset 实例
    """
    dataset_cls = SUPPORTED_PPI_DATASETS.get(dataset_name, PPIDataset)
    return dataset_cls(data_path, config, split=split, dataset_name=dataset_name, **kwargs)
