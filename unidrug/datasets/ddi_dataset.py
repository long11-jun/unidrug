

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DDIDataset(Dataset):
    
    
    def __init__(
        self,
        data_path: str,
        config: dict,
        scenario: str = "S1",  # S1, S2, S3
        split: str = "train",  # train, val, test
        max_samples: int = None,
    ):
        """
        Args:
            scenario: S1 (UnseenDDIs), S2 (unseen_onedrug), S3 (unseen_twodrugs)
            split: train, val, test
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.scenario = scenario
        self.split = split
        self.max_samples = max_samples
        
        self._load_data()
        
        logger.info(f"[DDIDataset] Loaded {len(self)} samples for {scenario}/{split}")
    
    def _load_data(self):
        
        
        if self.scenario == "S1":
            base_dir = os.path.join(self.data_path, "UnseenDDIs")
            if self.split == "train":
                csv_file = os.path.join(base_dir, "tr_dataset.csv")
            elif self.split == "val":
                csv_file = os.path.join(base_dir, "val_dataset.csv")
            elif self.split == "test":
                csv_file = os.path.join(base_dir, "tst_dataset.csv")
            else:
                raise ValueError(f"Unknown split: {self.split}")
                
        elif self.scenario == "S2":
            base_dir = os.path.join(self.data_path, "Unseendrugs")
            if self.split == "train":
                csv_file = os.path.join(base_dir, "tr_dataset.csv")
            elif self.split in ["val", "test"]:
                csv_file = os.path.join(base_dir, "val_dataset_unseen_onedrug.csv")
            else:
                raise ValueError(f"Unknown split: {self.split}")
                
        elif self.scenario == "S3":
            base_dir = os.path.join(self.data_path, "Unseendrugs")
            if self.split == "train":
                csv_file = os.path.join(base_dir, "tr_dataset.csv")
            elif self.split in ["val", "test"]:
                csv_file = os.path.join(base_dir, "val_dataset_unseen_twodrugs.csv")
            else:
                raise ValueError(f"Unknown split: {self.split}")
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        
        logger.info(f"Loading CSV from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        if self.max_samples is not None and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=42)
        
        self.smiles_a = df["drug_A"].tolist()
        self.smiles_b = df["drug_B"].tolist()
        self.labels = df["DDI"].astype(int).tolist()
        
        if not all(0 <= label < 4 for label in self.labels):
            invalid_labels = set([l for l in self.labels if not (0 <= l < 4)])
            raise ValueError(f"Invalid DDI labels found. Expected [0, 3], got: {invalid_labels}")
        
        label_counts = df["DDI"].value_counts().sort_index()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        return {
            "smiles_a": self.smiles_a[idx],
            "smiles_b": self.smiles_b[idx],
            "label": self.labels[idx]
        }


class DDICollator:
    
    
    def __init__(self, config: dict, drug_featurizer=None):
        """
        Args:
        """
        self.config = config
        self.drug_featurizer = drug_featurizer
    
    def __call__(self, batch: List[Dict]) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        Args:
            batch: List of dicts with keys: smiles_a, smiles_b, label
        
        Returns:
            smiles_a_list: List of Drug A SMILES
            smiles_b_list: List of Drug B SMILES
            labels: Tensor of shape [B]
        """
        smiles_a_list = [item["smiles_a"] for item in batch]
        smiles_b_list = [item["smiles_b"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        
        return smiles_a_list, smiles_b_list, labels


class MSSL2drugDataset(Dataset):
    
    
    def __init__(
        self,
        data_path: str,
        config: dict,
        split: str = "train",  # train, val, test
        split_ratio: Tuple[float, float] = (0.8, 0.1),  # train/val ratio, test = 1 - train - val
        random_seed: int = 42,
        max_samples: int = None,
    ):
        """
        Args:
            split: train, val, test
            split_ratio: (train_ratio, val_ratio)
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.split = split
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.max_samples = max_samples
        
        self._load_data()
        
        logger.info(f"[MSSL2drugDataset] Loaded {len(self)} samples for {split}")
    
    def _load_data(self):
        
        mssl_dir = os.path.join(self.data_path, "MSSL2drug")
        
        drug_file = os.path.join(mssl_dir, "drug.txt")
        with open(drug_file, 'r') as f:
            data = [line for line in f.read().split('\n') if line != '']
        drug_keys = [line[:7] for line in data]
        drug_keys2name = {line[:7]: line[8:] for line in data}
        
        smiles_file = os.path.join(mssl_dir, "drug_smiles.json")
        with open(smiles_file, 'r') as f:
            drug_name2smiles = json.load(f)
        
        drug_smiles_list = []
        for key in drug_keys:
            name = drug_keys2name[key]
            if name in drug_name2smiles:
                drug_smiles_list.append(drug_name2smiles[name])
            else:
                logger.warning(f"Drug {name} not found in drug_smiles.json, using empty SMILES")
                drug_smiles_list.append("")
        
        ddi_file = os.path.join(mssl_dir, "DDInet.txt")
        with open(ddi_file, 'r') as f:
            ddi_data = [line.split(' ') for line in f.read().split('\n') if line != '']
        
        assert len(ddi_data) == len(drug_keys), f"DDI matrix size mismatch: {len(ddi_data)} vs {len(drug_keys)}"
        
        all_pairs = []
        all_labels = []
        for i in range(len(ddi_data)):
            row = ddi_data[i]
            assert len(row) == len(drug_keys), f"Row {i} size mismatch"
            for j in range(i + 1):
                label = int(row[j])
                smiles_a = drug_smiles_list[i]
                smiles_b = drug_smiles_list[j]
                if smiles_a and smiles_b:
                    all_pairs.append((smiles_a, smiles_b))
                    all_labels.append(label)
        
        logger.info(f"Total DDI pairs: {len(all_pairs)}, Positive: {sum(all_labels)}, Negative: {len(all_labels) - sum(all_labels)}")
        
        np.random.seed(self.random_seed)
        n_total = len(all_pairs)
        indices = np.random.permutation(n_total)
        
        train_ratio, val_ratio = self.split_ratio
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if self.split == "train":
            selected_indices = indices[:n_train]
        elif self.split == "val":
            selected_indices = indices[n_train:n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val:]
        
        self.smiles_a = [all_pairs[i][0] for i in selected_indices]
        self.smiles_b = [all_pairs[i][1] for i in selected_indices]
        self.labels = [all_labels[i] for i in selected_indices]
        
        if self.max_samples is not None and len(self.labels) > self.max_samples:
            self.smiles_a = self.smiles_a[:self.max_samples]
            self.smiles_b = self.smiles_b[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        logger.info(f"[{self.split}] Positive: {pos_count}, Negative: {neg_count}, Ratio: {pos_count/len(self.labels):.4f}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "smiles_a": self.smiles_a[idx],
            "smiles_b": self.smiles_b[idx],
            "label": self.labels[idx]
        }


class DDInterDataset(Dataset):
    
    
    def __init__(
        self,
        data_path: str,
        config: dict,
        split: str = "train",
    ):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.split = split
        
        self._load_data()
    
    def _load_data(self):
        
        ddinter_dir = os.path.join(self.data_path, "DDInter")
        
        if not os.path.exists(ddinter_dir):
            logger.warning(f"DDInter directory not found: {ddinter_dir}")
            self.smiles_a = []
            self.smiles_b = []
            self.labels = []
            return
        
        possible_files = [
            os.path.join(ddinter_dir, f"{self.split}.csv"),
            os.path.join(ddinter_dir, f"{self.split}.tsv"),
            os.path.join(ddinter_dir, f"{self.split}_dataset.csv"),
        ]
        
        for csv_file in possible_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if "drug_A" in df.columns:
                    self.smiles_a = df["drug_A"].tolist()
                    self.smiles_b = df["drug_B"].tolist()
                    self.labels = df["DDI"].tolist() if "DDI" in df.columns else df["label"].tolist()
                    logger.info(f"[DDInterDataset] Loaded {len(self)} samples")
                    return
        
        logger.warning("DDInter data files not found or format unknown")
        self.smiles_a = []
        self.smiles_b = []
        self.labels = []
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "smiles_a": self.smiles_a[idx],
            "smiles_b": self.smiles_b[idx],
            "label": self.labels[idx]
        }


SUPPORTED_DDI_DATASETS = {
    "UnseenDDIs": DDIDataset,
    "Unseendrugs": DDIDataset,
    "DDInter": DDInterDataset,
    "S1": DDIDataset,
    "S2": DDIDataset,
    "S3": DDIDataset,
    "LUO": MSSL2drugDataset,
    "MSSL2drug": MSSL2drugDataset,
}


def get_ddi_dataset(
    dataset_name: str,
    data_path: str,
    config: dict,
    split: str = "train",
    scenario: str = None,
    **kwargs
) -> Dataset:
    """
    
    Args:
        split: train/val/test
    
    Returns:
    """
    if dataset_name in ["S1", "S2", "S3"]:
        scenario = dataset_name
        dataset_cls = DDIDataset
    elif dataset_name in ["LUO", "MSSL2drug"]:
        return MSSL2drugDataset(data_path, config, split=split, **kwargs)
    elif dataset_name == "DDInter":
        return DDInterDataset(data_path, config, split)
    else:
        dataset_cls = SUPPORTED_DDI_DATASETS.get(dataset_name, DDIDataset)
        if scenario is None:
            scenario = "S1"
    
    return dataset_cls(data_path, config, scenario=scenario, split=split, **kwargs)
