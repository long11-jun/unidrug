from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

from collections import OrderedDict
import copy
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from unidrug.feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from unidrug.feat.protein_featurizer import SUPPORTED_PROTEIN_FEATURIZER, ProteinMultiModalFeaturizer
from unidrug.utils.data_utils import balance_classes
from unidrug.utils.mol_utils import can_smiles
from unidrug.utils.kg_utils import SUPPORTED_KG, embed
from unidrug.utils.split import kfold_split, cold_drug_split, cold_protein_split, cold_cluster_split


class DTIDataset(Dataset, ABC):
    def __init__(self, path, config, split_strategy, in_memory=True):
        super(DTIDataset, self).__init__()
        self.path = path
        self.config = config
        self.split_strategy = split_strategy
        self.in_memory = in_memory
        self._load_data()
        self._train_test_split()
        self.kge = None

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _train_test_split(self):
        raise NotImplementedError

    def _build(self, eval_pair_index, save_path, train_pair_index=None, strict_cold_start=False):
        # build after train / test datasets are determined for filtering out edges
        if len(self.config["drug"]["modality"]) > 1:
            kg_config = self.config["drug"]["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            self.drug2kg, self.drug2text, self.protein2kg, self.protein2text = self.kg.link(self)
            self.concat_text_first = self.config["concat_text_first"]
            
            local_drug_count = 0
            local_prot_count = 0
            if hasattr(self, 'local_drug2text'):
                for smi, text in self.local_drug2text.items():
                    if text:
                        self.drug2text[smi] = text
                        local_drug_count += 1
            if hasattr(self, 'local_protein2text'):
                for seq, text in self.local_protein2text.items():
                    if text:
                        self.protein2text[seq] = text
                        local_prot_count += 1
            if local_drug_count > 0 or local_prot_count > 0:
                logger.info(f"[Text] Used local texts: {local_drug_count} drugs, {local_prot_count} proteins")
            
            kge = None
            if kg_config["kg_name"] == "BMKG":
                kg_base_path = kg_config.get("kg_path", "")
                
                embedding_path = os.path.join(kg_base_path, "embedding.pkl")
                if os.path.exists(embedding_path):
                    import pickle
                    logger.info(f"Loading pre-trained KGE from {embedding_path}")
                    with open(embedding_path, 'rb') as f:
                        kge = pickle.load(f)
                    logger.info(f"Loaded KGE with {len(kge)} entities (pre-trained)")
                else:
                    if os.path.exists(save_path):
                        import pickle
                        logger.info(f"Loading cached KGE from {save_path}")
                        with open(save_path, 'rb') as f:
                            kge = pickle.load(f)
                        logger.info(f"Loaded cached KGE with {len(kge)} entities")
                    else:
                        logger.warning(f"Pre-trained KGE not found, generating with ProNE...")
                        filter_out = []
                        for i_drug, i_protein in eval_pair_index:
                            smi = self.smiles[i_drug]
                            protein = self.proteins[i_protein]
                            if smi in self.drug2kg and protein in self.protein2kg:
                                filter_out.append((self.drug2kg[smi], self.protein2kg[protein]))
                        kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], 
                                    save=True, save_path=save_path)
                        logger.info(f"ProNE KGE has {len(kge)} entities")
            else:
                filter_out = []
                for i_drug, i_protein in eval_pair_index:
                    smi = self.smiles[i_drug]
                    protein = self.proteins[i_protein]
                    if smi in self.drug2kg and protein in self.protein2kg:
                        filter_out.append((self.drug2kg[smi], self.protein2kg[protein]))
                kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], 
                            save=True, save_path=save_path)
            
            # ============================================================
            # ============================================================
            #
            # ============================================================
            if strict_cold_start and kge is not None and train_pair_index is not None:
                train_drugs = set()
                train_proteins = set()
                for i_drug, i_protein in train_pair_index:
                    train_drugs.add(self.smiles[i_drug])
                    train_proteins.add(self.proteins[i_protein])
                
                test_drugs = set()
                test_proteins = set()
                for i_drug, i_protein in eval_pair_index:
                    test_drugs.add(self.smiles[i_drug])
                    test_proteins.add(self.proteins[i_protein])
                
                cold_drugs = test_drugs - train_drugs
                cold_proteins = test_proteins - train_proteins
                
                embed_dim = kg_config.get("embed_dim", 256)
                if kge:
                    sample_val = next(iter(kge.values()))
                    if hasattr(sample_val, 'shape'):
                        embed_dim = sample_val.shape[0]
                
                zeroed_drugs = 0
                for smi in cold_drugs:
                    if smi in self.drug2kg:
                        kg_id = self.drug2kg[smi]
                        if kg_id in kge:
                            kge[kg_id] = np.zeros(embed_dim, dtype=np.float32)
                            zeroed_drugs += 1
                
                zeroed_proteins = 0
                for seq in cold_proteins:
                    if seq in self.protein2kg:
                        kg_id = self.protein2kg[seq]
                        if kg_id in kge:
                            kge[kg_id] = np.zeros(embed_dim, dtype=np.float32)
                            zeroed_proteins += 1
                
                print(f"\n{'='*60}")
                print("[STRICT COLD START] Held-out entity KGE vectors were zeroed")
                print(f"  Cold drugs:    {len(cold_drugs)} (zeroed KGE: {zeroed_drugs}/{len(cold_drugs)})")
                print(f"  Cold proteins: {len(cold_proteins)} (zeroed KGE: {zeroed_proteins}/{len(cold_proteins)})")
                print(f"  Train drugs:   {len(train_drugs)}, Train proteins: {len(train_proteins)}")
                print(f"  Total KGE entities: {len(kge)}")
                print(f"{'='*60}\n")
            
            self.kge = kge
            self.config["drug"]["featurizer"]["kg"]["kge"] = kge
            self.config["protein"]["featurizer"]["kg"]["kge"] = kge
        else:
            self.concat_text_first = False
        self._configure_featurizer()
        # featurize all data pairs in one pass for training efficiency
        if self.in_memory:
            self._featurize()

    def index_select(self, indexes, balanced=False, keep_negatives=1.0, seed=42):
        """
        Creates new dataset objects by selecting a subset of indices
        :param indexes: indices of samples to select
        :param balanced: whether to balance classes (only relevant if self.type='classification')
        :param keep_balanced: probability in [0,1] to keep each negative sample (only relevant if self.type='classification')
        :param seed: seed to use for random sampling if balance=True
        :return: dataset with selected indices
        """
        assert keep_negatives >= 0.0 and keep_negatives <= 1.0
        new_dataset = copy.copy(self)
        new_dataset.pair_index = [self.pair_index[i] for i in indexes]
        new_dataset.labels = [self.labels[i] for i in indexes]
        
        if hasattr(self, 'featurized_drugs') and self.featurized_drugs:
            new_dataset.featurized_drugs = [self.featurized_drugs[i] for i in indexes]
        if hasattr(self, 'featurized_proteins') and self.featurized_proteins:
            new_dataset.featurized_proteins = [self.featurized_proteins[i] for i in indexes]
        
        if self.dataset_type == 'classification':
            if keep_negatives < 1.0:
                random.seed(seed)
                new_pair_index = []
                new_labels = []
                new_featurized_drugs = []
                new_featurized_proteins = []
                for i in range(len(new_dataset.labels)):
                    if new_dataset.labels[i] == 1:
                        append = True
                    elif new_dataset.labels[i] == 0:
                        if random.random() <= keep_negatives:
                            append = True
                        else:
                            append = False
                    else:
                        raise ValueError('Invalid label for classification dataset:', new_dataset.labels[i])
                    if append:
                        new_pair_index.append(new_dataset.pair_index[i])
                        new_labels.append(new_dataset.labels[i])
                        if hasattr(new_dataset, 'featurized_drugs') and new_dataset.featurized_drugs:
                            new_featurized_drugs.append(new_dataset.featurized_drugs[i])
                        if hasattr(new_dataset, 'featurized_proteins') and new_dataset.featurized_proteins:
                            new_featurized_proteins.append(new_dataset.featurized_proteins[i])
                new_dataset.pair_index = new_pair_index
                new_dataset.labels = new_labels
                if new_featurized_drugs:
                    new_dataset.featurized_drugs = new_featurized_drugs
                if new_featurized_proteins:
                    new_dataset.featurized_proteins = new_featurized_proteins
            if balanced:
                new_dataset.pair_index, new_dataset.labels = balance_classes(new_dataset.pair_index, new_dataset.labels, seed=seed)
        return new_dataset

    def _configure_featurizer(self):
        if len(self.config["drug"]["modality"]) > 1:
            self.drug_featurizer = DrugMultiModalFeaturizer(self.config["drug"])
            self.protein_featurizer = ProteinMultiModalFeaturizer(self.config["protein"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            self.protein_featurizer.set_protein2kgid_dict(self.protein2kg)
            if not self.concat_text_first:
                self.drug_featurizer.set_drug2text_dict(self.drug2text)
                self.protein_featurizer.set_protein2text_dict(self.protein2text)
        else:
            drug_feat_config = self.config["drug"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_DRUG_FEATURIZER[drug_feat_config["name"]](drug_feat_config)
            protein_feat_config = self.config["protein"]["featurizer"]["structure"]
            self.protein_featurizer = SUPPORTED_PROTEIN_FEATURIZER[protein_feat_config["name"]](protein_feat_config)

    def _featurize(self):
        logger.info("Featurizing...")
        self.featurized_drugs = []
        self.featurized_proteins = []
        for i_drug, i_protein in tqdm(self.pair_index):
            drug_smi, protein_seq = self.smiles[i_drug], self.proteins[i_protein]
            if len(self.config["drug"]["modality"]) > 1 and self.concat_text_first:
                processed_drug = self.drug_featurizer(drug_smi, skip=["text"])
                processed_protein = self.protein_featurizer(protein_seq)
                processed_drug["text"] = self.drug_featurizer["text"](
                    self.drug2text[drug_smi] + " [SEP] " + self.protein2text[protein_seq])
            else:
                processed_drug = self.drug_featurizer(drug_smi)
                processed_protein = self.protein_featurizer(protein_seq)
            
            processed_drug["smiles"] = drug_smi
            
            processed_protein["sequence"] = protein_seq
            
            self.featurized_drugs.append(processed_drug)
            self.featurized_proteins.append(processed_protein)
        
        self._load_tokenmol_cache()
        self._load_foldseek_cache()
    
    def _load_tokenmol_cache(self):
        
        import pickle
        
        dataset_name = self.__class__.__name__.lower()
        if dataset_name == "yamanishi08":
            dataset_name = "yamanishi08"
        elif dataset_name == "bmkg_dti":
            dataset_name = "bmkg_dti"
        elif dataset_name == "human":
            dataset_name = "human"
        
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_dir = os.path.join(repo_root, "cache", "tokenmol_3d")
        cache_file = os.path.join(cache_dir, f"tokenmol_3d_{dataset_name}.pkl")
        
        print(f"\n[DTI MolToken] Looking for cache: {cache_file}")
        
        if not os.path.exists(cache_file):
            print(f"[DTI MolToken]  Cache NOT FOUND: {cache_file}")
            print(f"[DTI MolToken] Run: python precompute_tokenmol_dti.py --dataset {dataset_name}")
            logger.warning(f"[DTI MolToken] Cache not found: {cache_file}")
            return
        
        try:
            with open(cache_file, "rb") as f:
                cache_dict = pickle.load(f)
            
            print(f"[DTI MolToken] Loaded cache with {len(cache_dict)} unique SMILES")
            
            hit, miss = 0, 0
            for i, (i_drug, i_protein) in enumerate(self.pair_index):
                smi = self.smiles[i_drug]
                if smi in cache_dict:
                    tokens = cache_dict[smi]
                    if isinstance(tokens, list):
                        self.featurized_drugs[i]["tokenmol_seq"] = " ".join(tokens)
                    else:
                        self.featurized_drugs[i]["tokenmol_seq"] = tokens
                    hit += 1
                else:
                    self.featurized_drugs[i]["tokenmol_seq"] = smi
                    miss += 1
            
            print(f"[DTI MolToken]  Cache applied: {hit} hits, {miss} misses")
            logger.info(f"[DTI MolToken] Cache loaded: {hit} hits, {miss} misses from {cache_file}")
        except Exception as e:
            print(f"[DTI MolToken]  Cache loading failed: {e}")
            logger.warning(f"[DTI MolToken] Cache loading failed: {e}")

    def _load_foldseek_cache(self):
        
        import pickle
        
        dataset_name = self.__class__.__name__.lower()
        if dataset_name == "yamanishi08":
            dataset_name = "yamanishi08"
        elif dataset_name == "bmkg_dti":
            dataset_name = "bmkg_dti"
        elif dataset_name == "human":
            dataset_name = "human"
        
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_dir = os.path.join(repo_root, "cache", "foldseek_3di")
        cache_file = os.path.join(cache_dir, f"foldseek_3di_{dataset_name}.pkl")
        
        print(f"\n[DTI FoldSeek] Looking for cache: {cache_file}")
        
        if not os.path.exists(cache_file):
            print(f"[DTI FoldSeek]  Cache NOT FOUND: {cache_file}")
            print(f"[DTI FoldSeek] Run: python scripts/precompute_foldseek_dti.py --dataset {dataset_name}")
            logger.warning(f"[DTI FoldSeek] Cache not found: {cache_file}")
            return
        
        try:
            with open(cache_file, "rb") as f:
                cache_dict = pickle.load(f)
            
            print(f"[DTI FoldSeek] Loaded cache with {len(cache_dict)} unique proteins")
            
            hit, miss = 0, 0
            for i, (i_drug, i_protein) in enumerate(self.pair_index):
                seq = self.proteins[i_protein]
                if seq in cache_dict:
                    self.featurized_proteins[i]["foldseek_3di"] = cache_dict[seq]
                    hit += 1
                else:
                    self.featurized_proteins[i]["foldseek_3di"] = ""
                    miss += 1
            
            print(f"[DTI FoldSeek]  Cache applied: {hit} hits, {miss} misses")
            logger.info(f"[DTI FoldSeek] Cache loaded: {hit} hits, {miss} misses from {cache_file}")
        except Exception as e:
            print(f"[DTI FoldSeek]  Cache loading failed: {e}")
            logger.warning(f"[DTI FoldSeek] Cache loading failed: {e}")
    
    def __getitem__(self, index):
        if not self.in_memory:
            drug, protein, label = self.smiles[self.pair_index[index][0]], self.proteins[self.pair_index[index][1]], \
            self.labels[index]
            processed_drug = self.drug_featurizer(drug)
            processed_protein = self.protein_featurizer(protein)
            if self.concat_text_first:
                processed_drug["text"] = self.drug_featurizer["text"](
                    self.drug2text[drug] + " [SEP] " + self.protein2text[protein])
            return processed_drug, processed_protein, label
        else:
            return self.featurized_drugs[index], self.featurized_proteins[index], self.labels[index]

    def __len__(self):
        return len(self.pair_index)


class DTIClassificationDataset(DTIDataset):
    def __init__(self, path, config, split_strategy):
        self.dataset_type = 'classification'
        super(DTIClassificationDataset, self).__init__(path, config, split_strategy)

    def _train_test_split(self):
        if self.split_strategy in ["warm", "cold_drug", "cold_protein"]:
            self.nfolds = 5
            if self.split_strategy == "warm":
                folds = kfold_split(len(self), 5)
            elif self.split_strategy == "cold_drug":
                folds = cold_drug_split(self, 5)
            else:
                folds = cold_protein_split(self, 5)
            self.folds = []
            for i in range(5):
                self.folds.append({
                    "train": np.concatenate(folds[:i] + folds[i + 1:], axis=0).tolist(),
                    "test": folds[i]
                })
        elif self.split_strategy == "cold_cluster":
            self.folds = cold_cluster_split(self, 3)
            self.nfolds = len(self.folds)
            logger.info(f"[cold_cluster] nfolds={self.nfolds} (some may have been skipped)")


class Yamanishi08(DTIClassificationDataset):
    def __init__(self, path, config, split_strategy):
        super(Yamanishi08, self).__init__(path, config, split_strategy)

    def _load_data(self):
        drug_data = json.load(open(os.path.join(self.path, "drug.json")))
        self.smiles = [drug_data[item]["SMILES"] for item in drug_data]
        drugsmi2index = dict(zip(self.smiles, range(len(self.smiles))))
        
        self.local_drug2text = {}
        for item in drug_data:
            smi = drug_data[item]["SMILES"]
            text = drug_data[item].get("text", "")
            if text:
                self.local_drug2text[smi] = text

        prot_data = json.load(open(os.path.join(self.path, "protein.json")))
        self.proteins = [prot_data[item]["sequence"] for item in prot_data]
        proteinseq2index = dict(zip(self.proteins, range(len(self.proteins))))
        
        self.local_protein2text = {}
        for item in prot_data:
            seq = prot_data[item]["sequence"]
            text = prot_data[item].get("text", "")
            if text:
                self.local_protein2text[seq] = text

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []

        for row in df.iterrows():
            row = row[1]
            self.pair_index.append(
                (drugsmi2index[row["compound_iso_smiles"]], proteinseq2index[row["target_sequence"]]))
            self.labels.append(int(row["affinity"]))
        
        logger.info(f"Yamanishi08's dataset, total {len(self)} samples")
        logger.info(f"  Local drug texts: {len(self.local_drug2text)}/{len(self.smiles)}")
        logger.info(f"  Local protein texts: {len(self.local_protein2text)}/{len(self.proteins)}")


class BMKG_DTI(DTIClassificationDataset):
    def __init__(self, path, config, split_strategy):
        super(BMKG_DTI, self).__init__(path, config, split_strategy)

    def _load_data(self):
        drug_data = json.load(open(os.path.join(self.path, "drug.json")))
        self.drug_keys = [item for item in drug_data]
        self.smiles = [drug_data[item]["SMILES"] for item in drug_data]
        self.drugid2index = dict(zip(drug_data.keys(), range(len(self.smiles))))
        
        self.local_drug2text = {}
        for item in drug_data:
            smi = drug_data[item]["SMILES"]
            text = drug_data[item].get("text", "")
            if text:
                self.local_drug2text[smi] = text

        prot_data = json.load(open(os.path.join(self.path, "protein.json")))
        self.prot_keys = [item for item in prot_data]
        self.proteins = [prot_data[item]["sequence"] for item in prot_data]
        self.proteinid2index = dict(zip(prot_data.keys(), range(len(self.proteins))))
        
        self.local_protein2text = {}
        for item in prot_data:
            seq = prot_data[item]["sequence"]
            text = prot_data[item].get("text", "")
            if text:
                self.local_protein2text[seq] = text

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []
        skipped = 0

        for row in df.iterrows():
            row = row[1]
            drug_id = row["drug_id"]
            prot_id = str(row["protein_id"])
            
            if drug_id not in self.drugid2index:
                skipped += 1
                continue
            if prot_id not in self.proteinid2index:
                skipped += 1
                continue
                
            self.pair_index.append((self.drugid2index[drug_id], self.proteinid2index[prot_id]))
            self.labels.append(int(row["affinity"]))
        
        if skipped > 0:
            logger.warning(f"BMKG_DTI: Skipped {skipped} samples due to missing drug/protein")
        logger.info(f"BMKG_DTI dataset, total {len(self)} samples")
        logger.info(f"  Local drug texts: {len(self.local_drug2text)}/{len(self.smiles)}")
        logger.info(f"  Local protein texts: {len(self.local_protein2text)}/{len(self.proteins)}")

class MSSL2drug(DTIClassificationDataset):
    def __init__(self, path, config, split_strategy):
        super(MSSL2drug, self).__init__(path, config, split_strategy)

    def _load_data(self):
        with open(os.path.join(self.path, 'drug.txt'), 'r') as f:
            data = [line for line in f.read().split('\n') if line != '']
        self.drug_keys = [line[:7] for line in data]
        drug_keys2name = {line[:7] : line[8:] for line in data}
        with open(os.path.join(self.path, 'drug_smiles.json'), 'r') as f:
            drug_name2smiles = json.load(f)
        self.smiles = [drug_name2smiles[drug_keys2name[key]] for key in self.drug_keys]
        self.drugid2index = dict(zip(self.drug_keys, range(len(self.smiles))))

        with open(os.path.join(self.path, 'protein.txt'), 'r') as f:
            data = [line for line in f.read().split('\n') if line != '']
        self.prot_keys = []
        prot_keys2name = {}
        for line in data:
            colon_idx = line.find(':')
            self.prot_keys.append(line[:colon_idx])
            prot_keys2name[line[:colon_idx]] = line[colon_idx+1:]
        with open(os.path.join(self.path, 'protein_seqs.json'), 'r') as f:
            prot_name2seq = json.load(f)
        self.proteins = [prot_name2seq[prot_keys2name[key]] for key in self.prot_keys]
        self.proteinid2index = dict(zip(self.prot_keys, range(len(self.proteins))))

        with open(os.path.join(self.path, 'DTInet.txt'), 'r') as f:
            data = [line.split(' ') for line in f.read().split('\n') if line != '']

        self.pair_index, self.labels = [], []
        for i in range(len(data)):
            row = data[i]
            for j in range(len(row)):
                label = int(row[j])
                self.pair_index.append((i, j))
                self.labels.append(label)

class Human(DTIClassificationDataset):
    """Human DTI dataset with pre-defined train/val/test splits.
    
    """
    def __init__(self, path, config, split_strategy):
        if split_strategy != "warm":
            logger.warning(f"Human dataset only supports 'warm' split (pre-defined), ignoring '{split_strategy}'")
        super(Human, self).__init__(path, config, "warm")

    def _load_data(self):
        train_df = pd.read_csv(os.path.join(self.path, "train.csv"))
        val_df = pd.read_csv(os.path.join(self.path, "val.csv"))
        test_df = pd.read_csv(os.path.join(self.path, "test.csv"))
        
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        unique_smiles = all_df["SMILES"].unique().tolist()
        unique_proteins = all_df["Protein"].unique().tolist()
        
        self.smiles = unique_smiles
        self.proteins = unique_proteins
        
        drugsmi2index = {smi: idx for idx, smi in enumerate(self.smiles)}
        proteinseq2index = {seq: idx for idx, seq in enumerate(self.proteins)}
        
        self.pair_index = []
        self.labels = []
        
        for _, row in all_df.iterrows():
            drug_idx = drugsmi2index[row["SMILES"]]
            prot_idx = proteinseq2index[row["Protein"]]
            self.pair_index.append((drug_idx, prot_idx))
            self.labels.append(int(row["Y"]))
        
        self.train_size = len(train_df)
        self.val_size = len(val_df)
        self.test_size = len(test_df)
        
        self.local_drug2text = {}
        self.local_protein2text = {}
        
        logger.info(f"Human DTI dataset loaded:")
        logger.info(f"  Total: {len(self)} samples")
        logger.info(f"  Drugs: {len(self.smiles)}, Proteins: {len(self.proteins)}")
        logger.info(f"  Train: {self.train_size}, Val: {self.val_size}, Test: {self.test_size}")
        logger.info(f"  Positive ratio: {sum(self.labels)/len(self.labels):.2%}")

    def _train_test_split(self):
        
        train_indices = list(range(self.train_size))
        val_indices = list(range(self.train_size, self.train_size + self.val_size))
        test_indices = list(range(self.train_size + self.val_size, len(self)))
        
        self.folds = [{
            "train": train_indices + val_indices,
            "val": val_indices,
            "test": test_indices
        }]
        self.nfolds = 1
        logger.info(f"[Human] Pre-defined split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


class DTIRegressionDataset(DTIDataset):
    def __init__(self, path, config, split_strategy):
        self.dataset_type = 'regression'
        super(DTIRegressionDataset, self).__init__(path, config, split_strategy)


class Davis_KIBA(DTIRegressionDataset):
    def __init__(self, path, config, split_strategy):
        self.is_davis = "davis" in path
        self.is_kiba = 'kiba' in path
        super(Davis_KIBA, self).__init__(path, config, split_strategy)

    def _load_data(self):
        Y = pickle.load(open(os.path.join(self.path, "Y"), "rb"), encoding='latin1')
        label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)

        can_smis_dict = json.load(open(os.path.join(self.path, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        can_smis = list(can_smis_dict.values())
        self.smiles = [can_smiles(smi) for smi in can_smis]

        proteins_dic = json.load(open(os.path.join(self.path, "proteins.txt")), object_pairs_hook=OrderedDict)
        self.proteins = list(proteins_dic.values())

        # data:
        self.pair_index = []
        self.labels = []
        train_folds = json.load(open(os.path.join(self.path, "folds/train_fold_setting1.txt")))
        for fold in train_folds:
            for i in fold:
                self.pair_index.append((label_row_inds[i], label_col_inds[i]))
                self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.train_index = list(range(len(self.labels)))
        test_fold = json.load(open(os.path.join(self.path, "folds/test_fold_setting1.txt")))
        for i in test_fold:
            self.pair_index.append((label_row_inds[i], label_col_inds[i]))
            self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.test_index = list(range(len(self.train_index), len(self.labels)))
        if self.is_davis:
            self.labels = [-float(np.log10(y / 1e9)) for y in self.labels]
        elif self.is_kiba:
            self.labels = [torch.tensor(label, dtype=torch.float32) for label in self.labels]
        else:
            raise RuntimeError

        logger.info("%s dataset, %d samples" % ("davis" if self.is_davis else "kiba", len(self)))

    def _train_test_split(self):
        self.val_index = []


SUPPORTED_DTI_DATASETS = {
    "yamanishi08": Yamanishi08,
    "bmkg-dti": BMKG_DTI,
    'mssl2drug': MSSL2drug,
    "human": Human,
    "davis": Davis_KIBA,
    "kiba": Davis_KIBA
}