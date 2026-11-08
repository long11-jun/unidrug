from abc import ABC, abstractmethod
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

import copy
from enum import Enum
import numpy as np
import pandas as pd
import os
from rdkit import Chem
import os
import sys
import pickle

import torch
from torch.utils.data import Dataset
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolTransforms
import math
from tqdm import tqdm

from feat.drug_featurizer import SUPPORTED_DRUG_FEATURIZER, DrugMultiModalFeaturizer
from utils.kg_utils import SUPPORTED_KG, embed
from utils.split import random_split, scaffold_split, kfold_split
from utils.utils import Normalizer

sys.path.insert(0, os.path.dirname(__file__))
import dp_dataset_load as dpload


class Task(Enum):
    CLASSFICATION = 0
    REGRESSION = 1


class DPDataset(Dataset, ABC):
    def __init__(self, path, config, in_memory=True):
        super(DPDataset, self).__init__()
        self.path = path
        self.config = config
        self.in_memory = in_memory
        # self._featurize()

    @abstractmethod
    def _load_data(self, path):
        raise NotImplementedError

    def _featurize(self, save=True):
        logger.info("Featurizing...")
        self.featurized_drugs = [self.drug_featurizer(drug) for drug in self.drugs]
        self.labels = [torch.tensor(label) for label in self.labels]

        try:
            if isinstance(self.featurized_drugs[0], dict):
                smiles_cfg = self.config["drug"]["featurizer"].get("smiles", {})
                tokenmol_vocab_file = smiles_cfg.get("tokenmol_vocab_file", None)
                precompute = bool(smiles_cfg.get("tokenmol_precompute", False))
                use_torsion = bool(smiles_cfg.get("tokenmol_use_torsion", False))
                max_torsions = int(smiles_cfg.get("tokenmol_max_torsions", 32))
                
                if precompute and tokenmol_vocab_file and os.path.exists(tokenmol_vocab_file):
                    from models.llm_encoder.tokenmol_tokenizer import TokenMolTokenizer
                    tok = TokenMolTokenizer(tokenmol_vocab_file)
                    vocab = tok.vocab
                    
                    cache_loaded = False
                    # unidrug/datasets/dp_dataset.py -> repository root
                    try:
                        from pathlib import Path
                        repo_root = str(Path(__file__).resolve().parents[2])
                    except Exception:
                        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    dataset_path = str(self.path) if hasattr(self, "path") else ""
                    dataset_name = os.path.basename(dataset_path.rstrip("/"))
                    if dataset_name.lower().endswith(".csv"):
                        dataset_name = os.path.splitext(dataset_name)[0]
                    if dataset_name.lower() in {"raw", "processed"}:
                        dataset_name = os.path.basename(os.path.dirname(dataset_path.rstrip("/")))
                        if dataset_name.lower().endswith(".csv"):
                            dataset_name = os.path.splitext(dataset_name)[0]
                    cache_candidates = [
                        os.path.join(repo_root, "cache", "tokenmol_3d"),
                        os.path.join(os.path.dirname(repo_root), "cache", "tokenmol_3d"),
                    ]
                    cache_file = None
                    possible_names = [
                        f"tokenmol_3d_{dataset_name.lower()}.pkl",
                        f"tokenmol_3d_{dataset_name.lower()}_data.pkl",
                    ]
                    for cache_dir in cache_candidates:
                        for fname in possible_names:
                            cand = os.path.join(cache_dir, fname)
                            if os.path.exists(cand):
                                cache_file = cand
                                break
                        if cache_file:
                            break
                    if cache_file is None:
                        cache_file = os.path.join(cache_candidates[0], f"tokenmol_3d_{dataset_name.lower()}.pkl")
                    
                    if os.path.exists(cache_file):
                        import pickle
                        logger.info(f"[TokenMol3D] Loading from cache: {cache_file}")
                        with open(cache_file, 'rb') as f:
                            cache_dict = pickle.load(f)
                        
                        hit, miss = 0, 0
                        for i, smi in enumerate(self.smiles):
                            if smi in cache_dict:
                                tokens = cache_dict[smi]
                                if isinstance(tokens, list):
                                    self.featurized_drugs[i]["tokenmol_seq"] = " ".join(tokens)
                                else:
                                    self.featurized_drugs[i]["tokenmol_seq"] = tokens
                                hit += 1
                            else:
                                smi_tokens = self._tokenmol_tokenize_smiles(smi, vocab=vocab)
                                self.featurized_drugs[i]["tokenmol_seq"] = " ".join(smi_tokens)
                                miss += 1
                        
                        logger.info(f"[TokenMol3D] Cache loaded: {hit} hits, {miss} misses")
                        cache_loaded = True
                    
                    if not cache_loaded:
                        logger.info(
                            f"[TokenMol3D] No cache found at {cache_file}. "
                            f"Precomputing tokenmol_seq for {len(self.smiles)} molecules... "
                            f"(torsion={use_torsion}, max_torsions={max_torsions})"
                        )
                        logger.info("[TokenMol3D] TIP: Run 'python scripts/precompute_tokenmol_3d.py --dataset SIDER' to speed up!")
                        for i, smi in enumerate(tqdm(self.smiles, desc="[TokenMol3D] precompute", total=len(self.smiles))):
                            seq = self._build_tokenmol_seq(
                                smi,
                                vocab=vocab,
                                use_torsion=use_torsion,
                                max_torsions=max_torsions,
                            )
                            self.featurized_drugs[i]["tokenmol_seq"] = seq
                        logger.info(f"[TokenMol3D] Precompute finished. Will be saved in: {self.processed_file_path}")
        except Exception as exc:
            logger.warning(f"[TokenMol3D] Precompute skipped due to error: {exc}")

        if save:
            normalizer_states = None
            if hasattr(self, 'normalizer') and self.normalizer:
                normalizer_states = []
                for norm in self.normalizer:
                    if norm is not None:
                        normalizer_states.append(norm.state_dict())
                    else:
                        normalizer_states.append(None)
            
            save_data = {
                'featurized_drugs': self.featurized_drugs,
                'labels': self.labels,
                'drugs': self.drugs,
                'normalizer_states': normalizer_states,
            }
            
            if hasattr(self, 'folds'):
                save_data['folds'] = self.folds
            else:
                save_data['train_index'] = self.train_index
                save_data['val_index'] = self.val_index
                save_data['test_index'] = self.test_index
            
            torch.save(save_data, self.processed_file_path)

    def _inject_tokenmol3d_from_cache(self, config):
        
        try:
            if not self.featurized_drugs or not isinstance(self.featurized_drugs[0], dict):
                return
            
            sample_seq = self.featurized_drugs[0].get("tokenmol_seq", "")
            if "GEO" in sample_seq or "TOR" in sample_seq:
                logger.info("[TokenMol3D] valid 3D tokenmol_seq in .pt cache, skip injection")
                return
            logger.info(f"[TokenMol3D] No 3D info in .pt (sample='{sample_seq[:40]}...'). Injecting from pkl...")
            
            smiles_cfg = config["drug"]["featurizer"].get("smiles", {})
            tokenmol_vocab_file = smiles_cfg.get("tokenmol_vocab_file", None)
            precompute = bool(smiles_cfg.get("tokenmol_precompute", False))
            
            if not precompute:
                logger.info("[TokenMol3D] tokenmol_precompute=False, skip injection")
                return
            if not tokenmol_vocab_file or not os.path.exists(tokenmol_vocab_file):
                logger.warning(f"[TokenMol3D] vocab file not found: {tokenmol_vocab_file}")
                return
            
            from models.llm_encoder.tokenmol_tokenizer import TokenMolTokenizer
            tok = TokenMolTokenizer(tokenmol_vocab_file)
            vocab = tok.vocab
            
            smiles_list = getattr(self, "smiles", None) or getattr(self, "drugs", None)
            if not smiles_list:
                logger.warning("[TokenMol3D] No smiles/drugs list available, skip injection")
                return
            
            from pathlib import Path
            try:
                repo_root = str(Path(__file__).resolve().parents[2])
            except Exception:
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            dataset_name = getattr(self, "name", "")
            if not dataset_name:
                dataset_path = str(self.path) if hasattr(self, "path") else ""
                dataset_name = os.path.basename(dataset_path.rstrip("/"))
                if dataset_name.lower().endswith(".csv"):
                    dataset_name = os.path.splitext(dataset_name)[0]
                if dataset_name.lower() in {"raw", "processed"}:
                    dataset_name = os.path.basename(os.path.dirname(dataset_path.rstrip("/")))
            
            cache_candidates = [
                os.path.join(repo_root, "cache", "tokenmol_3d"),
                os.path.join(os.path.dirname(repo_root), "cache", "tokenmol_3d"),
            ]
            possible_names = [
                f"tokenmol_3d_{dataset_name.lower()}.pkl",
                f"tokenmol_3d_{dataset_name.lower()}_data.pkl",
            ]
            cache_file = None
            for cache_dir in cache_candidates:
                for fname in possible_names:
                    cand = os.path.join(cache_dir, fname)
                    if os.path.exists(cand):
                        cache_file = cand
                        break
                if cache_file:
                    break
            
            if cache_file is None or not os.path.exists(cache_file):
                logger.warning(
                    f"[TokenMol3D] No cache found for '{dataset_name}'. "
                    f"Searched: {cache_candidates}. "
                    f"Run: python scripts/precompute_tokenmol_3d.py --dataset {dataset_name}"
                )
                return
            
            import pickle
            logger.info(f"[TokenMol3D] Injecting from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_dict = pickle.load(f)
            
            hit, miss = 0, 0
            for i, smi in enumerate(smiles_list):
                if smi in cache_dict:
                    tokens = cache_dict[smi]
                    if isinstance(tokens, list):
                        self.featurized_drugs[i]["tokenmol_seq"] = " ".join(tokens)
                    else:
                        self.featurized_drugs[i]["tokenmol_seq"] = tokens
                    hit += 1
                else:
                    smi_tokens = self._tokenmol_tokenize_smiles(smi, vocab=vocab)
                    self.featurized_drugs[i]["tokenmol_seq"] = " ".join(smi_tokens)
                    miss += 1
            
            logger.info(f"[TokenMol3D] Injected: {hit} hits, {miss} misses (total {hit+miss})")
        except Exception as exc:
            logger.warning(f"[TokenMol3D] Injection failed: {type(exc).__name__}: {exc}", exc_info=True)

    @staticmethod
    def _tokenmol_tokenize_smiles(smiles: str, vocab: dict) -> list:
        
        if not smiles or not vocab:
            return ["[UNK]"]
        max_len = 6
        out = []
        i = 0
        while i < len(smiles):
            if smiles[i].isspace():
                i += 1
                continue
            matched = None
            for L in range(min(max_len, len(smiles) - i), 0, -1):
                sub = smiles[i:i+L]
                if sub in vocab:
                    matched = sub
                    break
            if matched is None:
                out.append("[UNK]")
                i += 1
            else:
                out.append(matched)
                i += len(matched)
        return out

    @staticmethod
    def _torsion_bins(smiles: str, max_torsions: int = 32) -> list:
        
        if not smiles:
            return []
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            cid = AllChem.EmbedMolecule(mol, params)
            if cid < 0:
                return []
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()

            rot_bond_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
            matches = mol.GetSubstructMatches(rot_bond_smarts)
            torsions = []
            for (j, k) in matches:
                j_atom = mol.GetAtomWithIdx(j)
                k_atom = mol.GetAtomWithIdx(k)
                j_nbrs = [a.GetIdx() for a in j_atom.GetNeighbors() if a.GetIdx() != k]
                k_nbrs = [a.GetIdx() for a in k_atom.GetNeighbors() if a.GetIdx() != j]
                if not j_nbrs or not k_nbrs:
                    continue
                i_idx = j_nbrs[0]
                l_idx = k_nbrs[0]
                angle = rdMolTransforms.GetDihedralRad(conf, i_idx, j, k, l_idx)
                while angle <= -math.pi:
                    angle += 2 * math.pi
                while angle > math.pi:
                    angle -= 2 * math.pi
                q = max(-3.14, min(3.14, round(angle, 2)))
                if abs(q) < 0.0005:
                    q = 0.0
                torsions.append(f"{q:.2f}")
                if len(torsions) >= max_torsions:
                    break
            return torsions
        except Exception:
            return []

    def _build_tokenmol_seq(self, smiles: str, vocab: dict, use_torsion: bool, max_torsions: int) -> str:
        
        smi_tokens = self._tokenmol_tokenize_smiles(smiles, vocab=vocab)
        tokens = list(smi_tokens)
        if use_torsion:
            tors = self._torsion_bins(smiles, max_torsions=max_torsions)
            if tors:
                tokens += ["GEO"] + tors
        return " ".join(tokens)

    def _load_kg(self):

        if len(self.config["drug"]["modality"]) > 1:
            kg_config = self.config["drug"]["featurizer"]["kg"]
            self.kg = SUPPORTED_KG[kg_config["kg_name"]](kg_config["kg_path"])
            # TODO:
            if kg_config["kg_name"] == "BMKG":
                self.drug2kg, self.drug2text, _, _ = self.kg.link(self)
            else:
                self.drug2kg, self.drug2text, _, _ = self.kg.link_chebi20(self)
            # TODO: dp use TransE, don't need filter_out?
            filter_out = []
            """
            for i_drug in data_index:
                smi = self.smiles[i_drug]
                #if smi in self.drug2kg:
                #    filter_out.append((self.drug2kg[smi], self.protein2kg[protein]))
            """
            if kg_config["kg_name"] == "BMKG":
                save_path = self.config["drug"]["featurizer"]["kg"]["save_path"]
                kge_file = kg_config.get("kge_file", "embedding.pkl")
                kge_path = os.path.join(save_path, kge_file) if save_path else kge_file
                
                if os.path.exists(kge_path):
                    import pickle
                    logger.info(f"Loading pre-trained KGE from {kge_path}")
                    with open(kge_path, 'rb') as f:
                        kge = pickle.load(f)
                    logger.info(f"Loaded KGE with {len(kge)} entities (high-quality pre-trained)")
                else:
                    logger.warning(f"Pre-trained KGE not found at {kge_path}, falling back to ProNE...")
                    prone_cache_path = os.path.join(save_path, "prone_embedding.pkl") if save_path else "prone_embedding.pkl"
                    kge = embed(self.kg, 'ProNE', filter_out=filter_out, dim=kg_config["embed_dim"], 
                                save=True, save_path=prone_cache_path)
                    logger.info(f"ProNE KGE has {len(kge)} entities")
            else:
                kge = None
            self.config["drug"]["featurizer"]["kg"]["kge"] = kge
            self.kge = kge

    def _build(self):
        # featurize all data pairs in one pass for training efficency
        self._load_kg()
        self._configure_featurizer()
        if self.in_memory:
            self._featurize()

    def _configure_featurizer(self, save_path=""):

        if len(self.config["drug"]["modality"]) > 1:
            self.drug_featurizer = DrugMultiModalFeaturizer(self.config["drug"])
            self.drug_featurizer.set_drug2kgid_dict(self.drug2kg)
            self.drug_featurizer.set_drug2text_dict(self.drug2text)
        else:
            drug_feat_config = self.config["drug"]["featurizer"]["structure"]
            self.drug_featurizer = SUPPORTED_DRUG_FEATURIZER[drug_feat_config["name"]](drug_feat_config)

    def index_select(self, indexes):
        new_dataset = copy.copy(self)
        new_dataset.drugs = [new_dataset.drugs[i] for i in indexes]
        new_dataset.labels = [new_dataset.labels[i] for i in indexes]
        if hasattr(new_dataset, 'featurized_drugs'):
            new_dataset.featurized_drugs = [new_dataset.featurized_drugs[i] for i in indexes]
        if hasattr(new_dataset, 'smiles'):
            new_dataset.smiles = [new_dataset.smiles[i] for i in indexes]
        if hasattr(self, 'train_index'):
            if set(indexes).issubset(set(self.train_index)):
                new_dataset.train_index = set(range(len(indexes)))
            else:
                new_dataset.train_index = set()
        return new_dataset

    def __getitem__(self, index):
        if not self.in_memory:
            drug, label = self.drugs[index], self.labels[index]
            # if hasattr(self, 'train_index') and isinstance(drug, str):
            #     is_train = index in self.train_index
            #     
            #     if is_train:
            #         try:
            #             from rdkit.Chem import AllChem
            #             mol = AllChem.MolFromSmiles(drug)
            #             if mol is not None:
            #                 drug = AllChem.MolToSmiles(mol, isomericSmiles=True, canonical=False, doRandom=True)
            #         except Exception as e:
            #             pass
            return self.drug_featurizer(drug), label
        else:
            return self.featurized_drugs[index], self.labels[index]

    def __len__(self):
        return len(self.drugs)


class MoleculeNetDataset(DPDataset):
    name2target = {
        "BBBP": ["p_np"],
        "Tox21": ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],
        "ClinTox": ["CT_TOX", "FDA_APPROVED"],
        "HIV": ["HIV_active"],
        "Bace": ["class"],
        "SIDER": ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
                  "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
                  "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
                  "Reproductive system and breast disorders",
                  "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                  "General disorders and administration site conditions", "Endocrine disorders",
                  "Surgical and medical procedures", "Vascular disorders",
                  "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
                  "Congenital, familial and genetic disorders", "Infections and infestations",
                  "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
                  "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
                  "Ear and labyrinth disorders", "Cardiac disorders",
                  "Nervous system disorders", "Injury, poisoning and procedural complications"],
        "MUV": ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
                'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
                'MUV-652', 'MUV-466', 'MUV-832'],
        "Toxcast": [""],  # 617
        "FreeSolv": ["expt"],
        "ESOL": ["measured log solubility in mols per litre"],
        "Lipo": ["exp"],
        "qm7": ["u0_atom"],
        "qm8": ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
                "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"],
        "qm9": ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
    }
    name2task = {
        "BBBP": Task.CLASSFICATION,
        "Tox21": Task.CLASSFICATION,
        "ClinTox": Task.CLASSFICATION,
        "HIV": Task.CLASSFICATION,
        "Bace": Task.CLASSFICATION,
        "SIDER": Task.CLASSFICATION,
        "MUV": Task.CLASSFICATION,
        "Toxcast": Task.CLASSFICATION,
        "FreeSolv": Task.REGRESSION,
        "ESOL": Task.REGRESSION,
        "Lipo": Task.REGRESSION,
        "qm7": Task.REGRESSION,
        "qm8": Task.REGRESSION,
        "qm9": Task.REGRESSION
    }

    name2folder = {
        "Lipo": "lipophilicity",
        "FreeSolv": "freesolv",
    }
    
    def __init__(self, fold_path, config, name="BBBP", split_strategy="scaffold"):
        if name not in self.name2target:
            raise ValueError("%s is not a valid moleculenet task!" % name)
        folder_name = self.name2folder.get(name, name.lower())
        file_name = os.listdir(os.path.join(fold_path, folder_name, "raw"))[0]
        assert file_name[-4:] == ".csv"
        path = os.path.join(fold_path, folder_name, "raw", file_name)
        self._folder_name = folder_name
        self.name = name
        self.targets = self.name2target[name]
        
        if self.targets == [""] or (len(self.targets) == 1 and self.targets[0] == ""):
            import pandas as pd
            df = pd.read_csv(path, nrows=0)
            self.targets = [col for col in df.columns if col.lower() != 'smiles']
            logger.info(f"[{name}] Dynamic targets loaded: {len(self.targets)} tasks")
        # TODO: del: no use
        self.task = self.name2task[name]
        self.split_strategy = split_strategy
        super(MoleculeNetDataset, self).__init__(path, config)
        self.processed_file_path = self.get_processed_file_path(fold_path, self._folder_name, config)
        processed_dir = os.path.dirname(self.processed_file_path)
        os.makedirs(processed_dir, exist_ok=True)

        def _is_rank0():
            # best-effort: avoid multi-process cache migration races
            try:
                return int(os.environ.get("RANK", "0")) == 0
            except Exception:
                return True

        def _validate_cache(featurized_drugs) -> Tuple[bool, str]:
            
            try:
                modality = list(self.config["drug"]["modality"])
                if not modality:
                    return False, "empty modality"
                if len(modality) > 1:
                    if not featurized_drugs or (not isinstance(featurized_drugs[0], dict)):
                        return False, "expected dict samples for multi-modal cache"
                    keys = set(featurized_drugs[0].keys())
                    missing = [m for m in modality if m not in keys]
                    if missing:
                        return False, f"missing keys: {missing}"
                else:
                    if not featurized_drugs:
                        return False, "empty featurized_drugs"
                return True, "ok"
            except Exception as e:
                return False, f"validate exception: {e}"

        def _load_processed(p: str):
            return torch.load(p, weights_only=False)

        candidate_paths = []
        if os.path.exists(self.processed_file_path):
            candidate_paths.append(self.processed_file_path)
        else:
            # legacy fallback: scan all data_processed*.pt under processed/
            try:
                for fn in os.listdir(processed_dir):
                    if not (fn.startswith("data_processed") and fn.endswith(".pt")):
                        continue
                    candidate_paths.append(os.path.join(processed_dir, fn))
            except Exception:
                pass
            uniq = []
            seen = set()
            for p in [self.processed_file_path] + candidate_paths:
                if p in seen:
                    continue
                seen.add(p)
                if os.path.exists(p):
                    uniq.append(p)
            uniq.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            if os.path.exists(self.processed_file_path):
                uniq = [self.processed_file_path] + [p for p in uniq if p != self.processed_file_path]
            candidate_paths = uniq
        
        if bool(config.get("force_reprocess", False)) and os.path.exists(self.processed_file_path):
            logger.info(f"Deleting cached file to force re-processing: {self.processed_file_path}")
            os.remove(self.processed_file_path)
        if bool(config.get("force_reprocess", False)):
            candidate_paths = []
        
        loaded_ok = False
        for cand in candidate_paths:
            if not os.path.exists(cand):
                continue
            try:
                logger.info(f"Loading processed data from {cand}")
                loaded_data = _load_processed(cand)
                
                if isinstance(loaded_data, dict):
                    self.featurized_drugs = loaded_data['featurized_drugs']
                    self.labels = loaded_data['labels']
                    self.drugs = loaded_data['drugs']
                    if 'folds' in loaded_data:
                        self.folds = loaded_data['folds']
                    else:
                        self.train_index = loaded_data['train_index']
                        self.val_index = loaded_data['val_index']
                        self.test_index = loaded_data['test_index']
                    
                    normalizer_states = loaded_data.get('normalizer_states')
                    if normalizer_states:
                        from utils.utils import Normalizer
                        self.normalizer = []
                        for state in normalizer_states:
                            if state is not None:
                                norm = Normalizer.__new__(Normalizer)
                                norm.load_state_dict(state)
                                self.normalizer.append(norm)
                            else:
                                self.normalizer.append(None)
                        logger.info(f"[{self.name}]  Normalizer restored from cache")
                    else:
                        self.normalizer = [None] * len(self.targets)
                        
                elif isinstance(loaded_data, tuple):
                    if len(loaded_data) == 4:
                        self.featurized_drugs, self.labels, self.drugs, self.folds = loaded_data
                    elif len(loaded_data) == 6:
                        self.featurized_drugs, self.labels, self.drugs, self.train_index, self.val_index, self.test_index = loaded_data
                    else:
                        logger.warning(f" Unsupported processed cache format (len={len(loaded_data)}) at {cand}")
                        continue
                    self.normalizer = [None] * len(self.targets)
                    if self.name2task.get(self.name) == Task.REGRESSION:
                        logger.warning(f"[{self.name}]  Old cache format without normalizer. Please delete cache and re-process!")
                else:
                    logger.warning(f" Unknown cache format at {cand}")
                    continue

                ok, reason = _validate_cache(self.featurized_drugs)
                if not ok:
                    logger.warning(f" Processed cache mismatch for current modality: {reason}. Cache={cand}")
                    if bool(config.get("auto_delete_bad_cache", True)):
                        try:
                            logger.warning(f"Deleting bad cache: {cand}")
                            os.remove(cand)
                        except Exception as e:
                            logger.warning(f"Failed to delete bad cache {cand}: {e}")
                    continue
                
                self._load_kg()
                loaded_ok = True

                if cand != self.processed_file_path and _is_rank0() and bool(config.get("migrate_cache", True)):
                    try:
                        if not os.path.exists(self.processed_file_path):
                            torch.save(loaded_data, self.processed_file_path)
                            logger.info(f" Migrated legacy cache -> {self.processed_file_path}")
                    except Exception as e:
                        logger.warning(f"Cache migration failed: {e}")
                break
            except Exception as e:
                logger.warning(f" Failed to load cache {cand}: {e}")
                continue

        if loaded_ok:
            
            self._inject_tokenmol3d_from_cache(config)
            
            if split_strategy == 'kfold':
                if not hasattr(self, 'folds'):
                    logger.info(" Cached data has standard split, but kfold requested. Re-splitting...")
                    self._train_test_split(strategy='kfold')
                elif not hasattr(self, 'nfolds'):
                    self.nfolds = len(self.folds)
            else:
                if not hasattr(self, 'train_index'):
                    logger.info(f" Cached data has kfold split, but {split_strategy} requested. Re-splitting...")
                    self._train_test_split(strategy=split_strategy)
        else:
            logger.info(f"Processing raw data from {self.path}")
            self._load_data()
            self._train_test_split(strategy=split_strategy)
            self._normalize()
            self._build()

        # debug
        """    
        index_dict = {"train": self.train_index, "val": self.val_index, "test": self.test_index}
        with open("index.pkl", "wb") as f:
            pickle.dump(index_dict, f)
        """

    @property
    def processed_kg_file_names(self):
        return 'kg_processed.pkl'

    def get_processed_file_path(self, fold_path, folder_name, config):
        """
        get the .pt file path based on modality
        Args:
            fold_path (str): flod path of MoleculeNet
            config (dict): the config of task which used to get the modality
        """
        modality = config["drug"]["modality"]
        assert "structure" in modality
        processed_dir = os.path.join(fold_path, folder_name, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        if len(modality) == 2 and "kg" in modality:
            return os.path.join(processed_dir, "data_processed_kg.pt")
        elif len(modality) == 2 and "text" in modality:
            return os.path.join(processed_dir, "data_processed_text.pt")
        elif len(modality) == 3 and config["drug"]["featurizer"]["kg"]["kg_name"] == "BMKG":
            return os.path.join(processed_dir, "data_processed_bmkgv1.pt")
        elif len(modality) == 3:
            return os.path.join(processed_dir, "data_processed_text_kg.pt")
        elif len(modality) == 1 and config["drug"]["featurizer"]["structure"]["name"] in ["transformer"]:
            return os.path.join(processed_dir, "data_processed_kvplm.pt")
        elif len(modality) > 3:
            mods = [str(m).lower() for m in modality]
            sig = "mm_" + "_".join(mods)
            try:
                struct_name = str(config["drug"]["featurizer"]["structure"].get("name", "")).lower()
            except Exception:
                struct_name = ""
            if struct_name:
                sig += f"__struct_{struct_name}"
            if "kg" in mods:
                try:
                    kg_name = str(config["drug"]["featurizer"]["kg"].get("kg_name", "")).lower()
                except Exception:
                    kg_name = ""
                if kg_name:
                    sig += f"__kg_{kg_name}"
            # sanitize
            safe = []
            for ch in sig:
                safe.append(ch if (ch.isalnum() or ch in "_-") else "_")
            sig = "".join(safe)
            return os.path.join(processed_dir, f"data_processed_{sig}.pt")
        else:
            return os.path.join(processed_dir, "data_processed.pt")

    def _load_data(self):
        loader_name = f"_load_{self._folder_name}_dataset"
        smiles_list, rdkit_mol_objs, labels = getattr(globals()["dpload"], loader_name)(
            self.path)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        self.smiles, self.drugs, self.labels, self.mols = [], [], [], []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is None:
                continue
            # TODO: drugs and smiles are all get from AllChem.MolFromSmiles()
            self.smiles.append(smiles_list[i])
            self.drugs.append(smiles_list[i])
            self.mols.append(rdkit_mol_objs[i])
            self.labels.append(labels[i])

    """        
    def _load_data(self):
        # process the raw data which readed from file 
        # return: self.smiles, self.drugs, self.labels
        data = pd.read_csv(self.path)
        smiles = data['smiles'].to_numpy()
        if self.label_type == 1:
            labels = data[self.targets]
            labels = labels.fillna(0)
            labels = labels.to_numpy()
        else:
            labels = data[self.targets]
            # convert 0 to -1
            labels = labels.replace(0, -1)
            # convert nan to 0
            labels = labels.fillna(0)
            labels = labels.to_numpy()

        self.smiles, self.drugs, self.labels = [], [], []
        for i, drug in enumerate(smiles):
            mol = AllChem.MolFromSmiles(drug)
            # mol = Chem.MolFromSmiles(drug)

            if mol is not None:
                # self.smiles.append(drug)
                self.smiles.append(AllChem.MolToSmiles(mol))
                self.drugs.append(drug)
                self.labels.append(labels[i])
        # Debug

        # data_smiles_series = pd.Series(self.smiles)
        # floder_path = os.path.dirname(self.path)
        # file_name = os.path.join(floder_path, "smiles_2.csv")
        # data_smiles_series.to_csv(file_name, index=False, header=False)
        # TODO:
        if self.name == "qm9":
            for target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
                self.labels[:, self.targets.index(target)] *= 27.211386246
    """

    def _train_test_split(self, strategy="scaffold"):
        if strategy == "random":
            self.train_index, self.val_index, self.test_index = random_split(len(self), 0.1, 0.1)
        elif strategy == "scaffold":
            self.train_index, self.val_index, self.test_index = scaffold_split(self, 0.1, 0.1, is_standard=True)
        elif strategy == "kfold":
            self.nfolds = 5
            folds = kfold_split(len(self), self.nfolds)
            self.folds = []
            for i in range(self.nfolds):
                train_indices = np.concatenate(folds[:i] + folds[i + 1:], axis=0).tolist()
                test_indices = folds[i].tolist()
                val_size = int(len(train_indices) * 0.1)
                val_indices = train_indices[:val_size]
                train_indices = train_indices[val_size:]
                self.folds.append({
                    "train": train_indices,
                    "val": val_indices,
                    "test": test_indices
                })

    def _normalize(self):
        if self.name2task.get(self.name) == Task.REGRESSION:
            if isinstance(self.labels, list):
                labels_array = np.array(self.labels)
            else:
                labels_array = self.labels
            
            if labels_array.ndim == 1:
                labels_array = np.expand_dims(labels_array, axis=1)
            
            self.normalizer = []
            for i in range(len(self.targets)):
                col = torch.tensor(labels_array[:, i], dtype=torch.float32)
                self.normalizer.append(Normalizer(col))
                labels_array[:, i] = self.normalizer[i].norm(col).numpy()
            
            self.labels = [labels_array[j] for j in range(labels_array.shape[0])]
            logger.info(f"[{self.name}] Standardized {len(self.targets)} regression targets to mean 0 and std 1")
        else:
            self.normalizer = [None] * len(self.targets)


SUPPORTED_DP_DATASETS = {
    "MoleculeNet": MoleculeNetDataset
}