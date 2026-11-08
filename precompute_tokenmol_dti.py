#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算 DTI 数据集的 Token-Mol 3D Torsion Tokens
================================================
为 DTI 数据集（Yamanishi08, BMKG_DTI 等）预计算 MolToken

用法：
    python scripts/precompute_tokenmol_dti.py --dataset yamanishi08
    python scripts/precompute_tokenmol_dti.py --all
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import multiprocessing as mp
import time
import signal

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "unidrug"))

# RDKit 相关
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not available")

# DTI 数据集路径
DTI_DATASETS = {
    "yamanishi08": "unidrug/datasets/dti/Yamanishi08",
    "bmkg_dti": "unidrug/datasets/dti/BMKG_DTI",
    "human": "unidrug/datasets/dti/human",
}


def load_tokenmol_vocab(vocab_file: str) -> Dict[str, int]:
    """加载 TokenMol 词表"""
    vocab = {}
    with open(vocab_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) >= 1:
                token = parts[0]
                vocab[token] = i
    return vocab


def _tokenmol_tokenize_smiles(smiles: str, vocab: Dict[str, int]) -> List[str]:
    """贪心最长匹配 Token-Mol 词表"""
    if not smiles or not vocab:
        return ["[UNK]"]
    max_len = 6
    out: List[str] = []
    i = 0
    while i < len(smiles):
        if smiles[i].isspace():
            i += 1
            continue
        matched = None
        for L in range(min(max_len, len(smiles) - i), 0, -1):
            sub = smiles[i : i + L]
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


def _torsion_bins_rad_strict(smiles: str, max_torsions: int = 32) -> Optional[List[str]]:
    """严格 3D 扭转角提取"""
    if not smiles or not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxIterations = 200
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass
        
        conf = mol.GetConformer()
        torsion_smarts = "[!#1]~[!#1]~[!#1]~[!#1]"
        pattern = Chem.MolFromSmarts(torsion_smarts)
        matches = mol.GetSubstructMatches(pattern)
        
        torsion_strs = []
        for i, j, k, l in matches:
            try:
                angle = rdMolTransforms.GetDihedralRad(conf, i, j, k, l)
                torsion_strs.append(f"{angle:.2f}")
            except:
                continue
            if len(torsion_strs) >= max_torsions:
                break
        
        return torsion_strs if torsion_strs else None
    except Exception:
        return None


def build_tokenmol_seq_strict(
    smiles: str, 
    vocab: Dict[str, int], 
    use_torsion: bool = True, 
    max_torsions: int = 32
) -> Optional[str]:
    """构建 TokenMol 序列"""
    smiles_tokens = _tokenmol_tokenize_smiles(smiles, vocab)
    if not smiles_tokens:
        return None
    
    if use_torsion and RDKIT_AVAILABLE:
        torsion_strs = _torsion_bins_rad_strict(smiles, max_torsions)
        if torsion_strs:
            smiles_tokens.append("[GEO]")
            smiles_tokens.extend(torsion_strs)
    
    return " ".join(smiles_tokens)


class _Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _Timeout("timeout")


def process_single_smiles(args) -> Tuple[int, str, Optional[str], float, Optional[str]]:
    """处理单个 SMILES"""
    idx, smiles, vocab, use_torsion, max_torsions, timeout_sec = args
    t0 = time.time()
    
    old_handler = None
    try:
        if timeout_sec and timeout_sec > 0:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout_sec))
        
        seq = build_tokenmol_seq_strict(smiles, vocab, use_torsion, max_torsions)
        return idx, smiles, seq, time.time() - t0, None
    except _Timeout:
        return idx, smiles, None, time.time() - t0, "timeout"
    except Exception as e:
        return idx, smiles, None, time.time() - t0, str(e)
    finally:
        try:
            if timeout_sec and timeout_sec > 0:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        except:
            pass


def precompute_dti_dataset(
    dataset_name: str,
    project_root: Path,
    output_dir: Path,
    vocab_file: str,
    num_workers: int = 8,
    timeout_sec: int = 60,
):
    """预计算 DTI 数据集的 MolToken"""
    import pandas as pd
    
    dataset_path = project_root / DTI_DATASETS.get(dataset_name.lower(), "")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    smiles_list = []
    
    # Human 数据集：从 CSV 文件加载
    if dataset_name.lower() == "human":
        for csv_file in ["train.csv", "val.csv", "test.csv"]:
            csv_path = dataset_path / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if "SMILES" in df.columns:
                    smiles_list.extend(df["SMILES"].dropna().tolist())
        if not smiles_list:
            print(f"[ERROR] No SMILES found in human dataset")
            return
    else:
        # 其他数据集：从 drug.json 加载 SMILES
        drug_json = dataset_path / "drug.json"
        if not drug_json.exists():
            print(f"[ERROR] drug.json not found: {drug_json}")
            return
        
        with open(drug_json, 'r') as f:
            drug_data = json.load(f)
        
        for drug_id, info in drug_data.items():
            smi = info.get("SMILES", "")
            if smi:
                smiles_list.append(smi)
    
    unique_smiles = list(set(smiles_list))
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Total SMILES: {len(smiles_list)}, unique SMILES: {len(unique_smiles)}")
    
    # 加载词表
    vocab = load_tokenmol_vocab(vocab_file)
    print(f"[INFO] Vocab size: {len(vocab)}")
    
    # 并行处理
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"tokenmol_3d_{dataset_name.lower()}.pkl"
    failed_file = output_dir / f"tokenmol_3d_{dataset_name.lower()}_failed.txt"
    
    args_list = [
        (i, smi, vocab, True, 32, timeout_sec) 
        for i, smi in enumerate(unique_smiles)
    ]
    
    results = {}
    failed_list = []
    
    print(f"[INFO] Processing with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        for idx, smi, seq, elapsed, error in tqdm(
            pool.imap_unordered(process_single_smiles, args_list),
            total=len(args_list),
            desc=f"[TokenMol3D] {dataset_name}"
        ):
            if seq is not None:
                results[smi] = seq
            else:
                failed_list.append((smi, error))
    
    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    with open(failed_file, 'w') as f:
        for smi, err in failed_list:
            f.write(f"{smi}\t{err}\n")
    
    print(f"[INFO] Success: {len(results)}, Failed: {len(failed_list)}")
    print(f"[INFO] Saved to: {output_file}")
    print(f"[DONE] Precomputation complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yamanishi08",
                        help="Dataset name (yamanishi08, bmkg_dti)")
    parser.add_argument("--all", action="store_true", help="Process all DTI datasets")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "cache" / "tokenmol_3d"
    vocab_file = project_root / "torsion_version" / "torsion_voc_property.csv"
    
    if not vocab_file.exists():
        print(f"[ERROR] Vocab file not found: {vocab_file}")
        return
    
    if args.all:
        datasets = list(DTI_DATASETS.keys())
    else:
        datasets = [args.dataset]
    
    for ds in datasets:
        print(f"\n{'='*60}")
        precompute_dti_dataset(
            ds, project_root, output_dir, str(vocab_file),
            num_workers=args.workers, timeout_sec=args.timeout
        )


if __name__ == "__main__":
    main()

