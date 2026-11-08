#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build FoldSeek 3Di sequences for DTI proteins.

Modes: esmfold (GPU structure prediction), foldseek_api, from_pdb, seq_approx.

    python scripts/precompute_foldseek_dti.py --dataset yamanishi08 --mode esmfold
    python scripts/precompute_foldseek_dti.py --dataset yamanishi08 --mode from_pdb --pdb_dir /path/to/pdbs
    python scripts/precompute_foldseek_dti.py --dataset yamanishi08 --mode seq_approx
    python scripts/precompute_foldseek_dti.py --all --mode esmfold
"""

import os
import sys
import json
import argparse
import pickle
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DTI_DATASETS = {
    "yamanishi08": "unidrug/datasets/dti/Yamanishi08",
    "bmkg_dti": "unidrug/datasets/dti/BMKG_DTI",
    "human": "unidrug/datasets/dti/human",
}

FOLDSEEK_3DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

AA_TO_3DI_APPROX = {
    'A': 'D', 'R': 'N', 'N': 'Q', 'D': 'E', 'C': 'A',
    'Q': 'N', 'E': 'D', 'G': 'G', 'H': 'K', 'I': 'V',
    'L': 'I', 'K': 'R', 'M': 'L', 'F': 'W', 'P': 'P',
    'S': 'T', 'T': 'S', 'W': 'F', 'Y': 'H', 'V': 'C',
    'X': 'G', 'U': 'A', 'O': 'K', 'B': 'E', 'Z': 'D',
    'J': 'I',
}


def seq_hash(seq: str) -> str:
    
    return hashlib.md5(seq.encode()).hexdigest()[:12]


def load_protein_sequences(dataset_name: str, project_root: Path) -> Dict[str, str]:
    
    import pandas as pd
    
    dataset_path = project_root / DTI_DATASETS.get(dataset_name.lower(), "")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return {}
    
    sequences = set()
    
    if dataset_name.lower() == "human":
        for csv_file in ["train.csv", "val.csv", "test.csv"]:
            csv_path = dataset_path / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                seq_col = None
                for c in ["Sequence", "sequence", "target_sequence", "Target_Sequence"]:
                    if c in df.columns:
                        seq_col = c
                        break
                if seq_col:
                    sequences.update(df[seq_col].dropna().tolist())
    else:
        prot_json = dataset_path / "protein.json"
        if prot_json.exists():
            with open(prot_json, 'r') as f:
                prot_data = json.load(f)
            for pid, info in prot_data.items():
                seq = info.get("sequence", "")
                if seq:
                    sequences.add(seq)
    
    print(f"[INFO] Dataset: {dataset_name}, unique proteins: {len(sequences)}")
    return {seq: seq for seq in sequences}


def predict_structure_esmfold(sequence: str, output_pdb: str) -> bool:
    
    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
        
        global _esmfold_model, _esmfold_tokenizer
        
        if '_esmfold_model' not in globals() or _esmfold_model is None:
            print("[ESMFold] Loading model (first time)...")
            _esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            _esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            _esmfold_model = _esmfold_model.cuda().eval()
            print("[ESMFold] Model loaded!")
        
        max_len = 1000
        seq = sequence[:max_len]
        
        with torch.no_grad():
            inputs = _esmfold_tokenizer([seq], return_tensors="pt", 
                                         padding=False, add_special_tokens=False)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = _esmfold_model(**inputs)
        
        pdb_string = outputs.to_pdb()[0]
        with open(output_pdb, 'w') as f:
            f.write(pdb_string)
        
        return True
    except Exception as e:
        print(f"[ESMFold] Error: {e}")
        return False


def run_foldseek_createdb(pdb_dir: str, db_path: str) -> bool:
    
    try:
        cmd = ["foldseek", "createdb", pdb_dir, db_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"[FoldSeek] createdb error: {e}")
        return False


def extract_3di_from_db(db_path: str) -> Dict[str, str]:
    
    results = {}
    ss_file = db_path + "_ss"
    header_file = db_path + "_h"
    
    try:
        headers = []
        if os.path.exists(header_file):
            with open(header_file, 'r') as f:
                for line in f:
                    headers.append(line.strip().split('\t')[0].split()[0])
        
        seqs = []
        if os.path.exists(ss_file):
            with open(ss_file, 'r') as f:
                for line in f:
                    seqs.append(line.strip().split('\t')[0] if '\t' in line else line.strip())
        
        for i, (h, s) in enumerate(zip(headers, seqs)):
            results[h] = s
    except Exception as e:
        print(f"[FoldSeek] extract error: {e}")
    
    return results


def pdb_to_3di_foldseek(pdb_path: str) -> Optional[str]:
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "query_db")
            out_path = os.path.join(tmpdir, "3di_out")
            
            # createdb
            cmd1 = ["foldseek", "createdb", pdb_path, db_path]
            r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
            if r1.returncode != 0:
                return None
            
            ss_file = db_path + "_ss"
            if os.path.exists(ss_file):
                with open(ss_file, 'rb') as f:
                    content = f.read()
                cmd2 = ["foldseek", "convert2fasta", db_path + "_ss", out_path + ".fasta"]
                r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
                if r2.returncode == 0 and os.path.exists(out_path + ".fasta"):
                    with open(out_path + ".fasta", 'r') as f:
                        lines = f.readlines()
                    seq_3di = ""
                    for line in lines:
                        if not line.startswith('>'):
                            seq_3di += line.strip()
                    return seq_3di if seq_3di else None
            
            return None
    except Exception as e:
        print(f"[FoldSeek] pdb_to_3di error: {e}")
        return None


def sequence_to_approx_3di(aa_seq: str) -> str:
    """
    
    """
    return "".join(AA_TO_3DI_APPROX.get(aa.upper(), 'G') for aa in aa_seq)


def check_foldseek_installed() -> bool:
    
    try:
        result = subprocess.run(["foldseek", "version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False


def precompute_esmfold_mode(
    sequences: Dict[str, str],
    output_file: Path,
    pdb_cache_dir: Path,
):
    
    pdb_cache_dir.mkdir(parents=True, exist_ok=True)
    
    has_foldseek = check_foldseek_installed()
    if not has_foldseek:
        print("[WARNING] FoldSeek not installed. Will use ESMFold + seq_approx fallback.")
        print("[WARNING] Install: conda install -c conda-forge -c bioconda foldseek")
    
    results = {}
    
    for seq in tqdm(sequences.keys(), desc="ESMFold → 3Di"):
        h = seq_hash(seq)
        pdb_path = pdb_cache_dir / f"{h}.pdb"
        
        if not pdb_path.exists():
            if not predict_structure_esmfold(seq, str(pdb_path)):
                results[seq] = sequence_to_approx_3di(seq)
                continue
        
        # Step 2: PDB → 3Di
        if has_foldseek:
            seq_3di = pdb_to_3di_foldseek(str(pdb_path))
            if seq_3di:
                results[seq] = seq_3di
            else:
                results[seq] = sequence_to_approx_3di(seq)
        else:
            results[seq] = sequence_to_approx_3di(seq)
    
    return results


def precompute_from_pdb_mode(
    sequences: Dict[str, str],
    pdb_dir: Path,
):
    
    has_foldseek = check_foldseek_installed()
    if not has_foldseek:
        print("[ERROR] FoldSeek not installed! Install: conda install -c conda-forge -c bioconda foldseek")
        return {}
    
    results = {}
    
    for seq in tqdm(sequences.keys(), desc="PDB → 3Di"):
        h = seq_hash(seq)
        pdb_path = None
        for pattern in [f"{h}.pdb", f"{h}.cif"]:
            p = pdb_dir / pattern
            if p.exists():
                pdb_path = p
                break
        
        if pdb_path is None:
            results[seq] = sequence_to_approx_3di(seq)
            continue
        
        seq_3di = pdb_to_3di_foldseek(str(pdb_path))
        if seq_3di:
            results[seq] = seq_3di
        else:
            results[seq] = sequence_to_approx_3di(seq)
    
    return results


def precompute_seq_approx_mode(sequences: Dict[str, str]):
    
    results = {}
    for seq in tqdm(sequences.keys(), desc="Seq → Approx 3Di"):
        results[seq] = sequence_to_approx_3di(seq)
    return results


def precompute_dataset(
    dataset_name: str,
    project_root: Path,
    output_dir: Path,
    mode: str = "seq_approx",
    pdb_dir: Optional[Path] = None,
):
    
    sequences = load_protein_sequences(dataset_name, project_root)
    if not sequences:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"foldseek_3di_{dataset_name.lower()}.pkl"
    
    if mode == "esmfold":
        pdb_cache = output_dir / "pdb_cache" / dataset_name
        results = precompute_esmfold_mode(sequences, output_file, pdb_cache)
    elif mode == "from_pdb":
        if pdb_dir is None:
            print("[ERROR] --pdb_dir required for from_pdb mode")
            return
        results = precompute_from_pdb_mode(sequences, pdb_dir)
    elif mode == "seq_approx":
        results = precompute_seq_approx_mode(sequences)
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        return
    
    real_3di = sum(1 for v in results.values() if any(c in v for c in "ACDEFGHIKLMNPQRSTVWY"))
    approx_count = 0
    for seq, di3 in results.items():
        if di3 == sequence_to_approx_3di(seq):
            approx_count += 1
    
    print(f"\n[Results] Total: {len(results)}")
    print(f"[Results] With 3Di: {real_3di}, Approx fallback: {approx_count}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"[DONE] Saved to: {output_file}")
    
    avg_len = sum(len(v) for v in results.values()) / max(len(results), 1)
    print(f"[Stats] Avg 3Di length: {avg_len:.1f}")
    
    for i, (seq, di3) in enumerate(list(results.items())[:3]):
        print(f"[Sample {i+1}] AA: {seq[:50]}...")
        print(f"           3Di: {di3[:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Precompute FoldSeek 3Di for DTI proteins")
    parser.add_argument("--dataset", type=str, default="yamanishi08",
                        help="Dataset name (yamanishi08, bmkg_dti, human)")
    parser.add_argument("--all", action="store_true", help="Process all DTI datasets")
    parser.add_argument("--mode", type=str, default="seq_approx",
                        choices=["esmfold", "from_pdb", "seq_approx"],
                        help="3Di generation mode")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="PDB directory (for from_pdb mode)")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "cache" / "foldseek_3di"
    
    if args.all:
        datasets = list(DTI_DATASETS.keys())
    else:
        datasets = [args.dataset]
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {ds} (mode={args.mode})")
        print(f"{'='*60}")
        precompute_dataset(
            ds, project_root, output_dir, 
            mode=args.mode,
            pdb_dir=Path(args.pdb_dir) if args.pdb_dir else None,
        )


if __name__ == "__main__":
    main()
