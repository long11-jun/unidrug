#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the Human DTI TokenMol cache in one process.

Writes cache/tokenmol_3d/tokenmol_3d_human.pkl.

    python scripts/precompute_tokenmol_human.py
"""

import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms


def load_tokenmol_vocab(vocab_file: str):
    
    vocab = {}
    with open(vocab_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) >= 1:
                token = parts[0]
                vocab[token] = i
    return vocab


def tokenmol_tokenize_smiles(smiles, vocab):
    
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


def extract_torsions(smiles, max_torsions=32):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxIterations = 200
        
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass
        
        conf = mol.GetConformer()
        pattern = Chem.MolFromSmarts("[!#1]~[!#1]~[!#1]~[!#1]")
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


def build_tokenmol_seq(smiles, vocab):
    
    tokens = tokenmol_tokenize_smiles(smiles, vocab)
    if not tokens:
        return None
    
    torsions = extract_torsions(smiles)
    if torsions:
        tokens.append("[GEO]")
        tokens.extend(torsions)
    
    return " ".join(tokens)


def main():
    vocab_file = PROJECT_ROOT / "torsion_version" / "torsion_voc_property.csv"
    dataset_path = PROJECT_ROOT / "unidrug" / "datasets" / "dti" / "human"
    output_dir = PROJECT_ROOT / "cache" / "tokenmol_3d"
    output_file = output_dir / "tokenmol_3d_human.pkl"
    failed_file = output_dir / "tokenmol_3d_human_failed.txt"
    
    vocab = load_tokenmol_vocab(str(vocab_file))
    print(f"[INFO] Vocab size: {len(vocab)}")
    
    smiles_list = []
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = dataset_path / csv_file
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "SMILES" in df.columns:
                smiles_list.extend(df["SMILES"].dropna().tolist())
    
    unique_smiles = list(set(smiles_list))
    print(f"[INFO] Total SMILES: {len(smiles_list)}, unique: {len(unique_smiles)}")
    
    results = {}
    failed = []
    
    for smi in tqdm(unique_smiles, desc="Processing"):
        try:
            seq = build_tokenmol_seq(smi, vocab)
            if seq:
                results[smi] = seq
            else:
                failed.append((smi, "no_seq"))
        except Exception as e:
            failed.append((smi, str(e)))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    with open(failed_file, 'w') as f:
        for smi, err in failed:
            f.write(f"{smi}\t{err}\n")
    
    print(f"[DONE] Success: {len(results)}, Failed: {len(failed)}")
    print(f"[DONE] Saved to: {output_file}")


if __name__ == "__main__":
    main()
