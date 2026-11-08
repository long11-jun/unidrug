#!/usr/bin/env python3
"""
Rebuild PPI DFS split files from the deduplicated protein pairs used by PPIDataset.

    python scripts/regenerate_ppi_dfs_split.py
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.setrecursionlimit(10000)

PPI_CLASS_MAP = {
    "reaction": 0,
    "binding": 1,
    "ptmod": 2,
    "activation": 3,
    "inhibition": 4,
    "catalysis": 5,
    "expression": 6,
}


def load_deduplicated_pairs(data_path):
    """
    """
    seq_file = os.path.join(data_path, "sequences.tsv")
    proteins = []
    protname2id = {}
    
    with open(seq_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                prot_name = parts[0]
                sequence = parts[1]
                protname2id[prot_name] = i
                proteins.append({"id": prot_name, "sequence": sequence})
    
    print(f"Loaded {len(proteins)} proteins")
    
    interaction_file = os.path.join(data_path, "interactions.txt")
    ppi_dict = {}  # (prot1_id, prot2_id) -> [label_vector]
    
    with open(interaction_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
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
            
            if prot1_name not in protname2id or prot2_name not in protname2id:
                continue
            
            prot1_id = protname2id[prot1_name]
            prot2_id = protname2id[prot2_name]
            
            if prot1_id > prot2_id:
                prot1_id, prot2_id = prot2_id, prot1_id
            
            key = (prot1_id, prot2_id)
            if key not in ppi_dict:
                ppi_dict[key] = [0] * 7
            
            if interaction_type in PPI_CLASS_MAP:
                class_idx = PPI_CLASS_MAP[interaction_type]
                ppi_dict[key][class_idx] = 1
    
    pairs = []
    for (prot1_id, prot2_id), label_vector in ppi_dict.items():
        pairs.append({
            "prot1_id": prot1_id,
            "prot2_id": prot2_id,
            "labels": label_vector
        })
    
    print(f"Loaded {len(pairs)} deduplicated protein pairs")
    return proteins, pairs, protname2id


def build_protein_graph(pairs, n_proteins):
    
    graph = defaultdict(set)
    for pair in pairs:
        prot1_id = pair["prot1_id"]
        prot2_id = pair["prot2_id"]
        graph[prot1_id].add(prot2_id)
        graph[prot2_id].add(prot1_id)
    return graph


def dfs_split(pairs, n_proteins, test_ratio=0.1, seed=42):
    """
    
    """
    np.random.seed(seed)
    
    graph = build_protein_graph(pairs, n_proteins)
    n_pairs = len(pairs)
    n_test = int(n_pairs * test_ratio)
    
    visited = set()
    order = []
    
    degrees = [(i, len(graph[i])) for i in range(n_proteins)]
    degrees.sort(key=lambda x: -x[1])
    
    for start_node, _ in degrees:
        if start_node in visited:
            continue
        
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            
            neighbors = sorted(graph[node], key=lambda x: -len(graph[x]))
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    n_test_proteins = int(n_proteins * test_ratio * 1.5)
    test_proteins = set(order[-n_test_proteins:])
    
    print(f"Test proteins: {len(test_proteins)} / {n_proteins}")
    
    train_indices = []
    test_indices = []
    
    for i, pair in enumerate(pairs):
        prot1_id = pair["prot1_id"]
        prot2_id = pair["prot2_id"]
        
        if prot1_id in test_proteins or prot2_id in test_proteins:
            test_indices.append(i)
        else:
            train_indices.append(i)
    
    if len(test_indices) < n_test * 0.5:
        extra_count = min(n_test - len(test_indices), len(train_indices) // 10)
        if extra_count > 0:
            extra = np.random.choice(train_indices, size=extra_count, replace=False).tolist()
            test_indices.extend(extra)
            train_set = set(extra)
            train_indices = [i for i in train_indices if i not in train_set]
    elif len(test_indices) > n_test * 1.5:
        np.random.shuffle(test_indices)
        extra = test_indices[n_test:]
        test_indices = test_indices[:n_test]
        train_indices.extend(extra)
    
    train_indices.sort()
    test_indices.sort()
    
    print(f"DFS Split: train={len(train_indices)}, test={len(test_indices)}")
    return train_indices, test_indices


def main():
    datasets = ["SHS27k", "SHS148k"]
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unidrug", "datasets", "ppi")
    
    for dataset in datasets:
        data_path = os.path.join(base_path, dataset)
        if not os.path.exists(data_path):
            print(f"Skipping {dataset} (not found)")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {dataset}")
        print(f"{'='*50}")
        
        proteins, pairs, protname2id = load_deduplicated_pairs(data_path)
        n_proteins = len(proteins)
        
        train_idx, test_idx = dfs_split(pairs, n_proteins, test_ratio=0.1, seed=42)
        
        split_file = os.path.join(data_path, "split_dfs.json")
        backup_file = os.path.join(data_path, "split_dfs_backup.json")
        
        if os.path.exists(split_file):
            import shutil
            shutil.copy(split_file, backup_file)
            print(f"Backed up to {backup_file}")
        
        with open(split_file, 'w') as f:
            json.dump({"train": train_idx, "test": test_idx}, f)
        
        print(f"Saved new DFS split to {split_file}")
        
        total = len(train_idx) + len(test_idx)
        print(f"Verification: total pairs = {len(pairs)}, split covers = {total}")
        assert total == len(pairs), f"Mismatch! {total} vs {len(pairs)}"
        
        max_idx = max(max(train_idx), max(test_idx))
        print(f"Max index: {max_idx}, should be < {len(pairs)}")
        assert max_idx < len(pairs), f"Index out of range!"


if __name__ == "__main__":
    main()
