import logging
logger = logging.getLogger(__name__)

import math
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from tqdm import tqdm
import json
import collections

from utils.cluster import cluster_with_sim_matrix, merge_cluster
from utils.prot_utils import get_normalized_ctd

def random_split(n, r_val, r_test):
    r_train = 1 - r_val - r_test
    perm = np.random.permutation(n)
    train_cutoff = int(r_train * n)
    val_cutoff = int((r_train + r_val) * n)
    return perm[:train_cutoff], perm[train_cutoff : val_cutoff], perm[val_cutoff:]

def kfold_split(n, k):
    perm = np.random.permutation(n)
    return [perm[i * n // k: (i + 1) * n // k] for i in range(k)]

def _generate_scaffold(smiles, include_chirality=False, is_standard=False):
    if is_standard:
        scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)
    else:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset, log_every_n=1000, sort=True, is_standard=False):
    scaffolds = {}
    data_len = len(dataset)

    logger.info("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles):
        if log_every_n > 0 and ind % log_every_n == 0:
            logger.info("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles, is_standard=is_standard)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    if sort:
        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), 
                key=lambda x: (len(x[1]), x[1][0]), 
                reverse=True
            )
        ]
    else:
        scaffold_sets = [value for key, value in scaffolds.items()]
   
    # TODO: DEBUG
    """
    scaffold_index = collections.OrderedDict()
    for i, value in enumerate(scaffold_sets):
        scaffold_index[i] = str(value)
    scaffold_index = json.dumps(scaffold_index)
    with open("scaffold_set_2.json","w") as f:
        f.write(scaffold_index)
    """
    return scaffold_sets

def scaffold_split(dataset, r_val, r_test, log_every_n=1000, is_standard=False):
    r_train = 1.0 - r_val - r_test
    scaffold_sets = generate_scaffolds(dataset, log_every_n, is_standard=is_standard)

    train_cutoff = r_train * len(dataset)
    valid_cutoff = (r_train + r_val) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

def cold_drug_split(dataset, nfolds):
    scaffold_sets = generate_scaffolds(dataset, -1, sort=False)

    drug_pair_index = {}
    for i, (i_drug, i_prot) in enumerate(dataset.pair_index):
        if i_drug not in drug_pair_index:
            drug_pair_index[i_drug] = [i]
        else:
            drug_pair_index[i_drug].append(i)

    scaffold_pairs = []
    for scaffold_set in scaffold_sets:
        pair_in_scaffold_set = []
        for i_drug in scaffold_set:
            if i_drug in drug_pair_index:
                pair_in_scaffold_set += drug_pair_index[i_drug]
        if len(pair_in_scaffold_set) > 0:
            scaffold_pairs.append(pair_in_scaffold_set)

    scaffold_pairs.sort(key=len, reverse=True)

    import heapq
    heap = [(0, i) for i in range(nfolds)]
    heapq.heapify(heap)
    folds = [[] for _ in range(nfolds)]

    for pair_set in scaffold_pairs:
        size, fold_idx = heapq.heappop(heap)
        folds[fold_idx] += pair_set
        heapq.heappush(heap, (size + len(pair_set), fold_idx))

    labels = dataset.labels
    for i, fold in enumerate(folds):
        n_pos = sum(1 for idx in fold if labels[idx] == 1)
        n_neg = len(fold) - n_pos
        pos_rate = n_pos / max(len(fold), 1)
        logger.info(
            f"[cold_drug] Fold {i}: {len(fold)} samples "
            f"(pos={n_pos}, neg={n_neg}, pos_rate={pos_rate:.2%})"
        )
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                f"[cold_drug] Fold {i} has NO {'positive' if n_pos == 0 else 'negative'} "
                f"samples! This fold will produce unreliable metrics."
            )

    return folds

def cold_protein_split(dataset, nfolds):
    ctds = get_normalized_ctd(dataset.proteins)
    prot_sim = ctds @ ctds.T
    clusters = cluster_with_sim_matrix(prot_sim, 0.3)

    prot_pair_index = {}
    for i, (i_drug, i_prot) in enumerate(dataset.pair_index):
        if i_prot not in prot_pair_index:
            prot_pair_index[i_prot] = [i]
        else:
            prot_pair_index[i_prot].append(i)

    n_cutoff = len(dataset.pair_index) // nfolds
    folds = [[] for i in range(nfolds)]
    cur = 0
    for cluster in clusters:
        pair_in_cluster = []
        for i_protein in cluster:
            if i_protein in prot_pair_index:
                pair_in_cluster += prot_pair_index[i_protein]
        if cur != nfolds - 1 and len(folds[cur]) + len(pair_in_cluster) >= n_cutoff:
            if len(folds[cur]) + len(pair_in_cluster) - n_cutoff > n_cutoff - len(folds[cur]):
                cur += 1
                folds[cur] += pair_in_cluster
            else:
                folds[cur] += pair_in_cluster
                cur += 1
        else:
            folds[cur] += pair_in_cluster
    return folds

def cold_cluster_split(dataset, ngrids):
    """
    
    """
    import random
    
    drug_clusters = generate_scaffolds(dataset, -1)
    drug_clusters = merge_cluster(drug_clusters, ngrids)

    ctds = get_normalized_ctd(dataset.proteins)
    prot_sim = ctds @ ctds.T
    prot_clusters = cluster_with_sim_matrix(prot_sim, 0.3)
    prot_clusters = merge_cluster(prot_clusters, ngrids)

    pair_in_grid = []
    for i in range(ngrids):
        pair_in_grid.append([])
        for j in range(ngrids):
            pair_in_grid[i].append([])
            for k, (i_drug, i_prot) in enumerate(dataset.pair_index):
                if i_drug in drug_clusters[i] and i_prot in prot_clusters[j]:
                    pair_in_grid[i][j].append(k)
    
    labels = dataset.labels
    
    folds = []
    skipped = 0
    for i in range(ngrids):
        for j in range(ngrids):
            test_indices = pair_in_grid[i][j]
            
            if len(test_indices) == 0:
                skipped += 1
                logger.warning(f"[cold_cluster] Skipping fold ({i},{j}): empty test set")
                continue
            
            test_labels = [labels[k] for k in test_indices]
            n_pos = sum(1 for l in test_labels if l == 1)
            n_neg = len(test_labels) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                needed_label = 0 if n_pos > 0 else 1
                extra_indices = []
                
                for k in range(ngrids):
                    if k != i:
                        for idx in pair_in_grid[k][j]:
                            if labels[idx] == needed_label:
                                extra_indices.append(idx)
                    if k != j:
                        for idx in pair_in_grid[i][k]:
                            if labels[idx] == needed_label:
                                extra_indices.append(idx)
                
                if len(extra_indices) >= 10:
                    random.shuffle(extra_indices)
                    extra_count = min(50, len(extra_indices) // 2)
                    test_indices = test_indices + extra_indices[:extra_count]
                    logger.info(f"[cold_cluster] Fold ({i},{j}): added {extra_count} samples with label={needed_label}")
                else:
                    skipped += 1
                    logger.warning(f"[cold_cluster] Skipping fold ({i},{j}): only has label={1 if n_pos > 0 else 0}, cannot fix")
                    continue
            
            train = []
            for k in range(ngrids):
                if k != i:
                    for l in range(ngrids):
                        if l != j:
                            train += pair_in_grid[k][l]
            
            train = [idx for idx in train if idx not in test_indices]
            
            folds.append({"test": test_indices, "train": train})
    
    logger.info(f"[cold_cluster] Generated {len(folds)} valid folds (skipped {skipped})")
    return folds

def cold_mssl2drug_split(dataset, nfolds, test=0.05):
    folds = []
    test_count = int(test * len(dataset.smiles))
    logger.info("About to generate " + str(nfolds) + " folds")
    for n in tqdm(range(nfolds)):
        test_i_drug = random.sample(range(len(dataset.smiles)), test_count)
        train = []
        test = []
        test_pairs = []
        for i, pair in enumerate(dataset.pair_index):
            if (pair[0] in test_i_drug or pair[1] in test_i_drug):
                if (pair[1], pair[0]) not in test_pairs:
                    test.append(i)
                    test_pairs.append(pair)
                else:
                    continue
            else:
                train.append(i)
        folds.append({'train': train, 'test': test})
    return folds
