import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import os
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import random
import torch

from rdkit import Chem

class KG(object):
    def  __init__(self):
        super(KG, self).__init__()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def link(self, dataset):
        raise NotImplementedError

class BMKG(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()
        self.drugs = json.load(open(os.path.join(path, "drug.json"), "r"))
        self.smi2drugid = {}
        for key in self.drugs:
            mol = Chem.MolFromSmiles(self.drugs[key]["SMILES"])
            if mol is not None:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                self.smi2drugid[smi] = key

        self.proteins = json.load(open(os.path.join(path, "protein.json"), "r"))
        self.seq2proteinid = {}
        for key in self.proteins:
            self.seq2proteinid[self.proteins[key]["sequence"]] = key

        self.edges = pd.read_csv(os.path.join(path, "links.csv"), dtype=str).values.tolist()

    def link(self, dataset):
        link_drug, link_protein = 0, 0
        fallback_drug, fallback_protein = 0, 0

        # 尝试从数据集路径加载fallback文本（如果存在）
        dataset_drugs = None
        dataset_proteins = None
        dataset_smi2drug = {}
        dataset_seq2protein = {}
        
        if hasattr(dataset, 'path'):
            dataset_drug_path = os.path.join(dataset.path, "drug.json")
            dataset_protein_path = os.path.join(dataset.path, "protein.json")
            
            if os.path.exists(dataset_drug_path):
                try:
                    dataset_drugs = json.load(open(dataset_drug_path, "r"))
                    # 建立SMILES到drug条目的映射
                    for key, drug_data in dataset_drugs.items():
                        if "SMILES" in drug_data:
                            iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(drug_data["SMILES"]), isomericSmiles=True)
                            dataset_smi2drug[iso_smi] = drug_data
                except Exception as e:
                    logger.warning(f" 无法加载数据集drug.json作为fallback: {e}")
            
            if os.path.exists(dataset_protein_path):
                try:
                    dataset_proteins = json.load(open(dataset_protein_path, "r"))
                    # 建立sequence到protein条目的映射
                    for key, protein_data in dataset_proteins.items():
                        if "sequence" in protein_data:
                            dataset_seq2protein[protein_data["sequence"]] = protein_data
                except Exception as e:
                    logger.warning(f" 无法加载数据集protein.json作为fallback: {e}")

        drug2kg, drug2text = {}, {}
        if hasattr(dataset, 'smiles'):
            for smi in dataset.smiles:
                iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
                if iso_smi in self.smi2drugid:
                    link_drug += 1
                    drug2kg[smi] = self.smi2drugid[iso_smi]
                    drug2text[smi] = self.drugs[self.smi2drugid[iso_smi]]["text"].lower()
                else:
                    drug2kg[smi] = None
                    # Fallback: 从数据集路径下的drug.json提取text
                    if iso_smi in dataset_smi2drug:
                        drug_data = dataset_smi2drug[iso_smi]
                        if "text" in drug_data and drug_data["text"]:
                            drug2text[smi] = drug_data["text"].lower()
                            fallback_drug += 1
                        else:
                            drug2text[smi] = ""
                    else:
                        drug2text[smi] = ""
            logger.info("Linked drug %d/%d (fallback: %d)" % (link_drug, len(dataset.smiles), fallback_drug))

        protein2kg, protein2text = {}, {}
        if hasattr(dataset, 'proteins'):
            for seq in dataset.proteins:
                if seq in self.seq2proteinid:
                    link_protein += 1
                    protein2kg[seq] = self.seq2proteinid[seq]
                    protein2text[seq] = self.proteins[self.seq2proteinid[seq]]["text"].lower()
                else:
                    protein2kg[seq] = None
                    # Fallback: 从数据集路径下的protein.json提取text
                    if seq in dataset_seq2protein:
                        protein_data = dataset_seq2protein[seq]
                        if "text" in protein_data and protein_data["text"]:
                            protein2text[seq] = protein_data["text"].lower()
                            fallback_protein += 1
                        else:
                            protein2text[seq] = ""
                    else:
                        protein2text[seq] = ""
            logger.info("Linked protein %d/%d (fallback: %d)" % (link_protein, len(dataset.proteins), fallback_protein))

        return drug2kg, drug2text, protein2kg, protein2text
 
class BMKGv2(KG):
    def __init__(self, path):
        super(BMKGv2, self).__init__()
        try:
            self.kg = pickle.load(open(path, "rb"))
        except (ModuleNotFoundError, AttributeError) as e:
            if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                logger.error(f" Pickle文件与当前numpy版本不兼容: {e}")
                logger.error(f"请删除文件 {path} 并重新生成，或降级numpy版本")
                raise RuntimeError(f"Pickle文件版本不兼容，请删除 {path} 并重新生成")
            else:
                raise
        self.adj = {}
        for triplet in self.kg["triplets"]:
            if triplet[0] not in self.adj:
                self.adj[triplet[0]] = [triplet]
            else:
                self.adj[triplet[0]].append(triplet)
            if triplet[2] not in self.adj:
                self.adj[triplet[2]] = [triplet]
            else:
                self.adj[triplet[2]].append(triplet)

SUPPORTED_KG = {"BMKG": BMKG}

def subgraph_sample(num_nodes, edge_index, strategy, num_samples, directed=False):
    ### Inputs:
    # edge_index: edge index
    # strategy: sampling strategy, e.g. bfs
    ### Output:
    # indexes of sampled edges
    adj = []
    for i in range(num_nodes):
        adj.append([])
    for i, edge in enumerate(edge_index):
        adj[edge[0]].append(i)
    node_queue = []
    visited = [0] * num_nodes
    selected_edges = []
    
    random_node = random.randint(0, num_nodes - 1)
    while len(adj[random_node]) > 5:
        random_node = random.randint(0, num_nodes - 1)
    node_queue.append(random_node)
    visited[random_node] = 1

    def dfs(u):
        visited[u] = 1
        for i in adj[u]:
            if i not in selected_edges:
                selected_edges.append(i)
                selected_edges.append(i ^ 1)
            if len(selected_edges) >= num_samples:
                return
        for i in adj[u]:
            v = edge_index[i][1]
            if visited[v]:
                continue
            dfs(v)
            if len(selected_edges) >= num_samples:
                return

    if strategy == 'dfs':
        dfs(random_node)
    else:
        while len(selected_edges) < num_samples:
            u = node_queue.pop(0)
            for i in adj[u]:
                v = edge_index[i][1]
                if i not in selected_edges:
                    selected_edges.append(i)
                    selected_edges.append(i ^ 1)
                if not visited[v]:
                    visited[v] = 1
                    node_queue.append(v)

    return selected_edges

def embed(graph, model='ProNE', filter_out={}, dim=256, save=True, save_path=''):
    ### Inputs:
    # G: object of KG
    # model: network embedding model, e.g. ProNE
    ### Outputs:
    # emb: numpy array, |G| * dim
    if save and os.path.exists(save_path):
        logger.info("Load KGE from saved file.")
        try:
            return pickle.load(open(save_path, "rb"))
        except (ModuleNotFoundError, AttributeError) as e:
            if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                logger.warning(f" Pickle文件与当前numpy版本不兼容，将重新生成: {e}")
                # 删除旧的pickle文件，重新生成
                if os.path.exists(save_path):
                    os.remove(save_path)
                    logger.info(f"已删除旧的pickle文件: {save_path}")
            else:
                raise

    from cogdl.data import Adjacency
    from cogdl.models.emb.prone import ProNE
    name2id = {}
    cnt = 0
    filtered = 0
    row = []
    col = []
    for h, t, r in graph.edges:
        if (h, t) in filter_out:
            filtered += 1
            continue
        if h not in name2id:
            cnt += 1
            name2id[h] = cnt
        if t not in name2id:
            cnt += 1
            name2id[t] = cnt
        row.append(name2id[h])
        col.append(name2id[t])
    logger.info("Filtered out %d edges in val/test set" % (filtered))
        
    row = torch.tensor(row)
    col = torch.tensor(col)
    graph = Adjacency(row, col)
    emb_model = ProNE(dim, 5, 0.2, 0.5)
    logger.info("Generating KGE...")
    emb = emb_model(graph)
    
    kg_emb = {}
    for key in name2id:
        kg_emb[key] = emb[name2id[key]]

    if save:
        pickle.dump(kg_emb, open(save_path, "wb"))

    return kg_emb

def bfs(graph, node_id, max_depth):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the starting node
    # max_depth: the max number of steps to go
    ### Outputs:
    # dist: a list, dist[i] is the list of i-hop neighbors
    pass