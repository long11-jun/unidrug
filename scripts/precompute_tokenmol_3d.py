#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算 Token-Mol 3D Torsion Tokens
==================================
一次性处理所有 SMILES，生成 3D 构象并提取扭转角 tokens，保存到缓存文件。
之后训练时直接加载，无需重复计算。

用法：
    python scripts/precompute_tokenmol_3d.py --dataset SIDER
    python scripts/precompute_tokenmol_3d.py --dataset Tox21
    python scripts/precompute_tokenmol_3d.py --all  # 处理所有数据集
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import multiprocessing as mp
import math
import time
import signal

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 兼容仓库内部大量使用的 `from models...` 绝对导入（models 位于 unidrug/models）
sys.path.insert(0, str(PROJECT_ROOT / "unidrug"))

#
# 注意：本脚本不依赖 numpy；如需后续统计再引入即可

# RDKit 相关
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not available, 3D features will be disabled")


def _tokenmol_tokenize_smiles(smiles: str, vocab: Dict[str, int]) -> List[str]:
    """
    与训练端 `DPDataset._tokenmol_tokenize_smiles` 保持一致：贪心最长匹配 Token-Mol 词表。
    """
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
    """
    严格 3D 扭转角提取：仅使用固定 randomSeed 的 ETKDGv3 生成构象，失败则返回 None（不做随机坐标 fallback）。
    输出与训练端一致：[-3.14, 3.14] 的 0.01 rad 字符串（如 '1.23'）。
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # 固定 seed，保证可复现（但不做随机坐标 fallback）
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0 or mol.GetNumConformers() == 0:
        return None

    # 轻量优化：优先 MMFF（更合理），失败再 UFF；迭代数收紧，避免极慢样本拖死
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=120)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=120)
        except Exception:
            pass

    conf = mol.GetConformer()
    rot_bond_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(rot_bond_smarts)
    torsions: List[str] = []

    for (j, k) in matches:
        j_atom = mol.GetAtomWithIdx(j)
        k_atom = mol.GetAtomWithIdx(k)
        j_nbrs = [a.GetIdx() for a in j_atom.GetNeighbors() if a.GetIdx() != k]
        k_nbrs = [a.GetIdx() for a in k_atom.GetNeighbors() if a.GetIdx() != j]
        if not j_nbrs or not k_nbrs:
            continue
        i_idx = j_nbrs[0]
        l_idx = k_nbrs[0]
        try:
            angle = rdMolTransforms.GetDihedralRad(conf, i_idx, j, k, l_idx)
        except Exception:
            continue

        while angle <= -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        q = max(-3.14, min(3.14, round(float(angle), 2)))
        if abs(q) < 0.0005:
            q = 0.0
        torsions.append(f"{q:.2f}")
        if len(torsions) >= max_torsions:
            break
    return torsions


def build_tokenmol_seq_strict(
    smiles: str, vocab: Dict[str, int], use_torsion: bool, max_torsions: int
) -> Optional[str]:
    """
    构造训练端一致的 `tokenmol_seq`（空格分隔字符串）。
    严格模式：如果 use_torsion=True 且 3D 构象失败，则返回 None（明确失败，不做随机/降级）。
    """
    smi_tokens = _tokenmol_tokenize_smiles(smiles, vocab=vocab)
    tokens = list(smi_tokens)
    if use_torsion:
        tors = _torsion_bins_rad_strict(smiles, max_torsions=max_torsions)
        if tors is None:
            return None
        if len(tors) > 0:
            tokens += ["GEO"] + tors
    return " ".join(tokens)


def get_torsion_tokens(smiles: str, num_torsion_bins: int = 36) -> Optional[List[str]]:
    """
    从 SMILES 生成 3D 构象并提取扭转角 tokens
    
    Args:
        smiles: 分子 SMILES
        num_torsion_bins: 扭转角离散化的 bin 数量（默认 36，每 bin 10°）
    
    Returns:
        tokens: ['C', 'C', 'O', ..., 'GEO', 'TOR_0', 'TOR_90', ...]
                如果失败返回 None
    """
    # 旧接口保留但不使用：严谨模式下不再输出 TOR_*，改用 build_tokenmol_seq_strict
    return None


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("per-molecule timeout")


def process_single_smiles(args) -> Tuple[int, str, Optional[str], float, Optional[str]]:
    """处理单个 SMILES（用于多进程）。返回 (idx, smiles, seq_or_None, elapsed_sec, error_or_None)"""
    idx, smiles, vocab, use_torsion, max_torsions, per_mol_timeout_sec, max_heavy_atoms = args
    t0 = time.time()

    # 预筛：分子太大时 3D 构象会非常慢且不稳定 —— 直接标记失败更可控/可复现
    if max_heavy_atoms is not None and max_heavy_atoms > 0:
        try:
            mol0 = Chem.MolFromSmiles(smiles)
            if mol0 is None:
                return idx, smiles, None, time.time() - t0, "MolFromSmiles(None)"
            if mol0.GetNumHeavyAtoms() > max_heavy_atoms:
                return idx, smiles, None, time.time() - t0, f"heavy_atoms>{max_heavy_atoms}"
        except Exception as e:
            return idx, smiles, None, time.time() - t0, f"prefilter_error:{type(e).__name__}"

    old_handler = None
    try:
        if per_mol_timeout_sec and per_mol_timeout_sec > 0:
            # SIGALRM 仅在类 Unix 生效；本环境是 linux，OK
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(per_mol_timeout_sec))
        seq = build_tokenmol_seq_strict(
            smiles, vocab=vocab, use_torsion=use_torsion, max_torsions=max_torsions
        )
        return idx, smiles, seq, time.time() - t0, None
    except _Timeout:
        return idx, smiles, None, time.time() - t0, "timeout"
    except Exception as e:
        return idx, smiles, None, time.time() - t0, f"error:{type(e).__name__}"
    finally:
        try:
            if per_mol_timeout_sec and per_mol_timeout_sec > 0:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def precompute_dataset(
    dataset_name: str,
    data_dir: str,
    output_dir: str,
    num_workers: int = 4,
    per_mol_timeout_sec: int = 60,
    max_heavy_atoms: int = 80,
    slow_sec: float = 20.0,
    vocab_file: str = "",
    only_failed: bool = False,
):
    """
    预计算一个数据集的所有 3D torsion tokens
    
    Args:
        dataset_name: 数据集名称（SIDER, Tox21 等）
        data_dir: 数据目录
        output_dir: 输出目录
        num_workers: 并行进程数
    """
    # 查找数据文件
    dataset_path = Path(data_dir) / "dp" / "moleculenet" / dataset_name.lower()
    if not dataset_path.exists():
        # 尝试其他路径
        dataset_path = Path(data_dir) / dataset_name.lower()
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return
    
    # 查找 CSV 文件：优先在 raw/ 子目录，其次在根目录
    csv_files = list((dataset_path / "raw").glob("*.csv")) if (dataset_path / "raw").exists() else []
    if not csv_files:
        csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {dataset_path} or {dataset_path / 'raw'}")
        return
    
    # 输出路径（用于 only_failed 模式下读取历史 cache/failed）
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / f"tokenmol_3d_{dataset_name.lower()}.pkl"
    failed_file = output_path / f"tokenmol_3d_{dataset_name.lower()}_failed.txt"

    # 收集所有 SMILES（或只重试 failed）
    smiles_list: List[str] = []
    if only_failed:
        if not failed_file.exists():
            print(f"[ERROR] only_failed=True but failed file not found: {failed_file}")
            return
        with open(failed_file, "r") as f:
            smiles_list = [ln.strip() for ln in f if ln.strip()]
        # 去重但保持顺序
        seen = set()
        smiles_list = [s for s in smiles_list if not (s in seen or seen.add(s))]
        print(f"[INFO] only_failed=True: loaded {len(smiles_list)} SMILES from {failed_file}")
    else:
        all_smiles = set()
        import csv
        for csv_file in csv_files:
            print(f"[INFO] Reading {csv_file.name}...")
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles = row.get('smiles') or row.get('SMILES') or row.get('mol')
                    if smiles:
                        all_smiles.add(smiles.strip())
        smiles_list = list(all_smiles)
        print(f"[INFO] Found {len(smiles_list)} unique SMILES")
        if len(smiles_list) == 0:
            print("[ERROR] No SMILES found!")
            return
    
    # 载入 Token-Mol 词表（必须与训练端一致，否则 token->id 映射会错位，3D/SMILES 信息会失真）
    vocab_file = (vocab_file or "").strip()
    if not vocab_file:
        default_vocab_candidates = [
            os.path.join(PROJECT_ROOT, "torsion_version", "torsion_voc_property.csv"),
            os.path.join(PROJECT_ROOT, "ckpts", "pretrained", "vocab.csv"),
        ]
        for cand in default_vocab_candidates:
            if os.path.exists(cand):
                vocab_file = cand
                break
    if not os.path.exists(vocab_file):
        print(f"[ERROR] Token-Mol vocab file not found: {vocab_file}")
        print("[ERROR] 请通过 --vocab_file 指定正确的 Token-Mol 词表文件。")
        return
    from models.llm_encoder.tokenmol_tokenizer import TokenMolTokenizer
    tok = TokenMolTokenizer(vocab_file)
    vocab = tok.vocab

    # 准备任务（严格：use_torsion=True；max_torsions 与训练端默认一致）
    use_torsion = True
    max_torsions = 32
    tasks = [
        (i, smi, vocab, use_torsion, max_torsions, per_mol_timeout_sec, max_heavy_atoms)
        for i, smi in enumerate(smiles_list)
    ]
    
    # 多进程处理
    # only_failed 模式下：优先加载已有 cache，再把成功的补上
    results = {}
    if only_failed and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                results = pickle.load(f)
            if not isinstance(results, dict):
                print(f"[WARN] Existing cache is not dict, ignoring: {cache_file}")
                results = {}
            else:
                print(f"[INFO] Loaded existing cache: {cache_file} (keys={len(results)})")
        except Exception as e:
            print(f"[WARN] Failed to load existing cache {cache_file}: {e}")
            results = {}
    failed = []
    slow_cases: List[Tuple[float, str, str]] = []  # (elapsed, reason, smiles)
    
    print(f"[INFO] Processing with {num_workers} workers...")
    
    if num_workers > 1:
        pool = mp.Pool(num_workers)
        try:
            it = pool.imap_unordered(process_single_smiles, tasks, chunksize=8)
            for idx, smiles, tokens, elapsed, err in tqdm(it, total=len(tasks), desc=f"[TokenMol3D] {dataset_name}"):
                if elapsed >= slow_sec:
                    slow_cases.append((elapsed, err or ("ok" if tokens is not None else "failed"), smiles))
                if tokens is not None:
                    results[smiles] = tokens
                else:
                    failed.append(smiles)
        except KeyboardInterrupt:
            # 允许用户安全中断，避免 multiprocessing 内部异常噪声
            print("\n[WARN] Interrupted by user, terminating workers...")
            pool.terminate()
            pool.join()
            raise
        finally:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass
    else:
        for task in tqdm(tasks, desc=f"[TokenMol3D] {dataset_name}"):
            idx, smiles, tokens, elapsed, err = process_single_smiles(task)
            if elapsed >= slow_sec:
                slow_cases.append((elapsed, err or ("ok" if tokens is not None else "failed"), smiles))
            if tokens is not None:
                results[smiles] = tokens
            else:
                failed.append(smiles)
    
    print(f"[INFO] Success: {len(results)}, Failed: {len(failed)}")
    
    # 保存结果
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"[INFO] Saved to {cache_file}")
    print(f"[INFO] Cache size: {cache_file.stat().st_size / 1024:.1f} KB")
    
    # 保存失败的 SMILES（用于调试）
    if failed:
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed))
        print(f"[INFO] Failed SMILES saved to {failed_file}")

    # 保存慢样本（帮助定位“最后几条特别慢”的根因）
    if slow_cases:
        slow_cases.sort(key=lambda x: x[0], reverse=True)
        slow_file = output_path / f"tokenmol_3d_{dataset_name.lower()}_slow.txt"
        with open(slow_file, "w") as f:
            for elapsed, reason, smi in slow_cases[:2000]:
                f.write(f"{elapsed:.2f}\t{reason}\t{smi}\n")
        print(f"[INFO] Slow cases saved to {slow_file} (top {min(2000, len(slow_cases))})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Precompute Token-Mol 3D Torsion Tokens")
    parser.add_argument("--dataset", type=str, default="SIDER", 
                        help="Dataset name (SIDER, Tox21, ClinTox, BBBP, etc.)")
    parser.add_argument("--all", action="store_true", 
                        help="Process all datasets")
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="",
        help=(
            "Token-Mol vocab file (CSV, one token per line). "
            "Default: torsion_version/torsion_voc_property.csv if exists, else ckpts/pretrained/vocab.csv"
        ),
    )
    parser.add_argument("--data_dir", type=str,
                        default=str(PROJECT_ROOT / "unidrug" / "datasets"),
                        help="Data directory")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "cache" / "tokenmol_3d"),
                        help="Output directory for cache files")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--per_mol_timeout_sec", type=int, default=60,
                        help="Timeout (sec) for a single molecule; timeout -> failed (strict). 0 disables.")
    parser.add_argument("--max_heavy_atoms", type=int, default=80,
                        help="Skip molecules with heavy atoms > N (strict failure). 0 disables.")
    parser.add_argument("--slow_sec", type=float, default=20.0,
                        help="Log molecules slower than this threshold (sec) into *_slow.txt")
    parser.add_argument(
        "--only_failed",
        action="store_true",
        help=(
            "Only retry molecules listed in cache/tokenmol_3d/tokenmol_3d_<ds>_failed.txt "
            "and merge successes into existing cache."
        ),
    )
    
    args = parser.parse_args()
    
    if not RDKIT_AVAILABLE:
        print("[ERROR] RDKit is required but not available!")
        sys.exit(1)
    
    if args.all:
        # 处理所有数据集
        datasets = ["SIDER", "Tox21", "ClinTox", "BBBP", "Bace", "HIV", "MUV"]
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"Processing {ds}...")
            print(f"{'='*60}")
            try:
                precompute_dataset(
                    ds,
                    args.data_dir,
                    args.output_dir,
                    args.num_workers,
                    per_mol_timeout_sec=args.per_mol_timeout_sec,
                    max_heavy_atoms=args.max_heavy_atoms,
                    slow_sec=args.slow_sec,
                    vocab_file=args.vocab_file,
                    only_failed=args.only_failed,
                )
            except Exception as e:
                print(f"[ERROR] Failed to process {ds}: {e}")
    else:
        precompute_dataset(
            args.dataset,
            args.data_dir,
            args.output_dir,
            args.num_workers,
            per_mol_timeout_sec=args.per_mol_timeout_sec,
            max_heavy_atoms=args.max_heavy_atoms,
            slow_sec=args.slow_sec,
            vocab_file=args.vocab_file,
            only_failed=args.only_failed,
        )
    
    print("\n[DONE] Precomputation complete!")
    print(f"Cache files are in: {args.output_dir}")
    print("Training will automatically use these cached tokens.")


if __name__ == "__main__":
    main()

