#!/usr/bin/env python3
"""
Fig E: Digest Token LLM Attention Flow (Publication-grade)

Three panels:
  E1 — Layerwise attention flow (stream chart: attention to 3 digest tokens across layers)
  E2 — Attention source heatmap (4×3) with marginals
  E3 — Cross-modal attention asymmetry (same- vs cross-modality to each digest)

Usage:
  python scripts/vis_digest_attention.py --attn_data path/to/attn.npz --output figures/digest_attention.pdf
  Or with model: --config configs/dp/dp.json --checkpoint ckpts/dp/bace/best_model.pt
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from _style import apply_style, PALETTE, PALETTE_LIST, DOUBLE_COL
except ImportError:
    from scripts._style import apply_style, PALETTE, PALETTE_LIST, DOUBLE_COL


def _get_attn_layers(data):
    """Handle npz attn_to_digest as object array or list."""
    attn = data["attn_to_digest"]
    if hasattr(attn, "shape") and attn.shape == ():
        attn = attn.item()
    return list(attn) if not isinstance(attn, list) else attn


def plot_from_saved(attn_data_path: str, output: str):
    data = np.load(attn_data_path, allow_pickle=True)
    prefix_len = int(data["prefix_len"])
    struct_len = int(data["struct_seq_len"])
    kg_len = int(data["kg_seq_len"])
    mol_len = int(data["mol_seq_len"])
    attn_layers = _get_attn_layers(data)

    n_layers = len(attn_layers)
    digest_labels = ["Struct Digest", "KG Digest", "MolToken Digest"]
    source_labels = ["Struct", "KG", "MolToken", "Text"]

    # (n_layers, 3) mean attention to each digest per layer
    avg_per_layer = np.zeros((n_layers, 3))
    for layer_idx in range(n_layers):
        layer_attn = attn_layers[layer_idx]
        if hasattr(layer_attn, "shape") and layer_attn.shape:
            avg_per_layer[layer_idx] = np.asarray(layer_attn).mean(axis=(0, 1))
        else:
            avg_per_layer[layer_idx] = 0.0

    # Last few layers: (seq_len, 3) for source breakdown
    n_analyze = min(4, n_layers)
    deep = attn_layers[-n_analyze:]
    deep_avg = np.mean([np.asarray(a).mean(axis=0) for a in deep], axis=0)
    if deep_avg.ndim == 1:
        deep_avg = np.zeros((prefix_len + 50, 3))
    seq_len = deep_avg.shape[0]
    struct_range = slice(0, struct_len)
    kg_range = slice(struct_len, struct_len + kg_len)
    mol_range = slice(struct_len + kg_len, struct_len + kg_len + mol_len)
    text_range = slice(prefix_len, min(seq_len, prefix_len + 200))

    source_attn = np.zeros((4, 3))
    for src_idx, rng in enumerate([struct_range, kg_range, mol_range, text_range]):
        for dig_idx in range(3):
            sl = deep_avg[rng, dig_idx]
            source_attn[src_idx, dig_idx] = np.mean(sl) if sl.size else 0.0

    # Same- vs cross-modality: for digest i, same = source_attn[i,i], cross = mean of source_attn[j,i] for j!=i
    same_attn = np.array([source_attn[i, i] for i in range(3)])
    cross_attn = np.array([
        (source_attn[0, i] + source_attn[1, i] + source_attn[2, i] + source_attn[3, i] - source_attn[i, i]) / 3.0
        for i in range(3)
    ])

    apply_style()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig = plt.figure(figsize=(DOUBLE_COL, 5.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1], height_ratios=[1, 1])
    ax_e1 = fig.add_subplot(gs[0, :2])
    ax_e2 = fig.add_subplot(gs[0, 2])
    ax_e3 = fig.add_subplot(gs[1, :])

    # ---------- E1: Layerwise attention flow (stream) ----------
    layers = np.arange(1, n_layers + 1)
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    for dig_idx in range(3):
        ax_e1.fill_between(layers, 0, avg_per_layer[:, dig_idx], alpha=0.5, color=colors[dig_idx])
        ax_e1.plot(layers, avg_per_layer[:, dig_idx], color=colors[dig_idx], lw=1.5, label=digest_labels[dig_idx])
    ax_e1.set_xlabel("LLM layer")
    ax_e1.set_ylabel("Attention to digest token")
    ax_e1.set_ylim(0, None)
    ax_e1.set_title("E1  Layerwise attention flow to digest tokens", fontsize=10)
    ax_e1.legend(loc="upper right", fontsize=8)
    ax_e1.spines["top"].set_visible(False)
    ax_e1.spines["right"].set_visible(False)

    # ---------- E2: 4×3 heatmap with marginals ----------
    # Simple 4×3 heatmap in ax_e2; marginals as small bars on sides
    im = ax_e2.imshow(source_attn, aspect="auto", cmap="Blues", vmin=0, vmax=np.percentile(source_attn, 98))
    ax_e2.set_xticks(range(3))
    ax_e2.set_xticklabels([d.replace(" Digest", "") for d in digest_labels], fontsize=8)
    ax_e2.set_yticks(range(4))
    ax_e2.set_yticklabels(source_labels, fontsize=8)
    ax_e2.set_xlabel("Digest token")
    ax_e2.set_ylabel("Source")
    ax_e2.set_title("E2  Attention source → digest", fontsize=10)
    for i in range(4):
        for j in range(3):
            ax_e2.text(j, i, f"{source_attn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if source_attn[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax_e2, shrink=0.6, label="Attention")

    # ---------- E3: Cross-modal asymmetry ----------
    x = np.arange(3)
    w = 0.32
    ax_e3.bar(x - w / 2, same_attn, width=w, label="Same modality", color=PALETTE["blue"], alpha=0.8)
    ax_e3.bar(x + w / 2, cross_attn, width=w, label="Cross modality", color=PALETTE["orange"], alpha=0.8)
    ax_e3.set_xticks(x)
    ax_e3.set_xticklabels([d.replace(" Digest", "") for d in digest_labels])
    ax_e3.set_ylabel("Avg attention weight")
    ax_e3.set_title("E3  Same- vs cross-modality attention to digest", fontsize=10)
    ax_e3.legend(loc="upper right", fontsize=8)
    ax_e3.spines["top"].set_visible(False)
    ax_e3.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def extract_and_save(config_path, checkpoint_path, dataset, n_samples, output):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import torch
    from unidrug.models.task_model.dp import DPModel

    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loading model from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_cfg = config.get("network", config)
    model = DPModel(net_cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    model._capture_digest_attn = True

    from unidrug.datasets.dp_dataset import DPDataset
    from torch.utils.data import DataLoader
    ds_cfg = config.get("dataset", {})
    ds = DPDataset(ds_cfg, split="test")
    loader = DataLoader(ds, batch_size=min(n_samples, len(ds)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    with torch.no_grad():
        _ = model(batch)
    if not hasattr(model, "_last_digest_attn"):
        print("WARNING: No digest attention captured. Is CMON enabled?")
        return
    attn_info = model._last_digest_attn
    npz_path = output.replace(".pdf", ".npz")
    np.savez(
        npz_path,
        positions=np.array(attn_info["positions"]),
        prefix_len=attn_info["prefix_len"],
        struct_seq_len=attn_info["struct_seq_len"],
        kg_seq_len=attn_info["kg_seq_len"],
        mol_seq_len=attn_info["mol_seq_len"],
        attn_to_digest=np.array(attn_info["attn_to_digest"], dtype=object),
    )
    print(f"Attention data saved: {npz_path}")
    plot_from_saved(npz_path, output)


def build_demo_npz(output_npz: str):
    """Write a minimal synthetic .npz for testing the figure layout."""
    n_layers = 28
    seq_len = 200
    prefix_len = 170
    struct_len, kg_len, mol_len = 66, 50, 58
    rng = np.random.default_rng(42)
    attn_to_digest = []
    for _ in range(n_layers):
        # (batch=4, seq_len, 3)
        a = rng.uniform(0.001, 0.02, (4, seq_len, 3))
        a[:, :prefix_len, :] += rng.uniform(0, 0.03, (4, prefix_len, 3))
        attn_to_digest.append(a)
    np.savez(
        output_npz,
        positions=np.array([1, struct_len + 1, struct_len + kg_len + 1]),
        prefix_len=prefix_len,
        struct_seq_len=struct_len,
        kg_seq_len=kg_len,
        mol_seq_len=mol_len,
        attn_to_digest=np.array(attn_to_digest, dtype=object),
    )
    print(f"Demo data written: {output_npz}")
    return output_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_data", default=None, help="Pre-saved .npz (skip model load)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="bace")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--output", default="figures/digest_attention.pdf")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic data and plot")
    args = parser.parse_args()
    if args.demo:
        npz = args.output.replace(".pdf", ".npz")
        build_demo_npz(npz)
        plot_from_saved(npz, args.output)
    elif args.attn_data:
        plot_from_saved(args.attn_data, args.output)
    elif args.config and args.checkpoint:
        extract_and_save(args.config, args.checkpoint, args.dataset, args.n_samples, args.output)
    else:
        parser.error("Provide --attn_data, --demo, or both --config and --checkpoint")


if __name__ == "__main__":
    main()
