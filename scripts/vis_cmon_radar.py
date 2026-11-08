#!/usr/bin/env python3
"""
Fig C: CMON Cross-Modal Reliability Profile (Publication-grade)

Three panels:
  C1 — Multi-epoch radar overlay (early / mid / final)
  C2 — Reliability decomposition waterfall (Uniqueness × Confidence = Reliability)
  C3 — Parallel coordinates (6 CMON metrics across epochs)

Usage:
  python scripts/vis_cmon_radar.py --explain_dir ckpts/dp/bace/explain --output figures/cmon_radar.pdf
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from _style import apply_style, PALETTE, PALETTE_LIST, DOUBLE_COL
except ImportError:
    from scripts._style import apply_style, PALETTE, PALETTE_LIST, DOUBLE_COL


def load_cmon_diag(explain_dir: str):
    path = os.path.join(explain_dir, "cmon_diag.jsonl")
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def radar_plot(ax, categories, values_list, labels, colors, alphas, linestyles):
    """Draw multiple polygons on polar axes. values_list: list of [v1,v2,v3,v4] per curve."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    for vals, label, color, alpha, ls in zip(values_list, labels, colors, alphas, linestyles):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, lw=1.5, label=label, color=color, alpha=alpha, linestyle=ls, marker="o", markersize=2)
        ax.fill(angles, v, alpha=0.08, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain_dir", required=True)
    parser.add_argument("--output", default="figures/cmon_radar.pdf")
    parser.add_argument("--demo", action="store_true", help="Write demo cmon_diag.jsonl if missing")
    args = parser.parse_args()

    records = load_cmon_diag(args.explain_dir)
    if not records and getattr(args, "demo", False):
        os.makedirs(args.explain_dir, exist_ok=True)
        path = os.path.join(args.explain_dir, "cmon_diag.jsonl")
        demo = []
        for ep in [1, 5, 10, 20, 40, 60, 80, 100]:
            demo.append({
                "epoch": ep,
                "cmon_priv_ratio_s": 0.3 + 0.4 * (ep / 100),
                "cmon_priv_ratio_k": 0.25 + 0.35 * (ep / 100),
                "cmon_priv_ratio_m": 0.35 + 0.3 * (ep / 100),
                "cmon_lodc_s": 0.5 + 0.2 * (ep / 100),
                "cmon_lodc_k": 0.45 + 0.25 * (ep / 100),
                "cmon_lodc_m": 0.55 + 0.2 * (ep / 100),
                "cmon_conf_s": 0.6 + 0.15 * (ep / 100),
                "cmon_conf_k": 0.5 + 0.2 * (ep / 100),
                "cmon_conf_m": 0.55 + 0.2 * (ep / 100),
                "cmon_rel_s": 0.2 + 0.5 * (ep / 100),
                "cmon_rel_k": 0.18 + 0.45 * (ep / 100),
                "cmon_rel_m": 0.22 + 0.48 * (ep / 100),
                "cmon_ortho_s": 0.1 - 0.05 * (ep / 100),
                "cmon_ortho_k": 0.12 - 0.06 * (ep / 100),
                "cmon_ortho_m": 0.08 - 0.04 * (ep / 100),
            })
        with open(path, "w") as f:
            for r in demo:
                f.write(json.dumps(r) + "\n")
        records = demo
        print(f"Wrote demo {path}")
    if not records:
        raise ValueError(
            f"No CMON diag records in {args.explain_dir}/cmon_diag.jsonl. "
            "Run DP training with CMON enabled to generate cmon_diag.jsonl, or use --demo."
        )
    apply_style()
    print(f"Loaded {len(records)} epoch records")

    # Representative epochs: early, mid, final
    n = len(records)
    idx_early = 0
    idx_mid = n // 2 if n > 2 else 0
    idx_final = n - 1
    early, mid, final = records[idx_early], records[idx_mid], records[idx_final]
    epoch_early = early["epoch"]
    epoch_mid = mid["epoch"]
    epoch_final = final["epoch"]

    categories = ["Uniqueness", "LODC", "Confidence", "Reliability"]
    mod_keys = [
        ("Structure", "s"),
        ("KG", "k"),
        ("MolToken", "m"),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
    ax_c1 = fig.add_subplot(gs[0], polar=True)
    ax_c2 = fig.add_subplot(gs[1])
    ax_c3 = fig.add_subplot(gs[2])

    # ---------- C1: Multi-epoch radar overlay ----------
    values_list = []
    labels_list = []
    colors_list = []
    alphas_list = []
    linestyles_list = []
    for epoch_label, rec, alpha, ls in [
        (f"Epoch {epoch_early}", early, 0.45, ":"),
        (f"Epoch {epoch_mid}", mid, 0.7, "--"),
        (f"Epoch {epoch_final}", final, 1.0, "-"),
    ]:
        for mod_name, suffix in mod_keys:
            vals = [
                rec.get(f"cmon_priv_ratio_{suffix}", 0),
                rec.get(f"cmon_lodc_{suffix}", 0),
                rec.get(f"cmon_conf_{suffix}", 0),
                rec.get(f"cmon_rel_{suffix}", 0),
            ]
            values_list.append(vals)
            labels_list.append(f"{mod_name} ({epoch_label})")
            colors_list.append(PALETTE_LIST[["Structure", "KG", "MolToken"].index(mod_name) % len(PALETTE_LIST)])
            alphas_list.append(alpha)
            linestyles_list.append(ls)
    radar_plot(ax_c1, categories, values_list, labels_list, colors_list, alphas_list, linestyles_list)
    ax_c1.set_title("C1  Multi-epoch modality profile", fontsize=10, pad=12)
    ax_c1.legend(loc="upper right", bbox_to_anchor=(1.42, 1.05), fontsize=6, ncol=1)

    # ---------- C2: Reliability decomposition waterfall ----------
    x_groups = [0, 1.5, 3]
    bar_w = 0.28
    for g, (mod_name, suffix) in enumerate(mod_keys):
        u = final.get(f"cmon_priv_ratio_{suffix}", 0)
        c = final.get(f"cmon_conf_{suffix}", 0)
        r = final.get(f"cmon_rel_{suffix}", 0)
        x0 = x_groups[g] - bar_w
        x1 = x_groups[g]
        x2 = x_groups[g] + bar_w
        color = PALETTE_LIST[g % len(PALETTE_LIST)]
        ax_c2.bar(x0, u, width=bar_w * 0.9, label="Uniqueness" if g == 0 else None, color=color, alpha=0.7)
        ax_c2.bar(x1, c, width=bar_w * 0.9, label="Confidence" if g == 0 else None, color=color, alpha=0.5)
        ax_c2.bar(x2, r, width=bar_w * 0.9, label="Reliability" if g == 0 else None, color=color, alpha=1.0)
        ax_c2.text(x0, u + 0.02, f"{u:.2f}", ha="center", fontsize=7)
        ax_c2.text(x1, c + 0.02, f"{c:.2f}", ha="center", fontsize=7)
        ax_c2.text(x2, r + 0.02, f"{r:.2f}", ha="center", fontsize=7)
    ax_c2.set_xticks(x_groups)
    ax_c2.set_xticklabels([m[0] for m in mod_keys])
    ax_c2.set_ylabel("Score")
    ax_c2.set_ylim(0, 1.15)
    ax_c2.set_title("C2  Reliability decomposition (final epoch)", fontsize=10)
    ax_c2.legend(loc="upper right", fontsize=7)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["right"].set_visible(False)

    # ---------- C3: Parallel coordinates ----------
    dims = ["Uniq_S", "Uniq_K", "Uniq_M", "Conf_S", "Conf_K", "Conf_M"]
    key_map = [
        "cmon_priv_ratio_s", "cmon_priv_ratio_k", "cmon_priv_ratio_m",
        "cmon_conf_s", "cmon_conf_k", "cmon_conf_m",
    ]
    n_epochs = len(records)
    cmap = plt.cm.viridis
    for i, rec in enumerate(records):
        vals = [rec.get(k, 0) for k in key_map]
        color = cmap(i / max(n_epochs - 1, 1))
        ax_c3.plot(range(6), vals, color=color, alpha=0.6, lw=0.8)
    ax_c3.set_xticks(range(6))
    ax_c3.set_xticklabels(dims, fontsize=7, rotation=25, ha="right")
    ax_c3.set_ylabel("Value")
    ax_c3.set_ylim(0, 1.05)
    ax_c3.set_title("C3  Parallel coordinates (all epochs)", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_epochs - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_c3, shrink=0.6, label="Epoch index")
    ax_c3.spines["top"].set_visible(False)
    ax_c3.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()

    print(f"\nCMON summary (final epoch {epoch_final}):")
    for mod_name, suffix in mod_keys:
        u = final.get(f"cmon_priv_ratio_{suffix}", 0)
        c = final.get(f"cmon_conf_{suffix}", 0)
        r = final.get(f"cmon_rel_{suffix}", 0)
        print(f"  {mod_name}: Uniq={u:.3f} Conf={c:.3f} Rel={r:.3f}")


if __name__ == "__main__":
    main()
