#!/usr/bin/env python3
"""
SA-CMSA KG Retrieval Pattern Analysis (Publication-grade)

Four separate figures organised by analysis dimension:
  Fig 1 — Selectivity:   Lorenz/Gini curve + Modality attention profile
  Fig 2 — Discriminability: Clustered sample–entity attention heatmap
  Fig 3 — Coverage:      Top-20 entity frequency + Entity coverage curve
  Fig 4 — Structure:     Entity co-retrieval network

Supports two data formats:
  - DP format:  topk_entities  (16 weights, single modality)
  - DTI format: topk_entity_ids (32 weights, drug[0:16] + protein[16:32])

Usage:
  python scripts/vis_sacmsa_heatmap.py --explain_file kg_explain_epoch66.jsonl
  python scripts/vis_sacmsa_heatmap.py --explain_dir ckpts/dp/sider/explain
  python scripts/vis_sacmsa_heatmap.py --explain_file kg_explain_epoch66.jsonl --outdir figures/sacmsa_analysis
"""
import argparse
import json
import os
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy

try:
    from _style import apply_style, PALETTE, PALETTE_LIST, DOUBLE_COL, SINGLE_COL, gini_coefficient, lorenz_curve
except ImportError:
    from scripts._style import apply_style, PALETTE, PALETTE_LIST, DOUBLE_COL, SINGLE_COL, gini_coefficient, lorenz_curve


# ──────────── Data loading ────────────

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def detect_format(records, fmt_override=None):
    """Detect data format.

    Formats:
      - single:    DP with topk_entities (N weights, one pool)
      - multihead: DP with topk_entity_ids (num_heads*K weights, all from struct→KGE bank)
      - dual:      DTI with topk_entity_ids AND dual_modality marker (drug + protein)

    fmt_override: 'single', 'multihead', or 'dual' to skip auto-detection.
    """
    rec0 = records[0]

    if fmt_override:
        fmt = fmt_override.lower()
        if fmt == "dual":
            n_total = len(rec0.get("topk_entity_ids", rec0.get("topk_entities", [])))
            return "dual", n_total // 2
        else:
            n = len(rec0.get("topk_entity_ids", rec0.get("topk_entities", [])))
            return fmt, n

    if "topk_entity_ids" in rec0:
        if rec0.get("dual_modality"):
            n_total = len(rec0["topk_entity_ids"])
            return "dual", n_total // 2
        else:
            return "multihead", len(rec0["topk_entity_ids"])
    elif "topk_entities" in rec0:
        return "single", len(rec0.get("topk_entities", []))
    else:
        raise ValueError(f"Unknown format. Keys: {list(rec0.keys())}")


def extract_modalities(records, fmt, n_per):
    """Extract per-head or per-modality entity lists and weight lists.

    Formats:
      - dual:      Struct(0:n_per) + KG(n_per:2*n_per), two separate modality queries
      - multihead: Head1(0:n_per//2) + Head2(n_per//2:n_per), same struct query
      - single:    All weights in one pool
    """
    head1_ents, head1_ws = [], []
    head2_ents, head2_ws = [], []
    all_ents, all_ws = [], []

    for rec in records:
        eids = rec.get("topk_entity_ids", rec.get("topk_entities", []))
        ws = rec.get("topk_weights", [])

        if fmt == "dual":
            h1_e, h1_w = eids[:n_per], ws[:n_per]
            h2_e, h2_w = eids[n_per:], ws[n_per:]
            head1_ents.append(h1_e); head1_ws.append(h1_w)
            head2_ents.append(h2_e); head2_ws.append(h2_w)
            all_ents.append(h1_e + h2_e)
            all_ws.append(h1_w + h2_w)
        elif fmt == "multihead":
            n_half = len(eids) // 2
            h1_e, h1_w = eids[:n_half], ws[:n_half]
            h2_e, h2_w = eids[n_half:], ws[n_half:]
            head1_ents.append(h1_e); head1_ws.append(h1_w)
            head2_ents.append(h2_e); head2_ws.append(h2_w)
            all_ents.append(eids)
            all_ws.append(ws)
        else:
            all_ents.append(eids)
            all_ws.append(ws)
            head1_ents.append(eids)
            head1_ws.append(ws)

    return {
        "head1": {"ents": head1_ents, "ws": head1_ws},
        "head2": {"ents": head2_ents, "ws": head2_ws},
        "all":   {"ents": all_ents,   "ws": all_ws},
    }


# ──────────── Graph building ────────────

def build_coretrieval_graph(ent_lists, weight_lists, top_n=50):
    entity_counter = Counter()
    entity_weight = Counter()
    cooccur = Counter()
    for ents, ws in zip(ent_lists, weight_lists):
        for i, e in enumerate(ents):
            entity_counter[e] += 1
            entity_weight[e] += ws[i] if i < len(ws) else 0
        for (a, b) in combinations(set(ents), 2):
            key = tuple(sorted([a, b]))
            cooccur[key] += 1

    top_ents = [e for e, _ in entity_counter.most_common(top_n)]
    top_set = set(top_ents)
    G = nx.Graph()
    for e in top_ents:
        G.add_node(e, count=entity_counter[e],
                   avg_weight=entity_weight[e] / max(entity_counter[e], 1))
    for (a, b), w in cooccur.items():
        if a in top_set and b in top_set and w > 1:
            G.add_edge(a, b, weight=w)
    return G, entity_counter


# ──────────── Figure 1: Selectivity (A2 + A6) ────────────

def plot_selectivity(mods, fmt, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL + 1.5, 4.0))
    ax_lorenz, ax_profile = axes

    # ── Left: Lorenz + Gini ──
    if fmt == "dual":
        mod_cfg = [("Drug",    mods["head1"]["ws"], PALETTE["blue"]),
                   ("Protein", mods["head2"]["ws"], PALETTE["orange"])]
    elif fmt == "multihead":
        mod_cfg = [("Head 1", mods["head1"]["ws"], PALETTE["blue"]),
                   ("Head 2", mods["head2"]["ws"], PALETTE["orange"])]
    else:
        mod_cfg = [("Retrieval", mods["all"]["ws"], PALETTE["blue"])]

    x_unif = np.linspace(0, 1, 50)
    ax_lorenz.plot(x_unif, x_unif, "k--", lw=0.8, label="Uniform", alpha=0.5)
    ax_lorenz.fill_between(x_unif, x_unif, 0, alpha=0.04, color="gray")

    for mod_name, ws_list, color in mod_cfg:
        ginis = [gini_coefficient(w) for w in ws_list if len(w) > 0]
        mean_gini = np.mean(ginis) if ginis else 0
        for w in ws_list[:30]:
            if w:
                x, y = lorenz_curve(w)
                ax_lorenz.plot(x, y, color=color, alpha=0.12, lw=0.5)
        concat_w = np.concatenate([np.array(w, dtype=float) for w in ws_list if w])
        xm, ym = lorenz_curve(concat_w)
        ax_lorenz.plot(xm, ym, color=color, lw=2.2,
                       label=f"{mod_name} (Gini={mean_gini:.3f})")

    ax_lorenz.set_xlabel("Cumulative share of entities")
    ax_lorenz.set_ylabel("Cumulative share of attention")
    ax_lorenz.set_title("(a)  Attention concentration", fontsize=11, fontweight="bold")
    ax_lorenz.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax_lorenz.set_xlim(0, 1); ax_lorenz.set_ylim(0, 1)
    ax_lorenz.set_aspect("equal")

    # ── Right: Attention profile ──
    if len(mod_cfg) >= 2:
        lbl1, lbl2 = mod_cfg[0][0], mod_cfg[1][0]
        h1_top1 = [w[0] if w else 0 for w in mods["head1"]["ws"]]
        h2_top1 = [w[0] if w else 0 for w in mods["head2"]["ws"]]
        h1_ginis = [gini_coefficient(w) for w in mods["head1"]["ws"] if w]
        h2_ginis = [gini_coefficient(w) for w in mods["head2"]["ws"] if w]

        bp_data = [h1_top1, h2_top1, h1_ginis, h2_ginis]
        bp_labels = [f"{lbl1}\nTop-1 w", f"{lbl2}\nTop-1 w",
                     f"{lbl1}\nGini", f"{lbl2}\nGini"]
        bp_colors = [PALETTE["blue"], PALETTE["orange"],
                     PALETTE["blue"], PALETTE["orange"]]
        bp = ax_profile.boxplot(bp_data, patch_artist=True, widths=0.5,
                                medianprops=dict(color="black", lw=1.5))
        for patch, c in zip(bp["boxes"], bp_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax_profile.set_xticklabels(bp_labels, fontsize=8)
        ax_profile.set_ylabel("Value")
        ax_profile.set_title("(b)  Attention profile", fontsize=11, fontweight="bold")
        for i, d in enumerate(bp_data):
            med = np.median(d)
            ax_profile.text(i + 1, med + 0.01, f"{med:.3f}", ha="center",
                            fontsize=7, color="#333", fontweight="bold")
    else:
        copartner = {}
        for ss in mods["all"]["ents"]:
            s = set(ss)
            for e in s:
                if e not in copartner:
                    copartner[e] = set()
                copartner[e] |= s - {e}
        degrees = [len(v) for v in copartner.values()]
        ax_profile.hist(degrees, bins=40, color=PALETTE["purple"], alpha=0.8,
                        edgecolor="white", linewidth=0.3)
        mean_deg = np.mean(degrees) if degrees else 0
        ax_profile.axvline(mean_deg, ls="--", color=PALETTE["red"], lw=1.2)
        ax_profile.text(mean_deg + 1, ax_profile.get_ylim()[1] * 0.85,
                        f"mean={mean_deg:.0f}", fontsize=8, color=PALETTE["red"])
        ax_profile.set_xlabel("# Co-retrieval partners")
        ax_profile.set_ylabel("# Entities")
        ax_profile.set_title("(b)  Entity hub degree", fontsize=11, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(outdir, "fig1_selectivity.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return mod_cfg


# ──────────── Figure 2: Discriminability (A3) ────────────

def plot_discriminability(mods, fmt, topn, n_rec, outdir):
    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, 5.5))

    focus_ents = mods["all"]["ents"]
    focus_ws = mods["all"]["ws"]
    focus_label = "Retrieved KG entity"

    ent_counter_focus = Counter()
    for ents in focus_ents:
        ent_counter_focus.update(ents)
    top_n_ents = [e for e, _ in ent_counter_focus.most_common(topn)]
    ent_to_idx = {e: i for i, e in enumerate(top_n_ents)}

    n_samp = min(200, n_rec)
    mat = np.zeros((n_samp, len(top_n_ents)), dtype=np.float32)
    for i in range(n_samp):
        ents = focus_ents[i]
        ws = focus_ws[i]
        for j, e in enumerate(ents):
            if e in ent_to_idx:
                mat[i, ent_to_idx[e]] = ws[j] if j < len(ws) else 0.0

    vmax = np.percentile(mat[mat > 0], 97) if np.any(mat > 0) else 1.0
    if mat.shape[0] > 2 and mat.shape[1] > 2 and np.any(mat > 0):
        row_link = hierarchy.linkage(mat, method="ward", metric="euclidean")
        col_link = hierarchy.linkage(mat.T, method="ward", metric="euclidean")
        row_ord = hierarchy.leaves_list(row_link)
        col_ord = hierarchy.leaves_list(col_link)
        mat_ord = mat[np.ix_(row_ord, col_ord)]
        ent_labels = [top_n_ents[c].replace("DB", "") for c in col_ord]
    else:
        mat_ord = mat
        ent_labels = [e.replace("DB", "") for e in top_n_ents]

    im = ax.imshow(mat_ord, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=vmax)
    step = max(1, len(ent_labels) // 10)
    ax.set_xticks(np.arange(len(ent_labels))[::step])
    ax.set_xticklabels(ent_labels[::step], rotation=90, fontsize=6)
    ax.set_xlabel(f"{focus_label} entity (top-{len(top_n_ents)}, clustered)")
    ax.set_ylabel("Sample (clustered)")
    ax.set_title("")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Attention weight")

    fig.tight_layout()
    path = os.path.join(outdir, "fig2_discriminability.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────── Figure 3: Coverage (A4 + A5) ────────────

def plot_coverage(mods, all_counter, n_unique, outdir):
    top_20 = all_counter.most_common(20)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL + 1.5, 4.5))
    ax_freq, ax_cov = axes

    # ── Left: Top-20 entity frequency ──
    if top_20:
        ent_names_20 = [e for e, _ in top_20]
        ent_counts_20 = [c for _, c in top_20]
        y_pos = np.arange(len(ent_names_20))
        short = [e.replace("DB", "") for e in ent_names_20]
        colors = [PALETTE["blue"] if i < 5 else PALETTE["orange"] if i < 10
                  else PALETTE["green"] for i in range(len(ent_names_20))]
        ax_freq.barh(y_pos, ent_counts_20, height=0.6, color=colors, alpha=0.85,
                     edgecolor="white", linewidth=0.5)
        for i, (cnt, nm) in enumerate(zip(ent_counts_20, short)):
            ax_freq.text(cnt + 0.3, i, str(cnt), va="center", fontsize=7, color="#555")
        ax_freq.set_yticks(y_pos)
        ax_freq.set_yticklabels(short, fontsize=7)
        ax_freq.invert_yaxis()
        ax_freq.set_xlabel("Retrieval frequency")
        ax_freq.set_title("(a)  Top-20 entity frequency", fontsize=11, fontweight="bold")

        all_freqs = sorted(all_counter.values(), reverse=True)
        ins = ax_freq.inset_axes([0.55, 0.60, 0.40, 0.35])
        ranks = np.arange(1, len(all_freqs) + 1, dtype=float)
        ins.loglog(ranks, all_freqs, "-", color=PALETTE["blue"], lw=1.0, alpha=0.8)
        ins.scatter(ranks[:20], all_freqs[:20], s=10, color=PALETTE["red"],
                    zorder=3, edgecolors="white", linewidths=0.3)
        ins.set_xlabel("Rank", fontsize=6)
        ins.set_ylabel("Freq", fontsize=6)
        ins.set_title("Zipf distribution", fontsize=7)
        ins.tick_params(labelsize=5)
    else:
        ax_freq.text(0.5, 0.5, "No entity data", ha="center", va="center",
                     transform=ax_freq.transAxes, fontsize=9, color="gray")

    # ── Right: Entity coverage curve ──
    seen = set()
    coverage = []
    for ents in mods["all"]["ents"]:
        seen |= set(ents)
        coverage.append(len(seen))
    ax_cov.fill_between(range(len(coverage)), coverage, alpha=0.25, color=PALETTE["blue"])
    ax_cov.plot(coverage, color=PALETTE["blue"], lw=1.5)
    ax_cov.axhline(n_unique, ls="--", color=PALETTE["red"], lw=0.8, alpha=0.7)
    ax_cov.text(len(coverage) * 0.55, n_unique * 1.02, f"Total: {n_unique}",
                fontsize=8, color=PALETTE["red"])
    ax_cov.set_xlabel("# Samples seen")
    ax_cov.set_ylabel("# Unique entities retrieved")
    ax_cov.set_title("(b)  Entity coverage curve", fontsize=11, fontweight="bold")
    ax_cov.set_xlim(0, len(coverage))

    fig.tight_layout()
    path = os.path.join(outdir, "fig3_coverage.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────── Figure 4: Structural coherence (A1) ────────────

def plot_structure(mods, outdir):
    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    G, _ = build_coretrieval_graph(mods["all"]["ents"], mods["all"]["ws"], top_n=50)
    if G.number_of_nodes() < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    else:
        pos = nx.spring_layout(G, k=2.0, seed=42, iterations=100)
        counts = np.array([G.nodes[n]["count"] for n in G.nodes()])
        avg_ws = np.array([G.nodes[n]["avg_weight"] for n in G.nodes()])
        sizes = 80 + 600 * (counts - counts.min()) / (counts.max() - counts.min() + 1e-9)
        node_colors = avg_ws
        edge_ws = np.array([G.edges[u, v]["weight"] for u, v in G.edges()])
        if len(edge_ws) > 0:
            e_alpha = np.clip(edge_ws / (edge_ws.max() + 1e-9), 0.08, 0.55)
            nx.draw_networkx_edges(G, pos, alpha=e_alpha, width=0.6,
                                   edge_color="silver", ax=ax)
        sc = nx.draw_networkx_nodes(G, pos, node_size=sizes,
                                    node_color=node_colors, cmap="YlOrRd",
                                    alpha=0.9, edgecolors="white",
                                    linewidths=0.5, ax=ax)
        deg = dict(G.degree(weight="weight"))
        thresh = np.percentile(list(deg.values()), 85) if deg else 0
        labels = {n: n.replace("DB", "") for n in G.nodes() if deg.get(n, 0) >= thresh}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="#333", ax=ax)
        plt.colorbar(sc, ax=ax, shrink=0.55, label="Avg attn weight", pad=0.02)

    ax.set_title("")
    ax.axis("off")

    fig.tight_layout()
    path = os.path.join(outdir, "fig4_structure.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────── Figure 5: 3D Conformer contribution ────────────

def plot_3d_contribution(records, outdir):
    """Visualise the role of 3D conformer in cross-modal 2-hop retrieval.

    Panels:
      (a) Hop gate distribution — how much 3D-informed novelty is injected
      (b) Entity novelty — fraction of hop2 entities NOT in hop1
      (c) Route distribution — struct vs mol3D contribution in generation
    """
    has_hop2 = any("hop2_entity_ids" in r for r in records)
    has_gate = any("hop_gate" in r for r in records)
    has_route = any("route_mean" in r for r in records)

    if not (has_hop2 or has_gate or has_route):
        print("  Skipped: no hop2/gate/route data in records (re-train to collect)")
        return

    n_panels = sum([has_gate, has_hop2, has_route])
    fig, axes = plt.subplots(1, n_panels, figsize=(SINGLE_COL * n_panels + 1.0, 4.0))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # ── (a) Hop gate distribution ──
    if has_gate:
        gates = [r["hop_gate"] for r in records if "hop_gate" in r]
        ax = axes[ax_idx]; ax_idx += 1
        ax.hist(gates, bins=40, color=PALETTE["purple"], alpha=0.85,
                edgecolor="white", linewidth=0.3)
        med_g = np.median(gates)
        ax.axvline(med_g, ls="--", color=PALETTE["red"], lw=1.5)
        ax.text(med_g + 0.01, ax.get_ylim()[1] * 0.9,
                f"median={med_g:.3f}", fontsize=8, color=PALETTE["red"])
        ax.set_xlabel("Hop gate value")
        ax.set_ylabel("# Samples")
        chr_label = chr(ord('a') + ax_idx - 1)
        ax.set_title(f"({chr_label})  Hop gate distribution", fontsize=11, fontweight="bold")

    # ── (b) Entity novelty ──
    if has_hop2:
        novelty_fracs = []
        for r in records:
            if "hop2_entity_ids" not in r:
                continue
            hop1_set = set(r.get("topk_entity_ids", []))
            hop2_set = set(r["hop2_entity_ids"])
            if len(hop2_set) == 0:
                continue
            novel = hop2_set - hop1_set
            novelty_fracs.append(len(novel) / len(hop2_set))

        if novelty_fracs:
            ax = axes[ax_idx]; ax_idx += 1
            ax.hist(novelty_fracs, bins=30, color=PALETTE["cyan"], alpha=0.85,
                    edgecolor="white", linewidth=0.3)
            med_n = np.median(novelty_fracs)
            ax.axvline(med_n, ls="--", color=PALETTE["red"], lw=1.5)
            ax.text(med_n + 0.02, ax.get_ylim()[1] * 0.9,
                    f"median={med_n:.2%}", fontsize=8, color=PALETTE["red"])
            ax.set_xlabel("Fraction of novel entities in hop2")
            ax.set_ylabel("# Samples")
            chr_label = chr(ord('a') + ax_idx - 1)
            ax.set_title(f"({chr_label})  Entity novelty from 3D",
                         fontsize=11, fontweight="bold")

    # ── (c) Route distribution ──
    if has_route:
        routes = [r["route_mean"] for r in records if "route_mean" in r]
        ax = axes[ax_idx]; ax_idx += 1
        ax.hist(routes, bins=40, color=PALETTE["orange"], alpha=0.85,
                edgecolor="white", linewidth=0.3)
        med_r = np.median(routes)
        ax.axvline(med_r, ls="--", color=PALETTE["red"], lw=1.5)
        ax.text(med_r + 0.01, ax.get_ylim()[1] * 0.9,
                f"median={med_r:.3f}", fontsize=8, color=PALETTE["red"])
        ax.axvline(0.5, ls=":", color="gray", lw=0.8, alpha=0.6)
        ax.set_xlabel("Route weight (0=mol3D, 1=struct)")
        ax.set_ylabel("# Samples")
        chr_label = chr(ord('a') + ax_idx - 1)
        ax.set_title(f"({chr_label})  Struct vs Mol3D routing",
                     fontsize=11, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(outdir, "fig5_3d_contribution.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────── Main ────────────

def main():
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--explain_file", help="Direct path to a single .jsonl file")
    grp.add_argument("--explain_dir", help="Directory containing kg_explain_epoch*.jsonl")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Epoch (-1 = latest, only with --explain_dir)")
    parser.add_argument("--topn", type=int, default=40)
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "single", "multihead", "dual"],
                        help="Data format: auto-detect, or force single/multihead/dual")
    parser.add_argument("--outdir", default="figures/sacmsa_analysis",
                        help="Output subfolder for the four figures")
    args = parser.parse_args()

    # Resolve file path
    if args.explain_file:
        fpath = args.explain_file
    else:
        files = [f for f in os.listdir(args.explain_dir)
                 if f.startswith("kg_explain_epoch") and f.endswith(".jsonl")]
        epochs = sorted(int(f.replace("kg_explain_epoch", "").replace(".jsonl", ""))
                        for f in files)
        if not epochs:
            raise FileNotFoundError(f"No kg_explain files in {args.explain_dir}")
        ep = args.epoch if args.epoch >= 0 else epochs[-1]
        fpath = os.path.join(args.explain_dir, f"kg_explain_epoch{ep}.jsonl")
        print(f"Auto-selected epoch {ep}")

    records = load_records(fpath)
    fmt_override = None if args.format == "auto" else args.format
    fmt, n_per = detect_format(records, fmt_override)
    mods = extract_modalities(records, fmt, n_per)
    n_rec = len(records)

    apply_style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loaded {n_rec} samples | Format: {fmt} | Entities/modality: {n_per}")
    print(f"Output directory: {args.outdir}\n")

    all_counter = Counter()
    for ents in mods["all"]["ents"]:
        all_counter.update(ents)
    n_unique = len(all_counter)

    # ── Generate four figures ──
    print("[1/5] Selectivity ...")
    mod_cfg = plot_selectivity(mods, fmt, args.outdir)

    print("[2/5] Discriminability ...")
    plot_discriminability(mods, fmt, args.topn, n_rec, args.outdir)

    print("[3/5] Coverage ...")
    plot_coverage(mods, all_counter, n_unique, args.outdir)

    print("[4/5] Structural coherence ...")
    plot_structure(mods, args.outdir)

    print("[5/5] 3D conformer contribution ...")
    plot_3d_contribution(records, args.outdir)

    # ── Summary ──
    print(f"\n=== SA-CMSA Retrieval Analysis ===")
    print(f"  Format: {fmt}, Samples: {n_rec}, Unique entities: {n_unique}")
    for mod_name, ws_list, _ in mod_cfg:
        ginis = [gini_coefficient(w) for w in ws_list if w]
        top1s = [w[0] for w in ws_list if w]
        print(f"  {mod_name}: Gini={np.mean(ginis):.4f}±{np.std(ginis):.4f}"
              f"  Top-1={np.mean(top1s):.4f}±{np.std(top1s):.4f}")
    total_slots = sum(len(e) for e in mods["all"]["ents"])
    print(f"  Coverage: {n_unique} unique / {total_slots} total slots"
          f" = {n_unique/total_slots*100:.1f}% unique")
    print(f"\nAll figures saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
