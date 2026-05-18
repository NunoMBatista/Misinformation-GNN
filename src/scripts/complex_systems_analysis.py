import sys
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from scipy import stats
from collections import defaultdict
from torch_geometric.utils import to_networkx

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import load_data

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs/complex_systems")
COLOURS = {"Non-Rumour": "#4C72B0", "Rumour": "#DD8452"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_nx_graphs(dataset):
    graphs = []
    for data in dataset:
        G = to_networkx(data, to_undirected=False)
        graphs.append((G, int(data.y.item()), data.event))
    return graphs


def cascade_depth(G):
    """Mean distance from root (node 0) to all other nodes."""
    #if G.number_of_nodes() == 0:
    if G.number_of_nodes() <= 1:
        return 0
    try:
        lengths = nx.single_source_shortest_path_length(G, 0)
        # return max(lengths.values())  # longest chain (outlier-sensitive)
        return float(np.mean(list(lengths.values())))
    except Exception:
        return 0


def rank_biserial(U, n1, n2):
    """Effect size for Mann-Whitney U: 0=none, 0.1=small, 0.3=medium, 0.5=large."""
    return abs(1 - (2 * U) / (n1 * n2))


def coloured_boxplot(ax, data_nr, data_r, ylabel, title):
    bp = ax.boxplot(
        [data_nr, data_r],
        labels=["Non-Rumour", "Rumour"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, colour in zip(bp["boxes"], COLOURS.values()):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ─────────────────────────────────────────────────────────────────────────────
# Cascade Depth / Size Correlation
# ─────────────────────────────────────────────────────────────────────────────

def analysis_depth_size_correlation(graphs):
    print("\nCascade Depth / Size Correlation")

    sizes, depths, labels = [], [], []
    for G, label, _ in graphs:
        sizes.append(G.number_of_nodes())
        depths.append(cascade_depth(G))
        labels.append(label)

    sizes  = np.array(sizes)
    depths = np.array(depths)
    labels = np.array(labels)

    r_size,  p_size  = stats.spearmanr(sizes,  labels)
    r_depth, p_depth = stats.spearmanr(depths, labels)
    mask_r  = np.array(labels) == 1
    mask_nr = np.array(labels) == 0
    print(f"  Size  Rumour:    mean={sizes[mask_r].mean():.2f},  median={np.median(sizes[mask_r]):.1f}")
    print(f"  Size  Non-R:     mean={sizes[mask_nr].mean():.2f}, median={np.median(sizes[mask_nr]):.1f}")
    print(f"  Size  vs Label:  rho={r_size:.3f}, p={p_size:.4f}")
    print(f"  Depth Rumour:    mean={depths[mask_r].mean():.3f}, median={np.median(depths[mask_r]):.3f}")
    print(f"  Depth Non-R:     mean={depths[mask_nr].mean():.3f}, median={np.median(depths[mask_nr]):.3f}")
    print(f"  Depth vs Label:  rho={r_depth:.3f}, p={p_depth:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cascade Depth & Size by Label", fontsize=13, fontweight="bold")

    coloured_boxplot(
        axes[0], sizes[mask_nr], sizes[mask_r],
        "Node Count (log scale)", f"Cascade Size\n(Spearman rho={r_size:.3f}, p={p_size:.4f})"
    )
    axes[0].set_yscale("log")

    coloured_boxplot(
        axes[1], depths[mask_nr], depths[mask_r],
        "Mean Depth (log scale)", f"Cascade Mean Depth\n(Spearman rho={r_depth:.3f}, p={p_depth:.4f})"
    )
    axes[1].set_yscale("log")

    plt.tight_layout()
    out = OUTPUT_DIR / "depth_size_correlation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return pd.DataFrame({"size": sizes, "depth": depths, "label": labels})


# ─────────────────────────────────────────────────────────────────────────────
# Tipping Point Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analysis_tipping_point(graphs):
    print("\nTipping Point Analysis (Cascade Size Buckets)")

    records = [{"size": G.number_of_nodes(), "label": label} for G, label, _ in graphs]
    df = pd.DataFrame(records)

    bins       = [0, 5, 10, 20, 50, 100, 200, float("inf")]
    bin_labels = ["1–5", "6–10", "11–20", "21–50", "51–100", "101–200", "200+"]
    df["bucket"] = pd.cut(df["size"], bins=bins, labels=bin_labels)

    grouped = (
        df.groupby("bucket", observed=True)["label"]
        .agg(total="count", rumour_count="sum")
    )
    grouped["rumour_ratio"] = grouped["rumour_count"] / grouped["total"]
    print(grouped.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tipping Point: Cascade Size vs Rumour Prevalence",
                 fontsize=13, fontweight="bold")

    x = grouped.index.astype(str)

    axes[0].bar(x, grouped["total"], color="#4C72B0", alpha=0.8)
    axes[0].set_xlabel("Cascade Size Bucket")
    axes[0].set_ylabel("Number of Cascades")
    axes[0].set_title("Cascade Count per Size Bucket")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(x, grouped["rumour_ratio"], marker="o", color="#DD8452", linewidth=2, label="Rumour ratio")
    axes[1].axhline(df["label"].mean(), linestyle="--", color="grey",
                    label=f"Dataset mean ({df['label'].mean():.2f})")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Cascade Size Bucket")
    axes[1].set_ylabel("Fraction Labelled Rumour")
    axes[1].set_title("Rumour Ratio by Size Bucket")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = OUTPUT_DIR / "tipping_point.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# Structural Virality (Average Shortest Path Length)
# ─────────────────────────────────────────────────────────────────────────────

def _avg_path_length(G_directed):
    """APL on the largest connected component (undirected). Approximated for large graphs."""
    G_ud = G_directed.to_undirected()
    if G_ud.number_of_nodes() <= 1:
        return np.nan
    lcc_nodes = max(nx.connected_components(G_ud), key=len)
    lcc = G_ud.subgraph(lcc_nodes).copy()
    n = lcc.number_of_nodes()
    if n <= 1:
        return np.nan
    return nx.average_shortest_path_length(lcc)


def analysis_structural_virality(graphs):
    print("\nStructural Virality (Average Path Length)")

    records = []
    for G, label, _ in graphs:
        apl = _avg_path_length(G)
        records.append({"apl": apl, "label": label})

    df = pd.DataFrame(records).dropna()
    rumour     = df[df["label"] == 1]["apl"]
    non_rumour = df[df["label"] == 0]["apl"]

    stat, p = stats.mannwhitneyu(rumour, non_rumour, alternative="two-sided")
    r = rank_biserial(stat, len(rumour), len(non_rumour))
    print(f"  Rumour     APL: mean={rumour.mean():.3f}, median={rumour.median():.3f}")
    print(f"  Non-Rumour APL: mean={non_rumour.mean():.3f}, median={non_rumour.median():.3f}")
    print(f"  Mann-Whitney U={stat:.0f}, p={p:.4f}, r={r:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Structural Virality: Average Path Length",
                 fontsize=13, fontweight="bold")

    coloured_boxplot(
        axes[0], non_rumour, rumour,
        "Avg Shortest Path Length", f"APL Distribution\n(Mann-Whitney p={p:.4f}, r={r:.3f})"
    )
    axes[1].hist(non_rumour, bins=40, alpha=0.65, color=COLOURS["Non-Rumour"],
                 label="Non-Rumour", density=True)
    axes[1].hist(rumour,     bins=40, alpha=0.65, color=COLOURS["Rumour"],
                 label="Rumour", density=True)
    axes[1].set_xlabel("Average Path Length")
    axes[1].set_ylabel("Density")
    axes[1].set_title("APL Distribution Overlay")
    axes[1].legend()

    plt.tight_layout()
    out = OUTPUT_DIR / "structural_virality.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Network Robustness / Immunization
# ─────────────────────────────────────────────────────────────────────────────

def _lcc_fraction_after_removal(G_directed, fraction):
    G_ud = G_directed.to_undirected()
    n = G_ud.number_of_nodes()
    if n == 0:
        return np.nan
    k = max(1, int(n * fraction))
    top_nodes = [node for node, _ in sorted(G_ud.degree(), key=lambda x: x[1], reverse=True)[:k]]
    G_pruned = G_ud.copy()
    G_pruned.remove_nodes_from(top_nodes)
    if G_pruned.number_of_nodes() == 0:
        return 0.0
    return len(max(nx.connected_components(G_pruned), key=len)) / n


def analysis_robustness(graphs, fractions=None):
    if fractions is None:
        fractions = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("\nNetwork Robustness / Immunization")

    buckets = defaultdict(lambda: {"rumour": [], "non_rumour": []})
    for G, label, _ in graphs:
        for frac in fractions:
            lcc = _lcc_fraction_after_removal(G, frac)
            key = "rumour" if label == 1 else "non_rumour"
            buckets[frac][key].append(lcc)

    rows = []
    for frac in fractions:
        r_mean  = np.nanmean(buckets[frac]["rumour"])
        nr_mean = np.nanmean(buckets[frac]["non_rumour"])
        rows.append({"removal_%": frac * 100, "rumour_lcc": r_mean, "non_rumour_lcc": nr_mean})
        print(f"  Remove {frac*100:.0f}%: Rumour LCC={r_mean:.3f}, Non-Rumour LCC={nr_mean:.3f}")

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["removal_%"], df["rumour_lcc"],     marker="o", color=COLOURS["Rumour"],
            linewidth=2, label="Rumour")
    ax.plot(df["removal_%"], df["non_rumour_lcc"], marker="s", color=COLOURS["Non-Rumour"],
            linewidth=2, label="Non-Rumour")
    ax.set_xlabel("Removed Nodes (%)")
    ax.set_ylabel("LCC Fraction (relative to original size)")
    ax.set_title("Network Robustness: LCC after Hub Removal")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "robustness.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Echo Chambers (Clustering Coefficient + Branching Factor)
# ─────────────────────────────────────────────────────────────────────────────

def analysis_echo_chambers(graphs):
    print("\nEcho Chambers (Branching Factor)")

    records = []
    for G, label, _ in graphs:
        if G.number_of_nodes() < 3:
            continue
        out_degs = [d for _, d in G.out_degree() if d > 0]
        branching = float(np.mean(out_degs)) if out_degs else 0.0
        records.append({"branching": branching, "label": label})

    df = pd.DataFrame(records)
    rumour     = df[df["label"] == 1]
    non_rumour = df[df["label"] == 0]

    n_r, n_nr = len(rumour), len(non_rumour)
    U, p = stats.mannwhitneyu(rumour["branching"], non_rumour["branching"], alternative="two-sided")
    r = rank_biserial(U, n_r, n_nr)
    print(f"  Branching     Rumour={rumour['branching'].mean():.4f}, "
          f"Non-Rumour={non_rumour['branching'].mean():.4f}, p={p:.4f}, r={r:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Echo Chambers: Branching Factor",
                 fontsize=13, fontweight="bold")

    coloured_boxplot(
        ax, non_rumour["branching"], rumour["branching"],
        "Mean Branching Factor",
        f"Mean Branching Factor\n(Mann-Whitney p={p:.4f}, r={r:.3f})"
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "echo_chambers.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# New Metrics
# ─────────────────────────────────────────────────────────────────────────────
#
# Each metric is a scalar computed per cascade graph and then compared between
# Rumour and Non-Rumour classes via Mann-Whitney U + rank-biserial effect size.
# All metrics are size-normalised or inherently scale-invariant to avoid the
# confound that plagued the original analyses.
#
# Course connections:
#   Epidemic R₀          → Network Dynamics (SIS epidemic model)
#   Root Dominance        → Network Topology (broadcast vs. viral spreading)
#   Gini Out-Degree       → Network Science (scale-free / preferential attachment)
#   Strahler Number       → Fractal Theory (self-similar branching complexity)
#   Leaf Ratio            → Graph / Tree structure
#   Normalised Depth      → Network Science (small-world log-diameter baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _epidemic_r0(G):
    """Mean out-degree of internal nodes only (those with at least one reply).
    Directly maps to the branching ratio R₀ of the SIS/epidemic cascade model:
    R₀ > 1 → super-critical (growing cascade); R₀ < 1 → sub-critical (dying)."""
    internal = [d for _, d in G.out_degree() if d > 0]
    return float(np.mean(internal)) if internal else np.nan


def _root_dominance(G, root=0):
    """Fraction of all edges that originate from the root tweet.
    Value near 1 → star / broadcast (one source drives everything).
    Value near 0 → peer-to-peer viral chain.
    Size-invariant by construction."""
    m = G.number_of_edges()
    if m == 0:
        return np.nan
    return G.out_degree(root) / m


def _gini_out_degree(G):
    """Gini coefficient of the out-degree distribution.
    0 = perfectly equal attention across all tweets.
    1 = one tweet attracted all replies (maximum concentration).
    Captures whether spreading follows a scale-free (rich-get-richer) pattern."""
    arr = np.array([d for _, d in G.out_degree()], dtype=float)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.dot(index, arr) / (n * arr.sum())) - (n + 1) / n)


def _strahler(G, root=0):
    """Horton-Strahler order: a fractal-theoretic measure of branching complexity.
    Defined recursively on trees:
      - Leaf → order 1
      - Internal node → max(children orders), incremented by 1 if two or more
        children share that maximum (i.e. balanced branching detected).
    Higher order = more complex, self-similar branching structure.
    Implemented iteratively (post-order DFS) to avoid recursion limits."""
    if G.number_of_nodes() <= 1:
        return 1
    order = {}
    visited = set()
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            child_orders = [order[c] for c in G.successors(node) if c in order]
            if not child_orders:
                order[node] = 1
            else:
                mx = max(child_orders)
                order[node] = mx + 1 if child_orders.count(mx) >= 2 else mx
        elif node not in visited:
            visited.add(node)
            stack.append((node, True))
            for child in G.successors(node):
                if child not in visited:
                    stack.append((child, False))
    return order.get(root, 1)


def _leaf_ratio(G):
    """Fraction of nodes that are leaves (out-degree == 0).
    High → flat/broad cascade (many dead-end replies, little re-engagement).
    Low  → deep recursive cascade (replies spawn further replies)."""
    n = G.number_of_nodes()
    if n == 0:
        return np.nan
    leaves = sum(1 for _, d in G.out_degree() if d == 0)
    return leaves / n


def _norm_depth(G, root=0):
    """Max depth normalised by log₂(n).
    A perfectly balanced binary tree scores exactly 1.0.
    Score > 1 → chain-like (deeper than expected for its size).
    Score < 1 → unusually flat/broad for its size.
    Removes the raw size confound that breaks unnormalised depth comparisons."""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    try:
        lengths = nx.single_source_shortest_path_length(G, root)
        max_d = max(lengths.values())
    except Exception:
        return np.nan
    return max_d / np.log2(n)


# Ordered registry used by the analysis function and the plot loop
_NEW_METRICS = [
    ("r0",         "Epidemic R₀ (branching ratio)",      _epidemic_r0),
    ("root_dom",   "Root Dominance (broadcast index)",   _root_dominance),
    ("gini",       "Gini (out-degree inequality)",        _gini_out_degree),
    ("strahler",   "Strahler Number (tree complexity)",  _strahler),
    ("leaf_ratio", "Leaf Ratio",                          _leaf_ratio),
    ("norm_depth", "Normalised Depth (depth / log₂ n)",  _norm_depth),
]


def analysis_new_metrics(graphs):
    print("\nNew Graph Metrics Analysis")

    records = []
    for G, label, event in graphs:
        row = {"label": label, "event": event}
        for key, _, fn in _NEW_METRICS:
            try:
                row[key] = fn(G)
            except Exception:
                row[key] = np.nan
        records.append(row)

    df = pd.DataFrame(records)

    rumour     = df[df["label"] == 1]
    non_rumour = df[df["label"] == 0]
    n_r, n_nr  = len(rumour), len(non_rumour)

    print(f"\n  {'Metric':<42} {'R mean':>8} {'NR mean':>8}  {'p-value':>8}  {'r':>6}  Sig")
    print(f"  {'-'*42} {'-'*8} {'-'*8}  {'-'*8}  {'-'*6}  ---")
    for key, label_str, _ in _NEW_METRICS:
        r_vals  = rumour[key].dropna()
        nr_vals = non_rumour[key].dropna()
        if len(r_vals) < 2 or len(nr_vals) < 2:
            continue
        U, p = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        r_eff = rank_biserial(U, len(r_vals), len(nr_vals))
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {label_str:<42} {r_vals.mean():>8.3f} {nr_vals.mean():>8.3f}  {p:>8.4f}  {r_eff:>6.3f}  {sig}")

    # One figure per metric
    for key, label_str, _ in _NEW_METRICS:
        r_vals  = rumour[key].dropna()
        nr_vals = non_rumour[key].dropna()
        if len(r_vals) < 2 or len(nr_vals) < 2:
            continue
        U, p    = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        r_eff   = rank_biserial(U, len(r_vals), len(nr_vals))
        fig, ax = plt.subplots(figsize=(6, 5))
        coloured_boxplot(ax, nr_vals, r_vals, label_str,
                         f"{label_str}\n(Mann-Whitney p={p:.4f}, r={r_eff:.3f})")
        plt.tight_layout()
        out = OUTPUT_DIR / f"new_metric_{key}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALWAYS_EXCLUDE = {
    "prince-toronto-all-rnr-threads",
    "ebola-essien-all-rnr-threads",
    "ferguson-all-rnr-threads",
}

TRAIN_EVENTS = {
    "germanwings-crash-all-rnr-threads",
    "gurlitt-all-rnr-threads",
    "ottawashooting-all-rnr-threads",
    "putinmissing-all-rnr-threads",
}

TEST_EVENTS = {
    "charliehebdo-all-rnr-threads",
    "sydneysiege-all-rnr-threads",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["all", "train", "test"], default="all",
                        help="Which events to include: all (default), train, or test")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: outputs/complex_systems)")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_data()
    dataset = [d for d in dataset if d.event not in ALWAYS_EXCLUDE]

    if args.split == "train":
        dataset = [d for d in dataset if d.event in TRAIN_EVENTS]
    elif args.split == "test":
        dataset = [d for d in dataset if d.event in TEST_EVENTS]

    print(f"  {len(dataset)} graphs loaded (split={args.split}).")

    print("Converting to NetworkX...")
    graphs = build_nx_graphs(dataset)

    analysis_depth_size_correlation(graphs)
    analysis_tipping_point(graphs)
    analysis_structural_virality(graphs)
    analysis_robustness(graphs)
    analysis_echo_chambers(graphs)
    analysis_new_metrics(graphs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nAll outputs written to: {OUTPUT_DIR}/  (run at {timestamp})")


if __name__ == "__main__":
    main()
