"""
xai_analysis.py

Two XAI lenses applied to one fixed held-out fold (charliehebdo):

  1. GNNExplainer on ImprovedGNN/GCNConv
       → which feature groups and which cascade positions actually drive predictions?
  2. Attention-weight extraction from ImprovedGAT/GATv2Conv
       → where does the model "look" in the cascade tree, and does it differ
         between rumours and non-rumours?

Charliehebdo is chosen as the held-out event because it is the largest
(~1950 test graphs), giving the most stable aggregate statistics.

Hyperparameters and architectures match the tuned benchmark configs
(best_configs.json, gnn_full / gat_full entries).

Outputs: outputs/xai/
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, to_undirected
import networkx as nx

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import load_data, filter_features
from src.models.gnn import ImprovedGNN, ImprovedGAT
from src.models.trainer import _balance_per_event, focal_loss

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

TEST_EVENT  = "charliehebdo-all-rnr-threads"   # largest event → best statistics
SEED        = 42

# GNNExplainer is ~2 s/graph; 80 gives good statistics without a 30-min wait
MAX_EXPLAIN  = 80

OUT_DIR = Path("outputs/xai")

# Feature layout in the 390-dim node vector (matches dataset.py)
FEATURE_GROUPS = {
    "NLP (384-dim)": list(range(0, 384)),
    "Followers":     [384],
    "Verified":      [385],
    "PageRank":      [386],
    "Degree":        [387],
    "In-degree":     [388],
    "Out-degree":    [389],
}

COLOURS = {"Rumour": "#DD8452", "Non-Rumour": "#4C72B0"}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_split():
    """Load dataset with all features; split into train / test by TEST_EVENT."""
    dataset = load_data()
    all_on  = {k: True for k in ["use_nlp", "use_followers", "use_verified",
                                   "use_pagerank", "use_degree", "use_indegree", "use_outdegree"]}
    dataset, input_dim = filter_features(dataset, all_on)

    # Mirror the same exclusions used in experiment.yml
    excluded = {"prince-toronto-all-rnr-threads", "ebola-essien-all-rnr-threads"}
    dataset  = [d for d in dataset if d.event not in excluded and d.num_nodes >= 3]

    train = [d for d in dataset if d.event != TEST_EVENT]
    test  = [d for d in dataset if d.event == TEST_EVENT]
    return train, test, input_dim


def node_depths(data):
    """
    Shortest-path depth from node 0 (the source tweet / root) to every node.
    Returns a plain list, one depth per node; -1 if the node is unreachable
    (disconnected component).
    """
    G = to_networkx(data, to_undirected=False)
    try:
        lengths = nx.single_source_shortest_path_length(G, 0)
    except Exception:
        lengths = {}
    return [lengths.get(i, -1) for i in range(data.num_nodes)]


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_data, device, config):
    """Matches trainer.py train_nn: balanced sampling, focal loss, Adam + cosine annealing, gradient clipping."""
    balanced  = _balance_per_event(train_data, random_state=SEED)
    loader    = DataLoader(balanced, batch_size=32, shuffle=True)

    lr        = float(config["learning_rate"])
    wd        = float(config["weight_decay"])
    gamma     = float(config.get("focal_gamma", 2.0))
    epochs    = int(config["epochs"])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = lambda lo, tg: focal_loss(lo, tg, gamma=gamma)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss  = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    return model


def evaluate(model, test_data, device):
    """Quick accuracy / F1 check so we know the model we explain is reasonable."""
    from sklearn.metrics import accuracy_score, f1_score
    loader = DataLoader(test_data, batch_size=64, shuffle=False)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds.extend((torch.sigmoid(logits) > 0.5).int().cpu().tolist())
            labels.extend(batch.y.cpu().tolist())
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    return acc, f1


# ── 1. GNNExplainer ───────────────────────────────────────────────────────────

def run_gnnexplainer(gcn, test_data, input_dim, device):
    """
    GNNExplainer optimises soft masks over node features and edges per graph
    to best preserve the GCN's prediction.  We aggregate masks across many
    graphs to find global importance patterns.

    Returns:
      feat_by_label  – dict {"Rumour": [N × input_dim arrays], "Non-Rumour": [...]}
      depth_node_imp – dict {depth: [list of mean node importances]}
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=gcn,
        algorithm=GNNExplainer(epochs=100),  # 100 inner epochs per graph is enough for relative rankings
        explanation_type="model",            # explain the model's own prediction, not a ground-truth label
        node_mask_type="attributes",         # per-feature mask at each node → shape [N, F]
        edge_mask_type="object",             # scalar importance per edge
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",               # our model returns raw logits, not probabilities
        ),
    )

    feat_by_label  = defaultdict(list)   # class label → list of [input_dim] importance arrays
    depth_node_imp = defaultdict(list)   # depth → list of mean node importances

    # Balanced sample: equal Rumour / Non-Rumour so per-class stats are fair
    import random as _random
    _rng = _random.Random(SEED)
    _rumour    = [d for d in test_data if int(d.y.item()) == 1]
    _nonrumour = [d for d in test_data if int(d.y.item()) == 0]
    _n = min(MAX_EXPLAIN // 2, len(_rumour), len(_nonrumour))
    subset = _rng.sample(_rumour, _n) + _rng.sample(_nonrumour, _n)
    for i, data in enumerate(subset):
        print(f"  GNNExplainer {i+1}/{len(subset)}", end="\r")

        x          = data.x.to(device)
        edge_index = data.edge_index.to(device)
        # Single-graph forward: all nodes belong to graph 0
        batch      = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        label      = "Rumour" if int(data.y.item()) == 1 else "Non-Rumour"

        try:
            exp = explainer(x=x, edge_index=edge_index, batch=batch)
        except Exception:
            continue  # skip degenerate graphs (e.g. single-node, no edges)

        if exp.node_mask is None:
            continue

        node_mask = exp.node_mask.detach().cpu()   # [N, F]

        # Mean absolute mask across nodes → global feature importance for this graph
        feat_imp = node_mask.abs().mean(dim=0).numpy()
        feat_by_label[label].append(feat_imp)

        # Mean mask across feature dimensions → which nodes were structurally important?
        node_imp = node_mask.abs().mean(dim=1).numpy()   # [N]
        for depth, imp in zip(node_depths(data), node_imp):
            if depth >= 0:
                depth_node_imp[depth].append(float(imp))

    print()
    return feat_by_label, depth_node_imp


def plot_feature_groups(feat_by_label, out_dir):
    """
    Collapse the 390 raw feature dims into 7 semantic groups and compare
    importance between rumours and non-rumours.
    If NLP dominates both classes, the graph structure is not adding much.
    If structural features (PageRank, degree) differ by class, the GCN is
    actually using the cascade topology differently per class.
    """
    group_scores = {}
    for label, masks in feat_by_label.items():
        mean_mask = np.mean(masks, axis=0)
        group_scores[label] = {
            g: float(mean_mask[idx].mean()) for g, idx in FEATURE_GROUPS.items()
        }

    groups = list(FEATURE_GROUPS.keys())
    x      = np.arange(len(groups))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (label, colour) in enumerate(COLOURS.items()):
        if label not in group_scores:
            continue
        vals = [group_scores[label].get(g, 0) for g in groups]
        ax.bar(x + offset * width, vals, width, label=label, color=colour, alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Mean GNNExplainer node-feature mask")
    ax.set_title("GCN Feature Group Importance by Class\n(higher = GCN relies on this more for its predictions)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "gcn_feature_groups.png", dpi=150)
    plt.close(fig)

    # Flatten to a single importance score per group (mean across classes) for printing
    all_masks = [m for masks in feat_by_label.values() for m in masks]
    mean_all  = np.mean(all_masks, axis=0)
    return {g: float(mean_all[idx].mean()) for g, idx in FEATURE_GROUPS.items()}


def plot_node_depth_importance(depth_node_imp, out_dir):
    """
    Mean GNNExplainer node importance by cascade depth.
    Depth 0 = source tweet (root).
    If depth-0 always dominates, the source tweet's content drives predictions
    and the reply structure is mostly ignored by the GCN.
    """
    depths = sorted(k for k in depth_node_imp if k <= 8)
    means  = [np.mean(depth_node_imp[d]) for d in depths]
    stds   = [np.std(depth_node_imp[d]) for d in depths]
    counts = [len(depth_node_imp[d]) for d in depths]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(depths, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    for d, m, c in zip(depths, means, counts):
        ax.text(d, m + max(stds) * 0.1, f"n={c}", ha="center", fontsize=7, color="gray")

    ax.set_xlabel("Node depth in cascade  (0 = source tweet,  1 = direct reply, …)")
    ax.set_ylabel("Mean node importance (GNNExplainer mask)")
    ax.set_title("GCN: Which Position in the Cascade Matters Most?")
    plt.tight_layout()
    fig.savefig(out_dir / "gcn_node_depth_importance.png", dpi=150)
    plt.close(fig)


# ── 2. GAT Attention Weights ──────────────────────────────────────────────────

def extract_attention(gat, data, device):
    """
    Step through ImprovedGAT layers manually to intercept per-edge attention
    coefficients. Replicates ImprovedGAT.forward() exactly: edge direction
    transform, then for each layer conv → skip + relu + norm → dropout.

    GATv2Conv.forward(..., return_attention_weights=True) returns:
        (output, (edge_index, alpha))  where alpha is [num_edges, num_heads]
    """
    gat.eval()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Replicate ImprovedGAT.forward() edge direction transform
    if gat.edge_direction == 'bidirectional':
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    elif gat.edge_direction == 'inverted':
        edge_index = edge_index.flip(0)

    layer_alphas = []
    with torch.no_grad():
        for conv, skip, norm in zip(gat.convs, gat.skips, gat.norms):
            conv_out, (_, alpha) = conv(x, edge_index, return_attention_weights=True)
            layer_alphas.append(alpha.cpu())   # [E, num_heads]
            x = norm(F.relu(conv_out + skip(x)))
            x = F.dropout(x, p=gat.dropout_p, training=False)

    return layer_alphas   # list of [E, heads] tensors, one per layer


def build_attention_df(gat, test_data, device):
    """
    For every test graph, every GAT layer, every node: record the mean
    attention weight the node RECEIVES across all its incoming edges and heads.
    This tells us which positions in the cascade the model pays most attention to.
    """
    records = []
    for gidx, data in enumerate(test_data):
        if data.num_edges == 0:
            continue

        label  = "Rumour" if int(data.y.item()) == 1 else "Non-Rumour"
        depths = node_depths(data)
        # Use the same edge_index that extract_attention operates on (mirrors ImprovedGAT.forward)
        ei_t = data.edge_index
        if gat.edge_direction == 'bidirectional':
            ei_t = to_undirected(ei_t, num_nodes=data.num_nodes)
        elif gat.edge_direction == 'inverted':
            ei_t = ei_t.flip(0)
        ei = ei_t.numpy()   # [2, E_processed]

        try:
            layer_alphas = extract_attention(gat, data, device)
        except Exception:
            continue

        for layer_idx, alpha in enumerate(layer_alphas):
            alpha_per_edge = alpha.mean(dim=1).numpy()   # mean over heads → [E]

            # Collect incoming attention weights per destination node
            node_incoming = defaultdict(list)
            for eidx, dst in enumerate(ei[1]):
                node_incoming[dst].append(float(alpha_per_edge[eidx]))

            for nidx in range(data.num_nodes):
                incoming = node_incoming.get(nidx, [])
                records.append({
                    "graph_idx": gidx,
                    "layer":     layer_idx + 1,
                    "depth":     depths[nidx],
                    "attn":      np.mean(incoming) if incoming else np.nan,
                    "label":     label,
                })

    return pd.DataFrame(records)


def plot_attention_by_depth(attn_df, out_dir):
    """
    Mean attention received vs. cascade depth, one subplot per GAT layer.
    If the lines for Rumour and Non-Rumour diverge, the model is using
    cascade structure differently for each class — a meaningful signal.
    If they overlap, attention patterns are class-agnostic (topology not helping).
    """
    df  = attn_df[(attn_df["depth"] >= 0) & (attn_df["depth"] <= 7)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, layer in zip(axes, [1, 2]):
        ldf = df[df["layer"] == layer]
        for label, colour in COLOURS.items():
            grp   = ldf[ldf["label"] == label].groupby("depth")["attn"]
            means = grp.mean()
            stds  = grp.std().fillna(0)
            ax.plot(means.index, means.values, marker="o", label=label, color=colour)
            ax.fill_between(means.index, means - stds, means + stds, alpha=0.15, color=colour)
        ax.set_xlabel("Cascade depth  (0 = source tweet)")
        ax.set_ylabel("Mean attention weight received")
        ax.set_title(f"GAT Layer {layer}")
        ax.legend()

    plt.suptitle("GAT Attention by Cascade Depth — Rumour vs. Non-Rumour\n"
                 "Diverging lines = model uses structure differently per class",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "gat_attention_by_depth.png", dpi=150)
    plt.close(fig)


def plot_attention_entropy(attn_df, out_dir):
    """
    Attention entropy per graph (layer 2, the layer closest to the classifier).

    Low entropy  → attention concentrated on a few nodes (model found structure to exploit)
    High entropy → attention spread evenly (topology not helping; all nodes look the same)

    A significant Rumour vs. Non-Rumour difference in entropy is strong evidence
    that the GAT is actually leveraging cascade topology, not just node features.
    """
    from scipy.stats import entropy as sp_entropy, mannwhitneyu

    records = []
    layer2  = attn_df[(attn_df["layer"] == 2) & attn_df["attn"].notna()]

    for (gidx, label), grp in layer2.groupby(["graph_idx", "label"]):
        vals = grp["attn"].values
        vals = vals[vals > 0]
        if len(vals) < 2:
            continue
        p = vals / vals.sum()
        records.append({"label": label, "entropy": sp_entropy(p)})

    ent_df  = pd.DataFrame(records)
    r_ent   = ent_df[ent_df["label"] == "Rumour"]["entropy"]
    nr_ent  = ent_df[ent_df["label"] == "Non-Rumour"]["entropy"]
    _, p    = mannwhitneyu(r_ent, nr_ent, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, colour in COLOURS.items():
        vals = ent_df[ent_df["label"] == label]["entropy"]
        ax.hist(vals, bins=30, alpha=0.6,
                label=f"{label}  (μ={vals.mean():.2f})", color=colour)
    ax.set_xlabel("Attention entropy  (layer 2)")
    ax.set_ylabel("Number of graphs")
    ax.set_title(f"GAT: How Concentrated Is Attention?\n"
                 f"Mann-Whitney p = {p:.4f}  ({'sig.' if p < 0.05 else 'n.s.'})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "gat_attention_entropy.png", dpi=150)
    plt.close(fig)

    return ent_df, p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   |   Held-out fold: {TEST_EVENT}\n")

    # ── Load tuned hyperparameters ────────────────────────────────────────────
    best_configs_path = Path("outputs/hp_search/best_configs.json")
    with open(best_configs_path) as f:
        best_configs = json.load(f)
    gnn_hp = best_configs["gnn_full"]["hparams"]
    gat_hp = best_configs["gat_full"]["hparams"]
    print(f"GNN hparams (gnn_full): {gnn_hp}")
    print(f"GAT hparams (gat_full): {gat_hp}\n")

    # ── Load & split ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_data, test_data, input_dim = load_split()
    n_r  = sum(1 for d in test_data if d.y.item() == 1)
    n_nr = len(test_data) - n_r
    print(f"  Train: {len(train_data)}  |  Test: {len(test_data)} "
          f"({n_r} rumour, {n_nr} non-rumour)  |  Feature dim: {input_dim}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining ImprovedGNN...")
    gcn = ImprovedGNN(
        input_dim=input_dim,
        hidden_dims=gnn_hp["hidden_dims"],
        dropout=gnn_hp["dropout"],
        edge_direction=gnn_hp.get("edge_direction", "original"),
    ).to(device)
    gcn = train_model(gcn, train_data, device, gnn_hp)
    acc, f1 = evaluate(gcn, test_data, device)
    print(f"  ImprovedGNN test  ->  Acc: {acc:.4f}  F1: {f1:.4f}")

    print("Training ImprovedGAT...")
    gat = ImprovedGAT(
        input_dim=input_dim,
        hidden_dims=gat_hp["hidden_dims"],
        heads=gat_hp.get("heads", 2),
        dropout=gat_hp["dropout"],
        edge_direction=gat_hp.get("edge_direction", "original"),
    ).to(device)
    gat = train_model(gat, train_data, device, gat_hp)
    acc, f1 = evaluate(gat, test_data, device)
    print(f"  ImprovedGAT test  ->  Acc: {acc:.4f}  F1: {f1:.4f}")

    # ── GNNExplainer ──────────────────────────────────────────────────────────
    print(f"\n[1/2] GNNExplainer on ImprovedGNN — {min(MAX_EXPLAIN, len(test_data))} test graphs...")
    feat_by_label, depth_node_imp = run_gnnexplainer(gcn, test_data, input_dim, device)

    group_scores = plot_feature_groups(feat_by_label, OUT_DIR)
    plot_node_depth_importance(depth_node_imp, OUT_DIR)

    print("\nGCN feature group importance (mean across classes, high = relied upon more):")
    for g, s in sorted(group_scores.items(), key=lambda x: -x[1]):
        print(f"  {g:<20} {s:.5f}")

    print("\nGCN feature group importance split by class:")
    for label in ["Rumour", "Non-Rumour"]:
        if label not in feat_by_label:
            continue
        mean_mask = np.mean(feat_by_label[label], axis=0)
        print(f"  {label}:")
        for g, idx in FEATURE_GROUPS.items():
            print(f"    {g:<20} {float(mean_mask[idx].mean()):.5f}")

    # Mann-Whitney U per feature group: Rumour vs Non-Rumour importance
    from scipy.stats import mannwhitneyu as _mwu, rankdata as _rd
    def _rank_biserial(u, n1, n2):
        return 1 - (2 * u) / (n1 * n2)

    if "Rumour" in feat_by_label and "Non-Rumour" in feat_by_label:
        r_arr  = np.array(feat_by_label["Rumour"])    # [n_r, 390]
        nr_arr = np.array(feat_by_label["Non-Rumour"])# [n_nr, 390]
        print("\nGCN feature group importance: Mann-Whitney U (Rumour vs Non-Rumour)")
        print(f"  {'Group':<20}  {'R mean':>7}  {'NR mean':>7}  {'p-value':>8}  {'r':>6}  Sig")
        print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  ---")
        for g, idx in FEATURE_GROUPS.items():
            r_vals  = r_arr[:, idx].mean(axis=1)
            nr_vals = nr_arr[:, idx].mean(axis=1)
            u_stat, p = _mwu(r_vals, nr_vals, alternative="two-sided")
            r_val = _rank_biserial(u_stat, len(r_vals), len(nr_vals))
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {g:<20}  {r_vals.mean():>7.4f}  {nr_vals.mean():>7.4f}  {p:>8.4f}  {r_val:>6.3f}  {sig}")

    # ── GAT Attention ─────────────────────────────────────────────────────────
    print(f"\n[2/2] Extracting GAT attention from all {len(test_data)} test graphs...")
    attn_df = build_attention_df(gat, test_data, device)

    plot_attention_by_depth(attn_df, OUT_DIR)
    ent_df, mw_p = plot_attention_entropy(attn_df, OUT_DIR)

    print("\nGAT layer-2 mean attention received by depth:")
    depth_summary = (
        attn_df[(attn_df["layer"] == 2) & (attn_df["depth"] >= 0) & (attn_df["depth"] <= 6)]
        .groupby(["label", "depth"])["attn"]
        .mean()
        .unstack("depth")
        .round(4)
    )
    print(depth_summary.to_string())

    # Mann-Whitney U per depth: do Rumour and Non-Rumour attention differ at each depth?
    from scipy.stats import mannwhitneyu as _mwu2
    layer2_depth = attn_df[(attn_df["layer"] == 2) & (attn_df["depth"] >= 0) & (attn_df["depth"] <= 6)]
    print("\nGAT layer-2 attention per depth: Mann-Whitney U (Rumour vs Non-Rumour)")
    print(f"  {'Depth':>5}  {'R mean':>7}  {'NR mean':>7}  {'p-value':>8}  {'r':>6}  Sig")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  ---")
    for depth in sorted(layer2_depth["depth"].unique()):
        r_vals  = layer2_depth[(layer2_depth["label"] == "Rumour")     & (layer2_depth["depth"] == depth)]["attn"].dropna().values
        nr_vals = layer2_depth[(layer2_depth["label"] == "Non-Rumour") & (layer2_depth["depth"] == depth)]["attn"].dropna().values
        if len(r_vals) < 2 or len(nr_vals) < 2:
            continue
        u_stat, p = _mwu2(r_vals, nr_vals, alternative="two-sided")
        r_val = 1 - (2 * u_stat) / (len(r_vals) * len(nr_vals))
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {depth:>5}  {r_vals.mean():>7.4f}  {nr_vals.mean():>7.4f}  {p:>8.4f}  {r_val:>6.3f}  {sig}")

    print(f"\nGAT attention entropy (layer 2):")
    print(ent_df.groupby("label")["entropy"].agg(["mean", "median", "std"]).round(4).to_string())
    print(f"  Mann-Whitney p = {mw_p:.4f}  "
          f"({'significant difference between classes' if mw_p < 0.05 else 'no significant difference'})")

    # Save the per-depth attention summary as CSV for further use
    (
        attn_df[attn_df["depth"] >= 0]
        .groupby(["layer", "label", "depth"])["attn"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_csv(OUT_DIR / "gat_attention_summary.csv", index=False)
    )

    print(f"\nAll outputs written to: {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
