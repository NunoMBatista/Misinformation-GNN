"""
Final held-out evaluation using best hyperparameters from HP search.

Reads outputs/hp_search/best_configs.json, trains each model on the 4
training events, and evaluates on the 2 held-out test events:
  - charliehebdo-all-rnr-threads
  - sydneysiege-all-rnr-threads

Saves a timestamped CSV to outputs/final_eval_<timestamp>.csv and prints
a summary table.
"""
import json
import pickle
import sys
import datetime
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import load_data, filter_features, add_graph_features, compute_edge_features
from src.models.trainer import train_nn

# ── Constants ────────────────────────────────────────────────────────────────
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
# Events to exclude entirely (single-class or too small to be useful)
ALWAYS_EXCLUDE = {
    "prince-toronto-all-rnr-threads",
    "ebola-essien-all-rnr-threads",
    "ferguson-all-rnr-threads",
}
MIN_NODES = 3


def load_dataset_for_config(config_file: str):
    """Load + feature-filter the raw dataset for a given config YAML.
    Returns (dataset, input_dim).  Does NOT drop test events."""
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    dataset = load_data()
    features_cfg = cfg.get("features", {})
    edge_dim = 0
    if features_cfg.get("use_edge_features", False):
        compute_edge_features(dataset)
        edge_dim = 2

    dataset, input_dim = filter_features(dataset, features_cfg)

    use_root  = features_cfg.get("use_root", False)
    use_depth = features_cfg.get("use_depth", False)
    if use_root or use_depth:
        dataset, input_dim = add_graph_features(dataset, use_root=use_root, use_depth=use_depth)

    # Drop always-excluded events and tiny graphs; KEEP train + test events
    dataset = [
        d for d in dataset
        if d.event not in ALWAYS_EXCLUDE and d.num_nodes >= MIN_NODES
    ]
    return dataset, input_dim, edge_dim


def run_eval(key: str, entry: dict, dataset, input_dim: int, edge_dim: int):
    """Train on TRAIN_EVENTS, evaluate on each TEST_EVENT separately + combined.
    Returns (rows, predictions) where predictions = {'y_true': [...], 'y_pred': [...]}
    over the combined test set (used for McNemar's and bootstrap CI).
    """
    hparams    = entry["hparams"]
    model_type = entry["model_type"]

    train_data = [d for d in dataset if d.event in TRAIN_EVENTS]
    rows = []

    all_true_combined, all_pred_combined = [], []

    for test_event in sorted(TEST_EVENTS):
        test_data = [d for d in dataset if d.event == test_event]
        if not test_data:
            print(f"  [WARN] No graphs for {test_event} -- skipping")
            continue

        print(f"  [{key}] Train: {len(train_data)} graphs -> Test on {test_event} ({len(test_data)} graphs)")
        y_true, y_pred = train_nn(
            hparams, model_type, train_data, test_data, input_dim,
            edge_dim=edge_dim, fold_name=test_event,
        )
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        print(f"    -> F1={f1:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

        rows.append({
            "key": key,
            "model_type": model_type,
            "config_file": entry["config_file"],
            "test_event": test_event.replace("-all-rnr-threads", ""),
            "F1": round(f1, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
        })
        all_true_combined.extend(y_true)
        all_pred_combined.extend(y_pred)

    # Combined across both test events
    if all_true_combined:
        f1   = f1_score(all_true_combined, all_pred_combined, zero_division=0)
        acc  = accuracy_score(all_true_combined, all_pred_combined)
        prec = precision_score(all_true_combined, all_pred_combined, zero_division=0)
        rec  = recall_score(all_true_combined, all_pred_combined, zero_division=0)
        rows.append({
            "key": key,
            "model_type": model_type,
            "config_file": entry["config_file"],
            "test_event": "COMBINED",
            "F1": round(f1, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
        })
        print(f"  [{key}] COMBINED -> F1={f1:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    predictions = {"y_true": all_true_combined, "y_pred": all_pred_combined}
    return rows, predictions


def main():
    best_path = Path("outputs/hp_search/best_configs.json")
    with open(best_path) as f:
        best_configs = json.load(f)

    print(f"Loaded {len(best_configs)} best configs from {best_path}")

    # Group by config_file so we only load/filter the dataset once per config
    by_config: dict[str, list] = defaultdict(list)
    for key, entry in best_configs.items():
        by_config[entry["config_file"]].append((key, entry))

    all_rows = []
    all_preds = {}   # key -> {'y_true': [...], 'y_pred': [...]}

    for config_file, entries in sorted(by_config.items()):
        feature_label = (
            "full" if "full" in config_file
            else "struct" if "structural" in config_file
            else "text"
        )
        print(f"\n{'='*60}")
        print(f"Loading dataset: {config_file}  ({feature_label})")
        print(f"{'='*60}")
        dataset, input_dim, edge_dim = load_dataset_for_config(config_file)

        counts = defaultdict(int)
        for d in dataset:
            counts[d.event] += 1
        print(f"Dataset: {len(dataset)} graphs  |  input_dim={input_dim}  |  edge_dim={edge_dim}")
        print("  Events present:")
        for ev, n in sorted(counts.items()):
            tag = "TRAIN" if ev in TRAIN_EVENTS else ("TEST" if ev in TEST_EVENTS else "?")
            print(f"    [{tag}]  {ev}: {n}")

        for key, entry in entries:
            print(f"\n--- {key} ({entry['model_type']}) ---")
            rows, preds = run_eval(key, entry, dataset, input_dim, edge_dim)
            all_rows.extend(rows)
            all_preds[key] = preds

    # ── Save predictions pickle ────────────────────────────────────────────
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / "final_eval_predictions.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(all_preds, f)
    print(f"\nPredictions saved to: {pred_path}")

    # ── Summary table ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    print("\n\n" + "="*70)
    print("FINAL HELD-OUT EVALUATION RESULTS")
    print("="*70)

    combined = df[df["test_event"] == "COMBINED"].copy()
    combined = combined.sort_values("F1", ascending=False)
    print("\nCOMBINED (charliehebdo + sydneysiege):")
    print(combined[["key", "model_type", "config_file", "F1", "Accuracy", "Precision", "Recall"]].to_string(index=False))

    print("\nPER-EVENT breakdown:")
    per_event = df[df["test_event"] != "COMBINED"].pivot_table(
        index=["key", "model_type"], columns="test_event", values="F1"
    )
    print(per_event.to_string())

    # ── Bootstrap 95% CI for F1 ────────────────────────────────────────────
    def bootstrap_f1_ci(y_true, y_pred, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(y_true)
        yt, yp = np.array(y_true), np.array(y_pred)
        boot = [f1_score(yt[idx := rng.integers(0, n, n)], yp[idx], zero_division=0)
                for _ in range(n_boot)]
        return np.percentile(boot, [2.5, 97.5])

    print("\n\n" + "="*70)
    print("BOOTSTRAP 95% CI FOR F1 (combined test set, n_boot=2000)")
    print("="*70)
    ci_rows = []
    for key in sorted(all_preds):
        p = all_preds[key]
        if not p["y_true"]:
            continue
        point = f1_score(p["y_true"], p["y_pred"], zero_division=0)
        lo, hi = bootstrap_f1_ci(p["y_true"], p["y_pred"])
        print(f"  {key:<15}  F1={point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        ci_rows.append({"key": key, "F1": round(point, 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)})

    # ── McNemar's test ────────────────────────────────────────────────────
    def mcnemar_pair(pa, pb):
        """Two-sided McNemar's test between models a and b."""
        yt_a, yp_a = np.array(pa["y_true"]), np.array(pa["y_pred"])
        yt_b, yp_b = np.array(pb["y_true"]), np.array(pb["y_pred"])
        # y_true must be the same (same test set, same order)
        a_right = yp_a == yt_a
        b_right = yp_b == yt_b
        tb = int(np.sum( a_right &  b_right))
        tf = int(np.sum( a_right & ~b_right))
        ft = int(np.sum(~a_right &  b_right))
        ff = int(np.sum(~a_right & ~b_right))
        result = mcnemar([[tb, tf], [ft, ff]], exact=False, correction=True)
        return result.pvalue

    # Key comparisons (same config_file required so y_true aligns)
    comparisons = [
        ("gat_text",  "gnn_text",  "does attention help over GCN? (text)"),
        ("gnn_text",  "mlp_text",  "does message passing help? (text)"),
        ("gat_text",  "mlp_text",  "does GAT beat flat MLP? (text)"),
        ("gat_full",  "gnn_full",  "does attention help? (full)"),
        ("gnn_full",  "mlp_full",  "does message passing help? (full)"),
        ("gat_full",  "gat_text",  "does adding structural features help? (gat)"),
        ("gnn_full",  "gnn_text",  "does adding structural features help? (gnn)"),
        ("gat_text",  "gin_text",  "GAT vs GIN (text)"),
        ("gnn_text",  "gin_text",  "GNN vs GIN (text)"),
    ]

    print("\n\n" + "="*70)
    print("McNEMAR'S TEST (chi-squared with continuity correction)")
    print("="*70)
    print(f"  {'Comparison':<45}  {'p-value':>8}  Sig")
    print(f"  {'-'*45}  {'-'*8}  ---")
    for ka, kb, label in comparisons:
        if ka not in all_preds or kb not in all_preds:
            print(f"  {label:<45}  MISSING")
            continue
        pa, pb = all_preds[ka], all_preds[kb]
        if len(pa["y_true"]) != len(pb["y_true"]):
            print(f"  {label:<45}  MISMATCH (different test set sizes)")
            continue
        p = mcnemar_pair(pa, pb)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        f1_a = f1_score(pa["y_true"], pa["y_pred"], zero_division=0)
        f1_b = f1_score(pb["y_true"], pb["y_pred"], zero_division=0)
        direction = f"{ka}>{kb}" if f1_a > f1_b else f"{kb}>{ka}"
        print(f"  {label:<45}  {p:>8.4f}  {sig}  ({direction})")

    # ── Save CSV ───────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"final_eval_{ts}.csv"
    df.to_csv(out_path, index=False)

    ci_path = out_dir / f"final_eval_ci_{ts}.csv"
    pd.DataFrame(ci_rows).to_csv(ci_path, index=False)

    print(f"\nFull results saved to: {out_path}")
    print(f"Bootstrap CIs saved to: {ci_path}")


if __name__ == "__main__":
    main()
