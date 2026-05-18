"""
Final held-out evaluation (v2 — stratified 10% test split).

Loads the persisted test split from outputs/hp_search/test_split.json,
trains each best model on the full 90% train_val pool, and evaluates on
the held-out 10%.  Produces bootstrap CIs and McNemar comparisons.

Usage:
    python src/scripts/final_eval_v2.py
"""
import json
import pickle
import sys
import datetime
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import (load_data, filter_features, add_graph_features,
                               compute_edge_features, create_stratified_test_split)
from src.models.trainer import train_nn

SPLIT_FILE   = "outputs/hp_search/test_split.json"
BEST_CONFIGS = "outputs/hp_search/best_configs.json"


def load_dataset_for_config(config_file: str, split_file: str):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    features_cfg = cfg.get("features", {})

    dataset = load_data()
    edge_dim = 0
    if features_cfg.get("use_edge_features", False):
        compute_edge_features(dataset)
        edge_dim = 2

    dataset, input_dim = filter_features(dataset, features_cfg)

    use_root  = features_cfg.get("use_root", False)
    use_depth = features_cfg.get("use_depth", False)
    if use_root or use_depth:
        dataset, input_dim = add_graph_features(dataset, use_root=use_root, use_depth=use_depth)

    # Apply preprocessing exclusions
    excluded = set(cfg.get("preprocessing", {}).get("excluded_events", []))
    min_nodes = cfg.get("preprocessing", {}).get("min_nodes", 3)
    dataset = [d for d in dataset if d.event not in excluded and d.num_nodes >= min_nodes]

    train_val, test = create_stratified_test_split(dataset, save_path=split_file)
    return train_val, test, input_dim, edge_dim


def bootstrap_f1_ci(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt, yp = np.array(y_true), np.array(y_pred)
    boot = [f1_score(yt[idx := rng.integers(0, n, n)], yp[idx], zero_division=0)
            for _ in range(n_boot)]
    return np.percentile(boot, [2.5, 97.5])


def main():
    with open(BEST_CONFIGS) as f:
        best_configs = json.load(f)

    from collections import defaultdict
    by_config = defaultdict(list)
    for key, entry in best_configs.items():
        by_config[entry["config_file"]].append((key, entry))

    all_rows  = []
    all_preds = {}

    for config_file, entries in sorted(by_config.items()):
        feature_label = ("full" if "full" in config_file
                         else "struct" if "structural" in config_file else "text")
        print(f"\n{'='*60}\nLoading dataset: {config_file}  ({feature_label})\n{'='*60}")
        train_val, test, input_dim, edge_dim = load_dataset_for_config(config_file, SPLIT_FILE)
        print(f"  train_val={len(train_val)}  test={len(test)}  "
              f"input_dim={input_dim}  edge_dim={edge_dim}")

        for key, entry in entries:
            hparams    = entry["hparams"]
            model_type = entry["model_type"]
            print(f"\n--- {key} ({model_type}) ---")
            print(f"  Train on {len(train_val)} graphs -> test on {len(test)} graphs")

            torch.manual_seed(42)
            y_true, y_pred = train_nn(
                hparams, model_type, train_val, test, input_dim,
                edge_dim=edge_dim, fold_name="held_out_test",
            )
            f1   = f1_score(y_true, y_pred, zero_division=0)
            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            print(f"  F1={f1:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

            all_rows.append({
                "key": key, "model_type": model_type,
                "config_file": entry["config_file"],
                "F1": round(f1, 4), "Accuracy": round(acc, 4),
                "Precision": round(prec, 4), "Recall": round(rec, 4),
            })
            all_preds[key] = {"y_true": y_true, "y_pred": y_pred}

    # ── Save predictions ──────────────────────────────────────────────────────
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / "final_eval_v2_predictions.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(all_preds, f)
    print(f"\nPredictions saved to: {pred_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows).sort_values("F1", ascending=False)
    print("\n\n" + "="*70)
    print("FINAL HELD-OUT EVALUATION RESULTS (stratified 10% test split)")
    print("="*70)
    print(df[["key", "model_type", "F1", "Accuracy", "Precision", "Recall"]].to_string(index=False))

    # ── Bootstrap 95% CI ──────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("BOOTSTRAP 95% CI FOR F1 (n_boot=2000)")
    print("="*70)
    ci_rows = []
    for key in sorted(all_preds):
        p = all_preds[key]
        point = f1_score(p["y_true"], p["y_pred"], zero_division=0)
        lo, hi = bootstrap_f1_ci(p["y_true"], p["y_pred"])
        print(f"  {key:<15}  F1={point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        ci_rows.append({"key": key, "F1": round(point, 4),
                        "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)})

    # ── McNemar's test ────────────────────────────────────────────────────────
    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

        def mcnemar_pair(pa, pb):
            yt_a, yp_a = np.array(pa["y_true"]), np.array(pa["y_pred"])
            yt_b, yp_b = np.array(pb["y_true"]), np.array(pb["y_pred"])
            a_right = yp_a == yt_a
            b_right = yp_b == yt_b
            tb = int(np.sum( a_right &  b_right))
            tf = int(np.sum( a_right & ~b_right))
            ft = int(np.sum(~a_right &  b_right))
            ff = int(np.sum(~a_right & ~b_right))
            return _mcnemar([[tb, tf], [ft, ff]], exact=False, correction=True).pvalue

        comparisons = [
            ("gat_text", "gnn_text",  "attention vs GCN (text)"),
            ("gnn_text", "mlp_text",  "message passing vs MLP (text)"),
            ("gat_full", "gnn_full",  "attention vs GCN (full)"),
            ("gnn_full", "mlp_full",  "message passing vs MLP (full)"),
            ("gat_full", "gat_text",  "structural features in GAT"),
            ("gnn_full", "gnn_text",  "structural features in GNN"),
            ("gat_text", "gin_text",  "GAT vs GIN (text)"),
            ("gnn_text", "gin_text",  "GNN vs GIN (text)"),
        ]

        print("\n\n" + "="*70)
        print("McNEMAR'S TEST (chi-squared with continuity correction)")
        print("="*70)
        print(f"  {'Comparison':<40}  {'p-value':>8}  Sig")
        for ka, kb, label in comparisons:
            if ka not in all_preds or kb not in all_preds:
                print(f"  {label:<40}  MISSING")
                continue
            pa, pb = all_preds[ka], all_preds[kb]
            if len(pa["y_true"]) != len(pb["y_true"]):
                print(f"  {label:<40}  SIZE MISMATCH")
                continue
            p = mcnemar_pair(pa, pb)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            f1_a = f1_score(pa["y_true"], pa["y_pred"], zero_division=0)
            f1_b = f1_score(pb["y_true"], pb["y_pred"], zero_division=0)
            direction = f"{ka}>{kb}" if f1_a > f1_b else f"{kb}>{ka}"
            print(f"  {label:<40}  {p:>8.4f}  {sig}  ({direction})")
    except ImportError:
        print("\nstatsmodels not installed — skipping McNemar's test")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"final_eval_v2_{ts}.csv"
    ci_path   = out_dir / f"final_eval_v2_ci_{ts}.csv"
    df.to_csv(out_path, index=False)
    pd.DataFrame(ci_rows).to_csv(ci_path, index=False)
    print(f"\nFull results saved to: {out_path}")
    print(f"Bootstrap CIs saved to: {ci_path}")


if __name__ == "__main__":
    main()
