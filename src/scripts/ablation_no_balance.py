"""
Ablation: effect of per-event majority-class undersampling.

Trains each full-feature model twice on the 4 training events:
  - WITH balancing    (baseline, matches final_eval.py)
  - WITHOUT balancing (this ablation)

Evaluates on the same held-out test events (charliehebdo + sydneysiege).
Prints a side-by-side comparison table.
"""
import json
import sys
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import load_data, filter_features, add_graph_features
from src.models.trainer import train_nn

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
ALWAYS_EXCLUDE = {
    "prince-toronto-all-rnr-threads",
    "ebola-essien-all-rnr-threads",
    "ferguson-all-rnr-threads",
}
MIN_NODES = 3
CONFIG_FILE = "configs/benchmark_full.yml"
FULL_KEYS = ["gnn_full", "gat_full", "gin_full", "mlp_full"]


def load_dataset():
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    features_cfg = cfg.get("features", {})

    dataset = load_data()
    dataset, input_dim = filter_features(dataset, features_cfg)

    use_root  = features_cfg.get("use_root", False)
    use_depth = features_cfg.get("use_depth", False)
    if use_root or use_depth:
        dataset, input_dim = add_graph_features(dataset, use_root=use_root, use_depth=use_depth)

    dataset = [d for d in dataset if d.event not in ALWAYS_EXCLUDE and d.num_nodes >= MIN_NODES]
    return dataset, input_dim


def run(key, hparams, model_type, dataset, input_dim, balance):
    train_data = [d for d in dataset if d.event in TRAIN_EVENTS]
    all_true, all_pred = [], []
    for test_event in sorted(TEST_EVENTS):
        test_data = [d for d in dataset if d.event == test_event]
        y_true, y_pred = train_nn(
            hparams, model_type, train_data, test_data, input_dim,
            fold_name=test_event, balance=balance,
        )
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    return all_true, all_pred


def metrics(y_true, y_pred):
    return {
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Acc":       round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def main():
    best_path = Path("outputs/hp_search/best_configs.json")
    with open(best_path) as f:
        best_configs = json.load(f)

    print("Loading dataset (full features)...")
    dataset, input_dim = load_dataset()
    print(f"  {len(dataset)} graphs  |  input_dim={input_dim}\n")

    rows = []
    for key in FULL_KEYS:
        entry = best_configs[key]
        hparams = entry["hparams"]
        model_type = entry["model_type"]

        for balance in [True, False]:
            label = "balanced" if balance else "no_balance"
            print(f"[{key}  {label}]")
            y_true, y_pred = run(key, hparams, model_type, dataset, input_dim, balance)
            m = metrics(y_true, y_pred)
            print(f"  F1={m['F1']}  Acc={m['Acc']}  Prec={m['Precision']}  Rec={m['Recall']}")
            rows.append({"model": key, "balance": label, **m})

    print("\n\n" + "="*70)
    print("ABLATION: WITH vs WITHOUT PER-EVENT CLASS BALANCING")
    print("="*70)
    print(f"  {'Model':<12}  {'Balance':>11}  {'F1':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*12}  {'-'*11}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in rows:
        print(f"  {r['model']:<12}  {r['balance']:>11}  {r['F1']:>6}  {r['Acc']:>6}  {r['Precision']:>6}  {r['Recall']:>6}")

    print("\nDelta (balanced - no_balance):")
    print(f"  {'Model':<12}  {'dF1':>6}  {'dAcc':>6}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}")
    for i in range(0, len(rows), 2):
        bal = rows[i]
        nob = rows[i+1]
        print(f"  {bal['model']:<12}  {bal['F1']-nob['F1']:>+6.4f}  {bal['Acc']-nob['Acc']:>+6.4f}")


if __name__ == "__main__":
    main()
