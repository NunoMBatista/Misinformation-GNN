import argparse
import yaml
import datetime
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import load_data, filter_features, add_graph_features
from src.models.trainer import train_rf, train_nn

def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiments for Misinformation GNN")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def _event_stats(dataset):
    """Returns an OrderedDict: event -> {total, rumour, non_rumour}."""
    stats = defaultdict(lambda: {"total": 0, "rumour": 0, "non_rumour": 0})
    for d in dataset:
        label = int(d.y.item())
        stats[d.event]["total"] += 1
        if label == 1:
            stats[d.event]["rumour"] += 1
        else:
            stats[d.event]["non_rumour"] += 1
    return dict(sorted(stats.items()))


def _print_stats(dataset, header):
    stats = _event_stats(dataset)
    print(f"\n{header}  ({len(dataset)} graphs total)")
    print(f"  {'Event':<45} {'Total':>6}  {'R':>5}  {'NR':>5}")
    print(f"  {'-'*45} {'-'*6}  {'-'*5}  {'-'*5}")
    for event, s in stats.items():
        print(f"  {event:<45} {s['total']:>6}  {s['rumour']:>5}  {s['non_rumour']:>5}")


def preprocess_dataset(dataset, preprocessing_config):
    excluded = set(preprocessing_config.get("excluded_events", []))
    min_nodes = preprocessing_config.get("min_nodes", 3)

    _print_stats(dataset, "Before preprocessing:")

    removed_event = [d for d in dataset if d.event in excluded]
    removed_nodes = [d for d in dataset if d.event not in excluded and d.num_nodes < min_nodes]

    dataset = [d for d in dataset if d.event not in excluded and d.num_nodes >= min_nodes]

    if excluded:
        print(f"\n  Excluded events ({len(removed_event)} graphs removed): {sorted(excluded)}")
    print(f"  Removed {len(removed_nodes)} graphs with fewer than {min_nodes} nodes.")

    _print_stats(dataset, "After preprocessing:")
    print()
    return dataset


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    dataset = load_data()

    # Filter features dynamically from YAML
    features_config = config_data.get("features", {})
    dataset, input_dim = filter_features(dataset, features_config)

    # Append positional features (is_root, depth) if requested.
    # These are computed at load time — the .pt file is not modified.
    use_root  = features_config.get("use_root", False)
    use_depth = features_config.get("use_depth", False)
    if use_root or use_depth:
        dataset, input_dim = add_graph_features(dataset, use_root=use_root, use_depth=use_depth)
        print(f"Added positional features (is_root={use_root}, depth={use_depth}) → dim now {input_dim}")

    # Pre-processing: drop excluded events and tiny graphs
    preprocessing_config = config_data.get("preprocessing", {})
    dataset = preprocess_dataset(dataset, preprocessing_config)

    print(f"Dataset ready: {len(dataset)} graphs | Feature dimension: {input_dim}")

    # Identify unique events
    events = sorted(set(d.event for d in dataset))
    print(f"Events for LOEO cross-validation ({len(events)}): {events}")
    
    results = []
    
    for exp_name, config in config_data.get('experiments', {}).items():
        print(f"\n--- Running Experiment: {exp_name} ---")
        model_type = config.get("model_type")
        
        all_true = []
        all_pred = []
        
        pbar = tqdm(events, desc=f"CV: {exp_name}")
        for test_event in pbar:
            # Leave-One-Event-Out Split
            train_data = [d for d in dataset if d.event != test_event]
            test_data = [d for d in dataset if d.event == test_event]
            
            pbar.write(f"  Fold -> Test Event: {test_event} | Train: {len(train_data)} | Test: {len(test_data)}")
            
            if model_type == "rf":
                y_true, y_pred = train_rf(config, train_data, test_data)
            elif model_type in ["mlp", "gnn", "gat"]:
                y_true, y_pred = train_nn(config, model_type, train_data, test_data, input_dim)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
                
            # Calculate fold metrics
            f_acc = accuracy_score(y_true, y_pred)
            f_prec = precision_score(y_true, y_pred, zero_division=0)
            f_rec = recall_score(y_true, y_pred, zero_division=0)
            f_f1 = f1_score(y_true, y_pred, zero_division=0)
            
            pbar.write(f"    [Fold Results] Acc: {f_acc:.4f} | Prec: {f_prec:.4f} | Rec: {f_rec:.4f} | F1: {f_f1:.4f}")
            
            results.append({
                "Experiment": exp_name,
                "Model": model_type,
                "Fold": test_event,
                "Accuracy": f_acc,
                "Precision": f_prec,
                "Recall": f_rec,
                "F1-Score": f_f1
            })
            
            all_true.extend(y_true)
            all_pred.extend(y_pred)
            
        # Compute global metrics across all folds
        acc = accuracy_score(all_true, all_pred)
        prec = precision_score(all_true, all_pred, zero_division=0)
        rec = recall_score(all_true, all_pred, zero_division=0)
        f1 = f1_score(all_true, all_pred, zero_division=0)
        
        results.append({
            "Experiment": exp_name,
            "Model": model_type,
            "Fold": "GLOBAL",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    print("\n================ EXPERIMENT RESULTS ================")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"experiment_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()