import argparse
import yaml
import datetime
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import load_data, filter_features
from src.models.trainer import train_rf, train_nn

def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiments for Misinformation GNN")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    dataset = load_data()
    
    # Filter features dynamically from YAML
    features_config = config_data.get("features", {})
    dataset, input_dim = filter_features(dataset, features_config)
    
    print(f"Loaded dataset containing {len(dataset)} graphs. Active feature dimension: {input_dim}")
    
    # Identify unique events
    events = list(set([d.event for d in dataset]))
    print(f"Discovered {len(events)} events for Leave-One-Out cross validation: {events}")
    
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