"""
Single-trial HP evaluator for LLM-guided hyperparameter search.

Claude Code (the CLI) acts as the search loop: it reads the F1 output after
each trial, reasons about what to try next, and calls this script again with
a new --hparams JSON blob. The .jsonl log accumulates across calls so a
session can be resumed at any time.

Usage:
    # Warmup — sample a random config from the search space and run it:
    python src/scripts/hp_search.py --model gin --config configs/benchmark_full.yml \\
        --random --session my_session --trial 1

    # Guided trial — Claude supplies the config:
    python src/scripts/hp_search.py --model gin --config configs/benchmark_full.yml \\
        --hparams '{"learning_rate":0.0008,"dropout":0.35,"hidden_dims":[128,64],...}' \\
        --session my_session --trial 5

    # Inspect the resolved search space for a model+config combination:
    python src/scripts/hp_search.py --model gin --config configs/benchmark_full.yml --show-space

Output (stdout, read by Claude Code):
    Trial 3 | gin | F1=0.7234 | Acc=0.7456 | Prec=0.7123 | Rec=0.7350
    Hparams: {...}
    Saved: outputs/hp_search/gin_my_session.jsonl

Search parameters (all models unless noted):
    hidden_dims     categorical — choices adapt to input_dim (small for structural-only)
    dropout         continuous  — uniform [0.10, 0.60]
    learning_rate   continuous  — log-uniform [5e-5, 5e-3]
    weight_decay    continuous  — log-uniform [1e-5, 1e-3]
    focal_gamma     continuous  — uniform [1.0, 4.0]  (use_focal_loss always True)
    edge_direction  categorical — [original, bidirectional]  (not MLP)
    heads           categorical — [1, 2, 4]  (GAT only)
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
except ImportError:
    wandb = None

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import (load_data, filter_features, add_graph_features,
                               compute_edge_features, create_stratified_test_split,
                               stratified_kfold)
from src.models.trainer import train_nn
from src.scripts.run_experiments import preprocess_dataset
import yaml


# ---------------------------------------------------------------------------
# Search space definitions
# hidden_dims has two choice lists: "large" for full/text configs (386-392 dim
# input) and "small" for structural-only (8 dim input). resolve_space() picks
# the right one based on input_dim.
# ---------------------------------------------------------------------------
_SPACE_BASE = {
    "hidden_dims": {
        "type": "categorical",
        "choices_large": [[64, 32], [128, 64], [256, 128], [128, 64, 32], [256, 128, 64]],
        "choices_small": [[8, 4], [16, 8], [32, 16], [32, 16, 8]],
    },
    "dropout":       {"type": "uniform",    "low": 0.10, "high": 0.60},
    "learning_rate": {"type": "loguniform", "low": 5e-5, "high": 5e-3},
    "weight_decay":  {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "focal_gamma":   {"type": "uniform",    "low": 1.0,  "high": 4.0},
    "edge_direction":{"type": "categorical", "choices": ["original", "bidirectional"]},
}

SEARCH_SPACES = {
    # GIN: sum aggregation is WL-expressive; learning rate is the most
    # impactful knob since sum can blow up gradients with wide networks.
    "gin": _SPACE_BASE,

    # ImprovedGNN: GCN mean aggregation; generally more stable than GIN,
    # so the LR range can be explored more freely.
    "improved_gnn": _SPACE_BASE,

    # ImprovedGAT: attention heads multiply effective width — keep that in
    # mind when choosing hidden_dims (heads=4 + [256,128] is very wide).
    "improved_gat": {
        **_SPACE_BASE,
        "heads": {"type": "categorical", "choices": [1, 2, 4]},
    },

    # MLP: no message passing, no edge_direction. Benefits from wider
    # hidden layers since it can't leverage graph structure.
    "mlp": {k: v for k, v in _SPACE_BASE.items() if k != "edge_direction"},
}

# Fixed params not searched (use_focal_loss always True; epochs fixed per model)
_FIXED_PARAMS = {
    "gin":          {"use_focal_loss": True, "epochs": 150},
    "improved_gnn": {"use_focal_loss": True, "epochs": 150},
    "improved_gat": {"use_focal_loss": True, "epochs": 150},
    "mlp":          {"use_focal_loss": True, "epochs": 100},
}


def resolve_space(model_name: str, input_dim: int) -> dict:
    """Return the fully resolved search space (no choices_large/small split)."""
    space = copy.deepcopy(SEARCH_SPACES[model_name])
    use_small = input_dim <= 16
    hd = space["hidden_dims"]
    hd["choices"] = hd.pop("choices_small") if use_small else hd.pop("choices_large")
    if "choices_large" in hd:
        del hd["choices_large"]
    if "choices_small" in hd:
        del hd["choices_small"]
    return space


def _sample_random(space: dict) -> dict:
    config = {}
    for key, spec in space.items():
        if spec["type"] == "categorical":
            config[key] = random.choice(spec["choices"])
        elif spec["type"] == "uniform":
            config[key] = round(random.uniform(spec["low"], spec["high"]), 4)
        elif spec["type"] == "loguniform":
            config[key] = round(math.exp(random.uniform(math.log(spec["low"]), math.log(spec["high"]))), 6)
    return config


def _clamp(config: dict, space: dict) -> dict:
    """Snap any out-of-range values Claude may return to the nearest valid one."""
    for key, spec in space.items():
        if key not in config:
            config[key] = _sample_random({key: spec})[key]
            continue
        if spec["type"] == "categorical":
            if config[key] not in spec["choices"]:
                try:
                    config[key] = min(spec["choices"], key=lambda x: abs(x - config[key]))
                except TypeError:
                    config[key] = spec["choices"][0]
        elif spec["type"] in ("uniform", "loguniform"):
            config[key] = float(max(spec["low"], min(spec["high"], config[key])))
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run one HP trial and print results")
    parser.add_argument("--model",      type=str, required=True, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--config",     type=str, required=True, help="Base YAML config (preprocessing + feature flags)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hparams",     type=str, help="JSON string of hyperparameters to evaluate")
    group.add_argument("--random",      action="store_true", help="Sample a random config from the search space")
    group.add_argument("--show-space",  action="store_true", help="Print resolved search space and exit")
    parser.add_argument("--session",    type=str, default=None, help="Session ID for WandB grouping + log file naming")
    parser.add_argument("--trial",      type=int, default=None, help="Trial number (logged to WandB and jsonl)")
    parser.add_argument("--no-wandb",   action="store_true",    help="Disable WandB logging")
    parser.add_argument("--cv-folds",   type=int, default=5,    help="Number of CV folds (default 5)")
    parser.add_argument("--split-file", type=str,
                        default="outputs/hp_search/test_split.json",
                        help="Path to the persisted stratified test split JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    # ---- Dataset setup ----
    dataset = load_data()
    features_config = config_data.get("features", {})
    edge_dim = 0
    if features_config.get("use_edge_features", False):
        compute_edge_features(dataset)
        edge_dim = 2
    dataset, input_dim = filter_features(dataset, features_config)
    if features_config.get("use_root") or features_config.get("use_depth"):
        dataset, input_dim = add_graph_features(
            dataset,
            use_root=features_config.get("use_root", False),
            use_depth=features_config.get("use_depth", False),
        )
    dataset = preprocess_dataset(dataset, config_data.get("preprocessing", {}))

    # Strip out the held-out 10% test set — HP search never sees it
    train_val, _ = create_stratified_test_split(dataset, save_path=args.split_file)

    space = resolve_space(args.model, input_dim)

    # ---- --show-space: just print and exit ----
    if args.show_space:
        print(f"\nSearch space for {args.model} | input_dim={input_dim}")
        print(json.dumps(space, indent=2))
        print(f"\nFixed: {json.dumps(_FIXED_PARAMS[args.model])}")
        return

    # ---- Build hparams ----
    if args.random:
        hparams = _sample_random(space)
    else:
        hparams = _clamp(json.loads(args.hparams), space)

    # Merge fixed params (epochs, use_focal_loss) — searchable params take priority
    hparams = {**_FIXED_PARAMS[args.model], **hparams}

    session = args.session or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    use_wandb = (wandb is not None) and (not args.no_wandb) and bool(os.getenv("WANDB_API_KEY"))

    # ---- WandB init ----
    if use_wandb:
        trial_label = f"trial_{args.trial:03d}" if args.trial else "trial"
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "misinformation-gnn"),
            name=trial_label,
            group=f"hp_{args.model}_{session}",
            config={**hparams, "model": args.model, "trial": args.trial, "input_dim": input_dim},
            reinit=True,
        )

    # ---- 5-fold CV on train_val (test set never touched) ----
    all_true_flat, all_pred_flat = [], []
    for fold_idx, (train_data, val_data) in enumerate(
            stratified_kfold(train_val, k=args.cv_folds)):
        y_true, y_pred = train_nn(
            hparams, args.model, train_data, val_data, input_dim,
            edge_dim=edge_dim, fold_name=f"fold{fold_idx}",
        )
        all_true_flat.extend(y_true)
        all_pred_flat.extend(y_pred)

    f1   = f1_score(all_true_flat, all_pred_flat, zero_division=0)
    acc  = accuracy_score(all_true_flat, all_pred_flat)
    prec = precision_score(all_true_flat, all_pred_flat, zero_division=0)
    rec  = recall_score(all_true_flat, all_pred_flat, zero_division=0)

    trial_str = f"Trial {args.trial} | " if args.trial else ""
    print(f"\n{trial_str}{args.model} | F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    print(f"Hparams: {json.dumps(hparams)}")

    # ---- Persist to session log ----
    out_dir = Path("outputs/hp_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.model}_{session}.jsonl"
    with open(log_path, "a") as lf:
        lf.write(json.dumps({
            "trial": args.trial, "model": args.model,
            "f1": f1, "accuracy": acc, "precision": prec, "recall": rec,
            "hparams": hparams,
        }) + "\n")
    print(f"Saved: {log_path}")

    if use_wandb:
        wandb.log({"f1": f1, "accuracy": acc, "precision": prec, "recall": rec})
        wandb.run.summary.update({"f1": f1})
        wandb.finish()


if __name__ == "__main__":
    main()
