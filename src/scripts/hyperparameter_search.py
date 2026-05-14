import argparse
import copy
import json
import sys
import datetime
from pathlib import Path

import optuna
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import load_data, filter_features, add_graph_features
from src.models.trainer import train_nn

EXCLUDED_EVENTS = {
    "prince-toronto-all-rnr-threads",
    "ebola-essien-all-rnr-threads",
    "ferguson-all-rnr-threads",
}
MIN_NODES = 3

HIDDEN_DIMS_CHOICES = {
    "32-16":       [32, 16],
    "64-32":       [64, 32],
    "128-64":      [128, 64],
    "128-64-32":   [128, 64, 32],
    "256-128-64":  [256, 128, 64],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Misinformation GNN")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "gnn", "gat"],
                        help="Model architecture to optimise")
    parser.add_argument("--no_nlp", action="store_true",
                        help="Fix use_nlp=False for all trials (exclude NLP embeddings)")
    parser.add_argument("--n_trials", type=int, default=200,
                        help="Number of Optuna trials (default: 200)")
    parser.add_argument("--output_dir", type=str, default="outputs/hparam_search",
                        help="Directory to write results (default: outputs/hparam_search)")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name (auto-generated if omitted)")
    parser.add_argument("--seed", type=int, default=42,
                        help="TPE sampler seed (default: 42)")
    parser.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                        help="SQLite DB path for persistent storage "
                             "(e.g. outputs/hparam_search/study.db). "
                             "Each completed trial is saved immediately, so Ctrl+C is safe. "
                             "Re-run with the same --checkpoint and --study_name to resume.")
    return parser.parse_args()


def suggest_hyperparams(trial: optuna.Trial, model_type: str, no_nlp: bool):
    features_cfg = {
        "use_nlp":       not no_nlp,
        "use_followers": trial.suggest_categorical("use_followers", [True, False]),
        "use_verified":  trial.suggest_categorical("use_verified",  [True, False]),
        "use_pagerank":  trial.suggest_categorical("use_pagerank",  [True, False]),
        "use_degree":    trial.suggest_categorical("use_degree",    [True, False]),
        "use_indegree":  trial.suggest_categorical("use_indegree",  [True, False]),
        "use_outdegree": trial.suggest_categorical("use_outdegree", [True, False]),
        "use_root":      trial.suggest_categorical("use_root",      [True, False]),
        "use_depth":     trial.suggest_categorical("use_depth",     [True, False]),
    }

    hidden_key = trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMS_CHOICES))

    model_cfg = {
        "model_type":    model_type,
        "hidden_dims":   HIDDEN_DIMS_CHOICES[hidden_key],
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "dropout":       trial.suggest_float("dropout", 0.1, 0.7),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs":        trial.suggest_categorical("epochs", [50, 100, 150, 200]),
    }

    if model_type == "gat":
        model_cfg["heads"] = trial.suggest_categorical("heads", [1, 2, 4])

    return features_cfg, model_cfg


def run_loeo(dataset_raw, model_type: str, features_cfg: dict, model_cfg: dict) -> float:
    dataset = copy.deepcopy(dataset_raw)

    dataset, input_dim = filter_features(dataset, features_cfg)
    use_root  = features_cfg.get("use_root",  False)
    use_depth = features_cfg.get("use_depth", False)
    if use_root or use_depth:
        dataset, input_dim = add_graph_features(dataset, use_root=use_root, use_depth=use_depth)

    dataset = [
        d for d in dataset
        if d.event not in EXCLUDED_EVENTS and d.num_nodes >= MIN_NODES
    ]

    events = sorted(set(d.event for d in dataset))
    fold_f1s = []

    for test_event in events:
        train_data = [d for d in dataset if d.event != test_event]
        test_data  = [d for d in dataset if d.event == test_event]
        if not train_data or not test_data:
            continue

        y_true, y_pred = train_nn(model_cfg, model_type, train_data, test_data, input_dim)
        fold_f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0


def save_results(study: optuna.Study, out_dir: Path, study_name: str) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\nNo completed trials to save.")
        return

    best = study.best_trial
    print(f"\n=== Best Trial #{best.number} ===")
    print(f"  Macro-F1 : {best.value:.4f}")
    print(f"  Params   : {best.params}")

    best_path = out_dir / f"{study_name}_best.json"
    with open(best_path, "w") as f:
        json.dump({
            "trial":        best.number,
            "macro_f1":     best.value,
            "params":       best.params,
            "features_cfg": best.user_attrs.get("features_cfg", {}),
            "model_cfg":    best.user_attrs.get("model_cfg",    {}),
        }, f, indent=2)
    print(f"Best params → {best_path}")

    csv_path = out_dir / f"{study_name}_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"All trials  → {csv_path}")


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"hpsearch_{args.model}_{timestamp}"

    storage = None
    if args.checkpoint:
        db_path = Path(args.checkpoint)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"
        print(f"Checkpoint storage: {db_path}")
        if db_path.exists():
            print("  (resuming existing study — completed trials will be skipped)\n")
        else:
            print("  (new study — trials will be saved as they complete)\n")

    print("Loading dataset...")
    dataset_raw = load_data()
    print(f"Dataset loaded: {len(dataset_raw)} graphs\n")

    def objective(trial: optuna.Trial) -> float:
        features_cfg, model_cfg = suggest_hyperparams(trial, args.model, args.no_nlp)

        # Prune the degenerate case where every feature flag is False
        if not any(features_cfg.values()):
            raise optuna.TrialPruned("No features selected.")

        macro_f1 = run_loeo(dataset_raw, args.model, features_cfg, model_cfg)

        trial.set_user_attr("features_cfg", {k: bool(v) for k, v in features_cfg.items()})
        trial.set_user_attr("model_cfg",    {k: str(v)  for k, v in model_cfg.items()})

        return macro_f1

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    completed_before = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    if completed_before:
        print(f"Resuming: {completed_before} trials already completed, "
              f"running up to {args.n_trials} more.\n")

    print(f"Study '{study_name}': {args.n_trials} trials | model={args.model} | no_nlp={args.no_nlp}\n")

    try:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted — saving results for completed trials...")

    save_results(study, out_dir, study_name)


if __name__ == "__main__":
    main()

# python src\scripts\hyperparameter_search.py --no_nlp --n_trials 100 --model mlp
# python src\scripts\hyperparameter_search.py --no_nlp --n_trials 100 --model mlp --checkpoint outputs/hparam_search/mlp_no_nlp.db
