"""
Read a hp_search .jsonl session file, find the best hparams by mean F1 across
all trials that share identical hparams, and upsert the result into
outputs/hp_search/best_configs.json.

Usage:
    python src/scripts/update_best_configs.py \\
        --key gnn_full \\
        --session gnn_full \\
        --model improved_gnn \\
        --config configs/benchmark_full.yml
"""
import argparse, json, pathlib

OUT = pathlib.Path("outputs/hp_search/best_configs.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key",     required=True, help="Key in best_configs.json (e.g. gnn_full)")
    ap.add_argument("--session", required=True, help="Session ID used in hp_search.py")
    ap.add_argument("--model",   required=True, help="model_type string (e.g. improved_gnn)")
    ap.add_argument("--config",  required=True, help="YAML config file path")
    args = ap.parse_args()

    # Find the jsonl for this session (model prefix may vary)
    candidates = list(pathlib.Path("outputs/hp_search").glob(f"*{args.session}.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No .jsonl found for session '{args.session}'")
    jsonl = candidates[0]

    trials = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if not trials:
        raise ValueError("jsonl is empty")

    # Group by hparams, compute mean F1 per config
    from collections import defaultdict
    groups = defaultdict(list)
    for t in trials:
        key = json.dumps(t["hparams"], sort_keys=True)
        groups[key].append(t["f1"])

    best_key = max(groups, key=lambda k: sum(groups[k]) / len(groups[k]))
    best_hparams = json.loads(best_key)
    best_mean = sum(groups[best_key]) / len(groups[best_key])
    best_single = max(t["f1"] for t in trials if json.dumps(t["hparams"], sort_keys=True) == best_key)
    n_repeats = len(groups[best_key])

    entry = {
        "model_type":  args.model,
        "config_file": args.config,
        "session":     args.session,
        "best_trial_f1":          round(best_single, 4),
        "mean_f1_over_repeats":   round(best_mean,   4),
        "n_repeats":   n_repeats,
        "hparams":     best_hparams,
    }

    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[args.key] = entry
    OUT.write_text(json.dumps(existing, indent=2))
    print(f"[{args.key}] best mean F1={best_mean:.4f} (single={best_single:.4f}, n={n_repeats})")
    print(json.dumps(best_hparams, indent=2))


if __name__ == "__main__":
    main()
