# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph Neural Network project for rumour/misinformation detection on Twitter cascades using the PHEME dataset. The task is graph-level binary classification (Rumour vs. Non-Rumour) evaluated with Leave-One-Event-Out (LOEO) cross-validation across 9 real-world events.

## Commands

All scripts must be run from the **project root** — paths like `data/processed/pheme_pyg_dataset.pt` are relative to it.

### Setup
```bash
pip install -r requirements.txt
```

### Download pre-processed dataset (from HuggingFace)
```bash
python src/scripts/download_dataset.py
```

### Run experiments (main entry point)
```bash
python src/scripts/run_experiments.py --config configs/experiment.yml
# Ablation (NLP features disabled — pure topology):
python src/scripts/run_experiments.py --config configs/experiment_ablation_no_nlp.yml
```

### XAI analysis (GNNExplainer + GAT attention)
```bash
python src/scripts/xai_analysis.py   # outputs to outputs/xai/
```

### Complex systems / topological analysis
```bash
python src/scripts/complex_systems_analysis.py   # outputs charts to outputs/complex_systems/
```

### Visualize cascades
```bash
python src/scripts/visualize.py --random          # random cascade
python src/scripts/visualize.py --thread-id <ID>  # specific thread
```

### Rebuild dataset from raw JSON (optional, raw data not in repo)
```bash
# First download raw PHEME JSON from Figshare → data/pheme-rnr-dataset/
python src/data/builder/preprocessing_pipeline.py
```

There is no test suite. Correctness is validated through LOEO cross-validation results written to `outputs/`.

## Architecture

### Data flow

```
Raw PHEME JSON (Twitter threads)
    → thread_processing.py       # JSON → NetworkX DiGraph (nodes=tweets, edges=replies)
    → node_embedding.py          # Tweet text → 384-dim vectors (all-MiniLM-L6-v2)
    → network_metrics.py         # PageRank, degree/in-degree/out-degree centrality
    → preprocessing_pipeline.py  # Orchestrates above + PyG conversion
    → pheme_pyg_dataset.pt       # Serialized PyG dataset (HuggingFace: NunoBatista/PHEME-Misinformation-Graphs)
```

`prepare_data.py` in the same builder directory is an older, simpler script that was superseded by `preprocessing_pipeline.py` — prefer the latter.

### Node features (390 dimensions total, stored in `.pt`)

- `[0:384]`: NLP text embedding (all-MiniLM-L6-v2)
- `[384]`: Log-normalized follower count
- `[385]`: User verified status (0/1)
- `[386]`: PageRank
- `[387]`: Degree centrality
- `[388]`: In-degree centrality
- `[389]`: Out-degree centrality

Feature channels are **dynamically masked** at load time by `filter_features()` in `src/data/dataset.py` based on YAML config toggles — the `.pt` file always stores all 390 features. This allows ablation studies without recomputing the dataset.

### Positional features (computed at load time, not stored)

`add_graph_features()` in `src/data/dataset.py` appends two positional features per node after loading:
- **`is_root`**: 1.0 for node 0 (source tweet), 0.0 elsewhere — breaks permutation invariance so the model knows which tweet started the cascade
- **`depth`**: BFS depth from node 0 — captures position in the cascade tree

Both are toggled via `use_root` / `use_depth` in the YAML `features` section.

### PyG Data object custom attributes

Beyond the standard `x`, `edge_index`, `y` fields, each graph carries:
- `thread_id` — original Twitter thread ID (for XAI traceability)
- `event` — event name used for LOEO splits
- `text` — list of raw tweet strings per node (for interpretability without loading the gpickle)

### Models

All models reduce to a per-graph vector then classify with a single logit:

| Model | Aggregation | Classifier |
|---|---|---|
| `RandomForestBaseline` | mean pool → sklearn RF | RF |
| `MLPBaseline` | mean pool → MLP | Linear |
| `SimpleGNN` | 2× GCNConv + mean pool | Linear |
| `GATModel` | 2× GATv2Conv (2 heads) + mean pool | Linear |

GATv2Conv output dim = `h_dim * heads` per layer, so each layer's output is wider than `h_dim` alone. The classifier head receives `hidden_dims[-1] * heads` features. This is handled in `src/models/gnn.py`.

### Class balancing

`_balance_per_event()` in `src/models/trainer.py` undersamples the majority class **within each event** before training, so no single event's class imbalance dominates training. This runs automatically inside `train_rf()` and `train_nn()`.

### Experiment orchestration

`configs/experiment.yml` drives everything:
- **`preprocessing`** section: `excluded_events` list (two events excluded by default) and `min_nodes` threshold (graphs with fewer nodes are dropped)
- **`features`** section: toggle individual feature channels on/off (including `use_root`, `use_depth`)
- **`experiments`** section: list of models with their hyperparameters

`run_experiments.py` reads the YAML, calls `load_data()` + `filter_features()` + `add_graph_features()`, then runs LOEO CV for each experiment. Results are written to timestamped CSV in `outputs/`.

### Training

`src/models/trainer.py` provides:
- `train_rf()` — for RandomForest
- `train_nn()` — unified for MLP, GNN, and GAT; uses `BCEWithLogitsLoss`, Adam optimizer; prediction threshold is 0.5 after sigmoid

### XAI pipeline (`src/scripts/xai_analysis.py`)

Trains both SimpleGNN and GATModel fresh on charliehebdo as the held-out test event, then applies two independent explainability methods:

1. **GNNExplainer on SimpleGNN** — extracts per-node feature importance masks, collapses the 390 dims into 7 semantic groups, and compares Rumour vs. Non-Rumour importance patterns. Also plots importance by BFS depth.
2. **GAT attention extraction** — intercepts `return_attention_weights=True` from GATv2Conv; aggregates per-edge coefficients to the receiving node; computes per-graph attention entropy. Low entropy means attention is concentrated on a few nodes (structure is load-bearing); Mann-Whitney U test checks if entropy differs between classes.

### Complex systems analysis (`src/scripts/complex_systems_analysis.py`)

Five independent topological analyses with Mann-Whitney U significance testing and `rank_biserial` effect sizes:
1. Cascade size and mean depth vs. rumour label (Spearman correlation)
2. Tipping point — rumour ratio across 7 cascade-size bins
3. Structural virality — average path length on largest connected component
4. Network robustness — LCC fraction after removing top-k degree hubs (k = 1%, 5%, 10%, …)
5. Echo chambers — clustering coefficient and mean branching factor

## Key paths

| Purpose | Path |
|---|---|
| Main entry point | `src/scripts/run_experiments.py` |
| XAI analysis | `src/scripts/xai_analysis.py` |
| Complex systems analysis | `src/scripts/complex_systems_analysis.py` |
| Experiment config | `configs/experiment.yml` |
| Ablation config (no NLP) | `configs/experiment_ablation_no_nlp.yml` |
| GNN/GAT architectures | `src/models/gnn.py` |
| Training loop | `src/models/trainer.py` |
| Baseline models | `src/models/baselines.py` |
| Dataset loading + feature masking | `src/data/dataset.py` |
| Full preprocessing pipeline | `src/data/builder/preprocessing_pipeline.py` |
| HuggingFace dataset | `NunoBatista/PHEME-Misinformation-Graphs` |
