# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph Neural Network project for rumour/misinformation detection on Twitter cascades using the PHEME dataset. The task is graph-level binary classification (Rumour vs. Non-Rumour) evaluated with Leave-One-Event-Out (LOEO) cross-validation across 9 real-world events.

## Commands

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

### Node features (390 dimensions total)
- `[0:384]`: NLP text embedding (all-MiniLM-L6-v2)
- `[384]`: Log-normalized follower count
- `[385]`: User verified status (0/1)
- `[386]`: PageRank
- `[387]`: Degree centrality
- `[388]`: In-degree centrality
- `[389]`: Out-degree centrality

Feature channels are **dynamically masked** at load time by `filter_features()` in `src/data/dataset.py` based on YAML config toggles — the `.pt` file always stores all 390 features.

### Models

All models reduce to a per-graph vector then classify with a single logit:

| Model | Aggregation | Classifier |
|---|---|---|
| `RandomForestBaseline` | mean pool → sklearn RF | RF |
| `MLPBaseline` | mean pool → MLP | Linear |
| `SimpleGNN` | 2× GCNConv + mean pool | Linear |
| `GATModel` | 2× GATv2Conv (2 heads) + mean pool | Linear |

GATv2Conv doubles the hidden dimension at each layer (e.g. `hidden_dim=32` → intermediate is `64`), so the classifier head receives `hidden_dims[-1] * heads` features. This is handled in `src/models/gnn.py`.

### Experiment orchestration

`configs/experiment.yml` drives everything:
- **`features`** section: toggle individual feature channels on/off
- **`experiments`** section: list of models with their hyperparameters

`run_experiments.py` reads the YAML, calls `load_data()` + `filter_features()`, then runs LOEO CV for each experiment. Results are written to timestamped CSV in `outputs/`.

### Training

`src/models/trainer.py` provides:
- `train_rf()` — for RandomForest
- `train_nn()` — unified for MLP, GNN, and GAT; uses `BCEWithLogitsLoss`, Adam optimizer; prediction threshold is 0.5 after sigmoid

## Key paths

| Purpose | Path |
|---|---|
| Main entry point | `src/scripts/run_experiments.py` |
| Experiment config | `configs/experiment.yml` |
| GNN/GAT architectures | `src/models/gnn.py` |
| Training loop | `src/models/trainer.py` |
| Baseline models | `src/models/baselines.py` |
| Dataset loading + feature masking | `src/data/dataset.py` |
| Full preprocessing pipeline | `src/data/builder/preprocessing_pipeline.py` |
| HuggingFace dataset | `NunoBatista/PHEME-Misinformation-Graphs` |
