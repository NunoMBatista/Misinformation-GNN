# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph Neural Network project for rumour/misinformation detection on Twitter cascades using the PHEME dataset. The task is graph-level binary classification (Rumour vs. Non-Rumour) evaluated with Leave-One-Event-Out (LOEO) cross-validation across 9 real-world events.

## Commands

All scripts must be run from the **project root** — paths like `data/processed/pheme_pyg_dataset.pt` are relative to it. The ML virtualenv at `~/python_envs/ML` has all dependencies; system Python lacks PyTorch.

```bash
source ~/python_envs/ML/bin/activate
```

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
# Full benchmark suite (RF + MLP + GNN + GAT + GIN across 3 feature settings):
python src/scripts/run_experiments.py --config configs/benchmark_full.yml
python src/scripts/run_experiments.py --config configs/benchmark_text_only.yml
python src/scripts/run_experiments.py --config configs/benchmark_structural_only.yml
```

### Hyperparameter search (Claude Code as search loop)
`hp_search.py` runs one trial at a time; Claude Code reads the F1 output and supplies the next `--hparams` JSON, accumulating results in a `.jsonl` log.
```bash
# Random warmup trial:
python src/scripts/hp_search.py --model gin --config configs/benchmark_full.yml \
    --random --session my_session --trial 1

# Guided trial (Claude supplies hparams):
python src/scripts/hp_search.py --model gin --config configs/benchmark_full.yml \
    --hparams '{"learning_rate":0.0008,...}' --session my_session --trial 2

# Inspect the search space for a model+config:
python src/scripts/hp_search.py --model gin --config configs/benchmark_full.yml --show-space

# Promote best trial to best_configs.json:
python src/scripts/update_best_configs.py \
    --key gin_full --session my_session --model gin --config configs/benchmark_full.yml

# Final held-out evaluation (reads best_configs.json, trains on 4 events, tests on 2):
python src/scripts/final_eval.py
```

Logs written to `outputs/hp_search/<model>_<session>.jsonl`; best configs aggregated in `outputs/hp_search/best_configs.json`.

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

### Edge features (optional, computed at load time)

`compute_edge_features()` in `src/data/dataset.py` computes per-edge attributes and stores them as `data.edge_attr`. Must be called on the **raw** dataset before `filter_features()` since it reads from the full 390-dim feature vector.

- **`[0]` cos_sim**: cosine similarity between parent and child NLP embeddings — captures whether a reply is semantically aligned with its parent tweet
- **`[1]` log1p_time_gap**: log1p(seconds from parent tweet to child tweet) — reply latency

Enabled via `use_edge_features: true` in the YAML `features` section. Only `ImprovedGNN` and `ImprovedGAT` consume edge features (via `edge_dim` parameter); other models ignore `edge_attr`.

### PyG Data object custom attributes

Beyond the standard `x`, `edge_index`, `y` fields, each graph carries:
- `thread_id` — original Twitter thread ID (for XAI traceability)
- `event` — event name used for LOEO splits
- `text` — list of raw tweet strings per node (for interpretability without loading the gpickle)
- `edge_attr` — per-edge feature matrix [num_edges, 2], present only when `use_edge_features: true`

### Models

All models reduce to a per-graph vector then classify with a single logit:

| Model | Aggregation | Notes |
|---|---|---|
| `RandomForestBaseline` | mean pool → sklearn RF | |
| `MLPBaseline` | mean pool → MLP | no message passing |
| `SimpleGNN` | 2× GCNConv + mean pool | |
| `GATModel` | 2× GATv2Conv (2 heads) + mean pool | |
| `ImprovedGNN` | GCNConv + skip + LayerNorm; mean pool ∥ root node | supports edge_dim |
| `ImprovedGAT` | GATv2Conv + skip + LayerNorm; mean pool ∥ root node | supports edge_dim |
| `GINModel` | GINConv (sum, train_eps) + LayerNorm; mean pool ∥ root node | most expressive |

**GINModel** uses sum aggregation, which is as expressive as the Weisfeiler-Lehman graph isomorphism test. Unlike GCN's mean, sum preserves neighbour counts — letting the model distinguish "1 reply vs 10 replies", which is a direct proxy for cascade virality. This is the recommended model for graph-level classification on PHEME.

`ImprovedGNN` / `ImprovedGAT` / `GINModel` share three enhancements over the base models:
1. **Configurable edge direction** — `original` (parent→child), `inverted` (child→parent), or `bidirectional` (both); set via `edge_direction` in the YAML.
2. **Root readout** — the classifier concatenates mean-pooled graph embedding with the root node's representation, so the source tweet's signal is never diluted by pooling.
3. **Residual skip connections + LayerNorm** — stabilises training on shallow cascades. (`GINModel` uses LayerNorm without skip, since GINConv's internal MLP already provides depth.)

When `edge_dim > 0`, `ImprovedGNN` switches from `GCNConv` to `GATv2Conv(heads=1, edge_dim=edge_dim)` to consume edge features. `ImprovedGAT` uses the native `edge_dim` parameter of `GATv2Conv`.

### Class balancing

`_balance_per_event()` in `src/models/trainer.py` undersamples the majority class **within each event** before training, so no single event's class imbalance dominates training. This runs automatically inside `train_rf()` and `train_nn()`.

### Evaluation design (fixed train/test split)

The project uses a **fixed split**, not a full LOEO rotation, for the final evaluation:

| Role | Events |
|---|---|
| Training (HP search + final train) | germanwings, gurlitt, ottawashooting, putinmissing |
| Held-out test (never seen during HP search) | charliehebdo, sydneysiege |
| Excluded (near single-class) | prince-toronto (98.3% R), ebola-essien (100% R) |
| Excluded from main eval (distribution shift) | ferguson |

`run_experiments.py` still does LOEO over whatever events remain after `excluded_events` filtering. `final_eval.py` enforces the split above via hard-coded `TRAIN_EVENTS` / `TEST_EVENTS` constants and reads best hyperparameters from `outputs/hp_search/best_configs.json`.

### Experiment orchestration

Each YAML config has three top-level sections:
- **`preprocessing`**: `excluded_events` list and `min_nodes` threshold
- **`features`**: toggle individual feature channels on/off; `use_edge_features` triggers `compute_edge_features()` before filtering
- **`experiments`**: one block per run, each with its own model type and hyperparameters

`run_experiments.py` pipeline order:
1. `load_data()` — loads raw 390-dim dataset
2. `compute_edge_features()` — if `use_edge_features: true` (reads raw dims before filtering)
3. `filter_features()` — masks node feature columns per config
4. `add_graph_features()` — appends `is_root` / `depth` if requested
5. `preprocess_dataset()` — drops excluded events and small graphs
6. LOEO CV loop — trains and evaluates each experiment

Results are written to a timestamped CSV in `outputs/`.

### Training

`src/models/trainer.py` provides:
- `train_rf()` — for RandomForest
- `train_nn()` — unified for all neural models; Adam optimiser with cosine annealing LR schedule, gradient clipping (max_norm=1.0)

**Focal loss** is a per-experiment toggle: add `use_focal_loss: true` and `focal_gamma: 2.0` to any experiment block. Defaults to standard `BCEWithLogitsLoss` if omitted.

**Edge features** are passed through automatically when `data.edge_attr` is present and the model is `improved_gnn` or `improved_gat`.

### Benchmark configs

Three configs designed for a systematic ablation of graph structure vs. text vs. topology:

| Config | Features | Dim | Purpose |
|---|---|---|---|
| `benchmark_full.yml` | NLP + user + structural + positional | 392 | best overall features |
| `benchmark_text_only.yml` | NLP + user only | 386 | no graph-derived features; GNN advantage = pure message passing |
| `benchmark_structural_only.yml` | structural + positional only | 8 | no text; pure topology |

Each config runs RF, MLP, GNN, GAT, and GIN, enabling a clean 5×3 comparison.

### XAI pipeline (`src/scripts/xai_analysis.py`)

Trains both SimpleGNN and GATModel fresh on charliehebdo as the held-out test event, then applies two independent explainability methods:

1. **GNNExplainer on SimpleGNN** — extracts per-node feature importance masks, collapses the 390 dims into 7 semantic groups, and compares Rumour vs. Non-Rumour importance patterns. Also plots importance by BFS depth.
2. **GAT attention extraction** — intercepts `return_attention_weights=True` from GATv2Conv; aggregates per-edge coefficients to the receiving node; computes per-graph attention entropy. Low entropy means attention is concentrated on a few nodes (structure is load-bearing); Mann-Whitney U test checks if entropy differs between classes.

### Complex systems analysis (`src/scripts/complex_systems_analysis.py`)

Six independent topological analyses with Mann-Whitney U significance testing and `rank_biserial` effect sizes:
1. Cascade size and mean depth vs. rumour label (Spearman correlation)
2. Tipping point — rumour ratio across 7 cascade-size bins
3. Structural virality — average path length on largest connected component
4. Network robustness — LCC fraction after removing top-k degree hubs (k = 1%, 5%, 10%, …)
5. Echo chambers — clustering coefficient and mean branching factor
6. Novel topological metrics — Epidemic R₀, Root Dominance, Gini out-degree, Strahler Number, Leaf Ratio, Normalised Depth

## Design decisions and dead ends

These approaches were tried and deliberately reverted — do not re-introduce them without a clear new reason.

**BERTweet embeddings (768-dim) + temporal feature**: We built a new dataset using `vinai/bertweet-base` (768-dim) and added a per-node temporal feature (log1p seconds since root tweet), yielding 775-dim node features. BERTweet added ~+3.7 F1 over MiniLM, but the temporal feature contributed negligible improvement (+0.001). More importantly, the NLP quality gain benefited MLP and GNN equally — it did not widen the gap between flat aggregation and message passing. Since the study is about graph structure, not NLP quality, a better embedding would obscure the architectural comparison. We reverted to the HuggingFace dataset (all-MiniLM-L6-v2, 390-dim) to keep the focus clean.

**GIN overfits on text features**: In held-out evaluation, `GINModel` collapses on charliehebdo (F1 = 0.000, predicts all-negative in the text-only setting) and barely beats the random lower bound for full features (+0.003). Sum aggregation makes GIN sensitive to the absolute degree distribution, which shifts across events. Use `ImprovedGAT` or `ImprovedGNN` as the primary model for held-out generalisation; GIN is useful only as an upper bound on training-fold performance. Do not treat GIN's strong validation F1 (0.680) as reliable without held-out confirmation.

**Edge features (cos_sim + reply latency)**: We implemented `compute_edge_features()` to add 2-dim edge attributes (cosine similarity between parent/child embeddings, log1p reply latency). The approach hurt performance (-0.069 F1). Two reasons: (1) `ImprovedGNN` with `edge_dim>0` switches from GCNConv to GATv2Conv, which changes the base architecture entirely and conflates edge feature contribution with attention mechanism contribution; (2) bidirectional edge expansion assigns the same positive time gap to reverse edges (child→parent), which is semantically wrong. The `compute_edge_features()` function remains in `dataset.py` and the `use_edge_features` config flag still works, but no benchmark config uses it.

## Key paths

| Purpose | Path |
|---|---|
| Main entry point | `src/scripts/run_experiments.py` |
| HP search (one trial) | `src/scripts/hp_search.py` |
| Promote best HP trial | `src/scripts/update_best_configs.py` |
| Final held-out evaluation | `src/scripts/final_eval.py` |
| Best HP configs | `outputs/hp_search/best_configs.json` |
| XAI analysis | `src/scripts/xai_analysis.py` |
| Complex systems analysis | `src/scripts/complex_systems_analysis.py` |
| Benchmark configs | `configs/benchmark_full.yml`, `benchmark_text_only.yml`, `benchmark_structural_only.yml` |
| GNN/GAT/GIN architectures | `src/models/gnn.py` |
| Training loop | `src/models/trainer.py` |
| Baseline models | `src/models/baselines.py` |
| Dataset loading + feature masking | `src/data/dataset.py` |
| Full preprocessing pipeline | `src/data/builder/preprocessing_pipeline.py` |
| Experimental results | `docs/results.md` |
| HuggingFace dataset | `NunoBatista/PHEME-Misinformation-Graphs` |
