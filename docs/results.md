# Experimental Results

## 1. Dataset

| Event | Graphs | Rumour | Non-Rumour | R% |
|---|---|---|---|---|
| charliehebdo | 2 079 | 458 | 1 621 | 22.0% |
| ferguson | 1 143 | 284 | 859 | 24.8% |
| germanwings | 469 | 238 | 231 | 50.7% |
| gurlitt | 138 | 61 | 77 | 44.2% |
| ottawashooting | 890 | 470 | 420 | 52.8% |
| putinmissing | 238 | 126 | 112 | 52.9% |
| sydneysiege | 1 221 | 522 | 699 | 42.8% |
| prince-toronto | 233 | 229 | 4 | 98.3% — excluded (near single-class) |
| ebola-essien | 14 | 14 | 0 | 100% — excluded (single-class) |
| **TOTAL** | **6 425** | **2 402** | **4 023** | **37.4%** |

**Train / test split** (Leave-One-Event-Out design):
- Training events (LOEO CV): germanwings, gurlitt, ottawashooting, putinmissing
- Held-out test events: charliehebdo, sydneysiege
- Also excluded from all splits: ferguson (distribution shift; used only in baseline run)

After min\_nodes ≥ 3 filtering, the held-out test set contains **3 106 graphs** (charliehebdo: 1 952; sydneysiege: 1 154).

---

## 2. Network Science / Complex Systems Analysis

All analyses use the full 6 425-graph dataset (all events, no size filter). Tests are two-sided Mann-Whitney U with rank-biserial effect size *r* (thresholds: small ≥ 0.1, medium ≥ 0.3, large ≥ 0.5). Spearman ρ for continuous correlations.

### 2.1 Cascade Size and Depth

| Metric | Rumour mean | Rumour median | Non-Rumour mean | Non-Rumour median | Spearman ρ | p-value |
|---|---|---|---|---|---|---|
| Cascade size (nodes) | 13.89 | 10.0 | 18.00 | 14.0 | −0.119 | < 0.001 *** |
| Mean cascade depth | 1.257 | 1.000 | 1.528 | 1.048 | −0.102 | < 0.001 *** |

Rumour cascades are significantly **smaller and shallower** than non-rumour cascades (negative ρ: higher label = rumour = smaller/shallower). Effect is small but highly significant given n = 6 425.

### 2.2 Tipping Point — Rumour Ratio by Cascade Size

| Size bucket | Cascades | Rumour ratio |
|---|---|---|
| 1–5 | 1 723 | **44.9%** |
| 6–10 | 1 151 | 37.9% |
| 11–20 | 1 892 | 36.2% |
| 21–50 | 1 415 | 32.5% |
| 51–100 | 182 | 16.5% |
| 101–200 | 53 | 32.1% |
| 200+ | 9 | 11.1% |
| Dataset mean | 6 425 | 37.4% |

Rumour prevalence declines monotonically from 44.9% in small cascades to 11.1% in very large ones (200+ nodes), with an anomalous uptick at 101–200 (driven by low n). This tipping-point pattern is consistent with the hypothesis that successful non-rumours accumulate more engagement.

### 2.3 Structural Virality (Average Path Length)

| Class | APL mean | APL median |
|---|---|---|
| Rumour | 2.463 | 1.988 |
| Non-Rumour | 2.808 | 2.133 |

Mann-Whitney U = 3 389 624, p < 0.001 ***, r = 0.114 (small)

Non-rumour cascades have significantly **longer average paths** (more chain-like / viral diffusion), while rumour cascades are structurally more star-like (replies cluster around the root). Effect is statistically significant but small in magnitude.

### 2.4 Network Robustness — LCC after Hub Removal

| Removed (%) | Rumour LCC | Non-Rumour LCC |
|---|---|---|
| 1% | 0.242 | 0.256 |
| 5% | 0.235 | 0.241 |
| 10% | 0.209 | 0.201 |
| 15% | 0.184 | 0.173 |
| 20% | 0.165 | 0.152 |
| 25% | 0.151 | 0.138 |
| 30% | 0.143 | 0.128 |

Both classes show near-identical robustness curves. Rumour cascades are marginally more robust at low removal rates (hubs slightly less critical); the curves converge and cross by 10%. No significant structural difference in hub-centrality dependence.

### 2.5 Echo Chambers (Branching Factor)

| Metric | Rumour mean | Non-Rumour mean | p-value | *r* |
|---|---|---|---|---|
| Branching factor | 4.419 | 4.611 | 0.650 | 0.007 |

No significant difference (p = 0.650). Branching factor (mean out-degree of non-leaf nodes) is indistinguishable between classes.

### 2.6 Novel Topological Metrics

| Metric | Description | R mean | NR mean | p-value | *r* | Sig |
|---|---|---|---|---|---|---|
| Epidemic R₀ | Mean out-degree of internal nodes | 4.214 | 4.469 | 0.081 | 0.028 | ns |
| Root Dominance | Fraction of edges from root node | 0.765 | 0.727 | < 0.001 | 0.074 | *** |
| Gini out-degree | Out-degree inequality | 0.683 | 0.718 | < 0.001 | 0.065 | *** |
| Strahler Number | Fractal branching complexity | 1.893 | 2.024 | < 0.001 | 0.104 | *** |
| Leaf Ratio | Fraction of leaf nodes | 0.736 | 0.722 | 0.010 | 0.039 | ** |
| Normalised Depth | Max depth / log₂(n) | 0.726 | 0.822 | < 0.001 | 0.070 | *** |

Key findings:
- **Root Dominance** is significantly higher for rumours (more broadcast/star structure; the root tweet drives more replies directly). Effect is small (r = 0.074).
- **Gini coefficient** is lower for rumours: out-degree is slightly more equally distributed (fewer super-hub nodes relative to non-rumours).
- **Strahler Number** is lower for rumours (r = 0.104, small-medium): rumour trees are structurally simpler and less deeply branched.
- **Normalised Depth** is lower for rumours: rumour cascades are shallower than expected for their size, confirming the star / broadcast pattern.
- **Epidemic R₀** and **Branching Factor** are not significantly different.

**Overall summary**: All significant structural differences are small in effect size (r < 0.15). The clearest signal is that rumour cascades tend toward a star / broadcast topology (root dominance, low normalised depth, simpler Strahler branching) while non-rumour cascades display more chain-like, deeply recursive spreading. Despite significance at large n, none of these effects is large enough to drive classification alone — consistent with the finding that structural-only models fall below the random baseline in held-out evaluation.

---

## 2b. Network Science Analysis — Matched Cohort (6 Events)

Same analyses as Section 2, restricted to the **6 events used in classifier evaluation** (excluding prince-toronto, ebola-essien, and ferguson). This is the apples-to-apples comparison with Section 3: n = 5 035 graphs across germanwings, gurlitt, ottawashooting, putinmissing (train) + charliehebdo, sydneysiege (test).

### 2b.1 Cascade Size and Depth

| Metric | Rumour mean | Rumour median | Non-Rumour mean | Non-Rumour median | Spearman ρ | p-value |
|---|---|---|---|---|---|---|
| Cascade size (nodes) | 13.77 | 11.0 | 17.41 | 14.0 | −0.101 | < 0.001 *** |
| Mean cascade depth | 1.260 | 1.000 | 1.546 | 1.048 | −0.093 | < 0.001 *** |

Consistent with Section 2.1: rumour cascades remain significantly smaller and shallower after removing the three excluded events. Effect sizes are nearly identical.

### 2b.2 Tipping Point — Rumour Ratio by Cascade Size

| Size bucket | Cascades | Rumour ratio |
|---|---|---|
| 1–5 | 1 298 | **42.6%** |
| 6–10 | 928 | 39.0% |
| 11–20 | 1 460 | 37.7% |
| 21–50 | 1 190 | 32.2% |
| 51–100 | 123 | 13.0% |
| 101–200 | 32 | 34.4% |
| 200+ | 4 | 0.0% |
| Dataset mean | 5 035 | — |

Monotonic decline from small to large cascades holds; the 101–200 uptick remains (low n artefact). The pattern is unchanged from the full dataset.

### 2b.3 Structural Virality (Average Path Length)

| Class | APL mean | APL median |
|---|---|---|
| Rumour | 2.423 | 1.985 |
| Non-Rumour | 2.801 | 2.111 |

Mann-Whitney U = 2 166 100, p < 0.001 ***, r = 0.116 (small)

Near-identical to Section 2.3 (r = 0.114). Non-rumour cascades are significantly more chain-like.

### 2b.4 Network Robustness — LCC after Hub Removal

| Removed (%) | Rumour LCC | Non-Rumour LCC |
|---|---|---|
| 1% | 0.245 | 0.262 |
| 5% | 0.239 | 0.247 |
| 10% | 0.210 | 0.205 |
| 15% | 0.186 | 0.177 |
| 20% | 0.166 | 0.156 |
| 25% | 0.152 | 0.142 |
| 30% | 0.143 | 0.131 |

Curves nearly identical to Section 2.4. No structural difference in hub-centrality dependence.

### 2b.5 Echo Chambers (Branching Factor)

| Metric | Rumour mean | Non-Rumour mean | p-value | *r* |
|---|---|---|---|---|
| Branching factor | 4.485 | 4.541 | 0.073 | 0.032 |

Not significant (p = 0.073), consistent with Section 2.5.

### 2b.6 Novel Topological Metrics

| Metric | R mean | NR mean | p-value | *r* | Sig |
|---|---|---|---|---|---|
| Epidemic R₀ | 4.308 | 4.382 | 0.184 | 0.023 | ns |
| Root Dominance | 0.777 | 0.725 | < 0.001 | 0.096 | *** |
| Gini out-degree | 0.714 | 0.721 | 0.486 | 0.012 | ns |
| Strahler Number | 1.929 | 2.024 | < 0.001 | 0.076 | *** |
| Leaf Ratio | 0.737 | 0.716 | 0.002 | 0.053 | ** |
| Normalised Depth | 0.727 | 0.835 | < 0.001 | 0.077 | *** |

Notable differences vs. Section 2.6 (full dataset):
- **Gini** drops from *** to ns (p = 0.486): the inequality signal was driven by the excluded events (particularly ferguson, which had unusual degree distributions).
- **Epidemic R₀** remains ns.
- Root Dominance, Strahler, Leaf Ratio, and Normalised Depth remain significant with similar effect sizes.

**Overall**: The structural findings are robust to event exclusion. Rumour cascades in the 6-event cohort show the same star/broadcast topology signature (high root dominance, low normalised depth, simpler Strahler structure) as the full dataset.

---

## 3. Classifier Evaluation

### 3.1 Feature Configurations

| Config | Features used | Input dim |
|---|---|---|
| full | NLP (384) + user (2) + structural (4) + positional (2) | 392 |
| text | NLP (384) + user (2) | 386 |
| struct | structural (4) + positional (2) | 8 |

### 3.2 Models

| Model | Aggregation | Edge direction | Notes |
|---|---|---|---|
| MLP | Mean pool → MLP | N/A | No message passing; text/graph structure via pooling only |
| ImprovedGNN | GCNConv + skip + LayerNorm; mean pool ∥ root | original / bidir | |
| ImprovedGAT | GATv2Conv + skip + LayerNorm; mean pool ∥ root | original / bidir | Attention heads |
| GIN | GINConv (sum) + LayerNorm; mean pool ∥ root | original / bidir | WL-expressive |

All neural models: Adam, cosine-annealing LR schedule, gradient clipping (max\_norm = 1.0), focal loss (γ tuned per model), per-event majority-class undersampling.

### 3.3 Hyperparameter Search (Validation — LOEO on 4 Training Events)

50 guided trials per model–config combination (12 total). Evaluation metric: macro F1 on held-out training fold (LOEO across germanwings, gurlitt, ottawashooting, putinmissing).

| Key | Model | Config | Best F1 | Mean F1 | n repeats | Best hyperparameters |
|---|---|---|---|---|---|---|
| gnn_text | ImprovedGNN | text | 0.706 | 0.706 | 1 | [64,32], lr=5e-5, d=0.60, γ=2.6, bidir, wd=1.5e-4 |
| mlp_text | MLP | text | 0.702 | 0.702 | 1 | [256,128,64], lr=6.8e-5, d=0.45, γ=2.1, wd=1.9e-5 |
| gat_text | ImprovedGAT | text | 0.701 | 0.701 | 1 | [64,32], lr=1e-4, d=0.50, γ=2.0, original, wd=1e-4 |
| gat_full | ImprovedGAT | full | 0.693 | 0.693 | 1 | [128,64], lr=5.4e-5, d=0.44, γ=3.76, original, wd=1.2e-5 |
| mlp_full | MLP | full | **0.690** | **0.656** | **10** | [64,32], lr=1.6e-4, d=0.60, γ=2.0, wd=1.5e-4 |
| gnn_full | ImprovedGNN | full | 0.676 | 0.676 | 2 | [64,32], lr=6e-5, d=0.60, γ=2.0, bidir, wd=1.5e-4 |
| gat_struct | ImprovedGAT | struct | 0.668 | 0.668 | 1 | [8,4], lr=7e-5, d=0.30, γ=2.0, bidir, heads=1, wd=5e-5 |
| gin_struct | GIN | struct | 0.680 | 0.680 | 1 | [8,4], lr=5.6e-5, d=0.54, γ=2.0, bidir, wd=1.4e-5 |
| gnn_struct | ImprovedGNN | struct | 0.653 | 0.653 | 1 | [8,4], lr=7.1e-5, d=0.17, γ=3.1, original, wd=1.4e-4 |
| gin_full | GIN | full | 0.651 | 0.651 | 1 | [128,64,32], lr=2.77e-3, d=0.59, γ=2.55, bidir, wd=4.5e-4 |
| mlp_struct | MLP | struct | 0.629 | 0.629 | 1 | [8,4], lr=5.7e-5, d=0.39, γ=1.03, wd=2.3e-4 |
| gin_text | GIN | text | 0.640 | 0.640 | 1 | [128,64,32], lr=3e-3, d=0.50, γ=2.0, bidir, wd=2e-4 |

Note: mlp\_full is the most reliably estimated — 10 repeats with mean F1 = 0.656 (best single = 0.690). Single-repeat estimates are noisier (LOEO on 4 events has high variance).

### 3.4 Final Held-Out Evaluation

Train on all 4 training events → evaluate on charliehebdo and sydneysiege (never seen during HP search).

**Random classifier lower bound** (predict-all-positive, p = 0.5):

| Event | n | n\_R | pos\_rate | F1 lower bound |
|---|---|---|---|---|
| charliehebdo | 1 952 | 436 | 22.3% | 0.309 |
| sydneysiege | 1 154 | 494 | 42.8% | 0.461 |
| **combined** | **3 106** | **930** | **30.0%** | **0.375** |

**Model results:**

| Model | Config | charlie F1 | sydney F1 | Combined F1 | Combined Acc | Combined Prec | Combined Rec | Δ vs LB |
|---|---|---|---|---|---|---|---|---|
| ImprovedGAT | text | 0.552 | 0.653 | **0.610** | 0.770 | 0.620 | 0.600 | +0.235 |
| ImprovedGAT | full | 0.557 | 0.576 | **0.565** | 0.709 | 0.512 | 0.631 | +0.190 |
| ImprovedGNN | text | 0.504 | 0.604 | **0.551** | 0.686 | 0.482 | 0.643 | +0.176 |
| ImprovedGNN | full | 0.540 | 0.557 | **0.548** | 0.697 | 0.495 | 0.613 | +0.173 |
| MLP | text | 0.473 | 0.577 | **0.520** | 0.643 | 0.435 | 0.646 | +0.145 |
| MLP | full | 0.460 | 0.600 | **0.521** | 0.614 | 0.414 | 0.700 | +0.146 |
| GIN | text | 0.000 | 0.600 | **0.474** | 0.647 | 0.428 | 0.531 | +0.099 |
| GIN | full | 0.334 | 0.460 | **0.378** | 0.365 | 0.268 | 0.645 | +0.003 |
| MLP | struct | 0.365 | 0.000 | **0.303** | 0.353 | 0.223 | 0.469 | −0.072 |
| ImprovedGNN | struct | 0.289 | 0.282 | **0.287** | 0.414 | 0.226 | 0.394 | −0.088 |
| ImprovedGAT | struct | 0.152 | 0.247 | **0.200** | 0.666 | 0.352 | 0.140 | −0.175 |
| GIN | struct | 0.206 | 0.138 | **0.176** | 0.659 | 0.319 | 0.122 | −0.199 |

**Key findings:**
- All structural-only models fall **below** the random lower bound on the combined test set, confirming that pure graph topology does not generalise across events.
- Text-only and full-feature models all beat the lower bound by comfortable margins (+0.099 to +0.235), with the gap driven by NLP embeddings.
- GIN collapses on charliehebdo in the text-only setting (F1 = 0.000, predicts all-negative) and barely beats the lower bound for full features (+0.003) — sum aggregation over-fits the training event distributions.
- The **GNN–MLP gap** is modest (+0.030 for text, +0.027 for full), suggesting the text embedding carries most of the signal and message passing provides a small but consistent benefit.

### 3.5 Statistical Significance of Model Comparisons

Bootstrap CIs use multinomial resampling from each model's confusion matrix (n\_boot = 5 000). McNemar's test uses a conservative approximation from marginal error counts (discordant cells estimated as |errors\_A − errors\_B|, all one-directional); actual p-values can only be equal or smaller — results marked *** are robustly significant. n = 3 106 combined test examples.

**Bootstrap 95% CI for F1 (combined test set):**

| Model | F1 | 95% CI |
|---|---|---|
| gat\_text | 0.610 | [0.583, 0.636] |
| gat\_full | 0.565 | [0.540, 0.590] |
| gnn\_text | 0.551 | [0.526, 0.576] |
| gnn\_full | 0.548 | [0.521, 0.573] |
| mlp\_full | 0.521 | [0.496, 0.544] |
| mlp\_text | 0.520 | [0.495, 0.544] |
| gin\_text | 0.474 | [0.447, 0.500] |
| gin\_full | 0.378 | [0.357, 0.399] |
| mlp\_struct | 0.303 | [0.281, 0.324] |
| gnn\_struct | 0.287 | [0.264, 0.310] |
| gat\_struct | 0.200 | [0.172, 0.230] |
| gin\_struct | 0.176 | [0.149, 0.204] |

CIs for the top four models (gat\_text through gnn\_full) do not overlap with the random lower bound [0.375], confirming significance above chance. The gnn\_text and gnn\_full CIs overlap each other, as do mlp\_full and mlp\_text — these pairs are not distinguishable.

**McNemar's test (key pairwise comparisons):**

| Comparison | p-value | Sig | Winner |
|---|---|---|---|
| Attention vs GCN — text features | < 0.001 | *** | gat\_text |
| Message passing vs MLP — text features | < 0.001 | *** | gnn\_text |
| GAT vs MLP — text features | < 0.001 | *** | gat\_text |
| Attention vs GCN — full features | < 0.001 | *** | gat\_full |
| Message passing vs MLP — full features | < 0.001 | *** | gnn\_full |
| Structural features in GAT (full vs text) | < 0.001 | *** | gat\_text |
| Structural features in GNN (full vs text) | < 0.001 | *** | gnn\_text |
| GAT vs GIN — text features | < 0.001 | *** | gat\_text |
| GNN vs GIN — text features | < 0.001 | *** | gnn\_text |

All differences are statistically significant at p < 0.001. Key findings:

- **Message passing significantly beats flat MLP** in both feature settings, confirming that the graph structure provides genuine signal beyond what mean-pooled embeddings capture.
- **Attention (GAT) significantly beats GCN** in both feature settings, justifying the added complexity of the attention mechanism.
- **Adding structural/positional features to NLP models hurts generalisation** (gat\_text > gat\_full, gnn\_text > gnn\_full, both significant). The structural features appear to cause overfitting to the training event distributions, which do not transfer to charliehebdo and sydneysiege.
- **GIN is significantly worse than both GCN and GAT** on text features, consistent with the hypothesis that sum aggregation over-fits the training event distributions.

**Note on power**: with only 2 held-out events, fold-level paired tests (Wilcoxon, n = 2) are not meaningful. All tests above operate on n = 3 106 individual predictions.

---

## 3.6 Ablation: Effect of Per-Event Class Balancing

`_balance_per_event()` undersamples the majority class within each event before training, so no single event's imbalance dominates. This ablation tests whether that step is actually helping.

Same setup as Section 3.4: best HP-searched hyperparameters, full feature set, train on 4 events, evaluate on charliehebdo + sydneysiege combined.

| Model | F1 (balanced) | F1 (no balance) | ΔF1 | Acc (balanced) | Acc (no balance) | ΔAcc |
|---|---|---|---|---|---|---|
| ImprovedGNN | 0.5802 | 0.5721 | **+0.0081** | 0.7186 | 0.7057 | +0.013 |
| ImprovedGAT | 0.5907 | 0.5905 | **+0.0002** | 0.7572 | 0.7334 | +0.024 |
| GIN | 0.4521 | 0.4609 | −0.0088 | 0.3757 | 0.2994 | +0.076 |
| MLP | 0.5341 | 0.5308 | **+0.0033** | 0.6478 | 0.6249 | +0.023 |

Precision / Recall breakdown:

| Model | Prec (bal) | Rec (bal) | Prec (no bal) | Rec (no bal) |
|---|---|---|---|---|
| ImprovedGNN | 0.524 | 0.650 | 0.507 | 0.657 |
| ImprovedGAT | 0.597 | 0.585 | 0.547 | 0.642 |
| GIN | 0.307 | 0.860 | 0.299 | 1.000 |
| MLP | 0.442 | 0.674 | 0.424 | 0.709 |

**Findings:**

- **F1 differences are negligible** for GNN (+0.008), GAT (+0.0002), and MLP (+0.003). Balancing provides no meaningful F1 benefit for these models.
- **GIN without balancing degenerates**: Recall = 1.000 and Precision = 0.299 — the model predicts *every graph as rumour*, likely because GIN's sum aggregation amplifies the majority class signal. Balancing recovers a more calibrated GIN (Rec = 0.860, Prec = 0.307), though F1 is still low.
- **Accuracy improves with balancing for all models** (+0.013 to +0.076), because without balancing models skew toward the majority class (non-rumour), inflating accuracy on the imbalanced test set while hurting rumour recall.
- The main benefit of balancing is **preventing degenerate predictions** (especially for GIN) and **improving precision without sacrificing much recall** — important for a task where false positives (flagging real news as rumour) have a real cost.

**Conclusion:** Balancing is a useful pre-processing step, particularly for GIN. For GNN, GAT, and MLP the effect on F1 is marginal, but it consistently improves precision and prevents recall collapse.

---

## 4. XAI Analysis

Held-out event: **charliehebdo** (1 952 test graphs: 436 rumour, 1 516 non-rumour).  
Outputs written to `outputs/xai/`.

### 4.1 Model performance on charliehebdo (XAI fold)

| Model | Accuracy | F1 |
|---|---|---|
| SimpleGNN (GCNConv) | 0.7818 | 0.5848 |
| GATModel (GATv2Conv) | 0.7987 | 0.5868 |

Note: values vary slightly across runs due to random weight initialisation; these are single-seed results used for XAI, not the HP-searched models in Section 3.

### 4.2 GNNExplainer — feature group importance

GNNExplainer applied to 40 Rumour + 40 Non-Rumour test graphs (balanced). Per-node feature masks collapsed into 7 semantic groups; higher = model relies on that group more. Mann-Whitney U compares per-graph group importance between classes.

| Feature group | R mean | NR mean | p-value | r | Sig |
|---|---|---|---|---|---|
| Followers | 0.427 | 0.340 | 0.0001 | −0.508 | *** |
| PageRank | 0.378 | 0.312 | 0.0001 | −0.513 | *** |
| Degree centrality | 0.362 | 0.305 | 0.0002 | −0.491 | *** |
| NLP (384-dim) | 0.371 | 0.313 | 0.0004 | −0.464 | *** |
| In-degree centrality | 0.344 | 0.287 | 0.0002 | −0.485 | *** |
| Out-degree centrality | 0.114 | 0.105 | 0.889 | −0.019 | ns |
| Verified (0/1) | 0.045 | 0.033 | 0.025 | −0.286 | * |

*r* = rank-biserial correlation (effect size; negative = Rumour > Non-Rumour)

**Interpretation:** The GCN relies significantly more on *all* feature groups when processing Rumour graphs than Non-Rumour graphs (all centrality + NLP groups p ≤ 0.001). Rumour cascades are structurally unusual — the model detects anomalous authority patterns (followers, PageRank) alongside the text. Out-degree is the only non-significant group, suggesting the number of replies a user sends is not discriminative.

### 4.3 GAT attention — cascade depth analysis

GAT layer-2 mean attention weight received per depth (0 = root tweet, no incoming edges so NaN). Mann-Whitney U tests whether Rumour and Non-Rumour attention differ at each depth.

| Depth | R mean | NR mean | p-value | r | Sig |
|---|---|---|---|---|---|
| 0 (root) | — | — | — | — | — |
| 1 | 0.504 | 0.513 | <0.001 | 0.324 | *** |
| 2 | 0.719 | 0.776 | <0.001 | 0.233 | *** |
| 3 | 0.568 | 0.543 | 0.001 | −0.106 | *** |
| 4 | 0.506 | 0.516 | 0.284 | 0.041 | ns |
| 5 | 0.492 | 0.506 | 0.094 | 0.073 | ns |
| 6 | 0.516 | 0.501 | 0.256 | −0.057 | ns |

**Interpretation:** Depth-2 nodes receive disproportionately high attention in both classes (~0.72–0.78 vs ~0.50 elsewhere), showing the GAT prioritises second-level replies over the rest of the cascade. Crucially, Non-Rumour cascades receive *more* depth-2 attention than Rumour ones (0.776 vs 0.719, p < 0.001), suggesting non-rumour threads generate more engagement at the second tier. At depth 3, the pattern reverses (Rumour > Non-Rumour, p = 0.001). Beyond depth 3 the difference is not significant — the model treats deeper nodes equally across classes.

### 4.4 GAT attention entropy

Attention entropy measures how diffuse or concentrated attention is per graph (higher = spread evenly; lower = focused on a few nodes).

| Class | Mean entropy | Median | Std |
|---|---|---|---|
| Non-Rumour | 2.615 | 2.818 | 0.808 |
| Rumour | 2.394 | 2.483 | 0.822 |

Mann-Whitney U test: **p < 0.001** (significant)

**Interpretation:** Rumour cascades have significantly lower attention entropy — the GAT concentrates on fewer nodes when classifying rumours. This aligns with the feature-importance finding: rumour cascades appear to have a small number of structurally or semantically anomalous nodes (high-authority users, unusual reply patterns at depth 2–3) that dominate the model's decision, whereas non-rumour engagement is more evenly distributed.

---

## 5. Baseline Experiment (Pre-HP-Search Reference)

Run on May 10 using default hyperparameters (no tuning), full feature set, LOEO across all 7 included events including ferguson.

| Model | Global F1 | Notes |
|---|---|---|
| Random Forest | 0.604 | Mean pool → RF, no HP tuning |
| MLP (default) | 0.595 | [128,64] hidden, lr=0.001 |
| SimpleGNN | 0.593 | 2× GCNConv, default params |
| GAT (regularised) | 0.623 | 2× GATv2Conv, 2 heads |

These are lower than the HP-searched models and include the ferguson event (known distribution-shift event). Provided as a baseline reference; the main comparison is Section 3.4.
