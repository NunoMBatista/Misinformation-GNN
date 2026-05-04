# Experiments TO-DO List

This document tracks the specific experiments needed to satisfy the project's baseline, GNN, XAI, and Complex Systems analysis requirements.

## 1. Baselines (Non-Graph Models)
- [ ] **Baseline A (Text-Only):** Aggregate textual embeddings of a cascade (e.g., mean pool) and run through a Random Forest or Logistic Regression.
- [ ] **Baseline B (Structure & User Only):** Aggregate only `user_followers`, `user_verified`, and computed network metrics (PageRank, degree) into a Random Forest. (Ignores text entirely).
- [ ] **Baseline C (All Features):** Combine text and structural features into a single array per cascade for a Random Forest classifier.

## 2. GNN Experiments (Graph Classification)
- [ ] **GNN A (Full Model):** Train a GCN/GAT using the full 384-dim text embedding + structural/user features concatenated.
- [ ] **GNN B (Structure-Only Topology Test):** Train the GCN/GAT using *only* structural/user node features (no NLP text embeddings). This tests if the mathematical shape of information spread is enough to detect a rumour.
- [ ] **GNN C (Text-Only Topology Test):** Train the model with *only* the text embeddings as node features (information flow still relies on edges, but no explicit PageRank/Degree node features are passed).

## 3. Explainable AI (XAI)
- [ ] **Feature Importance Rank:** Compare outcomes of the experiments above to rank feature importance. 
- [ ] **GNNExplainer (Node/Edge Importance):** Run `GNNExplainer` on individual graph predictions to visualize which specific retweets/nodes or edges triggered the "Rumour" classification.
- [ ] **Attention Weight Analysis:** If using a GAT, extract and plot the attention weights for a large cascade to see which user replies the model attended to most.

## 4. Complex Systems Analysis
- [ ] **Cascade Depth/Size Correlation:** Correlate cascade depth and node count against the ground truth labels. Do rumours systematically spread deeper/wider?
- [ ] **Tipping Point Analysis:** Evaluate model confidence grouped by time or graph size (e.g., predict at 10 nodes vs. 100 nodes). Investigate at what cascade threshold misinformation characteristics become obvious to the model.
- [ ] **Structural Virality (Broadcast vs. Viral):** Measure the Wiener index or average path length of the cascades. This determines if misinformation spreads via a single massive influencer (star-graph / broadcast) or via peer-to-peer chains (deep viral trees).
- [ ] **Network Robustness / Immunization:** Simulate "banning" users. Remove the top 5% highest-degree nodes from the cascades and measure how the network shatters. Do rumour cascades rely on a few vulnerable hubs more than non-rumours?
- [ ] **Echo Chambers (Clustering):** Analyze the average clustering coefficient or triads to see if rumours circulate in tightly-knit "echo chambers" compared to normal news.