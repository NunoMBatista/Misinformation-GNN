# Implementation Goals

This document outlines the concrete, step-by-step implementation plan for the Misinformation GNN project using the PHEME dataset.

## 1. Data Preparation & Processing
* **Dataset Acquisition:** Download and parse the PHEME dataset of Twitter rumour threads.
* **Graph Construction:** 
  * **Nodes:** Represent each tweet in a rumour thread as a node.
  * **Edges:** Map user interactions by creating directed edges for replies and retweets between tweets.
* **Feature Extraction (Node Attributes):**
  * **Content Features:** Generate text embeddings for tweet content (e.g., using TF-IDF or a lightweight pre-trained language model, like MiniLM).
  * **User Credibility Features:** Include user metadata (e.g., follower count, account age, verified status).
  * **Structural Features:** Compute basic network centrality metrics (e.g., PageRank, degree centrality) for each node as input features.
* **Labels:** Map the PHEME annotations into a binary classification task (`False/Misinformation` vs. `True`) or multi-class (including `Unverified`).

## 2. Model Implementation
* **Baseline Formulation:** Implement a standard non-graph Machine Learning model (e.g., Random Forest or Logistic Regression) using only the extracted text and user features to classify rumours. This serves as the comparative baseline.
* **GNN Architecture:** 
  * Implement a Graph Neural Network using PyTorch Geometric (PyG) or DGL. 
  * Use a Graph Convolutional Network (GCN) or Graph Attention Network (GAT).
  * **Task:** Graph-level classification (where each separate rumour thread/cascade is a distinct graph to be classified as fake or real).

## 3. Explainable AI (XAI) & Evaluation
* **Performance Metrics:** Evaluate both the baseline and GNN models using Accuracy, Precision, Recall, and F1-Score.
* **Model Explainability (XAI):** Implement Explainable AI techniques to understand the GNN's decision-making process.
  * Use tools like **GNNExplainer** to extract the most important subgraphs, key nodes, and crucial node-features driving the classification.
  * If using a Graph Attention Network (GAT), analyze **attention weights** to visualize which user replies the network focuses on to detect rumours.
  * Compare the importance of user credibility features versus network centrality and text semantics natively.

## 4. Complex Systems & Network Analysis
* **Graph Topologies:** Analyze if specific graph shapes (e.g., deep, prolonged reply trees vs. shallow, rapid broadcast patterns) indicate a higher likelihood of fake news.
* **Cascade Dynamics:** Document how misinformation spreads. Analyze the macroscopic differences in size, depth, and duration between fake and real cascades.
* **Threshold / Emergent Effects:** Investigate if there is a tipping point (e.g., a specific graph density or cascade depth) where misinformation dominates the network quickly. Discuss how microscopic local interactions (user-to-user replies) lead to macroscopic misinformation cascades.