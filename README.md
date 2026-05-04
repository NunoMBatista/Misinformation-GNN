# Misinformation-GNN

This project models rumour propagation as a complex system using Graph Neural Networks (GNNs), tested specifically on the **PHEME dataset**.

## The Dataset

This project utilizes the **PHEME dataset of rumours and non-rumours**.

The PHEME dataset captures real-world breaking news events circulating on Twitter (such as the Sydney Siege or the Charlie Hebdo attack). It is composed of **5,802 conversational threads**. Each thread is annotated with ground-truth veracity labels denoting whether the circulating claim is a rumour/misinformation, or a verifiably true non-rumour claim.

### Setup & Installation

To run this repository locally, you must manually download the PHEME dataset.

1. **Download the data:** Download the zip file from [Figshare - PHEME dataset of rumours and non-rumours](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619).
2. **Extract to Directory:** Create a `data/` directory in the root of the project and extract the contents inside of it. Ensure the path aligns exactly as:
   `data/pheme-rnr-dataset/<event-folders>` (e.g. `data/pheme-rnr-dataset/charliehebdo`).
3. **Pre-Process:** Run the processing pipeline to parse the raw JSON tweets, extract network metrics, generate NLP embeddings, and compile them into NetworkX graph cascades:
   ```bash
   python src/data/preprocessing_pipeline.py
   ```

## How It Works

The goal of this task is **Graph-Level Classification**: determining whether a single breaking-news conversation thread is a "Rumour" (misinformation) or a "Non-Rumour".

To achieve this, the project translates social media cascades into mathematical graph representations:

* **The Graphs (Cascades):** Each graph represents a single conversational thread triggered by a piece of breaking news. There are 5,802 graphs in total.
* **The Nodes:** Each node represents a single **Tweet**. The root node is the *source claim* (the news being reported), and all subsequent nodes are user *reactions*. Every node holds attributes mapped from the tweet, including:
  * Content embeddings (semantic text features)
  * User credibility features (follower counts, verified status)
  * Network features (centrality)
* **The Edges:** Directed links represent the flow of interaction. If User B replies to User A, a directed edge connects their respective tweet nodes. This maps strictly *who interacted with whom*, tracing the cascade path.

## Project Scope

The project uses Graph Convolutional Networks (GCNs) or Graph Attention Networks (GATs) to learn these cascading patterns. Beyond standard accuracy metrics, the project strictly relies on two modes of analysis:

1. **Explainable AI (XAI):** Utilizing frameworks like GNNExplainer and attention-weight extraction to peek inside the neural network and understand exactly which nodes, features, and subgraphs trigger a "Rumour" prediction.
2. **Complex Systems Analysis:** Evaluating macroscopic topological structures (like cascade depth, tipping points, and network density) to understand emergent dynamics of misinformation spreading at scale.
