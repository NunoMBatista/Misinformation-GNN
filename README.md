---
license: mit
language:
- en
task_categories:
- graph-ml
- text-classification
tags:
- misinformation
- PyTorch Geometric
- NetworkX
- rumour-detection
- complex-systems
viewer: false
---
# Misinformation-GNN

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/NunoBatista/PHEME-Misinformation-Graphs)

This project models rumour propagation as a complex system using Graph Neural Networks (GNNs), tested specifically on the **PHEME dataset**. It evaluates pure topological properties, NLP node embeddings, and applies complex network theories to detect misinformation.

## The Dataset

This project utilizes the **9-Event PHEME dataset for Rumour Detection and Veracity Classification**. The complete source code used to construct the Hugging Face dataset is available within this repository.

The PHEME dataset captures real-world breaking news events circulating on Twitter (such as the Sydney Siege, Charlie Hebdo attack, Ferguson unrest, etc.). Each conversation is structured as a thread, heavily annotated with ground-truth veracity labels denoting whether the circulating claim is a rumour/misinformation, or a verifiably true non-rumour claim. The expanded 9-event dataset provides significantly more robust generalization targets.

### Quick Start & Installation

To run this repository locally, you can instantly download the fully processed PyTorch and NetworkX graphs directly from our Hugging Face repository. There is no need to manually parse the raw JSON data.

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the Dataset:** Run the included downloader script. This pulls the pre-compiled tensors directly from Hugging Face and places them intuitively in the `data/processed/` directory.
   ```bash
   python src/scripts/download_dataset.py
   ```
3. **(Optional) Rebuild from Scratch:** If you want to audit the raw JSON ingestion or alter the NLP bindings, you can still manually download the source [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) to `data/pheme-rnr-dataset/` and re-run `python src/data/preprocessing_pipeline.py`.

## Repository Structure

```text
.
├── configs/
│   └── experiment.yml         # Hyperparameters and pipeline run orchestration
├── data/
│   ├── processed/             # PyG Tensors & NetworkX gpickle graph outputs
│   └── pheme-rnr-dataset/     # Raw dataset JSONs (Git Ignored)
├── docs/
│   └── goals/                 # Project planning & experimental checklists
├── outputs/                   # Automatically generated CSV metric reports
├── src/
│   ├── data/                  # Thread parsers, structural metrics, embeddings, & PyG compilers
│   ├── models/                # PyTorch & Sklearn (RF, MLP Baseline, GNN modules)
│   └── scripts/               # Entrypoints (Execution Pipeline, Visualizations)
└── README.md
```

## How It Works

The goal of this task is **Graph-Level Classification**: determining whether a single breaking-news conversation cascade is a "Rumour" or a "Non-Rumour".

Social media cascades are translated into mathematical graph representations:

* **The Graphs (Cascades):** Each graph represents a single conversational thread triggered by a piece of breaking news.
* **The Nodes:** Each node represents a single **Tweet**. Every node holds attributes:
  * Dense textual NLP Embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
  * User credibility features (Log-normalized follower counts, verified status)
  * Topological metrics (PageRank, Out-Degree, In-Degree)
* **The Edges:** Directed links represent the flow of interaction. If User B replies to User A, a directed edge connects their respective tweet nodes.

## Running Experiments

The project uses a purely YAML-driven execution pipeline and relies on a highly rigorous **Leave-One-Event-Out Cross-Validation** approach. The model is trained on $N-1$ real-world events and tested on the completely unseen $N$th event, preventing lexical overfitting and forcing the network to learn pure cascade structure.

To run the experiments (Random Forest Baseline, MLP Baseline, Simple GCN):

```bash
python src/scripts/run_experiments.py --config configs/experiment.yml
```

Metrics will be automatically pooled across all subsets, outputting global `Accuracy, Precision, Recall, F1` to the terminal and saving them seamlessly as a timestamped `.csv` in `outputs/`.

## Project Scope & Next Steps

Beyond simple classification, the project strictly relies on two modes of analysis:

1. **Explainable AI (XAI):** Utilizing frameworks like GNNExplainer and attention-weight extraction to peek inside the network and understand exactly which nodes, features, and subgraphs trigger a "Rumour" prediction.
2. **Complex Systems Analysis:** Evaluating macroscopic topological structures (like cascade depth, tipping points, and network density) to understand emergent dynamics of misinformation spreading at scale.

---



## Citation & Acknowledgements

This project builds upon the 9-event PHEME dataset. If you utilize this processed dataset or pipeline in your own work, please ensure you cite the original dataset creators:

> Zubiaga, A., Kochkina, E., Liakata, M., Procter, R., Lukasik, M., et al. (2016). Fact-checking updates on the rumorous PHEME dataset. *Figshare*.
> Dataset DOI: [10.6084/m9.figshare.6392078](https://doi.org/10.6084/m9.figshare.6392078)
