import pickle
import math
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path

# Absolute imports for modular files
from thread_processing import parse_all_threads
from network_metrics import add_network_metrics
from node_embedding import embed_nodes

# ==========================================
# GLOBAL PIPELINE CONFIGURATION
# ==========================================
COMPUTE_PAGERANK = True
COMPUTE_DEGREE_CENTRALITY = True
COMPUTE_IN_OUT_DEGREE = True

NLP_DIM = 384   # all-MiniLM-L6-v2 output dimension

# Paths
DATA_DIR = Path("data/pheme-rnr-dataset/all-rnr-annotated-threads")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "pheme_cascades.gpickle"
PYG_OUTPUT_FILE = OUTPUT_DIR / "pheme_pyg_dataset.pt"


def create_pyg_dataset(graphs):
    """
    Converts networkx graphs into PyTorch Geometric Data objects.

    Node feature layout (390 dims total):
      [0:384]  all-MiniLM-L6-v2 text embedding
      [384]    log1p follower count
      [385]    user verified (0/1)
      [386]    PageRank
      [387]    degree centrality
      [388]    in-degree centrality
      [389]    out-degree centrality
    """
    pyg_graphs = []

    for G in tqdm(graphs, desc="Converting to PyTorch Tensors"):
        node_mapping = {node: i for i, node in enumerate(G.nodes())}

        # Build Edge Index [2, num_edges]
        edge_index = []
        for src, dst in G.edges():
            edge_index.append([node_mapping[src], node_mapping[dst]])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Build Node Feature Matrix
        x = []
        for node in G.nodes():
            data = G.nodes[node]
            feats = []

            # Text embedding (NLP_DIM dims)
            emb = list(data.get('text_embedding', [0.0] * NLP_DIM))
            feats.extend(emb)

            # User features
            feats.append(math.log1p(data.get('user_followers', 0)))
            feats.append(float(data.get('user_verified', 0)))

            # Structural features
            if COMPUTE_PAGERANK:
                feats.append(float(data.get('pagerank', 0.0)))
            if COMPUTE_DEGREE_CENTRALITY:
                feats.append(float(data.get('degree_centrality', 0.0)))
            if COMPUTE_IN_OUT_DEGREE:
                feats.append(float(data.get('in_degree', 0.0)))
                feats.append(float(data.get('out_degree', 0.0)))

            x.append(feats)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor([G.graph.get('label', 0)], dtype=torch.long)

        pyg_data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data.thread_id = G.graph.get('thread_id', 'unknown')
        pyg_data.event = G.graph.get('event', 'unknown')
        pyg_data.text = [G.nodes[n].get('text', '') for n in G.nodes()]

        pyg_graphs.append(pyg_data)

    return pyg_graphs


def main():
    print("Starting Misinformation Dataset Pipeline...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Parsing threads into NetworkX graphs...")
    graphs = parse_all_threads(DATA_DIR)
    print(f"   Done. Extracted {len(graphs)} distinct graph cascades.")

    print(f"2. Computing network metrics...")
    graphs = add_network_metrics(
        graphs,
        use_pagerank=COMPUTE_PAGERANK,
        use_degree=COMPUTE_DEGREE_CENTRALITY,
        use_in_out_degree=COMPUTE_IN_OUT_DEGREE
    )
    print("   Done.")

    print("3. Generating all-MiniLM-L6-v2 text embeddings for all nodes...")
    graphs = embed_nodes(graphs)
    print("   Done.")

    print("4. Formatting to PyTorch Geometric tensors...")
    pyg_dataset = create_pyg_dataset(graphs)
    print("   Done.")

    print("5. Saving datasets...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(graphs, f)
    torch.save(pyg_dataset, PYG_OUTPUT_FILE)

    print(f"Pipeline complete!\n - Graphs: {OUTPUT_FILE}\n - PyG tensors: {PYG_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
