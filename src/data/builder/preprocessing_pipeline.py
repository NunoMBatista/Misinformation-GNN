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
# Toggle these booleans to enable/disable specific feature extraction steps
COMPUTE_PAGERANK = True
COMPUTE_DEGREE_CENTRALITY = True
COMPUTE_IN_OUT_DEGREE = True

# Paths
DATA_DIR = Path("data/pheme-rnr-dataset/all-rnr-annotated-threads")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "pheme_cascades.gpickle"
PYG_OUTPUT_FILE = OUTPUT_DIR / "pheme_pyg_dataset.pt"

def create_pyg_dataset(graphs):
    """
    Converts networkx graphs into PyTorch Geometric Data objects.
    Constructs the feature matrix (X) combining text embeddings, 
    normalized user features, and topological metrics.
    """
    pyg_graphs = []
    
    for G in tqdm(graphs, desc="Converting to PyTorch Tensors"):
        # Map string IDs to integer indices for PyG 
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        # 1. Build Edge Index Array [2, num_edges]
        edge_index = []
        for src, dst in G.edges():
            edge_index.append([node_mapping[src], node_mapping[dst]])
            
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 2. Build Node Features Matrix (X)
        x = []
        for node in G.nodes():
            data = G.nodes[node]
            node_features = []
            
            # Text embedding (384 dims) 
            emb = list(data.get('text_embedding', [0.0] * 384))
            node_features.extend(emb)
            
            # User features
            followers = math.log1p(data.get('user_followers', 0))
            verified = float(data.get('user_verified', 0))
            node_features.extend([followers, verified])
            
            # Structural features (Dynamically included based on global toggles)
            if COMPUTE_PAGERANK:
                node_features.append(float(data.get('pagerank', 0.0)))
            if COMPUTE_DEGREE_CENTRALITY:
                node_features.append(float(data.get('degree_centrality', 0.0)))
            if COMPUTE_IN_OUT_DEGREE:
                node_features.append(float(data.get('in_degree', 0.0)))
                node_features.append(float(data.get('out_degree', 0.0)))
            
            x.append(node_features)
            
        x = torch.tensor(x, dtype=torch.float)

        # 3. Target Label (Y) - 1 for Rumour, 0 for Non-Rumour
        y = torch.tensor([G.graph.get('label', 0)], dtype=torch.long)
        
        # Create PyG Data object mapping
        pyg_data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data.thread_id = G.graph.get('thread_id', 'unknown') # Save for XAI tracking
        pyg_data.event = G.graph.get('event', 'unknown') # Save for Leave-One-Event-Out validation
        
        # Attach raw text to allow downstream users to interpret the graph semantics without loading gpickle
        text_list = [G.nodes[n].get('text', '') for n in G.nodes()]
        pyg_data.text = text_list
        
        pyg_graphs.append(pyg_data)
        
    return pyg_graphs

def main():
    print("Starting Misinformation Dataset Pipeline...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # Step 1: Thread Parsing
    # Extrapolates raw strings, integers, and edges into NetworkX
    # -------------------------------------------------------------
    print("1. Parsing threads into NetworkX graphs...")
    graphs = parse_all_threads(DATA_DIR)
    print(f"   Done. Extracted {len(graphs)} distinct graph cascades.")
    
    # -------------------------------------------------------------
    # Step 2: Structural Network Expansion
    # Injects mathematical network topology details directly into nodes
    # -------------------------------------------------------------
    print(f"2. Computing network metrics (PageRank={COMPUTE_PAGERANK}, Degree={COMPUTE_DEGREE_CENTRALITY}, InOutDegree={COMPUTE_IN_OUT_DEGREE})...")
    graphs = add_network_metrics(
        graphs, 
        use_pagerank=COMPUTE_PAGERANK, 
        use_degree=COMPUTE_DEGREE_CENTRALITY,
        use_in_out_degree=COMPUTE_IN_OUT_DEGREE
    )
    print("   Done. Metric enrichment complete.")
    
    # -------------------------------------------------------------
    # Step 3: NLP Node Embedding 
    # Maps human language into 384-dimensional dense vectors
    # -------------------------------------------------------------
    print("3. Generating NLP text embeddings for all nodes...")
    graphs = embed_nodes(graphs)
    print("   Done. Embeddings generated and mapped.")
    
    # -------------------------------------------------------------
    # Step 4: PyTorch Geometric Conversion
    # Compiles the arrays and structures into ML-ready tensor objects
    # -------------------------------------------------------------
    print("4. Formatting to PyTorch Geometric Tensors...")
    pyg_dataset = create_pyg_dataset(graphs)
    print("   Done. PyTorch Dataset constructed.")
    
    # -------------------------------------------------------------
    # Step 5: Storage
    # Freezes the finalized structures and PyG Tensors
    # -------------------------------------------------------------
    print("5. Saving comprehensive datasets...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(graphs, f)
    torch.save(pyg_dataset, PYG_OUTPUT_FILE)
        
    print(f"Pipeline complete! \n - Vis/Graph data: {OUTPUT_FILE} \n - PyTorch Tensors: {PYG_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
