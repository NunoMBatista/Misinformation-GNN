from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def embed_nodes(graphs):
    """
    Takes a list of graphs, extracts the 'text' attribute of every node,
    and uses a local pre-trained language model to generate dense numerical embeddings.
    
    The raw 'text' attribute is left intact to ensure visualizations still work.
    """
    # Load a lightweight, fast text-embedding model from HuggingFace
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for G in tqdm(graphs, desc="Embedding Nodes"):
        node_ids = list(G.nodes())
        
        # Extract text strings for the entire graph
        texts = [G.nodes[node].get('text', '') for node in node_ids]
        
        if texts:
            # Batch encode strings into NumPy arrays (384 dimensions)
            embeddings = model.encode(texts, show_progress_bar=False)
            
            # Map embeddings back onto their respective nodes
            for idx, node in enumerate(node_ids):
                G.nodes[node]['text_embedding'] = embeddings[idx]
                
    return graphs
