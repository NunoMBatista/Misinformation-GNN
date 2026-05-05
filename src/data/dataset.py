import torch
from pathlib import Path


def load_data():
    data_path = Path("data/processed/pheme_pyg_dataset.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocessing first or run download script.")
    # Use weights_only=False to allow loading custom objects if required by warnings
    return torch.load(data_path, weights_only=False)

def filter_features(dataset, features_config):
    """Dynamically filters the node feature matrix based on YAML config preferences"""
    indices = []
    
    # [0:384] NLP Embeddings
    if features_config.get("use_nlp", True):
        indices.extend(list(range(0, 384)))
        
    current_idx = 384
    # [384] Followers
    if features_config.get("use_followers", True):
        indices.append(current_idx)
    current_idx += 1
    
    # [385] Verified
    if features_config.get("use_verified", True):
        indices.append(current_idx)
    current_idx += 1
    
    # [386] PageRank
    if features_config.get("use_pagerank", True):
        indices.append(current_idx)
    current_idx += 1
    
    # [387] Degree Centrality
    if features_config.get("use_degree", True):
        indices.append(current_idx)
    current_idx += 1
    
    # [388] In-Degree
    if features_config.get("use_indegree", True):
        indices.append(current_idx)
    current_idx += 1
    
    # [389] Out-Degree
    if features_config.get("use_outdegree", True):
        indices.append(current_idx)
        
    # Apply mask slice to all graphs
    for data in dataset:
        data.x = data.x[:, indices]
        
    return dataset, len(indices)
