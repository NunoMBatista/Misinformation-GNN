import torch
from pathlib import Path


def load_data():
    data_path = Path("data/processed/pheme_pyg_dataset.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocessing first or run download script.")
    # Use weights_only=False to allow loading custom objects if required by warnings
    return torch.load(data_path, weights_only=False)


def _bfs_depths(edge_index, num_nodes):
    """BFS from node 0 (source tweet). Returns float tensor of depths; -1 = unreachable."""
    depths = [-1.0] * num_nodes
    if num_nodes == 0 or edge_index.shape[1] == 0:
        if num_nodes > 0:
            depths[0] = 0.0
        return torch.tensor(depths, dtype=torch.float)

    depths[0] = 0.0
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[int(src)].append(int(dst))

    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if depths[nb] == -1.0:
                depths[nb] = depths[node] + 1.0
                queue.append(nb)

    return torch.tensor(depths, dtype=torch.float)


def add_graph_features(dataset, use_root=True, use_depth=True):
    """
    Appends computed positional features to each graph's node matrix.
    Called after filter_features(), so these columns are always appended last.

      is_root  (use_root=True) : 1.0 for node 0 (source tweet), 0.0 elsewhere.
               Node 0 is always the source tweet by insertion order — verified
               across the full dataset via thread_id matching.

      depth    (use_depth=True): BFS depth from node 0. Lets the model learn
               position-conditioned representations without relying on attention
               to implicitly discover cascade depth.

    Returns (dataset, new_input_dim).
    """
    if not use_root and not use_depth:
        return dataset, dataset[0].x.shape[1] if dataset else 0

    for data in dataset:
        n       = data.num_nodes
        extras  = []

        if use_root:
            is_root       = torch.zeros(n, 1)
            is_root[0, 0] = 1.0
            extras.append(is_root)

        if use_depth:
            depth = _bfs_depths(data.edge_index, n).unsqueeze(1)
            extras.append(depth)

        data.x = torch.cat([data.x] + extras, dim=1)

    return dataset, dataset[0].x.shape[1] if dataset else 0


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
