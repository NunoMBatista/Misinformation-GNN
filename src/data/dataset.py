import torch
import torch.nn.functional as F
from pathlib import Path

# NLP embedding dimension — must match the model used in node_embedding.py
NLP_DIM = 384   # all-MiniLM-L6-v2 (sentence-transformers)


def load_data():
    data_path = Path("data/processed/pheme_pyg_dataset.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocessing first or run download script.")
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
      depth    (use_depth=True): BFS depth from node 0.

    Returns (dataset, new_input_dim).
    """
    if not use_root and not use_depth:
        return dataset, dataset[0].x.shape[1] if dataset else 0

    for data in dataset:
        n      = data.num_nodes
        extras = []

        if use_root:
            is_root       = torch.zeros(n, 1)
            is_root[0, 0] = 1.0
            extras.append(is_root)

        if use_depth:
            depth = _bfs_depths(data.edge_index, n).unsqueeze(1)
            extras.append(depth)

        data.x = torch.cat([data.x] + extras, dim=1)

    return dataset, dataset[0].x.shape[1] if dataset else 0


def compute_edge_features(dataset):
    """
    Computes relational edge features and stores as data.edge_attr.
    Call on the RAW dataset BEFORE filter_features() — reads dims 0:NLP_DIM
    (all-MiniLM-L6-v2) and NLP_DIM+6 (log1p seconds since root).

    Edge features (2 dims per edge):
      [0] Cosine similarity between parent and child NLP embeddings.
      [1] log1p(max(0, child_time_delta - parent_time_delta)) — reply latency.
    """
    for data in dataset:
        ei = data.edge_index
        if ei.shape[1] == 0:
            data.edge_attr = torch.zeros((0, 2), dtype=torch.float)
            continue

        src, dst = ei[0], ei[1]

        # Parent-child semantic similarity — requires knowing who replied to whom
        emb_src = data.x[src, :NLP_DIM]
        emb_dst = data.x[dst, :NLP_DIM]
        cos_sim = F.cosine_similarity(emb_src, emb_dst, dim=-1, eps=1e-8).unsqueeze(1)

        # Reply latency: x[:, NLP_DIM+6] = log1p(seconds since root), invert to get seconds
        sec_src = torch.expm1(data.x[src, NLP_DIM + 6])
        sec_dst = torch.expm1(data.x[dst, NLP_DIM + 6])
        time_gap = torch.log1p(torch.clamp(sec_dst - sec_src, min=0.0)).unsqueeze(1)

        data.edge_attr = torch.cat([cos_sim, time_gap], dim=1)

    return dataset


def filter_features(dataset, features_config):
    """
    Dynamically selects node feature columns based on YAML config.

    Stored feature layout (390 dims, set by preprocessing_pipeline.py):
      [0:NLP_DIM]       NLP text embedding (all-MiniLM-L6-v2, 384-dim)
      [NLP_DIM+0]  384  log1p follower count
      [NLP_DIM+1]  385  user verified (0/1)
      [NLP_DIM+2]  386  PageRank
      [NLP_DIM+3]  387  degree centrality
      [NLP_DIM+4]  388  in-degree centrality
      [NLP_DIM+5]  389  out-degree centrality
    """
    indices = []

    if features_config.get("use_nlp", True):
        indices.extend(range(0, NLP_DIM))

    base = NLP_DIM
    if features_config.get("use_followers", True):
        indices.append(base + 0)
    if features_config.get("use_verified", True):
        indices.append(base + 1)
    if features_config.get("use_pagerank", True):
        indices.append(base + 2)
    if features_config.get("use_degree", True):
        indices.append(base + 3)
    if features_config.get("use_indegree", True):
        indices.append(base + 4)
    if features_config.get("use_outdegree", True):
        indices.append(base + 5)

    for data in dataset:
        data.x = data.x[:, indices]

    return dataset, len(indices)
