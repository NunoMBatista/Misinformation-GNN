from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 64


def embed_nodes(graphs):
    """
    Embeds all tweet nodes using all-MiniLM-L6-v2 (384-dim).
    The raw 'text' attribute is left intact for downstream visualisations.
    """
    print(f"  Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Flatten all (graph_idx, node_id, text) so we can embed in large batches
    records = []
    for g_idx, G in enumerate(graphs):
        for node in G.nodes():
            records.append((g_idx, node, G.nodes[node].get('text', '') or ''))

    texts = [r[2] for r in records]
    all_embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    for (g_idx, node, _), emb in zip(records, all_embs):
        graphs[g_idx].nodes[node]['text_embedding'] = emb

    return graphs
