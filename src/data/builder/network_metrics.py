import networkx as nx
from tqdm import tqdm

def add_network_metrics(graphs, use_pagerank=True, use_degree=True, use_in_out_degree=True):
    """
    Given a list of graphs, computes structural network metrics and 
    assigns them as individual node attributes in-place.
    """
    for G in tqdm(graphs, desc="Computing Network Metrics"):
        
        # Compute PageRank: identifies influential nodes within the cascade
        if use_pagerank:
            try:
                pr = nx.pagerank(G)
            except nx.PowerIterationFailedConvergence:
                # Fallback if convergence fails on very sparse/weird cascades
                pr = {node: 0.0 for node in G.nodes()} 
                
            for node, score in pr.items():
                G.nodes[node]['pagerank'] = score

        # Compute Degree Centrality: identifies highly replied-to nodes
        if use_degree:
            dc = nx.degree_centrality(G)
            for node, score in dc.items():
                G.nodes[node]['degree_centrality'] = score
                
        # Compute Directed Degree Centralities (Crucial for information flow)
        if use_in_out_degree:
            in_dc = nx.in_degree_centrality(G)
            out_dc = nx.out_degree_centrality(G)
            for node in G.nodes():
                G.nodes[node]['in_degree'] = in_dc[node]
                G.nodes[node]['out_degree'] = out_dc[node]
                
    return graphs
