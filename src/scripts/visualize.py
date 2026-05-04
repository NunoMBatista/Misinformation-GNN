import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from pyvis.network import Network
import datetime
import argparse
import random

def visualize_cascade(thread_id=None, random_pick=False):
    file_path = Path("data/processed/pheme_cascades.gpickle")
    if not file_path.exists():
        print("Processed data not found. Please run prepare_data.py first.")
        return

    with open(file_path, "rb") as f:
        graphs = pickle.load(f)

    # Determine which graph to visualize
    if random_pick:
        g = random.choice(graphs)
    elif thread_id:
        g = next((graph for graph in graphs if graph.graph.get('thread_id') == thread_id), None)
        if not g:
            print(f"Thread ID {thread_id} not found.")
            return
    else:
        # Largest by default
        g = max(graphs, key=lambda graph: len(graph.nodes))
    
    # Define output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "visualizations" / f"viz_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tid = g.graph['thread_id']
    print(f"Visualizing cascade {tid} with {len(g.nodes)} nodes.")
    
    # Calculate layout once (k controls optimal distance between nodes)
    # Default is optimal distance. By dividing it by a much smaller factor
    # for larger graphs, we force the nodes to strongly repel and space out.
    optimal_distance = 1.0 / (len(g.nodes)**0.5) if len(g.nodes) > 0 else 0.15
    pos = nx.spring_layout(g, k=optimal_distance * 4, iterations=50)
    
    # ---------
    # 1. Static Plot (Matplotlib)
    # ---------
    plt.figure(figsize=(10, 8))
    node_colors = ['#ff4d4d' if data.get('is_source', False) else '#4da6ff' for node, data in g.nodes(data=True)]
    
    nx.draw_networkx(
        g, pos, node_color=node_colors, with_labels=False, 
        node_size=50, edge_color='gray', alpha=0.7, arrows=True, arrowsize=10
    )
    
    label_str = 'Rumour' if g.graph['label'] == 1 else 'Non-Rumour'
    plt.title(f"Thread ID: {tid} | Label: {label_str}")
    plt.axis('off')
    
    static_file = out_dir / f"cascade_{tid}_static.png"
    plt.savefig(static_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------
    # 2. Interactive Plot (PyVis)
    # ---------
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    for node, data in g.nodes(data=True):
        color = "#ff4d4d" if data.get('is_source') else "#4da6ff"
        title = data.get('text', 'No text mapping available')
        size = 20 if data.get('is_source') else 10
        # Pre-assign coordinates and scale them so PyVis doesn't need physics layout
        x, y = pos[node][0] * 800, pos[node][1] * 800
        net.add_node(node, label=str(node), title=title, color=color, size=size, x=x, y=y)
        
    for source, target in g.edges():
        net.add_edge(source, target, color="#aaaaaa")
        
    # Disable physics entirely. Nodes will be static where we placed them, but you can still drag them freely.
    net.toggle_physics(False)
    
    html_file = out_dir / f"cascade_{tid}_interactive.html"
    net.save_graph(str(html_file))
    
    print(f"Static visualization saved to {static_file}")
    print(f"Interactive visualization saved to {html_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a cascade")
    parser.add_argument("--thread-id", type=str, help="Specific thread ID to visualize")
    parser.add_argument("--random", action="store_true", help="Pick a random cascade to visualize")
    args = parser.parse_args()
    
    visualize_cascade(thread_id=args.thread_id, random_pick=args.random)
