import os
import json
import networkx as nx
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/pheme-rnr-dataset")
OUTPUT_DIR = Path("data/processed")

def load_tweet_json(file_path):
    """Loads a single tweet JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def process_thread(thread_dir, label):
    """
    Processes a single thread directory and returns a NetworkX graph.
    
    Nodes represent tweets, storing content and user metadata.
    Edges represent reply/retweet relationships forming a cascade.
    """
    G = nx.DiGraph()
    G.graph['label'] = 1 if label == 'rumours' else 0 # 1 for rumour, 0 for non-rumour
    G.graph['thread_id'] = thread_dir.name
    
    # 1. Load source tweet
    source_dir = thread_dir / "source-tweet"
    source_files = list(source_dir.glob("*.json"))
    if not source_files:
        return None
        
    source_tweet = load_tweet_json(source_files[0])
    if not source_tweet: return None
    
    source_id = source_tweet["id_str"]
    
    # Add source node
    G.add_node(
        source_id, 
        text=source_tweet.get("text", ""),
        user_followers=source_tweet["user"]["followers_count"],
        user_verified=int(source_tweet["user"]["verified"]),
        is_source=True
    )
    
    # 2. Load reaction tweets
    reactions_dir = thread_dir / "reactions"
    if reactions_dir.exists():
        for reaction_file in reactions_dir.glob("*.json"):
            reaction_tweet = load_tweet_json(reaction_file)
            if not reaction_tweet: continue
            
            reac_id = reaction_tweet["id_str"]
            reply_to_id = reaction_tweet.get("in_reply_to_status_id_str")
            
            # Add reaction node
            G.add_node(
                reac_id,
                text=reaction_tweet.get("text", ""),
                user_followers=reaction_tweet["user"]["followers_count"],
                user_verified=int(reaction_tweet["user"]["verified"]),
                is_source=False
            )
            
            # Add edge (reply_to -> reaction) indicating information flow
            if reply_to_id:
                G.add_edge(reply_to_id, reac_id)
            else:
                # If no explicit reply_to, attach to source as a fallback for cascade
                G.add_edge(source_id, reac_id)
                
    return G

def main():
    """Main pipeline to process all events and extract cascade graphs."""
    print("Starting data processing...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    graphs = []
    events = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "README"]
    
    for event in events:
        print(f"Processing event: {event.name}")
        for label in ["rumours", "non-rumours"]:
            label_dir = event / label
            if not label_dir.exists(): continue
            
            for thread_dir in label_dir.iterdir():
                if not thread_dir.is_dir(): continue
                
                G = process_thread(thread_dir, label)
                if G is not None and len(G.nodes) > 0:
                    graphs.append(G)
                    
    print(f"Successfully processed {len(graphs)} rumour threads (cascades).")
    
    # Example: saving processed data as picke
    output_file = OUTPUT_DIR / "pheme_cascades.gpickle"
    # nx.write_gpickle is deprecated in newer networkx versions, using pickle instead
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(graphs, f)
        
    print(f"Saved processed graph cascades to {output_file}")


if __name__ == "__main__":
    main()
