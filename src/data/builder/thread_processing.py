import json
import networkx as nx
from pathlib import Path
from tqdm import tqdm

def load_tweet_json(file_path):
    """Safely loads a single tweet JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def process_thread(thread_dir, label):
    """
    Parses a single directory into a NetworkX directed graph.
    Extracts raw features (text, followers, verification) and connections.
    """
    G = nx.DiGraph()
    G.graph['label'] = 1 if label == 'rumours' else 0
    G.graph['thread_id'] = thread_dir.name
    G.graph['event'] = thread_dir.parent.parent.name
    
    # 1. Source tweet processing
    source_dir = thread_dir / "source-tweets"
    if not source_dir.exists():
        source_dir = thread_dir / "source-tweet"
        
    source_files = list(source_dir.glob("*.json"))
    if not source_files:
        return None
        
    source_tweet = load_tweet_json(source_files[0])
    if not source_tweet: return None
    
    source_id = source_tweet["id_str"]
    G.add_node(
        source_id, 
        text=source_tweet.get("text", ""),
        user_followers=source_tweet["user"]["followers_count"],
        user_verified=int(source_tweet["user"]["verified"]),
        is_source=True
    )
    
    # 2. Reaction tweets processing
    reactions_dir = thread_dir / "reactions"
    if reactions_dir.exists():
        for reaction_file in reactions_dir.glob("*.json"):
            reaction_tweet = load_tweet_json(reaction_file)
            if not reaction_tweet: continue
            
            reac_id = reaction_tweet["id_str"]
            reply_to_id = reaction_tweet.get("in_reply_to_status_id_str")
            
            G.add_node(
                reac_id,
                text=reaction_tweet.get("text", ""),
                user_followers=reaction_tweet["user"]["followers_count"],
                user_verified=int(reaction_tweet["user"]["verified"]),
                is_source=False
            )
            
            # Add directed edge mapping information flow
            if reply_to_id:
                G.add_edge(reply_to_id, reac_id)
            else:
                G.add_edge(source_id, reac_id) # Attach to source if reply target is missing
                
    return G

def parse_all_threads(data_dir: Path):
    """Loops through the entire dataset returning a list of extracted cascades."""
    graphs = []
    events = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "README"]
    
    # Pre-collecting thread directories for an accurate progress bar
    thread_dirs = []
    for event in events:
        for label in ["rumours", "non-rumours"]:
            label_dir = event / label
            if label_dir.exists():
                for thread_dir in label_dir.iterdir():
                    if thread_dir.is_dir():
                        thread_dirs.append((thread_dir, label))
    
    for thread_dir, label in tqdm(thread_dirs, desc="Parsing Threads"):
        G = process_thread(thread_dir, label)
        if G is not None and len(G.nodes) > 0:
            graphs.append(G)
                    
    return graphs
