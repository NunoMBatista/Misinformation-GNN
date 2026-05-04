import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_dataset():
    """
    Downloads the pre-processed PHEME PyTorch Geometric and NetworkX datasets
    directly from Hugging Face into the local data/processed directory.
    """
    repo_id = "NunoBatista/PHEME-Misinformation-Graphs"
    local_dir = Path("data/processed")
    
    # Ensure directory exists
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset from Hugging Face: {repo_id}...")
    
    # Download PyTorch Geometric Tensor file
    print("Downloading PyG tensors (pheme_pyg_dataset.pt)...")
    hf_hub_download(
        repo_id=repo_id,
        filename="pheme_pyg_dataset.pt",
        local_dir=local_dir,
        repo_type="dataset"
    )
    
    # Download NetworkX graphs
    print("Downloading NetworkX graphs (pheme_cascades.gpickle)...")
    hf_hub_download(
        repo_id=repo_id,
        filename="pheme_cascades.gpickle",
        local_dir=local_dir,
        repo_type="dataset"
    )
    
    print(f"\nSuccess! Dataset successfully downloaded to: {local_dir.absolute()}")
    print("Ready to run experiments: python src/scripts/run_experiments.py --config configs/experiment.yml")

if __name__ == "__main__":
    download_dataset()
