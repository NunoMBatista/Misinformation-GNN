#!/bin/bash
# Master sequential script - runs all remaining HP searches one after another
# Runs one trial at a time to avoid Windows paging file exhaustion
cd C:/Users/Usuario/git/Misinformation-GNN

echo "=== Starting all remaining HP searches ==="
echo "Order: gat_struct → gin_struct → mlp_text → gnn_text → gat_text → gin_text"
echo "Started at: $(date)"

# ---- GAT structural ----
echo ""
echo "=== GAT structural ==="
bash run_gat_struct.sh

# Update best configs after each model
echo "Updating best configs after gat_struct..."
env/Scripts/python.exe src/scripts/update_best_configs.py --key gat_struct --session gat_struct --model improved_gat --config configs/benchmark_structural_only.yml

# ---- GIN structural ----
echo ""
echo "=== GIN structural ==="
bash run_gin_struct.sh

echo "Updating best configs after gin_struct..."
env/Scripts/python.exe src/scripts/update_best_configs.py --key gin_struct --session gin_struct --model gin --config configs/benchmark_structural_only.yml

# ---- MLP text-only ----
echo ""
echo "=== MLP text-only ==="
bash run_mlp_text.sh

echo "Updating best configs after mlp_text..."
env/Scripts/python.exe src/scripts/update_best_configs.py --key mlp_text --session mlp_text --model mlp --config configs/benchmark_text_only.yml

# ---- GNN text-only ----
echo ""
echo "=== GNN text-only ==="
bash run_gnn_text.sh

echo "Updating best configs after gnn_text..."
env/Scripts/python.exe src/scripts/update_best_configs.py --key gnn_text --session gnn_text --model improved_gnn --config configs/benchmark_text_only.yml

# ---- GAT text-only ----
echo ""
echo "=== GAT text-only ==="
bash run_gat_text.sh

echo "Updating best configs after gat_text..."
env/Scripts/python.exe src/scripts/update_best_configs.py --key gat_text --session gat_text --model improved_gat --config configs/benchmark_text_only.yml

# ---- GIN text-only ----
echo ""
echo "=== GIN text-only ==="
bash run_gin_text.sh

echo "Updating best configs after gin_text..."
env/Scripts/python.exe src/scripts/update_best_configs.py --key gin_text --session gin_text --model gin --config configs/benchmark_text_only.yml

echo ""
echo "=== All HP searches complete! ==="
echo "Finished at: $(date)"
echo ""
echo "Final best_configs.json:"
cat outputs/hp_search/best_configs.json
