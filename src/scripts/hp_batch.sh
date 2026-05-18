#!/usr/bin/env bash
# Runs one HP trial per line and appends clean results to outputs/hp_search/hp_log.txt
# Usage: bash src/scripts/hp_batch.sh
set -e
PY="./env/Scripts/python.exe"
SCRIPT="src/scripts/hp_search.py"
LOG="outputs/hp_search/hp_log.txt"
mkdir -p outputs/hp_search

run_trial() {
    local model=$1 config=$2 session=$3 trial=$4 hparams=$5
    echo ">>> $session trial $trial" | tee -a "$LOG"
    PYTHONIOENCODING=utf-8 "$PY" "$SCRIPT" \
        --model "$model" --config "$config" \
        --session "$session" --trial "$trial" \
        --hparams "$hparams" --no-wandb 2>&1 \
      | grep -E "^Trial|^Hparams|^Saved|F1=|Error|Traceback" \
      | tee -a "$LOG"
    echo "" >> "$LOG"
}

# ── Round 1: Intelligent warm starts ──────────────────────────────────────────
# Text-only (386 dims) — highest priority, likely to dominate

run_trial improved_gnn configs/benchmark_text_only.yml gnn_text 1 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.0001,"weight_decay":0.0001,"focal_gamma":2.5,"edge_direction":"bidirectional"}'

run_trial improved_gat configs/benchmark_text_only.yml gat_text 1 \
  '{"hidden_dims":[128,64],"dropout":0.45,"learning_rate":0.0001,"weight_decay":0.00005,"focal_gamma":2.5,"edge_direction":"original","heads":2}'

run_trial gin configs/benchmark_text_only.yml gin_text 1 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.0015,"weight_decay":0.0002,"focal_gamma":2.0,"edge_direction":"original"}'

run_trial mlp configs/benchmark_text_only.yml mlp_text 1 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.00015,"weight_decay":0.00005,"focal_gamma":2.0}'

# Full features (392 dims)

run_trial improved_gnn configs/benchmark_full.yml gnn_full 1 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.0001,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional"}'

run_trial improved_gat configs/benchmark_full.yml gat_full 1 \
  '{"hidden_dims":[128,64],"dropout":0.4,"learning_rate":0.0001,"weight_decay":0.00001,"focal_gamma":3.5,"edge_direction":"original","heads":2}'

run_trial gin configs/benchmark_full.yml gin_full 1 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.002,"weight_decay":0.0003,"focal_gamma":2.5,"edge_direction":"bidirectional"}'

run_trial mlp configs/benchmark_full.yml mlp_full 1 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.00015,"weight_decay":0.0001,"focal_gamma":2.0}'

# Structural (8 dims)

run_trial improved_gnn configs/benchmark_structural_only.yml gnn_struct 1 \
  '{"hidden_dims":[16,8],"dropout":0.3,"learning_rate":0.001,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional"}'

run_trial improved_gat configs/benchmark_structural_only.yml gat_struct 1 \
  '{"hidden_dims":[16,8],"dropout":0.3,"learning_rate":0.0005,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional","heads":2}'

run_trial gin configs/benchmark_structural_only.yml gin_struct 1 \
  '{"hidden_dims":[32,16,8],"dropout":0.35,"learning_rate":0.001,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional"}'

run_trial mlp configs/benchmark_structural_only.yml mlp_struct 1 \
  '{"hidden_dims":[16,8],"dropout":0.3,"learning_rate":0.0005,"weight_decay":0.0001,"focal_gamma":2.0}'

echo "=== Round 1 complete ===" | tee -a "$LOG"
