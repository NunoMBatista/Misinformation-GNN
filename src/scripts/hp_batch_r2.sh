#!/usr/bin/env bash
# Round 2 HP trials — exploit best R1 configs and explore capacity/LR space
# Usage: bash src/scripts/hp_batch_r2.sh
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

# ── Round 2: Capacity expansion + LR exploitation ─────────────────────────────

# gin_text: R1=0.7575 (LR=1.5e-3, [128,64], original, gamma=2.0)
# → try wider, try slightly higher LR
run_trial gin configs/benchmark_text_only.yml gin_text 2 \
  '{"hidden_dims":[256,128],"dropout":0.45,"learning_rate":0.002,"weight_decay":0.0002,"focal_gamma":2.0,"edge_direction":"original"}'

run_trial gin configs/benchmark_text_only.yml gin_text 3 \
  '{"hidden_dims":[128,64,32],"dropout":0.5,"learning_rate":0.0015,"weight_decay":0.0002,"focal_gamma":2.5,"edge_direction":"original"}'

# gat_text: R1=0.7409 (LR=1e-4, [128,64], heads=2, gamma=2.5)
# → try wider + more heads
run_trial improved_gat configs/benchmark_text_only.yml gat_text 2 \
  '{"hidden_dims":[256,128],"dropout":0.4,"learning_rate":0.00015,"weight_decay":0.00005,"focal_gamma":2.5,"edge_direction":"original","heads":4}'

# gnn_text: R1=0.7380 (LR=1e-4, [128,64], bidir, gamma=2.5)
# → higher LR + wider
run_trial improved_gnn configs/benchmark_text_only.yml gnn_text 2 \
  '{"hidden_dims":[256,128],"dropout":0.4,"learning_rate":0.0003,"weight_decay":0.0001,"focal_gamma":2.5,"edge_direction":"bidirectional"}'

# gin_full: R1=0.7424 (LR=2e-3, [128,64], bidir, gamma=2.5)
# → mirror gin_text R1 config (original, LR=1.5e-3) — bidir+high LR hurt it
run_trial gin configs/benchmark_full.yml gin_full 2 \
  '{"hidden_dims":[128,64],"dropout":0.5,"learning_rate":0.0015,"weight_decay":0.0002,"focal_gamma":2.0,"edge_direction":"original"}'

# gat_full: R1=0.7449 (LR=1e-4, [128,64], heads=2, gamma=3.5)
# → reduce focal (3.5 too aggressive), widen, try heads=4
run_trial improved_gat configs/benchmark_full.yml gat_full 2 \
  '{"hidden_dims":[256,128],"dropout":0.35,"learning_rate":0.00015,"weight_decay":0.00005,"focal_gamma":2.5,"edge_direction":"original","heads":4}'

# gnn_full: R1=0.7392 (LR=1e-4, [128,64], bidir, gamma=2.0)
# → higher LR + wider
run_trial improved_gnn configs/benchmark_full.yml gnn_full 2 \
  '{"hidden_dims":[256,128],"dropout":0.4,"learning_rate":0.0003,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional"}'

# mlp_text: R1=0.7098 (LR=1.5e-4, [128,64], gamma=2.0)
# → much higher LR + wider (MLP needs more capacity without graph structure)
run_trial mlp configs/benchmark_text_only.yml mlp_text 2 \
  '{"hidden_dims":[256,128],"dropout":0.4,"learning_rate":0.0004,"weight_decay":0.00005,"focal_gamma":2.5}'

# mlp_full: R1=0.7086
run_trial mlp configs/benchmark_full.yml mlp_full 2 \
  '{"hidden_dims":[256,128],"dropout":0.4,"learning_rate":0.0004,"weight_decay":0.00005,"focal_gamma":2.5}'

# Structural: all ~0.47-0.50 — widen + lower dropout + higher LR
run_trial improved_gnn configs/benchmark_structural_only.yml gnn_struct 2 \
  '{"hidden_dims":[32,16],"dropout":0.15,"learning_rate":0.003,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional"}'

run_trial improved_gat configs/benchmark_structural_only.yml gat_struct 2 \
  '{"hidden_dims":[32,16],"dropout":0.15,"learning_rate":0.0015,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional","heads":2}'

run_trial gin configs/benchmark_structural_only.yml gin_struct 2 \
  '{"hidden_dims":[32,16],"dropout":0.2,"learning_rate":0.002,"weight_decay":0.0001,"focal_gamma":2.0,"edge_direction":"bidirectional"}'

run_trial mlp configs/benchmark_structural_only.yml mlp_struct 2 \
  '{"hidden_dims":[32,16],"dropout":0.15,"learning_rate":0.002,"weight_decay":0.0001,"focal_gamma":2.0}'

echo "=== Round 2 complete ===" | tee -a "$LOG"
