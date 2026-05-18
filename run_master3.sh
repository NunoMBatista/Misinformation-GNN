#!/bin/bash
# Master HP search v3 — continues from gat_struct (2 done) through gin_text
# WANDB_MODE=offline prevents network hang on WinError 64
# Correct GAT param: "heads" not "num_heads"

cd C:/Users/Usuario/git/Misinformation-GNN
PYTHON=env/Scripts/python.exe

export WANDB_MODE=offline

echo "=== Master HP Search v3 ==="
echo "Started at: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"

# ─────────────────────────────────────────────────────────
# 1. GAT STRUCTURAL — continue (2 done, need 48+ more → run 50 fresh random)
# ─────────────────────────────────────────────────────────
echo ""
echo "=== GAT structural (50 more random trials) ==="
MODEL=improved_gat
CONFIG=configs/benchmark_structural_only.yml
SESSION=gat_struct

# Guided — correct param is "heads" not "num_heads"
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 101 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.17,"focal_gamma":3.14,"edge_direction":"original","weight_decay":1.4e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 102 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.3,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 103 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":5e-5,"dropout":0.2,"focal_gamma":2.5,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 104 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":1e-4,"dropout":0.2,"focal_gamma":2.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 105 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":7e-5,"dropout":0.2,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 106 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":1e-3,"dropout":0.5,"focal_gamma":2.4,"edge_direction":"bidirectional","weight_decay":1.4e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 107 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.2,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 108 \
  --hparams '{"hidden_dims":[32,16],"learning_rate":7e-5,"dropout":0.2,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'

# Random exploration (trials 109-152 → 44 random, gives 2+8+44=54 total)
for i in $(seq 109 152); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

$PYTHON src/scripts/update_best_configs.py --key gat_struct --session $SESSION --model $MODEL --config $CONFIG
echo "gat_struct done at $(date)"

# ─────────────────────────────────────────────────────────
# 2. GIN STRUCTURAL (50 trials)
# ─────────────────────────────────────────────────────────
echo ""
echo "=== GIN structural (50 trials) ==="
MODEL=gin
CONFIG=configs/benchmark_structural_only.yml
SESSION=gin_struct

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":3e-3,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":2e-3,"dropout":0.2,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":5e-3,"dropout":0.3,"focal_gamma":1.5,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":3e-3,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[32,16],"learning_rate":3e-3,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":1e-3,"dropout":0.2,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":5e-3,"dropout":0.1,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":2e-4}'

for i in $(seq 9 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

$PYTHON src/scripts/update_best_configs.py --key gin_struct --session $SESSION --model $MODEL --config $CONFIG
echo "gin_struct done at $(date)"

# ─────────────────────────────────────────────────────────
# 3. MLP TEXT-ONLY (50 trials)
# ─────────────────────────────────────────────────────────
echo ""
echo "=== MLP text-only (50 trials) ==="
MODEL=mlp
CONFIG=configs/benchmark_text_only.yml
SESSION=mlp_text

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1.6e-4,"dropout":0.6,"focal_gamma":2.0,"weight_decay":1.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":1.6e-4,"dropout":0.6,"focal_gamma":2.0,"weight_decay":1.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[256,128],"learning_rate":1.6e-4,"dropout":0.6,"focal_gamma":2.0,"weight_decay":2e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[64,32,16],"learning_rate":2e-4,"dropout":0.5,"focal_gamma":3.0,"weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":1.6e-4,"dropout":0.6,"focal_gamma":2.0,"weight_decay":1.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":2e-4,"dropout":0.7,"focal_gamma":1.5,"weight_decay":1e-4}'

for i in $(seq 9 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

$PYTHON src/scripts/update_best_configs.py --key mlp_text --session $SESSION --model $MODEL --config $CONFIG
echo "mlp_text done at $(date)"

# ─────────────────────────────────────────────────────────
# 4. GNN TEXT-ONLY (50 trials)
# ─────────────────────────────────────────────────────────
echo ""
echo "=== GNN text-only (50 trials) ==="
MODEL=improved_gnn
CONFIG=configs/benchmark_text_only.yml
SESSION=gnn_text

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":6e-5,"dropout":0.6,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":6e-5,"dropout":0.6,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"original","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":5e-5,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":7e-5,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":2e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":8e-5,"dropout":0.6,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":6e-5,"dropout":0.7,"focal_gamma":1.5,"edge_direction":"bidirectional","weight_decay":1e-4}'

for i in $(seq 9 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

$PYTHON src/scripts/update_best_configs.py --key gnn_text --session $SESSION --model $MODEL --config $CONFIG
echo "gnn_text done at $(date)"

# ─────────────────────────────────────────────────────────
# 5. GAT TEXT-ONLY (50 trials)
# ─────────────────────────────────────────────────────────
echo ""
echo "=== GAT text-only (50 trials) ==="
MODEL=improved_gat
CONFIG=configs/benchmark_text_only.yml
SESSION=gat_text

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":5.4e-5,"dropout":0.44,"focal_gamma":3.76,"edge_direction":"original","weight_decay":1.2e-5,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":5.4e-5,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":5e-5,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[256,128],"learning_rate":5.4e-5,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"original","weight_decay":2e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"original","weight_decay":1e-4,"heads":1}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":5.4e-5,"dropout":0.44,"focal_gamma":3.76,"edge_direction":"original","weight_decay":1.2e-5,"heads":2}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":1e-4,"heads":1}'

for i in $(seq 9 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

$PYTHON src/scripts/update_best_configs.py --key gat_text --session $SESSION --model $MODEL --config $CONFIG
echo "gat_text done at $(date)"

# ─────────────────────────────────────────────────────────
# 6. GIN TEXT-ONLY (50 trials)
# ─────────────────────────────────────────────────────────
echo ""
echo "=== GIN text-only (50 trials) ==="
MODEL=gin
CONFIG=configs/benchmark_text_only.yml
SESSION=gin_text

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":2.77e-3,"dropout":0.59,"focal_gamma":2.55,"edge_direction":"bidirectional","weight_decay":4.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":2.77e-3,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":2e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":5e-3,"dropout":0.6,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[256,128],"learning_rate":2.77e-3,"dropout":0.5,"focal_gamma":2.55,"edge_direction":"bidirectional","weight_decay":4.5e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":2e-3,"dropout":0.6,"focal_gamma":2.0,"edge_direction":"original","weight_decay":2e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[64,32,16],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":1e-4}'
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":1e-3,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":1e-4}'

for i in $(seq 9 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

$PYTHON src/scripts/update_best_configs.py --key gin_text --session $SESSION --model $MODEL --config $CONFIG
echo "gin_text done at $(date)"

# ─────────────────────────────────────────────────────────
echo ""
echo "=== ALL HP SEARCHES COMPLETE ==="
echo "Finished at: $(date)"
$PYTHON -c "
import json
cfg = json.load(open('outputs/hp_search/best_configs.json'))
for k, v in sorted(cfg.items()):
    print(f'  {k}: F1={v[\"best_trial_f1\"]:.4f}')
"
