#!/bin/bash
# Sequential GAT structural HP search
cd C:/Users/Usuario/git/Misinformation-GNN

PYTHON=env/Scripts/python.exe
MODEL=improved_gat
CONFIG=configs/benchmark_structural_only.yml
SESSION=gat_struct

echo "Starting sequential GAT structural HP search..."

# Guided trials - similar to gnn_struct but GAT-specific (low LR, heads=1)
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":6e-5,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":1e-4,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"original","weight_decay":5e-5,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[32,16,8],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":5e-5,"dropout":0.6,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":2e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":8e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":2}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":6e-5,"dropout":0.3,"focal_gamma":1.5,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 9 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":5e-5,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 10 \
  --hparams '{"hidden_dims":[32,16],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

# Random exploration - trials 11-52
for i in $(seq 11 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

echo "Done!"
wc -l outputs/hp_search/improved_gat_gat_struct.jsonl
