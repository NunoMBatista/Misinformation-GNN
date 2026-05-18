#!/bin/bash
# Sequential GNN structural HP search - runs one trial at a time to avoid CUDA memory issues
cd C:/Users/Usuario/git/Misinformation-GNN

PYTHON=env/Scripts/python.exe
MODEL=improved_gnn
CONFIG=configs/benchmark_structural_only.yml
SESSION=gnn_struct

echo "Starting sequential GNN structural HP search..."

# Guided trials - cover key configurations
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":6e-5,"dropout":0.4,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":1e-4,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":5e-5}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":7e-5,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[32,16,8],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":5e-5,"dropout":0.6,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":2e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 9 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"original","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 10 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":8e-5,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 11 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":6e-5,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 12 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":1.5,"edge_direction":"bidirectional","weight_decay":5e-5}'

# Random exploration - trials 13-52
for i in $(seq 13 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

echo "Done! Total trials:"
wc -l outputs/hp_search/improved_gnn_gnn_struct.jsonl
