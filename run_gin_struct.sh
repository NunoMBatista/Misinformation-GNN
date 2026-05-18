#!/bin/bash
# Sequential GIN structural HP search
cd C:/Users/Usuario/git/Misinformation-GNN

PYTHON=env/Scripts/python.exe
MODEL=gin
CONFIG=configs/benchmark_structural_only.yml
SESSION=gin_struct

echo "Starting sequential GIN structural HP search..."

# Guided trials - GIN needs higher LR (~3e-3) due to sum aggregation
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":3e-3,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":3e-3,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[32,16,8],"learning_rate":3e-3,"dropout":0.4,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":2e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":4e-3,"dropout":0.3,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":2e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":2e-3,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"original","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[32,16,8],"learning_rate":2e-3,"dropout":0.5,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":5e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":5e-3,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 9 \
  --hparams '{"hidden_dims":[16,8],"learning_rate":3e-3,"dropout":0.3,"focal_gamma":1.5,"edge_direction":"bidirectional","weight_decay":2e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 10 \
  --hparams '{"hidden_dims":[8,4],"learning_rate":3e-3,"dropout":0.6,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

# Random exploration - trials 11-52
for i in $(seq 11 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

echo "Done!"
wc -l outputs/hp_search/gin_gin_struct.jsonl
