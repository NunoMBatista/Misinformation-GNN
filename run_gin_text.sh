#!/bin/bash
# Sequential GIN text-only HP search (386-dim input)
cd C:/Users/Usuario/git/Misinformation-GNN

PYTHON=env/Scripts/python.exe
MODEL=gin
CONFIG=configs/benchmark_text_only.yml
SESSION=gin_text

echo "Starting sequential GIN text-only HP search..."

# Guided trials - GIN needs higher LR (~3e-3), larger networks for 386-dim
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":2e-3,"dropout":0.6,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":4e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":5e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":3e-3,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":2e-3,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":3.0,"edge_direction":"bidirectional","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":3e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"original","weight_decay":4e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 9 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":3e-3,"dropout":0.3,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":2e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 10 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":5e-3,"dropout":0.5,"focal_gamma":2.5,"edge_direction":"bidirectional","weight_decay":4e-4}'

# Random exploration - trials 11-52
for i in $(seq 11 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

echo "Done!"
wc -l outputs/hp_search/gin_gin_text.jsonl
