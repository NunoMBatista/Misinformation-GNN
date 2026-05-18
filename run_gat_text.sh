#!/bin/bash
# Sequential GAT text-only HP search (386-dim input)
cd C:/Users/Usuario/git/Misinformation-GNN

PYTHON=env/Scripts/python.exe
MODEL=improved_gat
CONFIG=configs/benchmark_text_only.yml
SESSION=gat_text

echo "Starting sequential GAT text-only HP search..."

# Guided trials - low LR regime (~6e-5), heads=1 preferred
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":6e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"original","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":6e-5,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":5e-5,"dropout":0.4,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":7e-5,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":6e-5,"dropout":0.4,"focal_gamma":2.5,"edge_direction":"original","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":6e-5,"dropout":0.5,"focal_gamma":2.0,"edge_direction":"original","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":6e-5,"dropout":0.6,"focal_gamma":2.0,"edge_direction":"original","weight_decay":2e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":6e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"original","weight_decay":5e-5,"num_heads":2}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 9 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":7e-5,"dropout":0.4,"focal_gamma":2.0,"edge_direction":"bidirectional","weight_decay":1e-4,"num_heads":1}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 10 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":6e-5,"dropout":0.5,"focal_gamma":3.0,"edge_direction":"original","weight_decay":1e-4,"num_heads":1}'

# Random exploration - trials 11-52
for i in $(seq 11 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

echo "Done!"
wc -l outputs/hp_search/improved_gat_gat_text.jsonl
