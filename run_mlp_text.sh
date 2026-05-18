#!/bin/bash
# Sequential MLP text-only HP search
cd C:/Users/Usuario/git/Misinformation-GNN

PYTHON=env/Scripts/python.exe
MODEL=mlp
CONFIG=configs/benchmark_text_only.yml
SESSION=mlp_text

echo "Starting sequential MLP text-only HP search..."

# Guided trials - text-only (386-dim input), similar to full MLP but might prefer smaller nets
$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 1 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1.6e-4,"dropout":0.6,"focal_gamma":2.0,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 2 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":1.6e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 3 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.5,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 4 \
  --hparams '{"hidden_dims":[64,32,16],"learning_rate":1.6e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 5 \
  --hparams '{"hidden_dims":[128,64,32],"learning_rate":1e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 6 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":2e-4,"dropout":0.6,"focal_gamma":2.0,"weight_decay":5e-5}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 7 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":1.6e-4,"dropout":0.4,"focal_gamma":3.0,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 8 \
  --hparams '{"hidden_dims":[128,64],"learning_rate":1e-4,"dropout":0.6,"focal_gamma":2.5,"weight_decay":2e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 9 \
  --hparams '{"hidden_dims":[32,16],"learning_rate":2e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'

$PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial 10 \
  --hparams '{"hidden_dims":[64,32],"learning_rate":3e-4,"dropout":0.5,"focal_gamma":2.0,"weight_decay":1e-4}'

# Random exploration - trials 11-52
for i in $(seq 11 52); do
  $PYTHON src/scripts/hp_search.py --model $MODEL --config $CONFIG --session $SESSION --trial $i --random
done

echo "Done!"
wc -l outputs/hp_search/mlp_mlp_text.jsonl
