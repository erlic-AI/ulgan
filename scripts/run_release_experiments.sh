#!/usr/bin/env bash
set -euo pipefail

# Revised attention + ACT candidate. Uses the existing Python/PyTorch environment;
# installs nothing. Ablations are matched on deterministic evaluation examples.
python scripts/train.py --seeds 11 23 42 --epochs 20 --steps-per-epoch 100 \
  --batch-size 128 --train-max-length 128 --eval-samples 2000 \
  --disable-prefix-supervision --prefix-weight 0.0 \
  --out artifacts/ulgan_1_attention_act/prefix_off

python scripts/train.py --seeds 11 23 42 --epochs 20 --steps-per-epoch 100 \
  --batch-size 128 --train-max-length 128 --eval-samples 2000 \
  --prefix-weight 0.25 \
  --out artifacts/ulgan_1_attention_act/prefix_025
