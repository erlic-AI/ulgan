# ULGAN v0.3

**Erlic AI — Adaptive Computation research prototype**

ULGAN v0.3 adds an Adaptive Computation Time (ACT)-style latent reasoning loop to the task-conditioned recurrent model. After encoding the complete sequence, a recurrent thinker performs up to eight latent refinement steps. A learned halting controller assigns probability mass to stopping at each step. Training optimizes task loss plus an expected-computation (ponder) penalty.

## Tasks
- `parity`: odd/even count of ones (odd -> 1)
- `majority`: 1 when ones are strictly more than half
- `last_bit`: final bit in the sequence

## Install and train
```bash
pip install -r requirements.txt
python train.py
```

## Predict
```bash
python predict.py parity 1 0 1 1
python predict.py majority 1 0 1 1
python predict.py last_bit 1 0 1 1
```

## Architecture
- Token embedding: 0, 1, EOS
- Task embedding shared by encoder and thinker
- GRUCell sequence encoder
- GRUCell latent thinker, maximum 8 steps
- Learned scalar halting controller
- Shared binary output head
- ACT-style probability-weighted output and ponder penalty

## Evaluation protocol
`train.py` reports adaptive-threshold inference and fixed-eight-step inference on the same task families, separately for training-range lengths (1–16) and held-out lengths (17–32). The metrics file records seed, sample counts, parameter count, training time, and per-task metrics. This is a small synthetic benchmark, not evidence of general reasoning or language ability. Adaptive computation is only demonstrated if adaptive inference maintains accuracy while using fewer latent steps than the fixed-step baseline.

## Reproducibility
```bash
python train.py --seed 42 --epochs 8 --steps-per-epoch 80 --batch-size 96
```

## Measured reference run (CPU)

Reference configuration: seed 42, 10 epochs, 80 steps/epoch, batch size 128, ponder weight 0.0002; 102,400 training examples presented; 21,596 trainable parameters. The held-out evaluation used 300 examples per task in each length range.

| Evaluation | Adaptive accuracy | Adaptive steps | Fixed-8 accuracy | Fixed steps |
|---|---:|---:|---:|---:|
| Lengths 1–16 | 93.78% | 5.21 | 94.11% | 8.00 |
| Lengths 17–32 | 86.89% | 5.12 | 82.67% | 8.00 |

Adaptive inference used about 35% fewer latent steps on the seen-length test and 36% fewer on the unseen-length test. Macro accuracy was close to fixed computation on seen lengths and higher on unseen lengths. Important weakness: parity accuracy was 83.3% on lengths 1–16 but only 65.3% on lengths 17–32. This is a working adaptive-computation prototype, **not** a solved general reasoning system. See `artifacts/metrics.json` for per-task details.
