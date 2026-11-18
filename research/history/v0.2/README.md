# ULGAN v0.2

**Erlic AI — compact task-conditioned recurrent model**

ULGAN v0.2 is a trained research prototype. A shared recurrent GRU processes binary sequences, while a task embedding conditions the recurrent computation and separate small output heads produce task-specific predictions.

## Supported tasks

- `parity`: 1 when the number of ones is odd.
- `majority`: 1 when ones are strictly more than half of the sequence.
- `last_bit`: returns the final bit.

## Train

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

- Token embedding: 3 tokens (0, 1, EOS)
- Task embedding: 3 task IDs
- Shared GRUCell recurrent core (hidden size 48)
- Three task-specific binary output heads
- Shared EOS halt head

## Current measured run

Seed 42, CPU, 8 epochs, 80 steps/epoch, batch size 128; 81,920 training examples presented. See `artifacts/metrics.json` for per-task results.

- Seen lengths 1–16: 100.0% macro accuracy (1,000 examples per task)
- Unseen lengths 17–32: 98.93% macro accuracy (1,000 examples per task); majority 96.8%, parity 100%, last-bit 100%
- EOS halt detection: 100% across both reported test ranges
- Trainable parameters: 17,980

These are synthetic-task results from one seeded run, not evidence of general intelligence or language ability. EOS halting is termination detection, not adaptive reasoning-depth.
