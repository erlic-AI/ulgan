# Experimental methodology

## v0.4 reference protocol

- Seed: 42
- Device: CPU
- Training: 8 epochs × 40 steps × 128 examples
- Presented training examples: 40,960
- Training sequence lengths: 1–64
- Tasks: parity, majority, last_bit
- Evaluation: 500 generated examples per task in each range
- Seen-length evaluation: 1–64
- Held-out-length evaluation: 65–128
- Adaptive policy: cumulative halt threshold 0.95, maximum 8 latent steps
- Baseline: same checkpoint evaluated at fixed step 8

## Reporting rules

Report each task separately, macro accuracy, mean latent steps, and the fixed-step baseline. Keep seen and held-out ranges separate. Do not describe latent-step reduction as equivalent to wall-clock speedup unless runtime is measured. Do not infer statistical significance from one seed.

## Historical experiment records

The original research archive supplied for this rebuild contains executable snapshots for v0.1–v0.4 and the v0.4 metrics file. It does not contain the full raw logs or result tables for experiments 1–70. Any experiment notes not backed by a raw record are identified as provisional in `experiments/` rather than presented as reproduced benchmarks.
