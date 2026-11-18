# ULGAN v0.2 — Model Card

## Intended use
Research prototype for compact task-conditioned recurrent learning on binary sequence tasks.

## Tasks
Parity, strict majority, and last-bit prediction, selected by a task embedding. A shared recurrent core is paired with task-specific output heads.

## Actual training run
- Seed: 42
- Device: CPU
- Epochs: 8; steps per epoch: 80; batch size: 128
- Training examples presented: 81,920
- Trainable parameters: 17,980
- Seen-length evaluation (1–16): 1,000 examples/task; macro accuracy 100.0%
- Unseen-length evaluation (17–32): 1,000 examples/task; macro accuracy 98.93%
- Unseen-length majority accuracy: 96.8%; parity and last-bit accuracy: 100.0%
- EOS halt detection: 100% in both test ranges

See `artifacts/metrics.json` for complete epoch history and per-task measurements.

## Limitations
- Synthetic tasks only; no real-world benchmark or language modeling.
- One random seed; statistical robustness is not established.
- EOS halting detects sequence termination. It is not adaptive reasoning-depth or learned compute allocation.
- No baseline comparison, device latency, energy, or peak-memory profiling has been completed.
- The reported length extrapolation is limited to 17–32 tokens.
