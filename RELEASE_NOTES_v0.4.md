# ulgan v0.4 — Research Release

This release packages the v0.4 adaptive-computation prototype with a reorganized source layout, reference checkpoint, benchmark measurements, reproducibility configuration, model card, and historical development snapshots.

## Highlights
- 34,908 trainable parameters.
- Task-conditioned GRU encoder and recurrent latent thinker.
- Learned halting with a maximum of eight latent steps.
- Synthetic tasks: parity, majority, and last-bit.
- Reference evaluation on lengths 1–64 and held-out lengths 65–128.

## Reference measurements

| Range | Adaptive macro accuracy | Mean latent steps | Fixed-8 accuracy |
|---|---:|---:|---:|
| 1–64 | 80.67% | 2.11 | 83.27% |
| 65–128 | 80.27% | 2.06 | 79.60% |

Single seed (42), CPU. Parity remains weak. These are prototype measurements, not a general reasoning benchmark.

## Validation
- Smoke tests: 2 passed.
- CLI prediction commands: parity and majority executed successfully.
- SHA256 manifest: verified.
