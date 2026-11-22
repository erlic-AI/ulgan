# Historical pre-density pilot — not release benchmark

These runs predate the selected density-feature architecture and are retained only as historical context. They are not directly comparable with the current development benchmark. They are intentionally much smaller than the release protocol and must not be presented as final performance.

## Settings

- Seeds: 11, 23, 42
- Epochs: 2
- Steps per epoch: 10
- Batch size: 32
- Training maximum length: 128
- Evaluation: 256 examples per task in each range
- Device: CPU
- Prefix auxiliary loss weight: 0.25 in the enabled run
- Matched ablation: identical settings with prefix supervision disabled

## Pilot results

| Condition | Adaptive seen macro accuracy | Adaptive unseen macro accuracy | Seen mean thinking steps | Unseen mean thinking steps |
|---|---:|---:|---:|---:|
| Prefix supervision on | 0.7001 ± 0.0290 | 0.6723 ± 0.0189 | 3.0985 ± 0.1539 | 3.0807 ± 0.1416 |
| Prefix supervision off | 0.7005 ± 0.0293 | 0.6740 ± 0.0164 | 3.1050 ± 0.1480 | 3.0851 ± 0.1374 |

Mean ± sample standard deviation across the three seeds. Evaluation sample size is small and training is only 2 epochs; no statistically meaningful improvement is established. In this pilot, prefix supervision did not improve macro accuracy. Full-budget experiments and ablations are required before deciding whether to retain this intervention.

## Task-level pilot results (prefix supervision on)

| Range | Parity | Majority | Last bit |
|---|---:|---:|---:|
| Seen 1–128 | 0.4948 ± 0.0870 | 0.6055 ± 0.0000 | 1.0000 ± 0.0000 |
| Unseen 129–256 | 0.4740 ± 0.0352 | 0.5430 ± 0.0217 | 1.0000 ± 0.0000 |

## Decision

Do not claim a performance improvement and do not publish pilot checkpoints as release weights. Complete `scripts/run_release_experiments.sh`, inspect all generated metrics, and only then decide the final architecture and release contents.
