# ulgan 1 — development experiment report

## Executive status

**Historical baseline note:** all numerical results in this report refer to the pre-attention, pre-ACT development architecture. They must not be attributed to the revised candidate now in `src/ulgan/model.py`. The revised candidate has passed code-level smoke tests only; its full benchmark is pending.



This is a development snapshot, not the completed reference benchmark or a performance release. The exact parity-summary shortcut from an earlier experiment was removed. It directly computed the parity label, so its 100% result is invalid as evidence of learned parity and is withdrawn. The shortcut checkpoints and associated metrics have been removed from the active artifact set.

## Selected extended development run

- Seeds: 11, 23, 42
- Device: CPU
- Epochs: 5
- Steps per epoch: 20
- Batch size: 32
- Training examples per seed: 3,200
- Training lengths: 1–128
- Evaluation: 1,024 examples per task for each range
- Evaluation ranges: seen 1–128; unseen 129–256
- Prefix-parity auxiliary supervision: disabled
- Trainable parameters: 135,977

### Adaptive inference results

| Range | Macro accuracy (mean ± sample SD) | Parity | Majority | Last bit | Mean steps (mean ± SD) |
|---|---:|---:|---:|---:|---:|
| Seen 1–128 | 73.70% ± 1.07 pp | 49.84% | 71.26% | 100.00% | 1.84 ± 0.07 |
| Unseen 129–256 | 70.14% ± 0.62 pp | 49.97% | 60.45% | 100.00% | 1.83 ± 0.08 |

### Fixed-12-step comparison

| Range | Macro accuracy | Parity | Majority | Last bit |
|---|---:|---:|---:|---:|
| Seen 1–128 | 71.63% | 49.32% | 65.56% | 100.00% |
| Unseen 129–256 | 69.14% | 49.45% | 57.98% | 100.00% |

Adaptive inference uses about 84.7% fewer thinking steps than fixed-12 inference on average in this run. The macro accuracy difference is modest and is influenced by last-bit being solved perfectly. This does not establish general adaptive-computation superiority.

## Controlled development ablations

All rows below use three seeds and 3 epochs × 10 steps × batch 16 (480 examples per seed). Prefix weights 0.0, 0.10, and 0.25 use 256 test examples per task and range. Prefix weight 0.50 uses 128 test examples per task and range, so it is not strictly comparable on evaluation precision.

| Train max length | Prefix weight | Seen macro | Seen parity | Seen majority | Unseen macro | Unseen parity | Unseen majority |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.00 | 68.49% | 48.96% | 56.51% | 68.14% | 48.83% | 55.60% |
| 128 | 0.10 | 68.49% | 48.96% | 56.51% | 68.06% | 48.83% | 55.34% |
| 128 | 0.25 | 68.53% | 48.96% | 56.64% | 68.06% | 48.83% | 55.34% |
| 128 | 0.50 | 69.70% | 48.70% | 60.42% | 68.40% | 49.22% | 55.99% |
| 64 | 0.00 | 71.66% | 48.44% | 66.54% | 68.75% | 50.52% | 55.73% |

The 0.00/0.10/0.25 results show no material parity benefit from prefix supervision. The 0.50 run has fewer evaluation examples, and all these ablations are small-budget; no causal performance claim is made. Training only to length 64 produces stronger majority on seen 1–64 but does not solve generalization to 65–128.

### Encoder direction ablation

A matched 3-seed comparison (3 epochs × 10 steps × batch 16; 256 evaluation examples per task/range) compared the bidirectional encoder with a causal/unidirectional encoder, both using density features and no prefix supervision:

| Encoder | Seen macro | Seen parity | Seen majority | Unseen macro | Unseen parity | Unseen majority | Mean steps (seen) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bidirectional | 68.49% | 48.96% | 56.51% | 68.14% | 48.83% | 55.60% | 3.11 |
| Causal | 68.01% | 47.79% | 56.25% | 68.27% | 51.30% | 53.52% | 3.58 |

The causal encoder did not improve the main accuracy bottleneck and used more adaptive steps in this pilot. The bidirectional encoder remains the selected development architecture; this is not a strong statistical conclusion.

## Historical baseline architectural and implementation changes

- Removed exact parity-summary shortcut and its projection.
- Added learned bit-density projection (ones / valid length), a non-label statistic.
- Added packed-sequence processing for valid bits plus EOS so right-padding is ignored by the GRU.
- Added regression tests for padding invariance and finite representations.
- Kept prefix predictions restricted to forward encoder states.

## Remaining limitations

1. Parity remains at chance and is the main unsolved task.
2. Majority is moderate on seen lengths and degrades on unseen lengths.
3. The full reference benchmark (20 × 100 × 128, three seeds, 2,000 test examples per task/range) has not been completed.
4. Ablations are small-budget and do not establish robust architectural conclusions.
5. Broader distributions and lengths beyond 256 are untested.

## Artifact locations

- Selected development checkpoints and aggregate metrics: `artifacts/ulgan_1/extended_main/`
- Per-seed raw metrics: `artifacts/ulgan_1/extended_seed11/`, `extended_seed23/`, `extended_seed42/`
- Matched small-budget ablations: `density_no_parity/`, `density_prefix010/`, `density_prefix025/`, `density_prefix050/`, `density_train64/`, `density_causal/`

## Revised attention/ACT candidate — small CPU pilot (preliminary)

A matched three-seed pilot (11, 23, 42) was run on CPU for 3 epochs × 10 steps × batch 16, train lengths 1–32, and 256 examples per task/range. The same deterministic evaluation examples were used for the prefix-off and prefix-0.25 conditions and across model seeds. This is a small pilot, not the full reference benchmark.

| Prefix weight | Adaptive seen macro | Adaptive seen parity | Adaptive seen majority | Adaptive unseen macro | Adaptive unseen parity | Adaptive unseen majority | Mean adaptive steps (seen) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 82.55% ± 1.26 pp | 51.17% | 96.48% | 82.07% ± 1.54 pp | 49.61% | 96.61% | 4.50 |
| 0.25 | 82.34% ± 0.89 pp | 50.52% | 96.48% | 81.94% ± 1.66 pp | 49.35% | 96.48% | 4.50 |

Last-bit adaptive accuracy was 100% in both ranges in both conditions. Prefix weight 0.25 did not improve parity in this pilot. The fixed-12-step results were lower overall because last-bit accuracy degraded substantially over repeated thinker updates (approximately 100% at steps 1–2 to about 59% by step 12 on seen lengths). Majority accuracy peaked around steps 3–6 and then remained broadly flat. This is a diagnostic reason to train and evaluate the halting policy jointly with per-step predictive quality; it is not proof that adaptive halting is robustly superior.

The revised candidate still does not solve parity. Full 20 × 100 × 128 training on three seeds, the 1–128 / 129–256 benchmark, and more robust uncertainty analysis remain outstanding. Do not treat this pilot as release-level evidence.


---

## Revised attention/ACT candidate — full benchmark

### Experimental setup

- Seeds: 11, 23, 42
- Device: CUDA
- Epochs: 20
- Steps per epoch: 100
- Batch size: 128
- Training examples per seed: 256,000
- Training maximum length: 128
- Evaluation: 2,000 examples per task and range
- Seen evaluation lengths: 1–128
- Unseen evaluation lengths: 129–256
- Trainable parameters: 200,777
- Maximum thinker steps: 12
- Halting threshold: 0.95
- Prefix-parity auxiliary loss weight: 0.25
- Ponder penalty weight: 0.0005

This section reports the revised token-attention and ACT-style candidate. It must not be confused with the historical pre-attention development results above.

### Adaptive inference results

| Range | Macro accuracy (mean ± sample SD) | Parity | Majority | Last bit | Mean steps (mean ± SD) |
|---|---:|---:|---:|---:|---:|
| Seen 1–128 | 99.99% ± 0.02 pp | 100.00% | 99.97% | 100.00% | 1.93 ± 0.16 |
| Unseen 129–256 | 99.74% ± 0.10 pp | 100.00% | 99.23% | 100.00% | 2.13 ± 0.22 |

### Fixed-12-step recurrent rollout

| Range | Macro accuracy (mean ± sample SD) | Parity | Majority | Last bit |
|---|---:|---:|---:|---:|
| Seen 1–128 | 72.27% ± 7.53 pp | 64.33% | 99.83% | 52.65% |
| Unseen 129–256 | 71.69% ± 9.76 pp | 66.12% | 99.00% | 49.95% |

The fixed-12 result is a 12-step recurrent rollout, not an independently optimized fixed-compute baseline. Repeated thinker updates can degrade intermediate predictions; adaptive halting preserves predictions at the selected stopping step. Therefore, these numbers should not be presented as a general claim that adaptive computation beats every fixed-compute method.

### Independent parity and metamorphic verification

An additional parity test used a separate Python RNG seed (20261121), independent of the training and evaluation generators.

- Three checkpoints tested.
- Sequence lengths 1–256.
- 16 random sequences per length and checkpoint.
- 12,288 parity predictions; all correct.
- Single-bit flip tests: 720/720 changed the predicted parity.
- Two-bit flip tests: 660/660 preserved the predicted parity.

These results provide independent empirical evidence of parity behavior. They are not a formal proof and do not rule out every possible implementation or dataset shortcut.

### Checkpoint integrity

All three checkpoints loaded successfully with strict state-dictionary matching.

- Seed 11: 200,777 parameters; checkpoint approximately 0.78 MB.
- Seed 23: 200,777 parameters; checkpoint approximately 0.78 MB.
- Seed 42: 200,777 parameters; checkpoint approximately 0.78 MB.

### Limitations and remaining work

- The benchmark covers three synthetic binary-sequence tasks, not natural language or general-purpose reasoning.
- The full benchmark evaluates three training seeds; broader robustness testing remains useful.
- The previous architecture has not yet been compared in a fully matched experiment against this revised candidate.
- Equal-budget ablations for encoder direction, prefix supervision, prefix-loss weight, and training-length limits remain outstanding.
- Fixed-12 recurrent rollout results are substantially weaker than adaptive results and require careful interpretation.
