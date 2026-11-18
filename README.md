<div align="center">

# ulgan

### Compact adaptive-computation research

**Erlic AI · Independent research initiative**

[Research direction](research/research_direction.md) · [Methodology](research/methodology.md) · [Model card](MODEL_CARD.md)

</div>

---

## Overview

**ulgan** is an independent research project exploring a central question: *can a compact learned model decide how much computation an input deserves, instead of applying the same fixed computation to every case?*

The long-term research direction is adaptive internal computation: build a compact representation, refine it when additional processing is useful, and learn when further computation is unlikely to improve a decision. The broader ambition is to study whether this principle can support more capable structured problem-solving systems. That is a research objective—not a claim that the current checkpoint is a general reasoning model.

The v0.4 release is a deliberately narrow experimental prototype. It studies learned halting on three synthetic binary-sequence tasks. It is **not** a language model, chatbot, general-purpose AI, or production system.

## What v0.4 contains

- 34,908 trainable parameters.
- Task-conditioned GRU sequence encoder.
- Shared recurrent latent thinker using a GRUCell.
- Task-specific binary prediction heads.
- Learned halting controller with a cumulative threshold of 0.95.
- Up to 8 latent thinking steps.
- Deep supervision across thinking steps and an expected-computation penalty.

### Supported tasks

| Task | Target |
|---|---|
| `parity` | 1 when the sequence contains an odd number of ones |
| `majority` | 1 when strictly more than half of the bits are 1 |
| `last_bit` | The final bit in the sequence |

## Benchmark results

The committed reference run used CPU, seed 42, 8 epochs × 40 steps × 128 examples (40,960 presented examples), and 500 test examples per task in each evaluation range. The fixed-step baseline uses the eighth output from the **same checkpoint**.

### Adaptive computation vs fixed 8 steps

| Evaluation range | Adaptive macro accuracy | Adaptive mean steps | Fixed-8 macro accuracy |
|---|---:|---:|---:|
| Seen lengths 1–64 | **80.67%** | **2.11** | 83.27% |
| Held-out lengths 65–128 | **80.27%** | **2.06** | 79.60% |

### Per-task accuracy

| Evaluation range | Task | Adaptive | Fixed 8 |
|---|---|---:|---:|
| 1–64 | Parity | 44.8% | 52.0% |
| 1–64 | Majority | 97.2% | 97.8% |
| 1–64 | Last bit | 100.0% | 100.0% |
| 65–128 | Parity | 50.4% | 46.8% |
| 65–128 | Majority | 90.4% | 92.0% |
| 65–128 | Last bit | 100.0% | 100.0% |

Adaptive inference uses approximately **74% fewer latent steps** than the fixed-eight-step policy in this run. This is a reduction in model steps, not a measured 74% wall-clock speedup. Accuracy effects are mixed: adaptive inference is slightly lower on the seen-range macro score and slightly higher on the held-out-range macro score. Parity remains a major weakness.

The results are single-seed measurements without confidence intervals. They should be treated as an initial reference point, not statistically robust evidence or a state-of-the-art comparison. Exact values and training history are available in [`metrics.json`](artifacts/benchmarks/v0.4/metrics.json).

## Research history and experiment records

The repository preserves the executable v0.1–v0.3 snapshots under [`research/history/`](research/history/) and documents the v0.4 protocol in [`research/methodology.md`](research/methodology.md). The supplied archives did not include complete primary logs for every numbered research experiment up to 70. Experiments 66, 68, and 69 have provisional recovery notes only; they are explicitly separated from the verified v0.4 benchmark. See the [experiment index](research/experiment_index.md).

## Installation

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run inference

```bash
python scripts/predict.py parity 1 0 1 1
python scripts/predict.py majority 1 0 1 1
python scripts/predict.py last_bit 1 0 1 1
```

## Train from scratch

The reference configuration is recorded in [`configs/v0.4.json`](configs/v0.4.json). To reproduce the original run:

```bash
python scripts/train.py --seed 42 --epochs 8 --steps-per-epoch 40 --batch-size 128 --ponder-weight 0.001 --eval-samples 500 --train-max-length 64 --out artifacts/benchmarks/reproduced
```

This writes a new checkpoint and metrics report to the selected output directory; it does not overwrite the committed reference artifact.

## Tests

```bash
pytest -q
```

The smoke tests verify checkpoint loading, parameter count, output shapes, finite values, and prediction bounds. They are software checks, not evidence of task accuracy.

## Repository layout

```text
ulgan/
├── artifacts/
│   ├── benchmarks/v0.4/
│   └── checkpoints/
├── configs/
├── research/
│   ├── experiments/
│   ├── history/
│   ├── experiment_index.md
│   ├── methodology.md
│   └── research_direction.md
├── scripts/
├── src/ulgan/
├── tests/
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── MODEL_CARD.md
├── README.md
└── requirements.txt
```

## Limitations and responsible use

The current model only addresses three synthetic binary-sequence tasks. It has weak parity performance, has only one recorded random seed, and has no evidence of general reasoning, language capability, or real-world decision competence. Do not use it in consequential settings. Read the [model card](MODEL_CARD.md) before use.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
