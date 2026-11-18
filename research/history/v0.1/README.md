# ULGAN v0.1

**A compact recurrent learning prototype from Erlic AI.**

ULGAN v0.1 is the first executable model artifact in this research line. It learns binary parity over variable-length bit sequences using a shared recurrent core and a learned end-of-sequence halting head.

## First measured result

The model has **5,426 trainable parameters**. In one seeded CPU training run, it was trained on 61,440 generated sequences of lengths 1–12. It achieved:

| Held-out test | Accuracy | Correct halt | Premature halt |
|---|---:|---:|---:|
| 2,000 sequences, lengths 1–12 | 100% | 100% | 0% |
| 2,000 sequences, lengths 13–20 | 100% | 100% | 0% |

These are narrow synthetic-task results from one seed, not a general-purpose capability claim. See [MODEL_CARD.md](MODEL_CARD.md) and `artifacts/metrics.json` for details.

## Quick start

```bash
pip install -r requirements.txt
python train.py --epochs 6 --steps-per-epoch 80 --batch-size 128
python predict.py 1 0 1 1
```

The example predicts parity `1` for the sequence `1 0 1 1`.

## Architecture

- 3-token embedding (`0`, `1`, `EOS`), width 16
- Shared `GRUCell`, hidden width 32
- Binary output head
- Learned halting head

The recurrent weights are reused at each input step. The halting head learns to stop at EOS. This is variable-length recurrent processing, not yet a demonstration of adaptive reasoning depth.

## Files

- `model.py` — model definition
- `train.py` — training and evaluation
- `predict.py` — checkpoint inference
- `artifacts/ulgan_v0.1.pt` — trained checkpoint
- `artifacts/metrics.json` — recorded run metrics
- `MODEL_CARD.md` — model scope, results, and limitations

## Scope

This is a first working model, not the final ULGAN architecture. Dependency-aware abstraction, certified value-of-computation selection, broader problem solving, baseline comparisons, multi-seed tests, and device-level profiling remain future work. No novelty or state-of-the-art claim is made.
