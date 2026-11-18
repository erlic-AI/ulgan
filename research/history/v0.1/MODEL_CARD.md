# ULGAN v0.1 — Model Card

## Summary

ULGAN v0.1 is an initial compact recurrent neural-network prototype from Erlic AI. It learns binary parity over variable-length sequences and uses a learned halting head to identify the end-of-sequence token.

## Architecture

- Token embedding: 3 tokens (`0`, `1`, `EOS`)
- Embedding width: 16
- Shared recurrent computation: one `GRUCell`, hidden width 32
- Binary output head
- Learned halting head
- Trainable parameters: 5,426
- FP32 parameter payload: 21,704 bytes (excluding serialization metadata)
- Saved PyTorch checkpoint in this run: 26,293 bytes

## Training run

- Random seed: 42
- Optimizer: AdamW
- Learning rate: 0.002
- Weight decay: 0.00001
- Epochs: 6
- Batches per epoch: 80
- Batch size: 128
- Training examples presented: 61,440
- Training sequence lengths: 1–12 bits
- Task: parity (XOR of all bits)
- Runtime device: CPU

## Observed evaluation

A held-out evaluation generated 2,000 examples with lengths 1–12 and a separate 2,000 examples with lengths 13–20. The recorded run achieved:

| Evaluation | Accuracy | Correct EOS halt | Premature halt | Missing halt |
|---|---:|---:|---:|---:|
| Lengths 1–12 | 100% (2,000/2,000) | 100% | 0% | 0% |
| Lengths 13–20 | 100% (2,000/2,000) | 100% | 0% | 0% |

These are results from one seeded synthetic run, not a generalization guarantee. Additional seeds, distributions, and independent implementations are required.

## Intended use

Research and educational experimentation with compact recurrent computation, parity learning, and learned end-of-sequence halting.

## Out of scope

This version is not a language model, general-purpose assistant, validated reasoning engine, or production-ready system. It does not yet implement the earlier proposed dependency-aware abstraction, certified VOI planning, or general decision-preserving compression as an integrated neural model.

## Limitations and risks

- Evaluated on a narrow synthetic parity task only.
- No comparison yet against tuned external neural baselines.
- No multi-seed statistical evaluation.
- No power, latency, or peak-RAM profiling across target devices.
- Halting currently learns to recognize EOS; this is not yet evidence of adaptive multi-step reasoning.
- Checkpoint size is not the same as total runtime memory or application size.
