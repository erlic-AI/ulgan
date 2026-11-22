# ULGAN 1 — Model Card

## Summary
ULGAN 1 is a 200,777-parameter recurrent neural model for studying
adaptive computation on three synthetic binary-sequence tasks.

## Intended use
- Research on adaptive computation and learned halting.
- Reproduction of the included synthetic tasks.
- Educational and exploratory experimentation.

## Out of scope
ULGAN 1 is not a language model, chatbot, general-purpose reasoning
system, autonomous agent or production decision system. Do not use it
for medical, legal, financial, security or other consequential decisions.

## Tasks
- parity: odd number of ones.
- majority: strictly more than half of the bits are ones.
- last_bit: final input bit.

## Architecture
Task-conditioned bidirectional GRU encoder, recurrent latent thinker,
learned halting controller, prefix parity supervision and a density-aware
feature for majority processing.

## Training
Three independent seeds: 11, 23 and 42.
256,000 training examples per seed.
Maximum sequence length during training: 128.
Maximum thinking steps: 12.
Adaptive threshold used in reported inference: 0.95.

## Evaluation
Adaptive macro accuracy:
- Seen lengths 1–128: 99.9889% mean, standard deviation 0.0192 percentage
  points across seeds.
- Unseen lengths 129–256: 99.7444% mean, standard deviation 0.1005
  percentage points across seeds.

The model is evaluated on synthetic data. These measurements do not
establish performance on natural language or real-world data.

## Limitations
Constructed edge cases can still fail. Fixed-step inference is materially
worse than adaptive inference in the reported evaluation. Benchmark
protocols differ from v0.4, so cross-version comparisons are descriptive.

## Checkpoints
The three seed checkpoints are separate trained models, not an ensemble.
Use one checkpoint at a time and report which seed was used.

## License
Apache-2.0, subject to the repository license.
