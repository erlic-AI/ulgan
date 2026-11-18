# Model card — ulgan v0.4

## Summary

ulgan v0.4 is a 34,908-parameter, task-conditioned recurrent neural network with a learned halting controller. It is a controlled research prototype for studying computation/accuracy trade-offs on three synthetic binary-sequence tasks.

## Intended use

- Reproducing the included synthetic benchmark.
- Studying learned early stopping in a compact recurrent model.
- Educational and exploratory research.

## Out of scope

This checkpoint is not a language model, chatbot, general-purpose reasoning system, autonomous agent, or production-ready model. It must not be used for safety-critical, medical, legal, financial, or security decisions.

## Inputs and outputs

The model receives a non-empty binary sequence and a task identifier (`parity`, `majority`, or `last_bit`). It returns a binary prediction, the number of latent thinking steps, and cumulative halting probability.

## Architecture

A GRU encodes the sequence. A shared GRUCell performs up to eight latent updates. Task-specific heads produce binary logits, and a learned controller estimates when to halt. Training applies supervision at each latent step and an expected-step penalty.

## Data and evaluation

Training data are generated synthetically online. The recorded reference run used seed 42, CPU, 8 epochs, 40 batches per epoch, batch size 128, and 40,960 presented examples. Evaluation used 500 examples per task and length range. Exact measurements are in `artifacts/benchmarks/v0.4/metrics.json`.

## Limitations

- Parity accuracy is weak; the model does not reliably learn this task.
- Results are single-seed measurements without confidence intervals.
- Only three synthetic tasks are covered.
- Adaptive halting reduces latent steps but does not consistently improve accuracy.
- Parameter count and checkpoint size are not equivalent to runtime memory or deployment cost.
- No novelty, state-of-the-art, or general reasoning claim is made.
