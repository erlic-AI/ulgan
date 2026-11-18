# Model Card — ULGAN v0.3

- **Project:** Erlic AI / ULGAN
- **Type:** compact task-conditioned recurrent neural network with learned latent halting
- **Framework:** PyTorch
- **Intended use:** research and reproducible experimentation on small synthetic sequence tasks
- **Not intended for:** production, safety-critical decisions, natural-language generation, or claims of general intelligence

## Main change from v0.2
Adds a latent recurrent thinker and an ACT-style halting controller. The controller is trained jointly with task prediction using an expected-computation (ponder) penalty. Evaluation compares threshold-based adaptive inference against a fixed eight-step inference baseline.

## Reference results
Seed 42; 10 epochs; 80 steps/epoch; batch size 128; ponder weight 0.0002; 102,400 training examples presented; 21,596 trainable parameters. With 300 examples per task per length range:
- Lengths 1–16: adaptive macro accuracy 93.78%, 5.21 mean latent steps; fixed-8 macro accuracy 94.11%.
- Lengths 17–32: adaptive macro accuracy 86.89%, 5.12 mean latent steps; fixed-8 macro accuracy 82.67%.
- Adaptive inference used approximately 35–36% fewer latent steps.
- Parity accuracy: 83.3% (lengths 1–16), 65.3% (17–32); long-sequence parity remains a clear weakness.

## Limitations
Only three synthetic binary-sequence tasks are included. Results depend on seed, training budget, threshold, and ponder penalty. A learned halt score is not automatically evidence of useful adaptive computation; accuracy and compute must be considered together. The reference run suggests a compute/accuracy trade-off, but long-sequence parity performance is still inadequate. No external datasets, language tasks, energy measurements, or hardware latency benchmarks are included.
