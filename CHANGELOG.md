# Changelog

## v0.4 — research release packaging
- Preserved the v0.4 model, checkpoint, training implementation, and recorded metrics.
- Reorganized source, scripts, tests, configuration, checkpoints, and benchmark outputs.
- Added a package-based import layout and a command-line prediction entry point.
- Expanded repository documentation and made the scope and limitations explicit.

## Model development history
- v0.1: recurrent binary parity prototype with EOS termination.
- v0.2: shared task-conditioned model for parity, majority, and last-bit.
- v0.3: adaptive latent computation and learned halting.
- v0.4: batched GRU encoding, deep supervision, and evaluation on longer held-out sequences.
