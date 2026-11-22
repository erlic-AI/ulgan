# Validation record — ulgan 1 development snapshot

## Revised attention/ACT candidate code checks

- `pytest -q`: 10 passed.
- `python -m compileall -q src scripts tests`: passed.
- One-epoch CPU smoke training and paired seen/unseen evaluation completed successfully (tiny smoke settings only; not a performance benchmark).
- Attention uses valid bit positions only; EOS and padding are masked.
- The halting objective uses differentiable ACT-style halting mass for per-step classification loss, with final-step residual mass and a ponder penalty.
- Evaluation examples are generated from fixed task/range RNG seeds, ensuring matched evaluation sets across model seeds and adaptive/fixed comparisons.
- No exact parity-summary computation or parity-projection module exists in active model source.

## Limitations

The revised candidate has not completed the full GPU benchmark. No accuracy, generalization, or adaptive-computation improvement is claimed. Historical metrics/checkpoints describe the previous architecture and must not be mixed with candidate results. Full benchmark and ablations remain outstanding.
