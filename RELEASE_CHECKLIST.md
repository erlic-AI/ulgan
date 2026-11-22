# ulgan 1 — release checklist

## Current research candidate status

- [x] Revised attention-based model source and inference CLI
- [x] ACT-weighted training script and updated configuration
- [x] Three completed development checkpoints
- [x] Per-seed metrics and aggregate benchmark
- [x] Small-budget ablation artifacts
- [x] Model card, methodology, experiment report, changelog
- [x] No dependency installation in scripts
- [x] Invalid exact-parity shortcut removed from active source and artifacts
- [x] Regression tests and compile check
- [x] Small three-seed CPU pilot for prefix-off and prefix-0.25 conditions

## Not complete — do not mark as performance release

- [ ] Solve parity without label leakage
- [ ] Improve unseen-length majority accuracy
- [ ] Complete full reference benchmark (20 epochs × 100 steps × batch 128, 3 seeds)
- [ ] Complete full-budget matched ablations
- [ ] Evaluate larger test sets and uncertainty
- [ ] Validate beyond length 256 and on broader task distributions

## Packaging

- [ ] Build final `ulgan 1` source archive only after research and validation are complete
- [ ] Verify final archive contents and checkpoint loading
- [ ] Generate final SHA-256 manifest
