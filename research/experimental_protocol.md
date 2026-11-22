# ulgan 1 — Experimental protocol

## Scope

Three synthetic binary-sequence tasks: parity (XOR of all bits), strict majority (ones > length/2), and last bit. ULGAN is not a language model or general-purpose reasoner.

## Model safeguards

- No feature may directly compute or expose the target label.
- The exact parity-summary feature explored in an earlier experiment is prohibited; it computed the parity target directly and its 100% result is withdrawn.
- Bit density (ones / valid length) is allowed as a non-label statistic; it does not determine parity.
- Packed sequences include real bits plus EOS and exclude right-padding from recurrent computation.
- Prefix-parity auxiliary predictions use forward-direction states only, avoiding future-bit leakage.

## Revised architecture candidate (not yet benchmark-validated)

The current candidate adds token-level attention from each thinker state over valid encoded bit states. EOS and right-padding are excluded from the attention mask. The thinker receives the retrieved context at every step. The halting objective is changed to ACT-style differentiable weighting: per-step classification loss is weighted by the predicted halting mass, with the final step absorbing residual survival probability, plus a ponder penalty. These changes are hypotheses intended to address the previous thinker/halting limitations; they are not evidence of an accuracy or efficiency gain until the full matched benchmark completes.

Evaluation data are generated from fixed per-task/per-range RNG seeds, so each model seed and adaptive/fixed comparison sees the same examples. Training RNG remains seed-specific.

## Development benchmark completed (previous architecture)

Seeds 11, 23, 42; 5 epochs; 20 steps per epoch; batch 32; maximum training length 128; 1,024 evaluation examples per task in ranges 1–128 and 129–256; prefix supervision disabled; CPU. Adaptive and fixed-12-step inference use the same generated examples within each run. Each seed receives 3,200 training examples.

## Full reference benchmark completed

The full run was completed for seeds 11, 23, and 42 using 20 epochs × 100 steps × batch 128, with 2,000 evaluation examples per task and range. Each seed processed 256,000 training examples on CUDA.

The revised attention/ACT candidate was evaluated on seen lengths 1–128 and unseen lengths 129–256. Aggregate results and per-seed records are stored in `artifacts/ulgan_1_density_gated/metrics.json`.

The completed benchmark does not replace the required matched comparison against the previous architecture or the planned ablations. Those experiments remain outstanding.

## Required future experiments

1. Validate the revised attention/ACT candidate against the previous architecture with matched seeds and fixed test sets.
2. Improve parity without label leakage or hand-computing parity.
3. Improve majority accuracy and unseen-length generalization.
4. Complete the full reference benchmark.
5. Complete equal-budget ablations for encoder direction, prefix supervision, prefix weights, and train-length limits.
6. Evaluate additional sequence lengths and task distributions.
7. Report uncertainty from both seeds and finite test sets.

## Reporting rules

Report exact configuration, seed, device, examples, parameter count, runtime, per-task accuracy, macro accuracy, adaptive steps, and fixed-step comparison. Small-budget runs are development evidence only. Do not claim a gain from unmatched runs.
