# Research direction

## Long-term question

Can a compact learned system allocate computation according to the structure and difficulty of an input, rather than spending the same fixed amount of computation on every case?

The broader direction behind ulgan is to investigate adaptive internal computation: a model should form a compact representation, refine it when additional processing is useful, and learn when further computation is unlikely to change its decision. In later research stages, this question may extend from simple sequence tasks toward structured decision problems and evidence-sensitive inference.

## What v0.4 actually implements

The released v0.4 checkpoint is a narrow experimental instrument for this question. It uses a task-conditioned GRU encoder, a recurrent latent thinker, task-specific binary heads, and a learned halting controller. It has three synthetic tasks only: parity, majority, and last-bit.

## What is not yet demonstrated

The current release does not demonstrate general reasoning, open-domain problem solving, language understanding, robust uncertainty estimation, or a broadly useful autonomous decision process. The parity results in particular show that the current model is not a solved general sequence learner. The long-term direction is a research objective, not a capability claim about this checkpoint.

## Research priorities

1. Improve task performance without hiding hard cases behind aggregate metrics.
2. Compare adaptive computation with fixed-step baselines under identical checkpoints and data.
3. Evaluate multiple random seeds and report dispersion and confidence intervals.
4. Measure actual wall-clock latency and memory alongside latent-step counts.
5. Expand task diversity and test transfer to sequence lengths and distributions not seen in training.
6. Test whether the halting policy correlates with example difficulty and whether abstention is calibrated.
