# Experimental roadmap

## Phase 0: invariants and adapter equivalence

Verify exact zero-merge equivalence at every proposed site. Validate proportional attention, additive mass, complete spatial or span lineage, Qwen cached zero-merge generation, and bounded merge-operator output.

## Phase 1: rollout corpus

Collect balanced original and post-merge states at early, middle, and late vision layers. For Qwen collect only layer-0 states until mid-layer KV compaction is implemented.

Every record must contain policy context, immediate damage, multi-round terminal damage, suffix-compute saving, bounded oracle residual, rollout depth, and source-sample identity.

Initial state trajectories use stratified cosine rollouts. Later iterations aggregate states visited by the current learned policy.

## Phase 2: value and merge-operator pilot

Train the monotonic cosine baseline plus contextual residual using log-damage likelihood, direct regression, and within-state pairwise ranking. Train the merge operator against offline gradient-optimized residuals.

Compare terminal-damage ranking against raw cosine on held-out source samples. Stop if the value model cannot beat cosine by 0.05 Spearman and lower regret at 10% and 25% selection budgets.

## Phase 3: sequential policy frontier

Sweep compute value while holding the distortion budget fixed. For every learned trajectory, replay cosine with the same merge count at the same layers and rounds.

Report token trajectories, terminal distortion, task metric, matched-trajectory cosine damage, safe-set calibration, and win fraction. Include learned policy with and without the merge operator.

## Phase 4: on-policy aggregation

If the pilot clears pair-ranking gates but degrades after repeated rounds, collect active queries from the learned policy at rounds one through four. Prefer uncertain candidates and candidates near zero utility. Retrain with immutable held-out manifests.

## Phase 5: dense vision and continuation text

For vision, add DINO dense-token and segmentation-output preservation so global CLIP embeddings cannot hide local destruction.

For text, replace single next-token KL with multi-token teacher-forced continuation KL and a long-context retrieval metric. Continue to preserve exact original RoPE positions and causal span ordering.

## Phase 6: GPU speed gate

Measure complete CUDA latency including value scoring, matching, merge operator, backbone execution, and cache construction. Report time to first token, prefill throughput, vision images per second, memory pressure, and actual task quality.

No larger corpus or backbone is approved without end-to-end speedup at matched quality.

## Phase 7: scale and generalization

Only after all fixed gates pass, expand ImageNet100 plus COCO calibration and then consider Qwen 8B. Test one policy across multiple tasks sharing exactly the same frozen backbone before investigating cross-backbone projections.
