# AGENTS.md

## Mission

Develop FractalMergeCritic as a sequential rate–distortion policy for frozen pretrained transformers.

The policy decides where, when, which, and how tokens merge without enumerating merge trees. A model-specific offline calibration phase uses the frozen backbone as a counterfactual teacher. Inference uses an amortized value model, a bounded merge operator, sparse legal candidates, compute-aware matching, and an explicit stop rule.

## Scientific formulation

For transformer blocks `B_0 ... B_(L-1)`, token state `s_l`, policy `pi`, and merge operator `f`, optimize

`E[D(T(x), T_tilde(x; pi, f)) + lambda C(trajectory) + eta C(policy)]`.

The constrained form is to minimize measured compute subject to downstream distortion at most `epsilon`.

The state contains token hidden vectors, additive mass, spatial or span geometry, lineage, layer, merge round, current token count, target token count, and distortion budget.

Actions are `WAIT`, a compatible set of hard merges, and `STOP`. Scanning configured layers implements `WAIT`; no positive risk-adjusted merge utility implements `STOP`.

For cluster `C`, the invariant merge is

`h_bar_C = sum_i(m_i h_i) / sum_i(m_i)`.

The learned representation is

`z_C = h_bar_C + delta_phi(C, s_l, c_l)`

with

`||delta_phi|| <= rho ||h_bar_C||`.

Mass, region or span union, ordering, and provenance are deterministic and cannot be invented by the learned operator.

The value model predicts cumulative terminal distortion after future merge rounds, not isolated immediate damage. It is a monotonic per-layer cosine baseline plus a contextual residual and uncertainty estimate.

The deployed risk is a log-normal upper quantile. Candidate utility is

`compute_reward * normalized_suffix_compute_saving - risk_adjusted_terminal_damage`.

Only positive-utility edges are eligible. Greedy maximal matching chooses simultaneous disjoint actions. Candidates are regenerated after each round.

## Offline oracle and amortization

Gradient descent is allowed only during calibration to construct the best bounded merge residual for sampled actions. It must not run during inference.

The merge operator distills these oracle residuals. The value model trains on multi-round rollout terminal damage with log-damage likelihood, direct regression, and within-state pairwise ranking.

Calibration states must include uncompressed states and states after multiple hard merge rounds. Initial rollouts may use cosine; later corpora should aggregate states visited by the learned policy.

## Layer sharing

Use one model-specific shared trunk with continuous policy conditioning, early/middle/late stage FiLM, and small per-layer FiLM parameters. Do not create one full critic per layer without an ablation proving it necessary.

Vision may calibrate and deploy at several layers. Qwen deployment remains layer-0-only until correct mid-layer KV-cache compaction exists.

Do not claim one universal policy across different backbones. Hidden spaces and task distortions differ.

## Backbone invariants

The main method keeps the pretrained backbone frozen.

Merged mass is additive. Weighted means remain associative. Proportional attention uses `log(mass)` where the adapter supports it. Vision groups stay disjoint and cover every original patch. Text spans stay contiguous, ordered, disjoint, and complete. Special tokens cannot merge.

## Required baselines

- No compression.
- Random legal merges.
- Fixed local grouping.
- Raw cosine ranking.
- ToMe-like similarity matching.
- Exact one-step counterfactual oracle.
- Cosine rollout policy.
- Value policy without learned merge operator.
- Full value policy with learned merge operator.

## Required reporting

- Token trajectory by layer and round.
- Remaining tokens and estimated compute saving.
- Actual wall-clock latency and policy overhead.
- Terminal output distortion and task metric.
- Value-model Spearman correlation and selection regret versus cosine.
- AUROC, calibration error, false-safe rate, and nonempty safe selection counts.
- Merge-operator improvement and norm-cap compliance.
- Results across repeated hard merge rounds.

## Checkpoint and data policy

Sequential-policy calibration schema and checkpoints use version 2. Legacy isolated-damage corpora and checkpoints must fail closed and be regenerated.

Checkpoint resume must verify calibration SHA-256 and the complete training configuration. Train/validation splits must group all records from one source sample together.

## Validation requirements

- Exact zero-merge adapter equivalence.
- Cached Qwen zero-merge generation equivalence.
- Additive mass and complete lineage after repeated rounds.
- Symmetric pair features.
- Bounded learned merge residual.
- Policy context changes with layer, round, compression, target, and budget.
- Earlier equivalent merges expose more suffix-compute saving than later merges.
- Positive utility merges and nonpositive utility stops.
- Matching never overlaps edges.
- Multi-round rollout labels exist in every training record.
- Save-and-resume and incompatible-data failure tests.
- Real-model CPU smoke before an idle-GPU smoke.
- End-to-end GPU throughput before any acceleration claim.

## Scale gate

Do not approve large calibration or larger backbones unless the value policy beats raw cosine by at least 0.05 Spearman and lower selection regret at both 10% and 25% budgets, has a nonempty calibrated safe set with false-safe rate at most 10%, preserves task quality across repeated rounds, and improves measured end-to-end throughput after policy overhead.

## Engineering constraints

Run substantive work on the DGX through `ssh dgx-spark` after `source /home/boni/ai/envs/dgx-dl/bin/activate`.

Do not replace the NVIDIA stack. Do not compete with, stop, or overwrite active training. Use BF16 on CUDA after FP32 correctness. Keep imports at the beginning of Python files. Do not wrap imports in exception handlers, put imports inside functions, add command-line parsing, or add decorative code comments. Configuration belongs near the beginning of directly runnable scripts.
