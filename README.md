# FractalMergeCritic

Sequential rate–distortion token coarsening for frozen pretrained transformers.

The project learns a model-specific policy that decides:

- where in the transformer to merge;
- which legal token groups to merge;
- how one vector should represent each merged group;
- when further compression is no longer worth its predicted distortion.

It avoids enumerating merge trees. A sparse candidate graph is scored in one batched value-model call, positive-utility edges are reduced to a disjoint matching, and candidates are regenerated after every hard merge round.

## Current status

The original isolated-pair damage critic failed the 2026-08-24 held-out scale gate because raw cosine ranked safe merges better on both CLIP and Qwen. That evidence remains in [`AUDIT_2026-08-24.md`](AUDIT_2026-08-24.md).

The repository now implements the replacement sequential policy:

- multi-round rollout terminal-damage labels;
- layer, round, compression, target, and distortion-budget conditioning;
- a monotonic per-layer cosine baseline plus a learned contextual residual;
- early/middle/late stage adapters and small per-layer adapters;
- uncertainty-aware compute utility and learned stopping;
- an offline gradient oracle distilled into a bounded runtime merge operator;
- versioned calibration corpora and checkpoints that reject legacy artifacts;
- cosine iteration-zero rollouts and learned-policy `Q^pi` rollout aggregation for later iterations.

The latest recorded full DGX suite has 38 tests passing, with exact zero-merge checks and bounded real-model CPU smokes. Those checks establish implementation behavior, not a useful compression policy. The later CLIP pilot shows a limited benefit at aggressive compression but degrades at mild compression, has no calibrated safe selections, and runs slower than baseline under the default safe policy. The small Qwen pilot is worse than cosine and has a severe per-sample failure. Scale and deployment gates therefore remain red. See [Measured pilot results](#measured-pilot-results-2026-08-24) and the detailed reports in [`TRAINING_TEST_REPORT.md`](TRAINING_TEST_REPORT.md), [`VALIDATION.md`](VALIDATION.md), and [`AUDIT_2026-08-24.md`](AUDIT_2026-08-24.md).

## Measured pilot results (2026-08-24)

These are the latest result reports present in this checkout. The linked experiment artifacts live under `/home/boni/experiments/FractalMergeCritic/` on the DGX and are not included in this repository. This table records pilot evidence, not a benchmark suite or a claim of generalization.

| Run | Held-out evidence | Outcome | Decision |
| --- | --- | --- | --- |
| Controlled learnability | 8,292 training and 1,536 held-out records. Within-state Spearman: policy `0.8582`, cosine `0.3068`; top-25% regret: policy `0.00160`, cosine `0.02029`. | The objective can learn the controlled candidate-selection task when state diversity and candidate grouping are adequate. | Controlled result only; not evidence of real-backbone quality. |
| CLIP vision policy | 100 balanced ImageNet100/COCO training images and 9,600 rollout records; 20 held-out images and 2,560 candidate pairs. Spearman: policy `0.4626`, cosine `0.3758`; top-25% regret: policy `0.000403`, cosine `0.000494`. | At 197→98 tokens, mean damage was `0.0983` versus cosine `0.1511` and the policy won on 70% of samples. At mild 197→185 compression, damage was `0.000289` versus cosine `0.000166`. Safe-set AUROC was `0.666`, calibration error `0.166`, and no candidates met the configured `0.95` safe-confidence threshold. The default safe policy made no merges and measured `0.49×` baseline speed due to policy overhead. | Limited benefit at one aggressive compression level; not a uniformly better rate–distortion frontier and not an acceleration result. Gate remains red. |
| Qwen3-0.6B text policy | 12 training texts, 864 sequential records; four held-out texts and 86 candidate pairs. Spearman: policy `0.1703`, cosine `0.2858`. | Top-25% regret was slightly worse than cosine. At 65.5→32.75 mean tokens, policy damage was `1.786` versus cosine `0.860`; one held-out text measured `5.142` versus `0.340`. | Smoke-scale, data-starved, and unsafe. Do not scale this text result. |

The first sequential CLIP pilot has some ranking and aggressive-compression signal, but its mild-compression regression, empty safe set, poor calibration, and measured overhead dominate deployment readiness. The text pilot has negative evidence. Neither result justifies large calibration, a larger backbone, or an acceleration claim. The precise remaining gates are listed below.

## Objective

For frozen transformer `T`, compressed execution `T_tilde`, policy `pi`, merge operator `f`, terminal distortion `D`, and actual compute `C`, optimize

```text
E[D(T(x), T_tilde(x; pi, f)) + lambda C(trajectory) + eta C(policy)]
```

The equivalent constrained problem is

```text
minimize compute subject to terminal distortion <= epsilon.
```

At layer `l`, the state contains hidden tokens, additive mass, spatial or span geometry, lineage, merge history, current token count, target token count, and distortion budget.

## How a merge is represented

For a candidate group `C`, the invariant weighted representation is

```text
h_bar = sum_i(m_i h_i) / sum_i(m_i).
```

The learned merge operator produces

```text
z_C = h_bar + delta_phi(C, state, policy_context)
```

with the hard constraint

```text
||delta_phi|| <= rho ||h_bar||.
```

Mass, region or text-span union, ordering, and lineage are deterministic. The network cannot invent mass or provenance. Supported adapters apply proportional attention through `log(mass)`.

During calibration, a few gradient steps through the frozen transformer suffix optimize the best bounded `delta*` for each sampled merge. The runtime merge operator learns to reproduce that result in one forward pass. No gradient descent runs during inference.

## Value policy

The value model estimates the distribution of terminal distortion after taking a candidate action and continuing through future hard merge rounds.

Its mean is a monotonic cosine baseline plus a learned residual:

```text
mu_log_damage = layer_bias - softplus(layer_slope) * cosine + contextual_residual.
```

The policy uses an upper risk quantile:

```text
risk = exp(mu_log_damage + confidence_z * sigma_log_damage).
```

The normalized utility of removing a token at a particular layer is

```text
utility = compute_reward * suffix_compute_saving - risk.
```

Only positive-utility candidates are eligible. Greedy maximal matching chooses non-overlapping merges. If no candidate has positive utility, the policy waits for a later layer or stops. Earlier equivalent merges expose more suffix-compute saving, so the layer decision is compute-aware.

## Policy context

The shared model is conditioned on:

- normalized transformer depth;
- remaining transformer depth;
- merge round;
- current token fraction;
- target token fraction;
- remaining merge budget;
- log distortion budget.

Most parameters are shared across a backbone. Early, middle, and late stage FiLM plus small per-layer FiLM parameters handle depth-specific distributions.

Vision can merge at multiple selected layers. Qwen deployment is intentionally layer-0-only until correct mid-layer KV-cache compaction is implemented.

## DGX environment

The migrated project is

```text
/home/boni/projects/FractalMergeCritic
```

Always run on the DGX:

```bash
ssh dgx-spark
cd /home/boni/projects/FractalMergeCritic
source /home/boni/ai/envs/dgx-dl/bin/activate
```

Do not replace the existing NVIDIA PyTorch or CUDA stack.

The tested DGX environment recorded by `VALIDATION.md` is Python 3.12.3 with PyTorch 2.11.0+cu130. The project requires Python 3.10 or newer and PyTorch 2.4 or newer. The optional `research` extra in `pyproject.toml` contains the Transformers, Accelerate, Datasets, Pillow, NumPy, and pytest dependencies.

The DGX pipeline entry points import `telegram_training_helper`, which is supplied by the existing DGX environment and is not part of this repository. Setting `FRACTAL_MERGE_TELEGRAM=0` disables notifications but does not remove that import dependency. A standalone environment needs that module available before invoking the pipeline scripts.

For a separate environment, install a PyTorch build suited to that machine first, then install the project and research dependencies:

```bash
python -m pip install -e '.[research]'
```

On the shared DGX environment, activate the existing `dgx-dl` environment above and do not replace its PyTorch/CUDA packages.

## Validation

Run the complete suite:

```bash
pytest -q
python -m compileall -q merge_critic scripts tests
```

The latest recorded full DGX suite passed 38 tests. On 2026-09-29, a rerun could not collect `test_core.py` and `test_qwen_adapter.py`: the installed Transformers package requires `huggingface-hub>=1.5,<2`, while the existing shared environment has `huggingface-hub==2.0.0`. I left the shared environment untouched. See `TRAINING_TEST_REPORT.md` for the later policy-v2 pilot details; `VALIDATION.md` retains earlier smoke results and pending CUDA/throughput checks.

Run bounded real-model smokes:

```bash
FRACTAL_MERGE_SMOKE=1 \
FRACTAL_MERGE_TELEGRAM=0 \
FRACTAL_MERGE_RUN_ID=policy_v2_clip_smoke \
python scripts/run_dgx_clip_pipeline.py

FRACTAL_MERGE_SMOKE=1 \
FRACTAL_MERGE_TELEGRAM=0 \
FRACTAL_MERGE_RUN_ID=policy_v2_qwen_smoke \
python scripts/run_dgx_qwen_pipeline.py
```

The audit runner waits for the pre-existing GPU workload before using CUDA:

```bash
bash scripts/run_dgx_audit_smokes_when_idle.sh
```

## Pilot training

The CLIP pilot uses balanced ImageNet100 and COCO images already under `/mnt/hdd`:

```bash
FRACTAL_MERGE_RUN_ID=policy_v2_clip_pilot \
FRACTAL_MERGE_IMAGES_PER_SOURCE=100 \
FRACTAL_MERGE_VALIDATION_IMAGES_PER_SOURCE=20 \
FRACTAL_MERGE_TELEGRAM=1 \
python scripts/run_dgx_clip_pipeline.py
```

The Qwen3-0.6B pilot uses the existing multisource text corpus:

```bash
FRACTAL_MERGE_RUN_ID=policy_v2_qwen_pilot \
FRACTAL_MERGE_TELEGRAM=1 \
python scripts/run_dgx_qwen_pipeline.py
```

Both pipelines provide deterministic manifests, grouped source-sample splits, atomic best and last checkpoints, resume validation, JSONL metrics, Telegram progress, zero-merge equivalence, held-out cosine comparisons, and final machine-readable reports.

Do not reuse v1 calibration or checkpoints. Version 2 intentionally rejects isolated-damage artifacts.

## Scale gate

A larger image corpus or Qwen 8B/14B run is approved only when the held-out policy:

- beats cosine by at least `0.05` Spearman;
- has lower selection regret at both 10% and 25% merge budgets;
- produces a nonempty calibrated safe set;
- keeps false-safe rate at or below 10%;
- preserves the task metric through repeated merge rounds;
- improves measured end-to-end GPU throughput after policy, matching, and merge-operator overhead.

Smoke completion is not a green scientific result.

## Important files

```text
merge_critic/model.py         conditioned value model and merge operator
merge_critic/policy.py        policy context and transformer compute model
merge_critic/calibration.py   oracle optimization and rollout labels
merge_critic/runtime.py       risk utility, stopping, and matching
merge_critic/schedule.py      layer and round policy execution
merge_critic/training.py      grouped training, ranking, checkpoint/resume
scripts/run_dgx_clip_pipeline.py
scripts/run_dgx_qwen_pipeline.py
scripts/collect_rollout_vision.py
scripts/collect_rollout_qwen.py
tests/
```

## Related work

- Bolya et al., [Token Merging: Your ViT But Faster](https://openreview.net/forum?id=JroZRaRw7Eu)
- Rao et al., [DynamicViT](https://arxiv.org/abs/2106.02034)
- Ryoo et al., [TokenLearner](https://arxiv.org/abs/2106.11297)

The distinction here is that the frozen backbone supplies counterfactual terminal-damage supervision, while the deployment object is a compute-aware sequential merge policy with a learned bounded representation operator.
