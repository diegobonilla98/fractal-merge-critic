# Validation status

## Sequential policy v2 — 2026-08-24

Validated on the DGX checkout with Python 3.12.3 and PyTorch 2.11.0+cu130:

- 35 unit and integration tests passing.
- Complete Python compilation of package, scripts, and tests.
- No imports inside functions, import fallbacks, command-line parsing, FP16 shortcuts, or decorative divider comments.
- Policy context changes with layer, remaining depth, round, current tokens, target tokens, remaining merge budget, and distortion budget.
- Transformer cost model assigns more suffix-compute saving to an equivalent earlier merge.
- Positive risk-adjusted utility merges and nonpositive utility stops.
- A fixed risk can merge early and wait late because compute value decreases with remaining depth.
- Weighted merge associativity, mass conservation, complete lineage, pair symmetry, refiner norm cap, legal adjacency, and non-overlapping matching.
- Multi-round terminal-damage and policy-context fields in every schema-v2 calibration record.
- Multi-round targets can follow either the iteration-zero cosine policy or the learned deployment policy for `Q^pi` aggregation.
- Offline bounded gradient oracle reduces synthetic counterfactual damage.
- Grouped source-sample train/validation split.
- Atomic best/last checkpoints, deterministic epoch shuffle, SHA-256 data/config resume validation, exact completed-checkpoint resume, and explicit legacy-artifact rejection.

## Real CLIP CPU smoke

Artifact root:

```text
/home/boni/experiments/FractalMergeCritic/policy_v2_clip_cpu_smoke_20260824
```

- Real `openai/clip-vit-base-patch16` loaded in FP32.
- Exact zero-merge output passed at layers 0, 1, 6, and 12.
- Four balanced ImageNet100/COCO source images generated 64 records across two layers, two state rounds, and one future rollout round.
- Merge-operator and cumulative value-model forward, backward, optimizer, validation, checkpoint, load, evaluation, and exact resume completed.
- Held-out smoke Spearman was `0.7049` for both the initialized monotonic value model and cosine.
- A conservative undertrained policy correctly stopped with zero merges.
- A deliberately high compute value executed four hard rounds across layers 0 and 2, reducing 197 to 133 tokens with estimated suffix-compute saving `0.3062` and measured CLIP embedding distortion `0.01692`.

The high-compute test validates policy mechanics, not acceptable quality.

## Real Qwen CPU smoke

Artifact root:

```text
/home/boni/experiments/FractalMergeCritic/policy_v2_qwen_cpu_smoke_20260824
```

- Real `Qwen/Qwen3-0.6B` loaded in FP32.
- Exact zero-merge logits, cached prefill, and four-token greedy generation equivalence passed.
- Four training texts generated 24 schema-v2 rollout records.
- Merge-operator and cumulative value-model training, grouped validation, checkpoint, load, held-out evaluation, and exact resume completed.
- Held-out smoke Spearman was `0.4126` for both the initialized monotonic value model and cosine.

## Prior adapter validation retained

- Real CLIP and DINOv3 zero-merge CUDA equivalence passed before the policy replacement.
- DINOv3 protects class and register tokens and produces finite merged outputs.
- The DINOv3 circular RoPE reduction for merged regions remains heuristic and still requires an ablation.

## Pending before an acceleration or scale claim

The pre-existing ImageNet100 training process PID `1774955` was still using the GPU during this replacement. The tmux session `fractal_merge_critic_audit_smokes` is waiting for that process to exit before running CUDA precision, DINOv3, CLIP, and Qwen smokes.

Still required:

- Fresh policy-v2 BF16 CUDA forward/backward and resume.
- Actual value-model, matching, merge-operator, prefill, and full-pipeline latency.
- Repeated-round task-quality evaluation.
- A held-out pilot demonstrating improvement over cosine.
- Nonempty calibrated safe selections with acceptable false-safe rate.

The project is ready for bounded pilot validation, not large training.
