# Sequential merge-policy training tests

Date: 2026-08-24

## Decision

The replacement policy is learnable and shows real signal beyond cosine on CLIP vision activations, but it is not ready for large downstream training or deployment. The current scientific decision remains red because uncertainty calibration, mild-compression behavior, and end-to-end speed are not yet acceptable. Qwen text results are negative at the tested scale.

## Controlled learnability

Artifact: `/home/boni/experiments/FractalMergeCritic/policy_v2_controlled_scaled_20260824/FINAL_REPORT.json`

- 8,292 training and 1,536 held-out records.
- Held-out within-state Spearman: policy 0.8582, cosine 0.3068.
- Top-25% selection regret: policy 0.00160, cosine 0.02029.
- Learned merge-operator error was 0.2395 times the zero-delta error.
- All controlled checks passed.

The earlier 1,032-record control failed within-state selection. This establishes that candidate grouping and sufficient state diversity are necessary; global correlation alone is misleading.

## CLIP vision pilot

Artifact: `/home/boni/experiments/FractalMergeCritic/policy_v2_clip_gpu_learnability_20260824/FINAL_REPORT.json`

- Training corpus: 100 balanced ImageNet100 and COCO images, 9,600 sequential terminal-damage records.
- Held-out corpus: 20 untouched images and 2,560 candidate pairs.
- Critic Spearman: 0.4626 versus cosine 0.3758.
- Top-25% regret: 0.000403 versus cosine 0.000494.
- Refined merges improved 63.8% of individual candidates.
- At 197 to 98 tokens, mean policy damage was 0.0983 versus cosine 0.1511, with a 70% policy win rate.
- At mild compression, 197 to 185 tokens, policy damage was 0.000289 versus cosine 0.000166.
- Safety AUROC reached 0.666 at epsilon 0.001, but calibration error was 0.166 and no candidates reached the 0.95 safe-confidence threshold.
- The default safe policy made no merges and ran at 0.49 times baseline speed because policy overhead was added without token reduction.
- Scientific gate: eligible, red.

This is the first real evidence that the learned policy can beat cosine at aggressive image-token compression. It does not yet produce a uniformly better rate-distortion frontier.

## Qwen text pilot

Artifact: `/home/boni/experiments/FractalMergeCritic/policy_v2_qwen_cpu_bounded_20260824/FINAL_REPORT.json`

- Training: 12 texts, four transformer depths, 864 sequential records.
- Held-out: four untouched texts and 86 candidate pairs.
- Critic Spearman: 0.1703 versus cosine 0.2858.
- Top-25% regret was slightly worse than cosine.
- At 65.5 to 32.75 mean tokens, policy damage was 1.786 versus cosine 0.860.
- One held-out text had policy damage 5.142 versus cosine 0.340.
- Scientific gate: ineligible smoke-scale result, all substantive checks failed.

The text pilot is data-starved and unsafe. It is not evidence for scaling Qwen training.

## Repairs made during testing

- Candidate sets are kept intact in grouped batches and train/validation splits are sample-disjoint.
- Pairwise ranking is damage-gap weighted and a grouped top-k selection objective directly trains the decision the runtime makes.
- Qwen calibration and evaluation use four depths rather than only layer 0.
- Qwen rate-distortion evaluation now uses the same multi-depth schedule as training.
- Frontier reports include per-sample and worst-case damage to expose tail failures.
- CLIP zero-merge equivalence uses dtype-aware BF16 tolerances.
- CUDA AUROC indexing is device-safe and covered by a GPU regression test.
- Successful retries archive old failures and report attempt plus cumulative elapsed time.
- Full DGX test suite: 38 passed.

## Recommended next experiment

Continue with vision only. Train on a larger CLIP calibration corpus with explicit hard-negative and tail-risk sampling, then calibrate uncertainty per layer on a separate calibration split. Evaluate a dense compute-reward sweep and require policy dominance or non-inferiority at every matched token count. Benchmark aggressive compression end to end only after the policy reliably selects merges. Do not start downstream ImageNet/OCR/VLM training until those gates pass.
