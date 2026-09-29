#!/usr/bin/env bash
set -uo pipefail

project_root=/home/boni/projects/FractalMergeCritic
environment=/home/boni/ai/envs/dgx-dl/bin/activate
run_stamp=$(date +%Y%m%d_%H%M%S)
log_root=/home/boni/experiments/FractalMergeCritic/audit_smokes_${run_stamp}

mkdir -p "$log_root"
source "$environment"
cd "$project_root"

pytest -q 2>&1 | tee "$log_root/tests.log"
test_status=${PIPESTATUS[0]}

python scripts/audit_clip_precision.py 2>&1 | tee "$log_root/clip_precision.log"
precision_status=${PIPESTATUS[0]}

python scripts/audit_dinov3_adapter.py 2>&1 | tee "$log_root/dinov3.log"
dinov3_status=${PIPESTATUS[0]}

FRACTAL_MERGE_SMOKE=1 FRACTAL_MERGE_TELEGRAM=0 FRACTAL_MERGE_RUN_ID="audit_clip_smoke_${run_stamp}" python scripts/run_dgx_clip_pipeline.py 2>&1 | tee "$log_root/clip.log"
clip_status=${PIPESTATUS[0]}

FRACTAL_MERGE_SMOKE=1 FRACTAL_MERGE_TELEGRAM=0 FRACTAL_MERGE_RUN_ID="audit_qwen_smoke_${run_stamp}" python scripts/run_dgx_qwen_pipeline.py 2>&1 | tee "$log_root/qwen.log"
qwen_status=${PIPESTATUS[0]}

printf '{"tests":%d,"precision":%d,"dinov3":%d,"clip":%d,"qwen":%d}\n' "$test_status" "$precision_status" "$dinov3_status" "$clip_status" "$qwen_status" > "$log_root/status.json"

if (( test_status != 0 || precision_status != 0 || dinov3_status != 0 || clip_status != 0 || qwen_status != 0 )); then
    exit 1
fi
