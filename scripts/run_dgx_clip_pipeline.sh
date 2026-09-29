#!/usr/bin/env bash
set -euo pipefail

project_root="/home/boni/projects/FractalMergeCritic"
environment_path="/home/boni/ai/envs/dgx-dl/bin/activate"
run_id="${FRACTAL_MERGE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
session_name="fractal_merge_critic_clip_${run_id}"
output_root="/home/boni/experiments/FractalMergeCritic/${run_id}"
log_path="${output_root}/training.log"

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "Export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before launching."
    exit 1
fi

if [[ "${FRACTAL_MERGE_IN_TMUX:-0}" != "1" ]]; then
    mkdir -p "${output_root}"
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "tmux session already exists: ${session_name}"
        exit 1
    fi
    tmux new-session -d -s "${session_name}" \
        -e "TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}" \
        -e "TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}" \
        -e "FRACTAL_MERGE_RUN_ID=${run_id}" \
        -e "FRACTAL_MERGE_SMOKE=${FRACTAL_MERGE_SMOKE:-0}" \
        "cd '${project_root}' && FRACTAL_MERGE_IN_TMUX=1 bash scripts/run_dgx_clip_pipeline.sh"
    echo "Started ${session_name}"
    echo "Attach: tmux attach -t ${session_name}"
    echo "Log: tail -f ${log_path}"
    exit 0
fi

source "${environment_path}"
cd "${project_root}"
mkdir -p "${output_root}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=20
export FRACTAL_MERGE_TELEGRAM=1

python scripts/run_dgx_clip_pipeline.py 2>&1 | tee -a "${log_path}"
