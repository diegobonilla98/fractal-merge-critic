#!/usr/bin/env bash
set -uo pipefail

active_pid=1774955
smoke_script=/home/boni/projects/FractalMergeCritic/scripts/run_dgx_audit_smokes.sh

while kill -0 "$active_pid" 2>/dev/null; do
    sleep 30
done

exec "$smoke_script"
