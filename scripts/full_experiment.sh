#!/usr/bin/env bash
set -euo pipefail

# Run the full experiment suite across all algorithms and seeds.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT"

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_experiment_$(date +%Y%m%d_%H%M%S).log"

echo "=== Full experiment started at $(date) ===" | tee "$LOG_FILE"

run_and_log() {
  local name="$1"
  shift
  echo "--- ${name} ---" | tee -a "$LOG_FILE"
  if "$@" >>"$LOG_FILE" 2>&1; then
    echo "[SUCCESS] ${name}" | tee -a "$LOG_FILE"
    return 0
  else
    local status=$?
    echo "[FAILED] ${name} (exit code ${status})" | tee -a "$LOG_FILE"
    return $status
  fi
}

run_and_log "Training" python train.py --all-algos --seeds 0 1 2 3 4 --ablation --postprocess
if [ $? -eq 0 ]; then
  if [ -x "$SCRIPT_DIR/eval_all.sh" ]; then
    run_and_log "Evaluation" "$SCRIPT_DIR/eval_all.sh"
  fi
  if [ -f "generate_figures.py" ]; then
    run_and_log "Generate figures" python generate_figures.py
  fi
  if [ -f "generate_tables.py" ]; then
    run_and_log "Generate tables" python generate_tables.py
  fi
fi

echo "=== Full experiment finished at $(date) ===" | tee -a "$LOG_FILE"
