#!/usr/bin/env bash
set -euo pipefail

# Run all experiment configurations in a deterministic order.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT"

export PYTHONHASHSEED=0
ENV_CONFIG="configs/env_8x8.yaml"
ALGO_DIR="configs/algo"

# Explicitly list algorithms to ensure deterministic ordering
ALGOS=(dyna lppo planner_subgoal shielded)

# Seeds to run; override by setting the SEEDS environment variable
if [[ -z "${SEEDS+x}" ]]; then
  SEEDS=(0 1 2 3 4)
else
  read -r -a SEEDS <<<"${SEEDS}"
fi

for algo in "${ALGOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== Running ${algo} seed ${seed} ==="
    OUT_DIR="runs/${algo}/seed_${seed}"
    mkdir -p "$OUT_DIR"
    (
      cd "$OUT_DIR"
      python "$REPO_ROOT/train.py" \
        --env-config "$REPO_ROOT/$ENV_CONFIG" \
        --algo-config "$REPO_ROOT/$ALGO_DIR/${algo}.yaml" \
        --seed "$seed"
    )
    echo
  done
done

# Optionally generate summary figures/tables after all runs
if [[ "${POSTPROCESS:-0}" -eq 1 ]]; then
  python generate_figures.py
  python generate_tables.py
fi
