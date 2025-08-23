#!/usr/bin/env bash
set -euo pipefail

# Deterministic working dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

export PYTHONHASHSEED=0

# === Config ===
ENV_CONFIG="configs/env_8x8.yaml"
ALGO_DIR="configs/algo"

# Canonical algorithms to run (match YAMLs in configs/algo/)
# Ensure these YAMLs exist: ppo.yaml, ppo_icm.yaml, ppo_rnd.yaml, ppo_pc.yaml, ppo_planner.yaml,
# ppo_icm_planner.yaml, lppo.yaml, shielded.yaml, planner_subgoal.yaml, dyna.yaml
ALGOS=(ppo ppo_icm ppo_rnd ppo_pc ppo_planner ppo_icm_planner lppo shielded planner_subgoal dyna)

# Seeds (override via: SEEDS="0 1 2 3 4 5" ./scripts/run_all.sh)
if [[ -z "${SEEDS+x}" ]]; then
  SEEDS=(0 1 2 3 4)
else
  # shellcheck disable=SC2206
  SEEDS=(${SEEDS})
fi

# Map split(s) to evaluate during training; train.py should already export train/benchmark results.
SPLITS=(train test ood)

# === Run each algorithm across seeds ===
for algo in "${ALGOS[@]}"; do
  algo_yaml="${ALGO_DIR}/${algo}.yaml"
  if [[ ! -f "$algo_yaml" ]]; then
    echo "WARN: Missing ${algo_yaml}; skipping ${algo}"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    echo "=== Running ${algo} | seed ${seed} ==="
    OUT_DIR="runs/${algo}/seed_${seed}"
    mkdir -p "${OUT_DIR}"
    (
      cd "${OUT_DIR}"
      python "${REPO_ROOT}/train.py" \
        --env-config "${REPO_ROOT}/${ENV_CONFIG}" \
        --algo-config "${REPO_ROOT}/${algo_yaml}" \
        --seed "${seed}"
    )
    echo
  done
done

# === Ablation sweep (single entry point that toggles modules internally) ===
# If train.py supports --ablation with a CSV list, run it once per seed using the "full" config.
ABLATION_LIST="baseline,no_icm,no_rnd,no_planner"
if [[ -f "${ALGO_DIR}/ppo_icm_planner.yaml" ]]; then
  for seed in "${SEEDS[@]}"; do
    echo "=== Ablation sweep | seed ${seed} ==="
    OUT_DIR="runs/ablation/seed_${seed}"
    mkdir -p "${OUT_DIR}"
    (
      cd "${OUT_DIR}"
      python "${REPO_ROOT}/train.py" \
        --env-config "${REPO_ROOT}/${ENV_CONFIG}" \
        --algo-config "${REPO_ROOT}/${ALGO_DIR}/ppo_icm_planner.yaml" \
        --ablation "${ABLATION_LIST}" \
        --seed "${seed}"
    )
    echo
  done
else
  echo "WARN: Skipping ablation (configs/algo/ppo_icm_planner.yaml not found)"
fi

# === Optional post-processing (figures + tables) ===
# Run with: POSTPROCESS=1 ./scripts/run_all.sh
if [[ "${POSTPROCESS:-0}" -eq 1 ]]; then
  echo "=== Generating figures/tables ==="
  if [[ -f "generate_figures.py" ]]; then
    MPLBACKEND=Agg python generate_figures.py
  fi
  if [[ -f "generate_tables.py" ]]; then
    MPLBACKEND=Agg python generate_tables.py
  fi
fi

echo "All runs complete."
echo "Check per-run folders under runs/<algo>/seed_<k>/ and aggregated outputs under results/ and figures/."

