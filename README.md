# WarzoneRobo

WarzoneRobo explores how modern reinforcement learning (RL) techniques handle
survival-focused planning on adversarial grid maps. The project combines PPO,
intrinsic motivation, and symbolic reasoning to study how agents balance
exploration, risk, and long-horizon credit assignment.

## Why this project stands out to research scientists
- **Methodological depth** – A controlled environment for comparing curiosity
  modules, planning guidance, and policy optimization under identical
  conditions.
- **Analysis-first workflow** – Built-in notebooks, scripts, and logging
  pipelines accelerate ablation studies, statistical testing, and visualization
  of learned behaviors.
- **Reproducibility** – Deterministic seeds, environment manifests, and a
  Docker image capture the full experimental context for peer review or hiring
  evaluations.

## Core research contributions
1. **Intrinsic motivation under adversarial pressure** – Demonstrates how ICM
   and RND bonuses reshape exploration when each step carries survival cost.
2. **Planner-RL hybridization** – Integrates a symbolic planner that hands off
   control as learning progresses, highlighting the interaction between model-
   based priors and policy gradients.
3. **Dynamic hazard modeling** – Evaluates agents on maps with evolving risk and
   cost fields to surface brittleness that static benchmarks miss.

## Methodology overview
- **Environment** – Grid-based survival task with optional dynamic risk and
  cost schedules, enemy units, and map perturbations. RGB rendering enables
  qualitative inspection and video exports.
- **Learning algorithms** – PPO baseline augmented with curiosity (ICM, RND)
  and planner bonuses that decay over training to test autonomy transfer.
- **Evaluation** – Benchmarks log success rates, reward curves, and statistical
  tests against PPO across multiple seeds, producing CSV/HTML/TeX summaries in
  `results/`.

## Quickstart
Use the demonstration notebook for rapid exploration or run the training
pipeline from the command line.

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive walkthrough
jupyter notebook demo.ipynb

# Or execute the notebook headlessly
jupyter nbconvert --to notebook --execute demo.ipynb --output output.ipynb
```

### Command-line experimentation

```bash
# Minimal PPO run
python train.py --grid_size 8 --num_episodes 200

# Configuration-driven experiment
python train.py --config configs/default.yaml

# Mix-and-match environment and algorithm settings
python train.py --env-config configs/env_8x8.yaml --algo-config configs/algo/lppo.yaml

# Persist training figures for analysis
python train.py --config configs/default.yaml --plot-dir figures
```

`configs/default.yaml` seeds both NumPy and PyTorch, enabling deterministic
CuDNN behavior for reproducible hiring packets or academic artifacts. Key
environment options include:

```yaml
grid_size: 8
num_episodes: 200
dynamic_risk: true      # enemies increase risk over time
dynamic_cost: true      # cost near mines decays and rises dynamically
add_noise: true         # perturb loaded maps on reset
```

### Automating sweeps & ablations
The `scripts/` directory houses deterministic helpers for scaling experiments:

```bash
# Run all algorithm variants sequentially
./scripts/run_all.sh

# Evaluate every checkpoint on held-out benchmark maps
./scripts/eval_all.sh

# Full suite: algorithms × seeds × ablations with figure/table generation
./scripts/full_experiment.sh
```

Loop over seeds to report confidence intervals:

```bash
for s in 0 1 2 3 4; do
    python train.py --config configs/default.yaml --seed $s
done
```

Curiosity-specific controls let you probe causality:

```bash
python train.py --initial-beta 0.2 --final-beta 0.05
python train.py --initial-beta 0.2 --final-beta 0.05 --disable_icm
```

## Evaluation assets
- **Tables** – `results/benchmark_results.csv` plus optional HTML/TeX exports
  with significance stars (paired, Welch, Mann–Whitney, or ANOVA tests via
  `--stat-test`).
- **Figures** – Reward curves, state visitation heatmaps, and curiosity decay
  diagnostics in `figures/` when `--plot-dir` is specified.
- **Videos** – Episode rollouts stored under `videos/` for qualitative review,
  useful in interview packets to showcase behavior shifts.

## Testing & quality gates
Run the targeted unit tests before sharing results:

```bash
pytest -q
```

The suite completes in under a minute on CPU, making it practical for iterative
development or code review in hiring scenarios.

## Reproducibility checklist
1. Clone the repository and build the Docker image.
2. Launch experiments inside the container.
3. Archive `manifest.txt`, `configs/`, and generated artifacts for auditability.

```bash
git clone <repo-url>
cd WarzoneRobo
docker build -t warzonerobo .
docker run --rm warzonerobo python train.py --config configs/default.yaml
```

Checkpoints appear under `checkpoints/` and benchmark tables under `results/`.
`manifest.txt` stores the commit hash used to generate the release.

## Project structure
- `src/` – Environment, agent implementations, planners, and training loops.
- `configs/` – Canonical experiment definitions with environment and algorithm
  splits for ablation studies.
- `scripts/` – End-to-end automation for sweeps, evaluation, and reporting.
- `figures/`, `videos/`, `results/` – Generated analysis artifacts ready for
  inclusion in research dossiers or candidate portfolios.
- `demo.ipynb` – Guided tour of the environment and modeling components.

## Looking ahead
Potential next steps for collaborators or interview follow-ups include scaling
to continuous control maps, integrating model-based rollouts, and evaluating
transfer to procedurally generated terrains. These directions highlight the
project's flexibility for future research agendas.
