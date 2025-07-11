# WarzoneRobo

WarzoneRobo contains research code exploring reinforcement learning (RL) techniques for navigating grid based strategy maps. The project compares vanilla policy gradients with techniques that improve exploration or planning.

## Project goals
* Train an agent using Proximal Policy Optimization (PPO).
* Augment the agent with Intrinsic Curiosity Module (ICM) and Random Network Distillation (RND) to encourage exploration.
* Combine RL with a symbolic planner to reduce search space and guide decision making.

## Getting started
The repository includes a Jupyter notebook named `RLcode` containing the training and evaluation code. You can run the notebook interactively or execute its cells as a script.

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook RLcode
```

Alternatively, you can run the notebook from the command line using `jupyter nbconvert`:

```bash
jupyter nbconvert --to notebook --execute RLcode --output output.ipynb
```

### Command line training
You can also train the models directly with `train.py`. Hyperparameters can be
supplied via command line flags or a YAML configuration file:

```bash
python train.py --grid_size 8 --num_episodes 200
```

or

```bash
python train.py --config configs/default.yaml
```

The repository includes `configs/default.yaml` as a starting configuration.
Duplicate and modify this file to experiment with different training settings.
Additional environment options can also be specified in the YAML:

```yaml
grid_size: 8
num_episodes: 200
dynamic_risk: true      # enemies increase risk over time
add_noise: true         # perturb loaded maps on reset
```

### Generating benchmark tables
After training, `train.py` evaluates each agent on the exported benchmark maps.
The metrics are saved to `results/benchmark_results.csv` and, when the output
path ends with `.html` or `.tex`, an additional formatted table is produced.

```bash
python train.py --num_episodes 200
# CSV and HTML tables are written to the `results/` folder
```

Paired t-tests are performed against the PPO baseline for each variant.
Differences are considered significant when `p < 0.05`.

## Components
* **PPO** – The main reinforcement learning algorithm used to learn policies from environment interaction.
* **ICM** – Adds intrinsic rewards based on prediction error of the agent's dynamics model to promote exploring unseen states.
* **RND** – Provides exploration bonuses by comparing a fixed random network with a trained predictor network.
* **Planner** – A symbolic planner computes heuristic paths that the agent can follow, helping integrate classical planning with learned policies.

The notebook experiments with different combinations of these components to evaluate their effect on success rate and exploration.

## Environment features
The grid world includes an optional *dynamic risk* mode where risk levels grow as enemies move. Benchmark maps can be exported with `export_benchmark_maps` and loaded later for evaluation. The environment's `render()` method returns RGB frames so that `render_episode_video` can produce GIFs of agent behavior.

## Running Experiments
Train all models from a configuration file:
```bash
python train.py --config configs/default.yaml
```
Checkpoints are saved under `checkpoints/`, episode videos under `videos/`, and result tables under `results/`. Hyperparameters such as planner weights (`cost_weight`, `risk_weight`, etc.) can be edited in the YAML file or passed as command-line flags.

To repeat an experiment with multiple random seeds you can loop over the `--seed` argument:

```bash
for s in 0 1 2 3 4; do
    python train.py --config configs/default.yaml --seed $s
done
```

## Running Tests
Execute the unit tests with:
```bash
pytest -q
```
The full suite runs in well under a minute on a CPU.
