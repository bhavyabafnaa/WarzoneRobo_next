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

## Components
* **PPO** – The main reinforcement learning algorithm used to learn policies from environment interaction.
* **ICM** – Adds intrinsic rewards based on prediction error of the agent's dynamics model to promote exploring unseen states.
* **RND** – Provides exploration bonuses by comparing a fixed random network with a trained predictor network.
* **Planner** – A symbolic planner computes heuristic paths that the agent can follow, helping integrate classical planning with learned policies.

The notebook experiments with different combinations of these components to evaluate their effect on success rate and exploration.
