import argparse
import os
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as iio

# Ensure repository root on path so 'src' package is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.env import GridWorldICM
from src.visualization import plot_heatmap_with_path


def random_policy(env: GridWorldICM, _obs) -> int:
    """Sample a random action from the environment's action space."""
    return int(env.np_random.integers(0, 4))


def run_episode(env: GridWorldICM, policy, max_steps: int = 100):
    """Run an episode and collect rendered frames and the agent path."""
    obs, _ = env.reset()
    frames = [env.render()]
    path = [tuple(env.agent_pos)]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = policy(env, obs)
        obs, _, _, done, _, _ = env.step(action)
        frames.append(env.render())
        path.append(tuple(env.agent_pos))
        steps += 1
    return frames, path


def run_failure_episode(env: GridWorldICM):
    """Intentionally step onto the first mine to generate a failure reel."""
    obs, _ = env.reset()
    frames = [env.render()]
    path = [tuple(env.agent_pos)]
    mines = np.argwhere(env.mine_map)
    target = mines[0] if len(mines) else np.array([0, 0])
    done = False
    while not done:
        dx = target[0] - env.agent_pos[0]
        dy = target[1] - env.agent_pos[1]
        if dx != 0:
            action = 1 if dx > 0 else 0
        elif dy != 0:
            action = 3 if dy > 0 else 2
        else:
            action = 0
        obs, _, _, done, _, _ = env.step(action)
        frames.append(env.render())
        path.append(tuple(env.agent_pos))
        if len(path) > env.grid_size * 2:
            break
    return frames, path


def save_outputs(env: GridWorldICM, frames, path, prefix: str) -> None:
    """Save episode frames as MP4 and overlay path PDF."""
    out_path = Path(prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(str(out_path.with_suffix('.mp4')), frames, fps=5)
    plot_heatmap_with_path(env, path, str(out_path.with_suffix('.pdf')), show=False)


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative trajectory videos and overlays.")
    parser.add_argument('--methods', nargs='+', default=['demo'], help='Method names to label outputs.')
    parser.add_argument('--out-dir', default='qualitative', help='Directory to store generated files.')
    parser.add_argument('--num-failures', type=int, default=2, help='Number of failure reels per method.')
    args = parser.parse_args()

    seeds = [0, 1, 2]  # train, test, OOD
    for method in args.methods:
        for map_id, seed in enumerate(seeds):
            env = GridWorldICM(grid_size=8, dynamic_risk=True, dynamic_cost=True, seed=seed)
            frames, path = run_episode(env, random_policy)
            prefix = f"{args.out_dir}/{method}__map_{map_id}"
            save_outputs(env, frames, path, prefix)
        for fidx in range(args.num_failures):
            env = GridWorldICM(grid_size=8, dynamic_risk=True, dynamic_cost=True, seed=100 + fidx)
            frames, path = run_failure_episode(env)
            prefix = f"{args.out_dir}/{method}__failure_{fidx}"
            save_outputs(env, frames, path, prefix)


if __name__ == '__main__':
    main()
