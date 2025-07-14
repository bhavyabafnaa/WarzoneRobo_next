import os
from src.env import GridWorldICM
from src.ppo import PPOPolicy
from src.visualization import render_episode_video


def test_render_episode_video(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=5)
    policy = PPOPolicy(4 * env.grid_size * env.grid_size, 4)
    output = tmp_path / "episode.gif"
    render_episode_video(env, policy, str(output), max_steps=2, seed=0)
    assert output.exists()
