import pandas as pd

from src.env import GridWorldICM
from src.ppo import PPOPolicy
from src.visualization import render_episode_video, plot_pareto


def test_render_episode_video(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=5)
    policy = PPOPolicy(4 * env.grid_size * env.grid_size + 2, 4)
    output = tmp_path / "episode.gif"
    render_episode_video(env, policy, str(output), max_steps=2, seed=0)
    assert output.exists()


def test_plot_pareto(tmp_path):
    df = pd.DataFrame(
        {
            "Model": ["A", "B"],
            "Reward Mean": [1.0, 2.0],
            "Reward CI": [0.1, 0.2],
            "Cost Mean": [0.5, 0.7],
            "Cost CI": [0.05, 0.07],
        }
    )
    output = tmp_path / "pareto.pdf"
    plot_pareto(df, 0.6, str(output))
    assert output.exists()
