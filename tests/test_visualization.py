import numpy as np
import pandas as pd

from src.env import GridWorldICM
from src.ppo import PPOPolicy
from src.visualization import (
    render_episode_video,
    plot_pareto,
    plot_learning_panels,
    plot_violation_rate,
    plot_coverage_heatmap,
    plot_ablation_radar,
)


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


def test_plot_learning_panels(tmp_path):
    logs = {
        "MethodA": {
            "Reward": [list(np.linspace(0, 1, 5)), list(np.linspace(0, 1, 5))],
            "Success": [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        },
        "MethodB": {
            "Reward": [list(np.linspace(1, 2, 5)), list(np.linspace(1, 2, 5))],
            "Success": [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0]],
        },
    }
    output = tmp_path / "panels.pdf"
    plot_learning_panels(logs, str(output))
    assert output.exists()


def test_plot_pareto_multiple_methods(tmp_path):
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Reward Mean": [1.0, 2.0, 3.0],
            "Reward CI": [0.1, 0.2, 0.3],
            "Cost Mean": [0.4, 0.5, 0.6],
            "Cost CI": [0.04, 0.05, 0.06],
        }
    )
    output = tmp_path / "pareto_multi.pdf"
    plot_pareto(df, 0.55, str(output))
    assert output.exists()


def test_plot_violation_rate(tmp_path):
    logs = [[0, 1, 0, 1, 0], [0, 0, 1, 0, 0]]
    output = tmp_path / "violation.pdf"
    plot_violation_rate(logs, str(output))
    assert output.exists()


def test_plot_coverage_heatmap(tmp_path):
    counts = np.array([[0, 1], [2, 3]])
    output = tmp_path / "coverage.pdf"
    plot_coverage_heatmap(counts, str(output))
    assert output.exists()


def test_plot_ablation_radar(tmp_path):
    raw_metrics = {
        "baseline": {
            "Safety": [0.9, 0.8],
            "Reward": [100, 110],
            "Coverage": [50, 55],
            "Compute": [1.0, 1.1],
        },
        "no_icm": {
            "Safety": [0.7, 0.6],
            "Reward": [80, 90],
            "Coverage": [40, 42],
            "Compute": [0.9, 0.95],
        },
        "no_rnd": {
            "Safety": [0.75, 0.7],
            "Reward": [85, 88],
            "Coverage": [45, 47],
            "Compute": [0.85, 0.9],
        },
    }

    rows = []
    for setting, metrics in raw_metrics.items():
        rows.append(
            {
                "Setting": setting,
                "Safety": np.mean(metrics["Safety"]),
                "Reward": np.mean(metrics["Reward"]),
                "Coverage": np.mean(metrics["Coverage"]),
                "Compute": np.mean(metrics["Compute"]),
            }
        )
    metrics_df = pd.DataFrame(rows)

    output = tmp_path / "radar.pdf"
    plot_ablation_radar(metrics_df, str(output))
    assert output.exists()
