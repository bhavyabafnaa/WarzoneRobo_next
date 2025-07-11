import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import imageio


def _stack_logs(logs: list[list[float]] | None, name: str) -> pd.DataFrame:
    """Convert a list of episode logs into a long-form DataFrame."""
    if logs is None:
        return pd.DataFrame()
    frames = []
    for idx, log in enumerate(logs):
        frames.append(
            pd.DataFrame({"Episode": np.arange(len(log)), name: log, "Seed": idx})
        )
    return pd.concat(frames, ignore_index=True)


def plot_training_curves(
    reward_logs: list[list[float]],
    intrinsic_logs: list[list[float]] | None,
    success_logs: list[list[int]],
) -> None:
    """Plot mean extrinsic/intrinsic rewards and success rate across seeds."""

    if reward_logs and not isinstance(reward_logs[0], (list, np.ndarray)):
        reward_logs = [reward_logs]  # backwards compatibility
    if intrinsic_logs and intrinsic_logs and not isinstance(intrinsic_logs[0], (list, np.ndarray)):
        intrinsic_logs = [intrinsic_logs]
    if success_logs and not isinstance(success_logs[0], (list, np.ndarray)):
        success_logs = [success_logs]

    sns.set(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    reward_df = _stack_logs(reward_logs, "Value")
    sns.lineplot(data=reward_df, x="Episode", y="Value", ax=ax1, ci=95, label="Extrinsic")

    if intrinsic_logs is not None:
        intrinsic_df = _stack_logs(intrinsic_logs, "Value")
        sns.lineplot(data=intrinsic_df, x="Episode", y="Value", ax=ax1, ci=95, label="Intrinsic")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards")
    ax1.legend()

    success_rates = []
    for flags in success_logs:
        success_rates.append(np.cumsum(flags) / (np.arange(len(flags)) + 1))
    success_df = _stack_logs(success_rates, "Value")
    sns.lineplot(data=success_df, x="Episode", y="Value", ax=ax2, ci=95, color="green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("Success Rate")

    plt.tight_layout()
    plt.show()


def plot_heatmap_with_path(env, path):
    """Display cost and risk maps with an overlayed agent path."""
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, data, title in zip(
        axs,
        [env.cost_map, env.risk_map],
        ["Cost Map", "Risk Map"],
    ):
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.plot(xs, ys, color="red", marker="o")
        ax.plot(env.goal_pos[1], env.goal_pos[0], marker="*", color="lime", markersize=10)
        ax.set_title(title)
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def generate_results_table(df: pd.DataFrame, output_path: str) -> None:
    """Save benchmark results and optionally render an HTML or LaTeX table.

    A CSV file with the same base name as ``output_path`` is always written. If
    ``output_path`` ends with ``.html`` or ``.tex`` the table will also be
    exported in that format.
    """

    base, ext = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)

    if ext.lower() in {".html", ".htm"}:
        styled = df.style.hide_index().format(precision=2)
        styled.to_html(output_path)
    elif ext.lower() in {".tex", ".latex"}:
        latex = df.to_latex(index=False, float_format="%.2f")
        with open(output_path, "w") as f:
            f.write(latex)


def render_episode_video(env, policy, output_path: str, max_steps: int = 100, seed: int | None = None) -> None:
    """Run one episode and save a GIF of the agent interacting with the env."""
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    step = 0
    while not done and step < max_steps:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _ = policy.act(state_tensor)
        obs, _, done, _, _ = env.step(action)
        frames.append(env.render())
        step += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=5)
