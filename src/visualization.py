import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(reward_log, intrinsic_log, success_flags):
    """Plot extrinsic/intrinsic rewards and success rate across episodes."""
    episodes = np.arange(len(reward_log))
    sns.set(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(x=episodes, y=reward_log, ax=ax1, label="Extrinsic")
    if intrinsic_log is not None:
        sns.lineplot(x=episodes, y=intrinsic_log, ax=ax1, label="Intrinsic")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards")
    ax1.legend()

    success_rate = np.cumsum(success_flags) / (np.arange(len(success_flags)) + 1)
    sns.lineplot(x=episodes, y=success_rate, ax=ax2, color="green")
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
