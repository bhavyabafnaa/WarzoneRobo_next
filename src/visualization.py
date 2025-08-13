import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import imageio

from .planner import SymbolicPlanner


def _stack_logs(logs: list[list[float]] | None, name: str) -> pd.DataFrame:
    """Convert a list of episode logs into a long-form DataFrame."""
    if logs is None:
        return pd.DataFrame()
    frames = []
    for idx, log in enumerate(logs):
        frames.append(pd.DataFrame(
            {"Episode": np.arange(len(log)), name: log, "Seed": idx}))
    return pd.concat(frames, ignore_index=True)


def plot_training_curves(
    metrics: dict[str, list[list[float]]],
    output_path: str | None = None,
) -> None:
    """Plot training metrics across seeds with 95% confidence intervals."""

    sns.set(style="darkgrid")
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, logs) in zip(axes, metrics.items()):
        if logs and not isinstance(logs[0], (list, np.ndarray)):
            logs = [logs]

        if name.lower().startswith("success"):
            processed = [np.cumsum(flags) / (np.arange(len(flags)) + 1) for flags in logs]
            df = _stack_logs(processed, "Value")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Success Rate")
        else:
            df = _stack_logs(logs, "Value")
            ax.set_ylabel(name)

        sns.lineplot(
            data=df,
            x="Episode",
            y="Value",
            ax=ax,
            errorbar=("ci", 95),
        )
        ax.set_xlabel("Episode")
        ax.set_title(name)

    plt.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        fmt = "svg" if ext == ".svg" else "pdf"
        plt.savefig(output_path, format=fmt)
    plt.show()


def plot_pareto(
    df: pd.DataFrame, cost_limit: float, output_path: str | None = None
) -> None:
    """Scatter mean reward vs. mean cost with 95% confidence intervals.

    ``df`` should contain the columns ``Model``, ``Reward Mean``, ``Reward CI``,
    ``Cost Mean`` and ``Cost CI``. A vertical dashed line at ``cost_limit`` is
    drawn to indicate the budget ``d``.
    """

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in df.iterrows():
        ax.errorbar(
            row["Cost Mean"],
            row["Reward Mean"],
            xerr=row.get("Cost CI", 0.0),
            yerr=row.get("Reward CI", 0.0),
            fmt="o",
            label=row.get("Model", ""),
        )
    ax.axvline(cost_limit, color="red", linestyle="--", label=f"Budget d={cost_limit:.2f}")
    ax.set_xlabel("Mean Cost")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    plt.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        fmt = "svg" if ext == ".svg" else "pdf"
        plt.savefig(output_path, format=fmt)
    plt.show()


def plot_heatmap_with_path(env, path, output_path: str | None = None):
    """Display cost and risk maps with an overlayed agent path.

    The goal position is no longer plotted.
    """
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
        ax.set_title(title)
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        fmt = "svg" if ext == ".svg" else "pdf"
        plt.savefig(output_path, format=fmt)
    plt.show()


def generate_results_table(df: pd.DataFrame, output_path: str) -> None:
    """Save benchmark results and optionally render an HTML or LaTeX table.

    A CSV file with the same base name as ``output_path`` is always written. If
    ``output_path`` ends with ``.html`` or ``.tex`` the table will also be
    exported in that format.
    """

    base, ext = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    p_col = "Reward p-adj" if "Reward p-adj" in df.columns else "Reward p-value"
    if p_col in df.columns:
        def _mark(p: float) -> str:
            if pd.isna(p):
                return ""
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        df = df.copy()
        df["Significance"] = df[p_col].apply(_mark)

    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)

    if ext.lower() in {".html", ".htm"}:
        # ``hide_index`` was removed in newer versions of pandas in favour of
        # ``hide``. Use whichever is available for compatibility across
        # different pandas releases.
        styled = df.style.format(precision=2)
        if hasattr(styled, "hide_index"):
            styled = styled.hide_index()
        else:
            styled = styled.hide(axis="index")
        styled.to_html(output_path)
    elif ext.lower() in {".tex", ".latex"}:
        latex = df.to_latex(index=False, float_format="%.2f")
        with open(output_path, "w") as f:
            f.write(latex)


def render_episode_video(
        env,
        policy,
        output_path: str,
        max_steps: int = 100,
        seed: int | None = None,
        H: int = 8) -> None:
    """Run one episode and save a GIF of the agent interacting with the env."""
    obs, _ = env.reset(seed=seed)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    g = planner.get_subgoal(env.agent_pos, H)
    subgoal_timer = H
    dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
    obs = np.concatenate([obs, np.array([dx, dy], dtype=np.float32)])
    frames = [env.render()]
    done = False
    step = 0
    while not done and step < max_steps:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _, _ = policy.act(state_tensor)
        obs_base, _, _, done, _, _ = env.step(action)
        subgoal_timer -= 1
        if subgoal_timer <= 0 or tuple(env.agent_pos) == g:
            g = planner.get_subgoal(env.agent_pos, H)
            subgoal_timer = H
        dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
        obs = np.concatenate([obs_base, np.array([dx, dy], dtype=np.float32)])
        frames.append(env.render())
        step += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fps = 5
    duration = int(1000 / fps)
    imageio.mimsave(output_path, frames, duration=duration)
