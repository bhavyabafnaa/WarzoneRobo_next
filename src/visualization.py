import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import imageio
from pathlib import Path

from .planner import SymbolicPlanner

plt.switch_backend("Agg")


def _stack_logs(logs: list[list[float]] | None, name: str) -> pd.DataFrame:
    """Convert a list of episode logs into a long-form DataFrame."""
    if logs is None:
        return pd.DataFrame()
    frames = []
    for idx, log in enumerate(logs):
        frames.append(pd.DataFrame(
            {"Episode": np.arange(len(log)), name: log, "Seed": idx}))
    return pd.concat(frames, ignore_index=True)


def _save_fig(fig: plt.Figure, output_path: str) -> None:
    """Save figure to ``output_path`` as a PDF."""

    path = Path(output_path)
    os.makedirs(path.parent or ".", exist_ok=True)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    fig.savefig(path, format="pdf")


def _finalize_fig(fig: plt.Figure, output_path: str | None, show: bool = False) -> None:
    """Save and close a figure."""

    if output_path is not None:
        _save_fig(fig, output_path)
    plt.close(fig)


def plot_training_curves(
    metrics: dict[str, list[list[float]]],
    output_path: str | None = None,
    show: bool = False,
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
    _finalize_fig(fig, output_path, show)


def plot_coverage_heatmap(
    visit_counts: np.ndarray,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Visualize state visit frequencies as a normalized heatmap."""
    data = np.asarray(visit_counts, dtype=float)
    if data.ndim != 2:
        raise ValueError("visit_counts must be a 2D array")
    if data.max() > 0:
        data = data / data.max()

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data, origin="lower", cmap="viridis")
    ax.set_title("Visit Frequency")
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _finalize_fig(fig, output_path, show)


def plot_violation_rate(
    logs: list[list[float]] | None,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot running constraint violation probability with 95% CIs.

    ``logs`` should be a list of episode-wise violation flags for each seed.
    The function computes the cumulative probability of a violation over
    episodes and displays the mean trend with 95% confidence interval
    shading across seeds.
    """

    if not logs:
        return
    if logs and not isinstance(logs[0], (list, np.ndarray)):
        logs = [logs]

    # Compute running violation probabilities for each seed
    rates = [
        np.cumsum(seed) / (np.arange(len(seed)) + 1)
        for seed in logs
    ]
    min_len = min(len(r) for r in rates)
    arr = np.stack([r[:min_len] for r in rates])
    episodes = np.arange(1, min_len + 1)

    mean = arr.mean(axis=0)
    n = arr.shape[0]
    if n > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
        ci = 1.96 * sem
    else:
        ci = np.zeros_like(mean)

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes, mean, label="Violation Rate")
    ax.fill_between(episodes, mean - ci, mean + ci, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Violation Probability")
    ax.set_title("Constraint Violation Rate")

    plt.tight_layout()
    _finalize_fig(fig, output_path, show)


def plot_violation_comparison(
    method_logs: dict[str, list[list[float]]],
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Overlay mean violation rates with 95% CI bands for each method.

    Parameters
    ----------
    method_logs:
        Mapping from method name to a list of violation flag sequences, one per
        seed. Each sequence should be a list of binary values indicating whether
        a constraint violation occurred on that episode.
    output_path:
        Optional path to save the resulting figure. The file format is inferred
        from the extension and defaults to PDF.
    """

    if not method_logs:
        return

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted = False
    for name, logs in method_logs.items():
        if not logs:
            continue
        if logs and not isinstance(logs[0], (list, np.ndarray)):
            logs = [logs]

        # Convert violation flags to running probabilities for each seed
        rates = [
            np.cumsum(seed) / (np.arange(len(seed)) + 1)
            for seed in logs
        ]
        df = _stack_logs(rates, "Rate")
        if df.empty:
            continue
        sns.lineplot(
            data=df,
            x="Episode",
            y="Rate",
            errorbar=("ci", 95),
            ax=ax,
            label=name,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Violation Probability")
    ax.set_title("Constraint Violation Comparison")
    ax.legend()

    plt.tight_layout()
    _finalize_fig(fig, output_path, show)


def plot_learning_panels(
    metrics_dict: dict[str, dict[str, list[list[float]]]],
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot learning curves for multiple methods and metrics.

    ``metrics_dict`` should be structured as ``{method: {metric: logs}}`` where
    ``logs`` is a list of episode-value sequences for each seed. Subplots are
    arranged with methods as rows and metrics as columns. Shaded regions denote
    95% confidence intervals across seeds.
    """

    if not metrics_dict:
        return

    sns.set(style="darkgrid")
    methods = list(metrics_dict.keys())
    metric_names = sorted({m for logs in metrics_dict.values() for m in logs})
    n_rows = len(methods)
    n_cols = len(metric_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for i, method in enumerate(methods):
        for j, metric in enumerate(metric_names):
            ax = axes[i][j]
            logs = metrics_dict[method].get(metric)
            if not logs:
                ax.axis("off")
                continue
            if logs and not isinstance(logs[0], (list, np.ndarray)):
                logs = [logs]

            if metric.lower().startswith("success"):
                processed = [
                    np.cumsum(flags) / (np.arange(len(flags)) + 1) for flags in logs
                ]
                df = _stack_logs(processed, "Value")
                ax.set_ylim(0, 1)
            else:
                df = _stack_logs(logs, "Value")

            sns.lineplot(
                data=df,
                x="Episode",
                y="Value",
                ax=ax,
                errorbar=("ci", 95),
            )
            if i == n_rows - 1:
                ax.set_xlabel("Episode")
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(method)
            else:
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(metric)

    plt.tight_layout()
    _finalize_fig(fig, output_path, show)


def plot_ablation_radar(
    metrics_df: pd.DataFrame,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a radar chart comparing ablation metrics.

    The input DataFrame should contain a ``Setting`` column identifying each
    ablation configuration and one column for every metric to visualise
    (e.g. ``Safety``, ``Reward``, ``Coverage`` and ``Compute``). Each metric is
    linearly normalised to the ``[0, 1]`` range across all settings before being
    plotted on a polar (radar) chart.
    """

    if metrics_df.empty:
        return

    categories = [c for c in metrics_df.columns if c.lower() != "setting"]
    data = metrics_df[categories].astype(float)

    # Normalise metrics to [0, 1]
    mins = data.min()
    maxs = data.max()
    span = maxs - mins
    span[span == 0] = 1.0
    norm = (data - mins) / span
    norm["Setting"] = metrics_df["Setting"].values

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for _, row in norm.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["Setting"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title("Ablation Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    _finalize_fig(fig, output_path, show)


def plot_pareto(
    df: pd.DataFrame,
    cost_limit: float,
    output_path: str | None = None,
    show: bool = False,
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
    _finalize_fig(fig, output_path, show)


def plot_heatmap_with_path(
    env,
    path,
    output_path: str | None = None,
    show: bool = False,
):
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
    _finalize_fig(fig, output_path, show)


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
