import os
import warnings
import argparse
import sys
import yaml
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import subprocess
from pathlib import Path
from scipy import stats
from scipy.stats import (
    ttest_rel,
    ttest_ind,
    mannwhitneyu,
    friedmanchisquare,
)
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from collections import defaultdict

from src.env import (
    GridWorldICM,
    export_benchmark_maps,
    visualize_paths_on_benchmark_maps,
    evaluate_on_benchmarks,
)
from src.visualization import (
    plot_learning_panels,
    plot_pareto,
    plot_heatmap_with_path,
    plot_coverage_heatmap,
    generate_results_table,
    render_episode_video,
    plot_violation_rate,
    plot_violation_comparison,
)
from src.icm import ICMModule
from src.rnd import RNDModule
from src.pseudocount import PseudoCountExploration
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent, get_beta_schedule
from src.utils import save_model, count_intrinsic_spikes
from src.safety import (
    save_pareto_summaries,
    save_violation_curves,
    append_budget_sweep,
)


for d in ["videos", "results", "figures", "checkpoints"]:
    os.makedirs(d, exist_ok=True)


# Canonical method names used for the main results table
MAIN_METHODS = [
    "PPO",
    "PPO+ICM",
    "PPO+RND",
    "Count",
    "PC",
    "LPPO",
    "Shielded-PPO",
    "Planner-only",
    "Planner-Subgoal PPO",
    "Dyna-PPO(1)",
    "Hybrid PPO+ICM+Planner",
]

# Mapping from internal model names to canonical names
NAME_MAP = {
    "PPO Only": "PPO",
    "PPO + ICM": "PPO+ICM",
    "PPO + RND": "PPO+RND",
    "PPO + count": "Count",
    "PPO + PC": "PC",
    "PPO + ICM + Planner": "Hybrid PPO+ICM+Planner",
}


def build_main_table(df_train: pd.DataFrame) -> pd.DataFrame:
    """Return filtered table for ``MAIN_METHODS`` with key metrics and p-values."""

    col_map = {
        "Train Reward": "Reward (±CI)",
        "Reward AUC": "Reward AUC (±CI)",
        "Success": "Success (±CI)",
        "Train Cost": "Avg Cost (±CI)",
        "Pr[Jc > d]": "Violations % (±CI)",
        "Planner Adherence %": "Planner Adherence %",
        "Masked Action Rate": "Masked %",
        "Unique Cells": "Unique Cells",
        "Reward p-value": "p_reward",
        "Violation p-value": "p_violation",
    }
    cols = ["Model"] + list(col_map.values())
    if "Model" not in df_train.columns:
        return pd.DataFrame(columns=cols)
    table = df_train[df_train["Model"].isin(MAIN_METHODS)].copy()
    table = table[["Model"] + list(col_map.keys())]
    table = table.rename(columns=col_map)
    return table.reset_index(drop=True)


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Return the mean and 95% confidence interval for ``values``."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    ci = float(stats.sem(arr) * stats.t.ppf(0.975, len(arr) - 1))
    return mean, ci


def bootstrap_ci(values: list[float], n_resamples: int = 10_000) -> tuple[float, float]:
    """Return the mean and 95% bootstrap confidence interval for ``values``."""
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    resampled_means = np.mean(
        np.random.choice(arr, size=(n_resamples, len(arr)), replace=True), axis=1
    )
    lower, upper = np.percentile(resampled_means, [2.5, 97.5])
    return mean, float((upper - lower) / 2)


def compute_cohens_d(
    baseline: np.ndarray, method: np.ndarray, paired: bool = False
) -> float:
    """Compute Cohen's d effect size between two samples.

    If ``paired`` is ``True`` the effect size is computed on the differences
    between paired observations. Otherwise, the pooled standard deviation of
    the two samples is used. ``baseline`` and ``method`` correspond to the
    baseline and comparison samples respectively, so a positive value indicates
    the comparison sample has a higher mean.
    """

    x = np.asarray(baseline, dtype=float)
    y = np.asarray(method, dtype=float)
    if paired:
        if x.shape != y.shape:
            raise ValueError("Paired samples must have the same shape")
        diff = y - x
        denom = diff.std(ddof=1)
        if denom == 0:
            return 0.0
        return diff.mean() / denom
    n1 = x.size
    n2 = y.size
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = x.var(ddof=1)
    var2 = y.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (y.mean() - x.mean()) / pooled


def format_mean_ci(values: list[float], scale: float = 1.0) -> str:
    """Format mean ± 95% CI string for ``values`` with optional scaling."""
    scaled = np.asarray(values, dtype=float) * scale
    mean, ci = mean_ci(scaled)
    return f"{mean:.2f} ± {ci:.2f}"


def format_bootstrap_ci(
    values: list[float], scale: float = 1.0, n_resamples: int = 10_000
) -> str:
    """Format mean ± 95% bootstrap CI string for ``values``."""
    scaled = np.asarray(values, dtype=float) * scale
    mean, ci = bootstrap_ci(scaled, n_resamples=n_resamples)
    return f"{mean:.2f} ± {ci:.2f}"


def check_reward_difference_ci(
    baseline_rewards: list[float],
    method_rewards: list[float],
    threshold: float = 0.2,
) -> tuple[float, float]:
    """Compare reward difference against combined CI width.

    Emits a warning if the sum of 95% CI half-widths exceeds ``threshold``
    times the absolute difference in means, suggesting more seeds/maps are
    needed for confident comparison.

    Returns the observed difference and combined CI width.
    """
    base_mean, base_ci = mean_ci(baseline_rewards)
    meth_mean, meth_ci = mean_ci(method_rewards)
    diff = abs(meth_mean - base_mean)
    ci_width = base_ci + meth_ci
    if diff > 0 and ci_width > threshold * diff:
        warnings.warn(
            (
                f"Reward CI width {ci_width:.3f} exceeds {threshold * 100:.1f}% "
                f"of difference {diff:.3f}. Consider using more seeds/maps."
            ),
            RuntimeWarning,
        )
    return diff, ci_width


EPISODE_COLUMNS = [
    "reward",
    "success",
    "cost_sum",
    "steps",
    "unique_cells",
    "min_enemy_dist",
    "planner_adherence_pct",
    "masked_action_rate",
    "lambda_lagrange",
    "wall_clock",
    "near_miss_count",
    "intrinsic_icm_sum",
    "intrinsic_rnd_sum",
    "intrinsic_spike_count",
    "policy_entropy_mean",
    "value_loss",
    "kl_policy",
    "dyna_model_loss",
    "seed",
    "map_id",
]


def save_episode_metrics(method: str, run_seed: int, split: str, episode_records: list[dict]):
    """Save per-episode metrics to a CSV file."""
    if not episode_records:
        return
    os.makedirs("results/episodes", exist_ok=True)
    df = pd.DataFrame(episode_records)
    df = df[EPISODE_COLUMNS]
    safe_method = method.replace(" ", "_").replace("+", "_")
    out_path = f"results/episodes/{safe_method}__seed{run_seed}__split-{split}.csv"
    df.to_csv(out_path, index=False)


def evaluate_policy_on_maps(
    env: GridWorldICM,
    policy: PPOPolicy,
    map_folder: str,
    num_maps: int,
    H: int,
) -> tuple[list[float], list[int]]:
    """Run ``policy`` on each map in ``map_folder`` and record rewards and
    success flags.

    Returns lists of rewards and binary success values aligned by map index."""

    rewards: list[float] = []
    successes: list[int] = []
    for i in range(num_maps):
        map_path = f"{map_folder}/map_{i:02d}.npz"
        obs, _ = env.reset(load_map_path=map_path)
        planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
        g = planner.get_subgoal(env.agent_pos, H)
        subgoal_timer = H
        dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
        obs = np.concatenate([obs, np.array([dx, dy], dtype=np.float32)])
        done = False
        total_reward = 0.0
        step_count = 0
        alive = True
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _, _ = policy.act(state_tensor)
            obs_base, reward, _cost, done, _trunc, info = env.step(action)
            subgoal_timer -= 1
            step_count += 1
            if subgoal_timer <= 0 or tuple(env.agent_pos) == g:
                g = planner.get_subgoal(env.agent_pos, H)
                subgoal_timer = H
            dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
            obs = np.concatenate([obs_base, np.array([dx, dy], dtype=np.float32)])
            total_reward += reward
            if info.get("dead", False):
                alive = False
        rewards.append(total_reward)
        successes.append(1 if step_count >= env.max_steps and alive else 0)
    return rewards, successes


def evaluate_planner_on_maps(
    env: GridWorldICM, map_folder: str, num_maps: int
) -> tuple[list[float], list[int]]:
    """Evaluate the standalone planner on ``num_maps`` maps."""

    rewards: list[float] = []
    successes: list[int] = []
    for i in range(num_maps):
        map_path = f"{map_folder}/map_{i:02d}.npz"
        obs, _ = env.reset(load_map_path=map_path)
        planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
        done = False
        total_reward = 0.0
        step_count = 0
        alive = True
        while not done:
            action = planner.get_safe_subgoal(env.agent_pos)
            obs, reward, _cost, done, _trunc, info = env.step(action)
            step_count += 1
            total_reward += reward
            if info.get("dead", False):
                alive = False
        rewards.append(total_reward)
        successes.append(1 if step_count >= env.max_steps and alive else 0)
    return rewards, successes


def compute_visit_counts_on_map(
    env: GridWorldICM, policy: PPOPolicy, map_path: str, H: int
) -> np.ndarray:
    """Run ``policy`` on ``map_path`` once and return state visit counts."""

    obs, _ = env.reset(load_map_path=map_path)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    g = planner.get_subgoal(env.agent_pos, H)
    subgoal_timer = H
    dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
    obs = np.concatenate([obs, np.array([dx, dy], dtype=np.float32)])
    counts = np.zeros((env.grid_size, env.grid_size), dtype=np.int32)
    counts[tuple(env.agent_pos)] += 1
    done = False
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _, _ = policy.act(state_tensor)
        obs_base, _rew, _cost, done, _trunc, _info = env.step(action)
        subgoal_timer -= 1
        if subgoal_timer <= 0 or tuple(env.agent_pos) == g:
            g = planner.get_subgoal(env.agent_pos, H)
            subgoal_timer = H
        dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
        obs = np.concatenate([obs_base, np.array([dx, dy], dtype=np.float32)])
        counts[tuple(env.agent_pos)] += 1
    return counts


def plot_policy_coverage(
    env: GridWorldICM,
    policy: PPOPolicy,
    method: str,
    setting_name: str,
    plot_dir: str | None,
    H: int,
) -> None:
    """Compute visit frequencies for a map and plot a coverage heatmap."""

    map_path = "test_maps/map_00.npz"
    counts = compute_visit_counts_on_map(env, policy, map_path, H)
    out_file = None
    if plot_dir:
        safe_setting = setting_name.replace(" ", "_")
        safe_name = method.replace(" ", "_")
        out_file = os.path.join(
            plot_dir, f"{safe_setting}_{safe_name}_coverage.pdf"
        )
    plot_coverage_heatmap(counts, output_path=out_file)


def get_paired_arrays(
    baseline: dict[int, list[float]],
    method: dict[int, list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned arrays for paired statistical tests.

    For each seed present in both ``baseline`` and ``method`` the rewards are
    paired by map index and truncated to the minimum available length."""

    base_flat: list[float] = []
    method_flat: list[float] = []
    for seed, base_vals in baseline.items():
        if seed in method:
            meth_vals = method[seed]
            n = min(len(base_vals), len(meth_vals))
            base_flat.extend(base_vals[:n])
            method_flat.extend(meth_vals[:n])
    return np.asarray(base_flat, dtype=float), np.asarray(method_flat, dtype=float)


def flatten_metric(metric_dict: dict[int, list[float]]) -> list[float]:
    """Flatten nested seed→values dict into a single list."""
    values: list[float] = []
    for vals in metric_dict.values():
        values.extend(vals)
    return values


def compute_auc_reward(reward_log: list[float]) -> float:
    """Return area under the reward curve across episodes.

    Uses the trapezoidal rule over episode indices. An empty ``reward_log``
    yields ``0.0``.
    """

    if not reward_log:
        return 0.0
    episodes = np.arange(len(reward_log))
    return float(np.trapz(reward_log, episodes))


def write_aggregate_csv(
    method: str, data: dict, split: str, out_dir: str = "results/aggregates"
) -> pd.DataFrame:
    """Compute mean and SD for key metrics and write to a CSV file.

    Parameters
    ----------
    method : str
        Name of the evaluated method.
    data : dict
        Dictionary containing lists of metric values across runs.
    split : str
        Name of the dataset split; used in the output filename.
    out_dir : str, default "results/aggregates"
        Directory where the aggregate CSV will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the aggregate statistics.
    """

    metrics_to_aggregate: dict[str, list[float]] = {
        "reward": flatten_metric(data.get("rewards", {})),
        "success": flatten_metric(data.get("success", {})),
        "cost": data.get("episode_costs", []),
        "violation_rate": data.get("violation_flags", []),
        "unique_cells": data.get("unique_cells", []),
        "planner_adherence_pct": data.get("planner_adherence_pct", []),
        "masked_action_rate": data.get("masked_action_rate", []),
        "intrinsic_spike_count": data.get("spikes", []),
    }
    steps_values = data.get("steps_per_sec", [])
    if steps_values:
        metrics_to_aggregate["steps_per_sec"] = steps_values

    stats: dict[str, float] = {}
    for key, values in metrics_to_aggregate.items():
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean()) if arr.size else 0.0
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        stats[f"{key}_mean"] = mean
        stats[f"{key}_sd"] = sd

    df = pd.DataFrame([stats])
    os.makedirs(out_dir, exist_ok=True)
    safe_method = method.replace(" ", "_")
    out_path = os.path.join(out_dir, f"{safe_method}__split-{split}.csv")
    df.to_csv(out_path, index=False)
    return df


def parse_args(arg_list: list[str] | None = None):
    """Parse command line arguments with optional YAML config defaults.

    Parameters
    ----------
    arg_list : list[str] | None
        Optional list of arguments to parse. Useful for tests; when ``None``
        (the default) ``sys.argv`` is used.

    Configuration can be provided in up to three YAML files. ``--env-config``
    is loaded first and typically contains environment settings such as grid
    size or hazard densities. ``--algo-config`` is then applied on top to
    override or extend those defaults with algorithm-specific hyperparameters.
    Finally, ``--config`` can be used for a monolithic config file or to make
    run-specific tweaks. Command line flags always take precedence, allowing
    quick overrides without editing any YAML files.

    Example YAML snippet::

        grid_size: 8
        dynamic_risk: true
        add_noise: false

    Example usage::

        python train.py --env-config configs/env_8x8.yaml \
                         --algo-config configs/algo/lppo.yaml
    """

    parser = argparse.ArgumentParser(
        description="Train or evaluate PPO agents")
    parser.add_argument(
        "--env-config",
        type=str,
        help="Path to environment YAML config file",
        default=None,
    )
    parser.add_argument(
        "--algo-config",
        type=str,
        help="Path to algorithm YAML config file",
        default=None,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to additional YAML config file",
        default=None,
    )
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=100)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "ood"],
        default="train",
        help="Dataset split name for aggregate metrics",
    )
    parser.add_argument(
        "--cost_weight",
        type=float,
        nargs="+",
        default=[2.0],
        help="Planner cost weight(s) for sweeps",
    )
    parser.add_argument(
        "--risk_weight",
        type=float,
        nargs="+",
        default=[3.0],
        help="Planner risk weight(s) for sweeps",
    )
    parser.add_argument("--revisit_penalty", type=float, default=1.0)
    parser.add_argument(
        "--lambda_cost",
        type=float,
        default=0.0,
        help="Initial value for the Lagrange multiplier",
    )
    parser.add_argument(
        "--eta_lambda",
        type=float,
        choices=[0.01, 0.05],
        default=0.05,
        help="Learning rate for lambda update",
    )
    parser.add_argument(
        "--cost_limit",
        "--d",
        dest="cost_limit",
        type=float,
        default=0.05,
        help="Cost threshold for constraint",
    )
    parser.add_argument("--c1", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument(
        "--danger-distance",
        dest="danger_distance",
        type=int,
        default=2,
        help="Manhattan distance considered a near miss",
    )
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=[0.7],
        help="Risk threshold(s) for masking unsafe actions",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        nargs="+",
        choices=[2, 4],
        default=[4.0],
        help="Soft risk penalty weight(s) applied to policy logits",
    )
    parser.add_argument(
        "--disable-risk-penalty",
        action="store_true",
        help="Disable risk-based soft penalty on logits",
    )
    parser.add_argument(
        "--initial-beta",
        type=float,
        nargs="+",
        default=[0.1],
        help="Starting weight(s) for intrinsic reward",
    )
    parser.add_argument(
        "--final-beta",
        type=float,
        nargs="+",
        default=[0.01],
        help="Final beta value(s) after decay",
    )
    parser.add_argument(
        "--planner-bonus-decay",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multiplier(s) for planner bonus decay rate",
    )
    parser.add_argument(
        "--dynamic-risk",
        dest="dynamic_risk",
        action="store_true",
        help="Enable dynamic risk in env",
    )
    parser.add_argument(
        "--dynamic-cost",
        dest="dynamic_cost",
        action="store_true",
        help="Enable dynamic cost in env",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise when resetting maps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[10],
        help="List of seeds or single integer count",
    )
    parser.add_argument(
        "--all-algos",
        action="store_true",
        help="Run all algorithm configs found in configs/algo/",
    )
    parser.add_argument(
        "--disable_icm",
        action="store_true",
        help="Disable the intrinsic curiosity module",
    )
    parser.add_argument(
        "--disable_rnd",
        action="store_true",
        help="Disable random network distillation",
    )
    parser.add_argument(
        "--disable_planner",
        action="store_true",
        help="Disable the symbolic planner",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=10,
        help="Subgoal planning horizon",
    )
    parser.add_argument(
        "--waypoint_bonus",
        type=float,
        default=0.05,
        help="Reward bonus for moving toward subgoal",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Number of model-based transitions per real step",
    )
    parser.add_argument(
        "--world_model_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the world model",
    )
    parser.add_argument(
        "--mine_density_range",
        type=float,
        nargs=2,
        default=[0.03, 0.10],
        help="Range for mine density",
    )
    parser.add_argument(
        "--hazard_density_range",
        type=float,
        nargs=2,
        default=[0.0, 0.2],
        help="Range for initial hazard density",
    )
    parser.add_argument(
        "--enemy_speed_range",
        type=int,
        nargs=2,
        default=[1, 2],
        help="Range for enemy movement speed",
    )
    parser.add_argument(
        "--enemy_policies",
        type=str,
        nargs="+",
        default=["random", "aggressive", "stationary"],
        help="Enemy movement policies to sample from",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Loop over disabling ICM, RND and the planner individually",
    )
    parser.add_argument(
        "--log_backend",
        choices=["wandb", "tensorboard", "none"],
        default="none",
        help="Logging backend to use",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save training plots",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Generate figures and tables after training",
    )
    parser.add_argument(
        "--stat-test",
        choices=["paired", "welch", "mannwhitney", "anova", "friedman"],
        default="paired",
        help="Statistical test for result comparisons",
    )

    # Parse once to read the config file path and load defaults from YAML. We
    # intentionally parse without removing any of the original command line
    # arguments so they can still override the config on the second pass.
    config_args, _ = parser.parse_known_args(arg_list)
    merged_cfg: dict[str, object] = {}
    if config_args.env_config and os.path.exists(config_args.env_config):
        with open(config_args.env_config, "r") as f:
            merged_cfg.update(yaml.safe_load(f) or {})
    if config_args.algo_config and os.path.exists(config_args.algo_config):
        with open(config_args.algo_config, "r") as f:
            merged_cfg.update(yaml.safe_load(f) or {})
    if config_args.config and os.path.exists(config_args.config):
        with open(config_args.config, "r") as f:
            merged_cfg.update(yaml.safe_load(f) or {})
    if merged_cfg:
        parser.set_defaults(**merged_cfg)

    # Final parse with config defaults applied; command line flags take
    # precedence over YAML settings.
    args = parser.parse_args(arg_list)
    for name in [
        "cost_weight",
        "risk_weight",
        "tau",
        "kappa",
        "initial_beta",
        "final_beta",
        "planner_bonus_decay",
    ]:
        val = getattr(args, name)
        if isinstance(val, list) and len(val) == 1:
            setattr(args, name, val[0])
    return args
def run(args):
    budget_str = f"budget_{args.cost_limit:.2f}"
    dynamics_str = (
        f"risk_{'on' if args.dynamic_risk else 'off'}_"
        f"cost_{'on' if args.dynamic_cost else 'off'}"
    )
    video_dir = os.path.join("videos", budget_str, dynamics_str)
    result_dir = os.path.join("results", budget_str, dynamics_str)
    figure_dir = os.path.join("figures", budget_str, dynamics_str)
    checkpoint_dir = os.path.join("checkpoints", budget_str, dynamics_str)
    plot_dir = None
    if args.plot_dir:
        plot_dir = os.path.join(args.plot_dir, budget_str, dynamics_str)
        os.makedirs(plot_dir, exist_ok=True)
    for d in [video_dir, result_dir, figure_dir, checkpoint_dir]:
        os.makedirs(d, exist_ok=True)

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = None
    if args.log_backend == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("logs", "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)
    elif args.log_backend == "wandb":
        import wandb
        wandb.init(project="warzone", config=vars(args))
        logger = wandb

    grid_size = args.grid_size
    input_dim = 4 * grid_size * grid_size + 2
    action_dim = 4

    if args.seeds:
        seeds = args.seeds
        if len(seeds) == 1:
            seeds = list(range(seeds[0]))
        else:
            seeds = list(seeds)
    else:
        seeds = [args.seed]

    algo_name = Path(args.algo_config).stem if args.algo_config else "default"
    if len(seeds) == 1:
        seed_key = f"seed_{seeds[0]}"
    else:
        seed_key = f"seeds_{seeds[0]}-{seeds[-1]}"

    env = GridWorldICM(
        grid_size=grid_size,
        dynamic_risk=args.dynamic_risk,
        dynamic_cost=args.dynamic_cost,
        max_steps=args.max_steps,
        mine_density_range=tuple(args.mine_density_range),
        hazard_density_range=tuple(args.hazard_density_range),
        enemy_speed_range=tuple(args.enemy_speed_range),
        enemy_policy_options=args.enemy_policies,
        seed=seeds[0],
    )
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(
        env.cost_map,
        env.risk_map,
        env.np_random,
        cost_weight=args.cost_weight,
        risk_weight=args.risk_weight,
        revisit_penalty=args.revisit_penalty,
    )

    export_benchmark_maps(env)

    policy_demo = PPOPolicy(input_dim, action_dim)
    visualize_paths_on_benchmark_maps(
        env, policy_demo, map_folder="train_maps/", num_maps=9, H=args.H
    )

    planner_weights = {
        "cost_weight": args.cost_weight,
        "risk_weight": args.risk_weight,
        "revisit_penalty": args.revisit_penalty,
    }

    settings = [
        {
            "name": "baseline",
            "icm": args.disable_icm,
            "rnd": args.disable_rnd,
            "planner": args.disable_planner,
        }
    ]
    if args.ablation:
        settings = [
            {
                "name": "baseline",
                "icm": False,
                "rnd": False,
                "planner": False,
            },
            {
                "name": "no_icm",
                "icm": True,
                "rnd": False,
                "planner": False,
            },
            {
                "name": "no_rnd",
                "icm": False,
                "rnd": True,
                "planner": False,
            },
            {
                "name": "no_planner",
                "icm": False,
                "rnd": False,
                "planner": True,
            },
        ]

    all_results = []
    all_bench = []
    pareto_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"rewards": [], "costs": [], "violations": []}
    )

    for setting in settings:
        args.disable_icm = setting["icm"]
        args.disable_rnd = setting["rnd"]
        args.disable_planner = setting["planner"]

        metrics = {
            "PPO Only": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "PPO + ICM": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "PPO + ICM + Planner": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "PPO + count": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "PPO + RND": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "PPO + PC": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "LPPO": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "Shielded-PPO": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "Planner-only": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "Planner-Subgoal PPO": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
            "Dyna-PPO(1)": {
                "rewards": {},
                "success": {},
                "ood_rewards": {},
                "planner_pct": [],
                "masked_action_rate": [],
                "planner_adherence_pct": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "unique_cells": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
                "lambda_vals": [],
                "auc_reward": [],
            },
        }
        bench = {
            "PPO Only": [],
            "PPO + ICM": [],
            "PPO + ICM + Planner": [],
            "PPO + count": [],
            "PPO + RND": [],
            "PPO + PC": [],
            "LPPO": [],
            "Shielded-PPO": [],
            "Planner-only": [],
            "Planner-Subgoal PPO": [],
            "Dyna-PPO(1)": [],
        }
        bench_ood = {
            "PPO Only": [],
            "PPO + ICM": [],
            "PPO + ICM + Planner": [],
            "PPO + count": [],
            "PPO + RND": [],
            "PPO + PC": [],
            "LPPO": [],
            "Shielded-PPO": [],
            "Planner-only": [],
            "Planner-Subgoal PPO": [],
            "Dyna-PPO(1)": [],
        }

        curve_logs = {
            "PPO Only": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "PPO + ICM": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "PPO + ICM + Planner": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "PPO + count": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "PPO + RND": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "PPO + PC": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "LPPO": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "Shielded-PPO": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "Planner-only": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "Planner-Subgoal PPO": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
            "Dyna-PPO(1)": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
                "lambda": [],
            },
        }

        for run_seed in seeds:
            safe_setting = setting["name"].replace(" ", "_")
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            env.reset(seed=run_seed)
            icm = ICMModule(input_dim, action_dim)
            planner = SymbolicPlanner(
                env.cost_map,
                env.risk_map,
                env.np_random,
                cost_weight=args.cost_weight,
                risk_weight=args.risk_weight,
                revisit_penalty=args.revisit_penalty,
            )
            beta_schedule = get_beta_schedule(
                args.num_episodes, args.initial_beta, args.final_beta
            )

            # PPO only
            print("Training PPO Only")
            ppo_policy = PPOPolicy(input_dim, action_dim)
            opt_ppo = optim.Adam(ppo_policy.parameters(), lr=args.learning_rate)
            (
                rewards_ppo_only,
                intrinsic_ppo_only,
                _,
                _,
                paths_ppo_only,
                _,
                success_ppo_only,
                planner_rate_ppo_only,
                mask_counts_ppo_only,
                mask_rates_ppo_only,
                adherence_rates_ppo_only,
                coverage_ppo_only,
                min_dists_ppo_only,
                episode_costs_ppo_only,
                violation_flags_ppo_only,
                first_violation_episode_ppo_only,
                episode_times_ppo_only,
                steps_per_sec_ppo_only,
                wall_clock_times_ppo_only,
                beta_log_ppo_only,
                lambda_log_ppo_only,
                episode_data_ppo_only,
            ) = train_agent(
                env,
                ppo_policy,
                icm,
                planner,
                opt_ppo,
                opt_ppo,
                use_icm=False,
                use_planner=False,
                num_episodes=args.num_episodes,
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=args.lambda_cost,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=0,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("PPO Only", run_seed, args.split, episode_data_ppo_only)
            metrics["PPO Only"]["auc_reward"].append(
                compute_auc_reward(rewards_ppo_only)
            )
            metrics["PPO Only"]["planner_pct"].append(
                float(np.mean(planner_rate_ppo_only)))
            metrics["PPO Only"]["masked_action_rate"].append(
                float(np.mean(mask_rates_ppo_only)))
            metrics["PPO Only"]["planner_adherence_pct"].append(
                float(np.mean(adherence_rates_ppo_only)))
            metrics["PPO Only"]["min_dist"].append(
                float(np.mean(min_dists_ppo_only)))
            metrics["PPO Only"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_ppo_only))
            metrics["PPO Only"]["episode_costs"].append(
                float(np.mean(episode_costs_ppo_only)))
            metrics["PPO Only"]["violation_flags"].append(
                float(np.mean(violation_flags_ppo_only)))
            metrics["PPO Only"]["first_violation_episode"].append(
                first_violation_episode_ppo_only
            )
            metrics["PPO Only"]["unique_cells"].append(
                float(np.mean(coverage_ppo_only)))
            metrics["PPO Only"]["episode_time"].append(
                float(np.mean(episode_times_ppo_only)))
            metrics["PPO Only"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_ppo_only)))
            metrics["PPO Only"]["wall_time"].append(
                float(wall_clock_times_ppo_only[-1]))
            metrics["PPO Only"]["lambda_vals"].append(lambda_log_ppo_only)
            save_model(
                ppo_policy,
                os.path.join(
                    checkpoint_dir,
                    f"ppo_only_{run_seed}.pt"))
            curve_logs["PPO Only"]["rewards"].append(rewards_ppo_only)
            curve_logs["PPO Only"]["success"].append(success_ppo_only)
            curve_logs["PPO Only"]["episode_costs"].append(
                episode_costs_ppo_only)
            curve_logs["PPO Only"]["violation_flags"].append(
                violation_flags_ppo_only)
            curve_logs["PPO Only"]["lambda"].append(lambda_log_ppo_only)
            render_episode_video(
                env,
                ppo_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_only_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                ppo_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["PPO Only"]["rewards"][run_seed] = [id_res[0]]
            metrics["PPO Only"]["success"][run_seed] = [0.0]
            metrics["PPO Only"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["PPO Only"].append(id_res[0])
            bench_ood["PPO Only"].append(ood_res[0])
            plot_policy_coverage(
                env,
                ppo_policy,
                "PPO Only",
                setting["name"],
                plot_dir,
                args.H,
            )

            # LPPO
            print("Training LPPO")
            lppo_policy = PPOPolicy(input_dim, action_dim)
            opt_lppo = optim.Adam(lppo_policy.parameters(), lr=args.learning_rate)
            (
                rewards_lppo,
                intrinsic_lppo,
                _,
                _,
                paths_lppo,
                _,
                success_lppo,
                planner_rate_lppo,
                mask_counts_lppo,
                mask_rates_lppo,
                adherence_rates_lppo,
                coverage_lppo,
                min_dists_lppo,
                episode_costs_lppo,
                violation_flags_lppo,
                first_violation_episode_lppo,
                episode_times_lppo,
                steps_per_sec_lppo,
                wall_clock_times_lppo,
                beta_log_lppo,
                lambda_log_lppo,
                episode_data_lppo,
            ) = train_agent(
                env,
                lppo_policy,
                icm,
                planner,
                opt_lppo,
                opt_lppo,
                use_icm=False,
                use_planner=False,
                num_episodes=args.num_episodes,
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=args.lambda_cost,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=0,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("LPPO", run_seed, args.split, episode_data_lppo)
            metrics["LPPO"]["auc_reward"].append(
                compute_auc_reward(rewards_lppo)
            )
            metrics["LPPO"]["planner_pct"].append(float(np.mean(planner_rate_lppo)))
            metrics["LPPO"]["masked_action_rate"].append(float(np.mean(mask_rates_lppo)))
            metrics["LPPO"]["planner_adherence_pct"].append(float(np.mean(adherence_rates_lppo)))
            metrics["LPPO"]["min_dist"].append(float(np.mean(min_dists_lppo)))
            metrics["LPPO"]["spikes"].append(count_intrinsic_spikes(intrinsic_lppo))
            metrics["LPPO"]["episode_costs"].append(float(np.mean(episode_costs_lppo)))
            metrics["LPPO"]["violation_flags"].append(float(np.mean(violation_flags_lppo)))
            metrics["LPPO"]["first_violation_episode"].append(first_violation_episode_lppo)
            metrics["LPPO"]["unique_cells"].append(float(np.mean(coverage_lppo)))
            metrics["LPPO"]["episode_time"].append(float(np.mean(episode_times_lppo)))
            metrics["LPPO"]["steps_per_sec"].append(float(np.mean(steps_per_sec_lppo)))
            metrics["LPPO"]["wall_time"].append(float(wall_clock_times_lppo[-1]))
            metrics["LPPO"]["lambda_vals"].append(lambda_log_lppo)
            save_model(
                lppo_policy,
                os.path.join(checkpoint_dir, f"lppo_{run_seed}.pt"),
            )
            curve_logs["LPPO"]["rewards"].append(rewards_lppo)
            curve_logs["LPPO"]["success"].append(success_lppo)
            curve_logs["LPPO"]["episode_costs"].append(episode_costs_lppo)
            curve_logs["LPPO"]["violation_flags"].append(violation_flags_lppo)
            curve_logs["LPPO"]["lambda"].append(lambda_log_lppo)
            render_episode_video(
                env,
                lppo_policy,
                os.path.join(video_dir, f"{safe_setting}_lppo_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                lppo_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["LPPO"]["rewards"][run_seed] = [id_res[0]]
            metrics["LPPO"]["success"][run_seed] = [0.0]
            metrics["LPPO"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["LPPO"].append(id_res[0])
            bench_ood["LPPO"].append(ood_res[0])
            plot_policy_coverage(
                env,
                lppo_policy,
                "LPPO",
                setting["name"],
                plot_dir,
                args.H,
            )

            # Shielded-PPO
            print("Training Shielded-PPO")
            shield_policy = PPOPolicy(input_dim, action_dim)
            opt_shield = optim.Adam(shield_policy.parameters(), lr=args.learning_rate)
            (
                rewards_shield,
                intrinsic_shield,
                _,
                _,
                paths_shield,
                _,
                success_shield,
                planner_rate_shield,
                mask_counts_shield,
                mask_rates_shield,
                adherence_rates_shield,
                coverage_shield,
                min_dists_shield,
                episode_costs_shield,
                violation_flags_shield,
                first_violation_episode_shield,
                episode_times_shield,
                steps_per_sec_shield,
                wall_clock_times_shield,
                beta_log_shield,
                lambda_log_shield,
                episode_data_shield,
            ) = train_agent(
                env,
                shield_policy,
                icm,
                planner,
                opt_shield,
                opt_shield,
                use_icm=False,
                use_planner=True,
                num_episodes=args.num_episodes,
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=0.0,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=0.0,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=0,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("Shielded-PPO", run_seed, args.split, episode_data_shield)
            metrics["Shielded-PPO"]["auc_reward"].append(
                compute_auc_reward(rewards_shield)
            )
            metrics["Shielded-PPO"]["planner_pct"].append(float(np.mean(planner_rate_shield)))
            metrics["Shielded-PPO"]["masked_action_rate"].append(float(np.mean(mask_rates_shield)))
            metrics["Shielded-PPO"]["planner_adherence_pct"].append(float(np.mean(adherence_rates_shield)))
            metrics["Shielded-PPO"]["min_dist"].append(float(np.mean(min_dists_shield)))
            metrics["Shielded-PPO"]["spikes"].append(count_intrinsic_spikes(intrinsic_shield))
            metrics["Shielded-PPO"]["episode_costs"].append(float(np.mean(episode_costs_shield)))
            metrics["Shielded-PPO"]["violation_flags"].append(float(np.mean(violation_flags_shield)))
            metrics["Shielded-PPO"]["first_violation_episode"].append(first_violation_episode_shield)
            metrics["Shielded-PPO"]["unique_cells"].append(float(np.mean(coverage_shield)))
            metrics["Shielded-PPO"]["episode_time"].append(float(np.mean(episode_times_shield)))
            metrics["Shielded-PPO"]["steps_per_sec"].append(float(np.mean(steps_per_sec_shield)))
            metrics["Shielded-PPO"]["wall_time"].append(float(wall_clock_times_shield[-1]))
            metrics["Shielded-PPO"]["lambda_vals"].append(lambda_log_shield)
            save_model(
                shield_policy,
                os.path.join(checkpoint_dir, f"shielded_ppo_{run_seed}.pt"),
            )
            curve_logs["Shielded-PPO"]["rewards"].append(rewards_shield)
            curve_logs["Shielded-PPO"]["success"].append(success_shield)
            curve_logs["Shielded-PPO"]["episode_costs"].append(episode_costs_shield)
            curve_logs["Shielded-PPO"]["violation_flags"].append(violation_flags_shield)
            curve_logs["Shielded-PPO"]["lambda"].append(lambda_log_shield)
            render_episode_video(
                env,
                shield_policy,
                os.path.join(video_dir, f"{safe_setting}_shielded_ppo_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                shield_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["Shielded-PPO"]["rewards"][run_seed] = [id_res[0]]
            metrics["Shielded-PPO"]["success"][run_seed] = [0.0]
            metrics["Shielded-PPO"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["Shielded-PPO"].append(id_res[0])
            bench_ood["Shielded-PPO"].append(ood_res[0])
            plot_policy_coverage(
                env,
                shield_policy,
                "Shielded-PPO",
                setting["name"],
                plot_dir,
                args.H,
            )

            # Planner-only
            print("Running Planner-only")
            planner.reset()
            rewards_po: list[float] = []
            success_po: list[int] = []
            episode_costs_po: list[float] = []
            violation_flags_po: list[float] = []
            coverage_po: list[float] = []
            min_dists_po: list[float] = []
            episode_times_po: list[float] = []
            steps_per_sec_po: list[float] = []
            wall_clock_po: list[float] = []
            start_po = time.time()
            first_violation_po = args.num_episodes
            for ep in range(args.num_episodes):
                ep_start = time.time()
                obs, _ = env.reset(seed=run_seed)
                planner.reset()
                done = False
                total_reward = 0.0
                step_count = 0
                visit: set[tuple[int, int]] = set()
                min_dist = float("inf")
                alive = True
                while not done:
                    action = planner.get_safe_subgoal(env.agent_pos)
                    obs, reward, _cost, done, _, info = env.step(action)
                    total_reward += reward
                    step_count += 1
                    visit.add(tuple(env.agent_pos))
                    if env.enemy_positions:
                        x, y = env.agent_pos
                        curr = min(abs(x - ex) + abs(y - ey) for ex, ey in env.enemy_positions)
                        min_dist = min(min_dist, curr)
                    if info.get("dead", False):
                        alive = False
                rewards_po.append(total_reward)
                success_po.append(1 if step_count >= env.max_steps and alive else 0)
                Jc = env.episode_cost
                episode_costs_po.append(Jc)
                viol = float(Jc > args.cost_limit)
                violation_flags_po.append(viol)
                if viol == 1.0 and first_violation_po == args.num_episodes:
                    first_violation_po = ep
                coverage_po.append(len(visit))
                min_dists_po.append(min_dist if min_dist < float("inf") else env.grid_size * 2)
                ep_time = time.time() - ep_start
                episode_times_po.append(ep_time)
                steps_per_sec_po.append(step_count / ep_time if ep_time > 0 else 0.0)
                wall_clock_po.append(time.time() - start_po)
            metrics["Planner-only"]["planner_pct"].append(1.0)
            metrics["Planner-only"]["masked_action_rate"].append(0.0)
            metrics["Planner-only"]["planner_adherence_pct"].append(1.0)
            metrics["Planner-only"]["min_dist"].append(float(np.mean(min_dists_po)))
            metrics["Planner-only"]["spikes"].append(0)
            metrics["Planner-only"]["episode_costs"].append(float(np.mean(episode_costs_po)))
            metrics["Planner-only"]["violation_flags"].append(float(np.mean(violation_flags_po)))
            metrics["Planner-only"]["first_violation_episode"].append(first_violation_po)
            metrics["Planner-only"]["unique_cells"].append(float(np.mean(coverage_po)))
            metrics["Planner-only"]["episode_time"].append(float(np.mean(episode_times_po)))
            metrics["Planner-only"]["steps_per_sec"].append(float(np.mean(steps_per_sec_po)))
            metrics["Planner-only"]["wall_time"].append(float(wall_clock_po[-1]))
            metrics["Planner-only"]["auc_reward"].append(
                compute_auc_reward(rewards_po)
            )
            curve_logs["Planner-only"]["rewards"].append(rewards_po)
            curve_logs["Planner-only"]["success"].append(success_po)
            curve_logs["Planner-only"]["episode_costs"].append(episode_costs_po)
            curve_logs["Planner-only"]["violation_flags"].append(violation_flags_po)
            rewards_b, success_b = evaluate_planner_on_maps(env, "test_maps", 5)
            metrics["Planner-only"]["rewards"][run_seed] = rewards_b
            metrics["Planner-only"]["success"][run_seed] = success_b
            bench["Planner-only"].append(float(np.mean(rewards_b)))

            # Planner-Subgoal PPO
            print("Training Planner-Subgoal PPO")
            subgoal_policy = PPOPolicy(input_dim, action_dim)
            opt_subgoal = optim.Adam(subgoal_policy.parameters(), lr=args.learning_rate)
            (
                rewards_subgoal,
                intrinsic_subgoal,
                _,
                _,
                paths_subgoal,
                _,
                success_subgoal,
                planner_rate_subgoal,
                mask_counts_subgoal,
                mask_rates_subgoal,
                adherence_rates_subgoal,
                coverage_subgoal,
                min_dists_subgoal,
                episode_costs_subgoal,
                violation_flags_subgoal,
                first_violation_episode_subgoal,
                episode_times_subgoal,
                steps_per_sec_subgoal,
                wall_clock_times_subgoal,
                beta_log_subgoal,
                lambda_log_subgoal,
                episode_data_subgoal,
            ) = train_agent(
                env,
                subgoal_policy,
                icm,
                planner,
                opt_subgoal,
                opt_subgoal,
                use_icm=False,
                use_planner=True,
                num_episodes=args.num_episodes,
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=0.0,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=0,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("Planner-Subgoal PPO", run_seed, args.split, episode_data_subgoal)
            metrics["Planner-Subgoal PPO"]["auc_reward"].append(
                compute_auc_reward(rewards_subgoal)
            )
            metrics["Planner-Subgoal PPO"]["planner_pct"].append(float(np.mean(planner_rate_subgoal)))
            metrics["Planner-Subgoal PPO"]["masked_action_rate"].append(float(np.mean(mask_rates_subgoal)))
            metrics["Planner-Subgoal PPO"]["planner_adherence_pct"].append(float(np.mean(adherence_rates_subgoal)))
            metrics["Planner-Subgoal PPO"]["min_dist"].append(float(np.mean(min_dists_subgoal)))
            metrics["Planner-Subgoal PPO"]["spikes"].append(count_intrinsic_spikes(intrinsic_subgoal))
            metrics["Planner-Subgoal PPO"]["episode_costs"].append(float(np.mean(episode_costs_subgoal)))
            metrics["Planner-Subgoal PPO"]["violation_flags"].append(float(np.mean(violation_flags_subgoal)))
            metrics["Planner-Subgoal PPO"]["first_violation_episode"].append(first_violation_episode_subgoal)
            metrics["Planner-Subgoal PPO"]["unique_cells"].append(float(np.mean(coverage_subgoal)))
            metrics["Planner-Subgoal PPO"]["episode_time"].append(float(np.mean(episode_times_subgoal)))
            metrics["Planner-Subgoal PPO"]["steps_per_sec"].append(float(np.mean(steps_per_sec_subgoal)))
            metrics["Planner-Subgoal PPO"]["wall_time"].append(float(wall_clock_times_subgoal[-1]))
            metrics["Planner-Subgoal PPO"]["lambda_vals"].append(lambda_log_subgoal)
            save_model(
                subgoal_policy,
                os.path.join(checkpoint_dir, f"planner_subgoal_ppo_{run_seed}.pt"),
            )
            curve_logs["Planner-Subgoal PPO"]["rewards"].append(rewards_subgoal)
            curve_logs["Planner-Subgoal PPO"]["success"].append(success_subgoal)
            curve_logs["Planner-Subgoal PPO"]["episode_costs"].append(episode_costs_subgoal)
            curve_logs["Planner-Subgoal PPO"]["violation_flags"].append(violation_flags_subgoal)
            curve_logs["Planner-Subgoal PPO"]["lambda"].append(lambda_log_subgoal)
            render_episode_video(
                env,
                subgoal_policy,
                os.path.join(video_dir, f"{safe_setting}_planner_subgoal_ppo_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                subgoal_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["Planner-Subgoal PPO"]["rewards"][run_seed] = [id_res[0]]
            metrics["Planner-Subgoal PPO"]["success"][run_seed] = [0.0]
            metrics["Planner-Subgoal PPO"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["Planner-Subgoal PPO"].append(id_res[0])
            bench_ood["Planner-Subgoal PPO"].append(ood_res[0])
            plot_policy_coverage(
                env,
                subgoal_policy,
                "Planner-Subgoal PPO",
                setting["name"],
                plot_dir,
                args.H,
            )

            # Dyna-PPO(1)
            print("Training Dyna-PPO(1)")
            dyna_policy = PPOPolicy(input_dim, action_dim)
            opt_dyna = optim.Adam(dyna_policy.parameters(), lr=args.learning_rate)
            (
                rewards_dyna,
                intrinsic_dyna,
                _,
                _,
                paths_dyna,
                _,
                success_dyna,
                planner_rate_dyna,
                mask_counts_dyna,
                mask_rates_dyna,
                adherence_rates_dyna,
                coverage_dyna,
                min_dists_dyna,
                episode_costs_dyna,
                violation_flags_dyna,
                first_violation_episode_dyna,
                episode_times_dyna,
                steps_per_sec_dyna,
                wall_clock_times_dyna,
                beta_log_dyna,
                lambda_log_dyna,
                episode_data_dyna,
            ) = train_agent(
                env,
                dyna_policy,
                icm,
                planner,
                opt_dyna,
                opt_dyna,
                use_icm=False,
                use_planner=False,
                num_episodes=args.num_episodes,
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=0.0,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=0.0,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=args.K,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("Dyna-PPO(1)", run_seed, args.split, episode_data_dyna)
            metrics["Dyna-PPO(1)"]["auc_reward"].append(
                compute_auc_reward(rewards_dyna)
            )
            metrics["Dyna-PPO(1)"]["planner_pct"].append(float(np.mean(planner_rate_dyna)))
            metrics["Dyna-PPO(1)"]["masked_action_rate"].append(float(np.mean(mask_rates_dyna)))
            metrics["Dyna-PPO(1)"]["planner_adherence_pct"].append(float(np.mean(adherence_rates_dyna)))
            metrics["Dyna-PPO(1)"]["min_dist"].append(float(np.mean(min_dists_dyna)))
            metrics["Dyna-PPO(1)"]["spikes"].append(count_intrinsic_spikes(intrinsic_dyna))
            metrics["Dyna-PPO(1)"]["episode_costs"].append(float(np.mean(episode_costs_dyna)))
            metrics["Dyna-PPO(1)"]["violation_flags"].append(float(np.mean(violation_flags_dyna)))
            metrics["Dyna-PPO(1)"]["first_violation_episode"].append(first_violation_episode_dyna)
            metrics["Dyna-PPO(1)"]["unique_cells"].append(float(np.mean(coverage_dyna)))
            metrics["Dyna-PPO(1)"]["episode_time"].append(float(np.mean(episode_times_dyna)))
            metrics["Dyna-PPO(1)"]["steps_per_sec"].append(float(np.mean(steps_per_sec_dyna)))
            metrics["Dyna-PPO(1)"]["wall_time"].append(float(wall_clock_times_dyna[-1]))
            metrics["Dyna-PPO(1)"]["lambda_vals"].append(lambda_log_dyna)
            save_model(
                dyna_policy,
                os.path.join(checkpoint_dir, f"dyna_ppo1_{run_seed}.pt"),
            )
            curve_logs["Dyna-PPO(1)"]["rewards"].append(rewards_dyna)
            curve_logs["Dyna-PPO(1)"]["success"].append(success_dyna)
            curve_logs["Dyna-PPO(1)"]["episode_costs"].append(episode_costs_dyna)
            curve_logs["Dyna-PPO(1)"]["violation_flags"].append(violation_flags_dyna)
            curve_logs["Dyna-PPO(1)"]["lambda"].append(lambda_log_dyna)
            render_episode_video(
                env,
                dyna_policy,
                os.path.join(video_dir, f"{safe_setting}_dyna_ppo1_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                dyna_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["Dyna-PPO(1)"]["rewards"][run_seed] = [id_res[0]]
            metrics["Dyna-PPO(1)"]["success"][run_seed] = [0.0]
            metrics["Dyna-PPO(1)"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["Dyna-PPO(1)"].append(id_res[0])
            bench_ood["Dyna-PPO(1)"].append(ood_res[0])
            plot_policy_coverage(
                env,
                dyna_policy,
                "Dyna-PPO(1)",
                setting["name"],
                plot_dir,
                args.H,
            )

            # PPO + ICM
            if not args.disable_icm:
                print("Training PPO + ICM")
                ppo_icm_policy = PPOPolicy(input_dim, action_dim)
                opt_icm_policy = optim.Adam(
                    ppo_icm_policy.parameters(), lr=args.learning_rate)
                (
                    rewards_ppo_icm,
                    intrinsic_icm,
                    _,
                    _,
                    paths_icm,
                    _,
                    success_icm,
                    planner_rate_icm,
                mask_counts_icm,
                mask_rates_icm,
                adherence_rates_icm,
                coverage_icm,
                min_dists_icm,
                episode_costs_icm,
                violation_flags_icm,
                first_violation_episode_icm,
                episode_times_icm,
                steps_per_sec_icm,
                wall_clock_times_icm,
                beta_log_icm,
                lambda_log_icm,
                episode_data_icm,
            ) = train_agent(
                env,
                ppo_icm_policy,
                icm,
                planner,
                    opt_icm_policy,
                    opt_icm_policy,
                    use_icm=True,
                    use_planner=False,
                    num_episodes=args.num_episodes,
                    beta_schedule=beta_schedule,
                    final_beta=args.final_beta,
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                    lambda_cost=args.lambda_cost,
                    eta_lambda=args.eta_lambda,
                    cost_limit=args.cost_limit,
                    c1=args.c1,
                    c2=args.c2,
                    entropy_coef=args.entropy_coef,
                    tau=args.tau,
                    kappa=args.kappa,
                    use_risk_penalty=not args.disable_risk_penalty,
                    H=args.H,
                    waypoint_bonus=args.waypoint_bonus,
                    planner_bonus_decay=args.planner_bonus_decay,
                    imagination_k=0,
                    world_model_lr=args.world_model_lr,
                    clip_epsilon=args.clip_epsilon,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    danger_distance=args.danger_distance,
                    map_id=env.map_id,
                )
            save_episode_metrics("PPO + ICM", run_seed, args.split, episode_data_icm)
            metrics["PPO + ICM"]["auc_reward"].append(
                compute_auc_reward(rewards_ppo_icm)
            )
            metrics["PPO + ICM"]["planner_pct"].append(
                float(np.mean(planner_rate_icm)))
            metrics["PPO + ICM"]["masked_action_rate"].append(
                float(np.mean(mask_rates_icm)))
            metrics["PPO + ICM"]["planner_adherence_pct"].append(
                float(np.mean(adherence_rates_icm)))
            metrics["PPO + ICM"]["min_dist"].append(
                float(np.mean(min_dists_icm)))
            metrics["PPO + ICM"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_icm)
            )
            metrics["PPO + ICM"]["episode_costs"].append(
                float(np.mean(episode_costs_icm)))
            metrics["PPO + ICM"]["violation_flags"].append(
                float(np.mean(violation_flags_icm)))
            metrics["PPO + ICM"]["first_violation_episode"].append(
                first_violation_episode_icm
            )
            metrics["PPO + ICM"]["unique_cells"].append(
                float(np.mean(coverage_icm)))
            metrics["PPO + ICM"]["episode_time"].append(
                float(np.mean(episode_times_icm)))
            metrics["PPO + ICM"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_icm)))
            metrics["PPO + ICM"]["wall_time"].append(
                float(wall_clock_times_icm[-1]))
            metrics["PPO + ICM"]["lambda_vals"].append(lambda_log_icm)
            save_model(
                ppo_icm_policy,
                os.path.join(
                    checkpoint_dir,
                    f"ppo_icm_{run_seed}.pt"),
                icm=icm)
            curve_logs["PPO + ICM"]["rewards"].append(rewards_ppo_icm)
            curve_logs["PPO + ICM"]["intrinsic"].append(intrinsic_icm)
            curve_logs["PPO + ICM"]["success"].append(success_icm)
            curve_logs["PPO + ICM"]["episode_costs"].append(
                episode_costs_icm)
            curve_logs["PPO + ICM"]["violation_flags"].append(
                violation_flags_icm)
            curve_logs["PPO + ICM"]["lambda"].append(lambda_log_icm)
            render_episode_video(
                env,
                ppo_icm_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_icm_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                ppo_icm_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["PPO + ICM"]["rewards"][run_seed] = [id_res[0]]
            metrics["PPO + ICM"]["success"][run_seed] = [0.0]
            metrics["PPO + ICM"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["PPO + ICM"].append(id_res[0])
            bench_ood["PPO + ICM"].append(ood_res[0])
            plot_policy_coverage(
                env,
                ppo_icm_policy,
                "PPO + ICM",
                setting["name"],
                plot_dir,
                args.H,
            )

            # PPO + Pseudo-count exploration
            print("Training PPO + PC")
            ppo_pc_policy = PPOPolicy(input_dim, action_dim)
            opt_pc_policy = optim.Adam(ppo_pc_policy.parameters(), lr=args.learning_rate)
            pseudo = PseudoCountExploration()
            (
                rewards_pc,
                intrinsic_pc,
                _,
                _,
                paths_pc,
                _,
                success_pc,
                planner_rate_pc,
                mask_counts_pc,
                mask_rates_pc,
                adherence_rates_pc,
                coverage_pc,
                min_dists_pc,
                episode_costs_pc,
                violation_flags_pc,
                first_violation_episode_pc,
                episode_times_pc,
                steps_per_sec_pc,
                wall_clock_times_pc,
                beta_log_pc,
                lambda_log_pc,
                episode_data_pc,
            ) = train_agent(
                env,
                ppo_pc_policy,
                icm,
                planner,
                opt_pc_policy,
                opt_pc_policy,
                use_icm="pseudo",
                use_planner=False,
                pseudo=pseudo,
                num_episodes=args.num_episodes,
                beta_schedule=beta_schedule,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=args.lambda_cost,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=0,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("PPO + PC", run_seed, args.split, episode_data_pc)
            metrics["PPO + PC"]["auc_reward"].append(
                compute_auc_reward(rewards_pc)
            )
            metrics["PPO + PC"]["planner_pct"].append(
                float(np.mean(planner_rate_pc)))
            metrics["PPO + PC"]["masked_action_rate"].append(
                float(np.mean(mask_rates_pc)))
            metrics["PPO + PC"]["planner_adherence_pct"].append(
                float(np.mean(adherence_rates_pc)))
            metrics["PPO + PC"]["min_dist"].append(
                float(np.mean(min_dists_pc)))
            metrics["PPO + PC"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_pc))
            metrics["PPO + PC"]["episode_costs"].append(
                float(np.mean(episode_costs_pc)))
            metrics["PPO + PC"]["violation_flags"].append(
                float(np.mean(violation_flags_pc)))
            metrics["PPO + PC"]["first_violation_episode"].append(
                first_violation_episode_pc
            )
            metrics["PPO + PC"]["unique_cells"].append(
                float(np.mean(coverage_pc)))
            metrics["PPO + PC"]["episode_time"].append(
                float(np.mean(episode_times_pc)))
            metrics["PPO + PC"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_pc)))
            metrics["PPO + PC"]["wall_time"].append(
                float(wall_clock_times_pc[-1]))
            metrics["PPO + PC"]["lambda_vals"].append(lambda_log_pc)
            save_model(
                ppo_pc_policy,
                os.path.join(
                    checkpoint_dir,
                    f"ppo_pc_{run_seed}.pt"))
            curve_logs["PPO + PC"]["rewards"].append(rewards_pc)
            curve_logs["PPO + PC"]["success"].append(success_pc)
            curve_logs["PPO + PC"]["episode_costs"].append(
                episode_costs_pc)
            curve_logs["PPO + PC"]["violation_flags"].append(
                violation_flags_pc)
            curve_logs["PPO + PC"]["lambda"].append(lambda_log_pc)
            render_episode_video(
                env,
                ppo_pc_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_pc_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                ppo_pc_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["PPO + PC"]["rewards"][run_seed] = [id_res[0]]
            metrics["PPO + PC"]["success"][run_seed] = [0.0]
            metrics["PPO + PC"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["PPO + PC"].append(id_res[0])
            bench_ood["PPO + PC"].append(ood_res[0])
            plot_policy_coverage(
                env,
                ppo_pc_policy,
                "PPO + PC",
                setting["name"],
                plot_dir,
                args.H,
            )

            # PPO + ICM + Planner
            if not args.disable_icm and not args.disable_planner:
                print("Training PPO + ICM + Planner")
                ppo_icm_planner_policy = PPOPolicy(input_dim, action_dim)
                opt_plan_policy = optim.Adam(
                    ppo_icm_planner_policy.parameters(), lr=args.learning_rate)
                (
                    rewards_ppo_icm_plan,
                    intrinsic_plan,
                    _,
                    _,
                    paths_plan,
                    _,
                    success_plan,
                    planner_rate_plan,
                    mask_counts_icm_plan,
                    mask_rates_icm_plan,
                    adherence_rates_icm_plan,
                    coverage_icm_plan,
                    min_dists_icm_plan,
                    episode_costs_icm_plan,
                    violation_flags_icm_plan,
                    first_violation_episode_icm_plan,
                    episode_times_icm_plan,
                    steps_per_sec_icm_plan,
                    wall_clock_times_icm_plan,
                    beta_log_icm_plan,
                    lambda_log_icm_plan,
                    episode_data_icm_plan,
                ) = train_agent(
                    env,
                    ppo_icm_planner_policy,
                    icm,
                    planner,
                    opt_plan_policy,
                    opt_plan_policy,
                    use_icm=True,
                    use_planner=True,
                    num_episodes=args.num_episodes,
                    beta_schedule=beta_schedule,
                    final_beta=args.final_beta,
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                    lambda_cost=args.lambda_cost,
                    eta_lambda=args.eta_lambda,
                    cost_limit=args.cost_limit,
                    c1=args.c1,
                    c2=args.c2,
                    entropy_coef=args.entropy_coef,
                    tau=args.tau,
                    kappa=args.kappa,
                    use_risk_penalty=not args.disable_risk_penalty,
                    H=args.H,
                    waypoint_bonus=args.waypoint_bonus,
                    planner_bonus_decay=args.planner_bonus_decay,
                    imagination_k=0,
                    world_model_lr=args.world_model_lr,
                    clip_epsilon=args.clip_epsilon,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    danger_distance=args.danger_distance,
                    map_id=env.map_id,
                )
                save_episode_metrics("PPO + ICM + Planner", run_seed, args.split, episode_data_icm_plan)
                metrics["PPO + ICM + Planner"]["auc_reward"].append(
                    compute_auc_reward(rewards_ppo_icm_plan)
                )
                metrics["PPO + ICM + Planner"]["planner_pct"].append(
                    float(np.mean(planner_rate_plan)))
                metrics["PPO + ICM + Planner"]["masked_action_rate"].append(
                    float(np.mean(mask_rates_icm_plan)))
                metrics["PPO + ICM + Planner"]["planner_adherence_pct"].append(
                    float(np.mean(adherence_rates_icm_plan)))
                metrics["PPO + ICM + Planner"]["min_dist"].append(
                    float(np.mean(min_dists_icm_plan)))
                metrics["PPO + ICM + Planner"]["spikes"].append(
                    count_intrinsic_spikes(intrinsic_plan))
                metrics["PPO + ICM + Planner"]["episode_costs"].append(
                    float(np.mean(episode_costs_icm_plan)))
                metrics["PPO + ICM + Planner"]["violation_flags"].append(
                    float(np.mean(violation_flags_icm_plan)))
                metrics["PPO + ICM + Planner"]["first_violation_episode"].append(
                    first_violation_episode_icm_plan
                )
                metrics["PPO + ICM + Planner"]["unique_cells"].append(
                    float(np.mean(coverage_icm_plan)))
                metrics["PPO + ICM + Planner"]["episode_time"].append(
                    float(np.mean(episode_times_icm_plan)))
                metrics["PPO + ICM + Planner"]["steps_per_sec"].append(
                    float(np.mean(steps_per_sec_icm_plan)))
                metrics["PPO + ICM + Planner"]["wall_time"].append(
                    float(wall_clock_times_icm_plan[-1]))
                metrics["PPO + ICM + Planner"]["lambda_vals"].append(lambda_log_icm_plan)
                save_model(
                    ppo_icm_planner_policy,
                    os.path.join(
                        checkpoint_dir,
                        f"ppo_icm_planner_{run_seed}.pt"),
                    icm=icm,
                )
                curve_logs["PPO + ICM + Planner"]["rewards"].append(
                    rewards_ppo_icm_plan)
                curve_logs["PPO + ICM + Planner"]["intrinsic"].append(
                    intrinsic_plan)
                curve_logs["PPO + ICM + Planner"]["success"].append(
                    success_plan)
                curve_logs["PPO + ICM + Planner"]["episode_costs"].append(
                    episode_costs_icm_plan)
                curve_logs["PPO + ICM + Planner"]["violation_flags"].append(
                    violation_flags_icm_plan)
                curve_logs["PPO + ICM + Planner"]["lambda"].append(lambda_log_icm_plan)
                render_episode_video(
                    env,
                    ppo_icm_planner_policy,
                    os.path.join(
                        video_dir, f"{safe_setting}_ppo_icm_planner_{run_seed}.gif"
                    ),
                    H=args.H,
                )
                id_res, ood_res = evaluate_on_benchmarks(
                    env,
                    ppo_icm_planner_policy,
                    "test_maps",
                    5,
                    H=args.H,
                    ood_map_folder="ood_maps",
                    num_ood_maps=10,
                )
                metrics["PPO + ICM + Planner"]["rewards"][run_seed] = [id_res[0]]
                metrics["PPO + ICM + Planner"]["success"][run_seed] = [0.0]
                metrics["PPO + ICM + Planner"]["ood_rewards"][run_seed] = [ood_res[0]]
                bench["PPO + ICM + Planner"].append(id_res[0])
                bench_ood["PPO + ICM + Planner"].append(ood_res[0])
                plot_policy_coverage(
                    env,
                    ppo_icm_planner_policy,
                    "PPO + ICM + Planner",
                    setting["name"],
                    plot_dir,
                    args.H,
                )
                if paths_plan:
                    heat_path = None
                    if plot_dir:
                        safe_setting = setting["name"].replace(" ", "_")
                        heat_filename = (
                            f"heatmap_{safe_setting}_{run_seed}.pdf"
                        )
                        heat_path = os.path.join(
                            plot_dir,
                            heat_filename,
                        )
                    plot_heatmap_with_path(
                        env, paths_plan[-1], output_path=heat_path)

            # Count-based exploration
            print("Training PPO + count")
            ppo_count_policy = PPOPolicy(input_dim, action_dim)
            opt_count_policy = optim.Adam(
                ppo_count_policy.parameters(), lr=args.learning_rate)
            (
                rewards_ppo_count,
                intrinsic_count,
                _,
                _,
                paths_count,
                _,
                success_count,
                planner_rate_count,
                mask_counts_count,
                mask_rates_count,
                adherence_rates_count,
                coverage_count,
                min_dists_count,
                episode_costs_count,
                violation_flags_count,
                first_violation_episode_count,
                episode_times_count,
                steps_per_sec_count,
                wall_clock_times_count,
                beta_log_count,
                lambda_log_count,
                episode_data_count,
            ) = train_agent(
                env,
                ppo_count_policy,
                icm,
                planner,
                opt_count_policy,
                opt_count_policy,
                use_icm="count",
                use_planner=False,
                num_episodes=args.num_episodes,
                beta_schedule=beta_schedule,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
                lambda_cost=args.lambda_cost,
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                entropy_coef=args.entropy_coef,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                planner_bonus_decay=args.planner_bonus_decay,
                imagination_k=0,
                world_model_lr=args.world_model_lr,
                clip_epsilon=args.clip_epsilon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                danger_distance=args.danger_distance,
                map_id=env.map_id,
            )
            save_episode_metrics("PPO + count", run_seed, args.split, episode_data_count)
            metrics["PPO + count"]["auc_reward"].append(
                compute_auc_reward(rewards_ppo_count)
            )
            metrics["PPO + count"]["planner_pct"].append(
                float(np.mean(planner_rate_count)))
            metrics["PPO + count"]["masked_action_rate"].append(
                float(np.mean(mask_rates_count)))
            metrics["PPO + count"]["planner_adherence_pct"].append(
                float(np.mean(adherence_rates_count)))
            metrics["PPO + count"]["min_dist"].append(
                float(np.mean(min_dists_count)))
            metrics["PPO + count"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_count))
            metrics["PPO + count"]["episode_costs"].append(
                float(np.mean(episode_costs_count)))
            metrics["PPO + count"]["violation_flags"].append(
                float(np.mean(violation_flags_count)))
            metrics["PPO + count"]["first_violation_episode"].append(
                first_violation_episode_count
            )
            metrics["PPO + count"]["unique_cells"].append(
                float(np.mean(coverage_count)))
            metrics["PPO + count"]["episode_time"].append(
                float(np.mean(episode_times_count)))
            metrics["PPO + count"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_count)))
            metrics["PPO + count"]["wall_time"].append(
                float(wall_clock_times_count[-1]))
            metrics["PPO + count"]["lambda_vals"].append(lambda_log_count)
            save_model(
                ppo_count_policy,
                os.path.join(
                    checkpoint_dir,
                    f"ppo_count_{run_seed}.pt"))
            curve_logs["PPO + count"]["rewards"].append(rewards_ppo_count)
            curve_logs["PPO + count"]["success"].append(success_count)
            curve_logs["PPO + count"]["episode_costs"].append(
                episode_costs_count)
            curve_logs["PPO + count"]["violation_flags"].append(
                violation_flags_count)
            curve_logs["PPO + count"]["lambda"].append(lambda_log_count)
            render_episode_video(
                env,
                ppo_count_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_count_{run_seed}.gif"),
                H=args.H,
            )
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                ppo_count_policy,
                "test_maps",
                5,
                H=args.H,
                ood_map_folder="ood_maps",
                num_ood_maps=10,
            )
            metrics["PPO + count"]["rewards"][run_seed] = [id_res[0]]
            metrics["PPO + count"]["success"][run_seed] = [0.0]
            metrics["PPO + count"]["ood_rewards"][run_seed] = [ood_res[0]]
            bench["PPO + count"].append(id_res[0])
            bench_ood["PPO + count"].append(ood_res[0])
            plot_policy_coverage(
                env,
                ppo_count_policy,
                "PPO + count",
                setting["name"],
                plot_dir,
                args.H,
            )

            # RND exploration
            if not args.disable_rnd:
                print("Training PPO + RND")
                ppo_rnd_policy = PPOPolicy(input_dim, action_dim)
                opt_rnd_policy = optim.Adam(
                    ppo_rnd_policy.parameters(), lr=args.learning_rate)
                rnd = RNDModule(input_dim)
                opt_rnd = optim.Adam(rnd.predictor.parameters(), lr=1e-3)
                (
                    rewards_ppo_rnd,
                    intrinsic_rnd,
                    _,
                    _,
                    paths_rnd,
                    _,
                    success_rnd,
                    planner_rate_rnd,
                    mask_counts_rnd,
                    mask_rates_rnd,
                    adherence_rates_rnd,
                    coverage_rnd,
                    min_dists_rnd,
                    episode_costs_rnd,
                    violation_flags_rnd,
                    first_violation_episode_rnd,
                    episode_times_rnd,
                    steps_per_sec_rnd,
                    wall_clock_times_rnd,
                    beta_log_rnd,
                    lambda_log_rnd,
                    episode_data_rnd,
                ) = train_agent(
                    env,
                    ppo_rnd_policy,
                    icm,
                    planner,
                    opt_rnd_policy,
                    opt_rnd,
                    use_icm="rnd",
                    use_planner=False,
                    rnd=rnd,
                    num_episodes=args.num_episodes,
                    beta_schedule=beta_schedule,
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                    lambda_cost=args.lambda_cost,
                    eta_lambda=args.eta_lambda,
                    cost_limit=args.cost_limit,
                    c1=args.c1,
                    c2=args.c2,
                    entropy_coef=args.entropy_coef,
                    tau=args.tau,
                    kappa=args.kappa,
                    use_risk_penalty=not args.disable_risk_penalty,
                    H=args.H,
                    waypoint_bonus=args.waypoint_bonus,
                    planner_bonus_decay=args.planner_bonus_decay,
                    imagination_k=0,
                    world_model_lr=args.world_model_lr,
                    clip_epsilon=args.clip_epsilon,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    danger_distance=args.danger_distance,
                    map_id=env.map_id,
                )
                save_episode_metrics("PPO + RND", run_seed, args.split, episode_data_rnd)
                metrics["PPO + RND"]["auc_reward"].append(
                    compute_auc_reward(rewards_ppo_rnd)
                )
                metrics["PPO + RND"]["planner_pct"].append(
                    float(np.mean(planner_rate_rnd)))
                metrics["PPO + RND"]["masked_action_rate"].append(
                    float(np.mean(mask_rates_rnd)))
                metrics["PPO + RND"]["planner_adherence_pct"].append(
                    float(np.mean(adherence_rates_rnd)))
                metrics["PPO + RND"]["min_dist"].append(
                    float(np.mean(min_dists_rnd)))
                metrics["PPO + RND"]["spikes"].append(
                    count_intrinsic_spikes(intrinsic_rnd))
                metrics["PPO + RND"]["episode_costs"].append(
                    float(np.mean(episode_costs_rnd)))
                metrics["PPO + RND"]["violation_flags"].append(
                    float(np.mean(violation_flags_rnd)))
                metrics["PPO + RND"]["first_violation_episode"].append(
                    first_violation_episode_rnd
                )
                metrics["PPO + RND"]["unique_cells"].append(
                    float(np.mean(coverage_rnd)))
                metrics["PPO + RND"]["episode_time"].append(
                    float(np.mean(episode_times_rnd)))
                metrics["PPO + RND"]["steps_per_sec"].append(
                    float(np.mean(steps_per_sec_rnd)))
                metrics["PPO + RND"]["wall_time"].append(
                    float(wall_clock_times_rnd[-1]))
                metrics["PPO + RND"]["lambda_vals"].append(lambda_log_rnd)
                save_model(
                    ppo_rnd_policy,
                    os.path.join(checkpoint_dir, f"ppo_rnd_{run_seed}.pt"),
                    rnd=rnd,
                )
                curve_logs["PPO + RND"]["rewards"].append(rewards_ppo_rnd)
                curve_logs["PPO + RND"]["success"].append(success_rnd)
                curve_logs["PPO + RND"]["episode_costs"].append(
                    episode_costs_rnd)
                curve_logs["PPO + RND"]["violation_flags"].append(
                    violation_flags_rnd)
                curve_logs["PPO + RND"]["lambda"].append(lambda_log_rnd)
                render_episode_video(
                    env,
                    ppo_rnd_policy,
                    os.path.join(
                        video_dir, f"{safe_setting}_ppo_rnd_{run_seed}.gif"),
                    H=args.H,
                )
                id_res, ood_res = evaluate_on_benchmarks(
                    env,
                    ppo_rnd_policy,
                    "test_maps",
                    5,
                    H=args.H,
                    ood_map_folder="ood_maps",
                    num_ood_maps=10,
                )
                metrics["PPO + RND"]["rewards"][run_seed] = [id_res[0]]
                metrics["PPO + RND"]["success"][run_seed] = [0.0]
                metrics["PPO + RND"]["ood_rewards"][run_seed] = [ood_res[0]]
                bench["PPO + RND"].append(id_res[0])
                bench_ood["PPO + RND"].append(ood_res[0])
                plot_policy_coverage(
                    env,
                    ppo_rnd_policy,
                    "PPO + RND",
                    setting["name"],
                    plot_dir,
                    args.H,
                )

        for name, data in metrics.items():
            if data.get("rewards"):
                write_aggregate_csv(name, data, args.split)

        # Plot aggregated curves across seeds for all methods
        panel_logs: dict[str, dict[str, list[list[float]]]] = {}
        for name, logs_dict in curve_logs.items():
            if logs_dict["rewards"]:
                metrics_to_plot: dict[str, list[list[float]]] = {
                    "Reward": logs_dict["rewards"],
                    "Success": logs_dict["success"],
                }
                if logs_dict["intrinsic"]:
                    metrics_to_plot["Intrinsic Reward"] = logs_dict["intrinsic"]
                if logs_dict["episode_costs"]:
                    metrics_to_plot["Episode Cost"] = logs_dict["episode_costs"]
                if logs_dict["violation_flags"]:
                    metrics_to_plot["Constraint Violation"] = logs_dict["violation_flags"]
                    safe_setting = setting["name"].replace(" ", "_")
                    safe_name = name.replace(" ", "_")
                    out_file_vr = None
                    if plot_dir:
                        out_file_vr = os.path.join(
                            plot_dir, f"{safe_setting}_{safe_name}_violation_rate.pdf"
                        )
                    plot_violation_rate(
                        logs_dict["violation_flags"], output_path=out_file_vr
                    )
                panel_logs[name] = metrics_to_plot

        comparison_logs: dict[str, list[list[float]]] = {}
        for method in [
            "PPO Only",
            "LPPO",
            "Shielded-PPO",
            "PPO + ICM + Planner",
        ]:
            logs = curve_logs.get(method, {}).get("violation_flags", [])
            if logs:
                comparison_logs[method] = logs
        if comparison_logs:
            cmp_path = None
            if plot_dir:
                cmp_path = os.path.join(plot_dir, "violation_compare.pdf")
            plot_violation_comparison(comparison_logs, output_path=cmp_path)
        if panel_logs:
            out_file = None
            if plot_dir:
                safe_setting = setting["name"].replace(" ", "_")
                out_file = os.path.join(plot_dir, f"{safe_setting}_panels.pdf")
            plot_learning_panels(panel_logs, output_path=out_file)

        # Aggregate metrics across seeds for this setting
        baseline_rewards = np.array(
            flatten_metric(metrics["PPO Only"]["rewards"]))
        baseline_success = np.array(
            flatten_metric(metrics["PPO Only"]["success"]))
        baseline_violations = np.array(metrics["PPO Only"]["violation_flags"])

        overall_reward_p = np.nan
        overall_success_p = np.nan
        overall_violation_p = np.nan
        reward_posthoc: dict[str, float] = {}
        success_posthoc: dict[str, float] = {}
        violation_posthoc: dict[str, float] = {}
        if args.stat_test == "anova":
            reward_groups = []
            success_groups = []
            violation_groups = []
            for data in metrics.values():
                rewards_flat = flatten_metric(data["rewards"])
                success_flat = flatten_metric(data["success"])
                violations = data["violation_flags"]
                if (
                    len(rewards_flat) == len(baseline_rewards)
                    and rewards_flat
                    and len(violations) == len(baseline_violations)
                ):
                    reward_groups.append(rewards_flat)
                    success_groups.append(success_flat)
                    violation_groups.append(violations)
            if len(reward_groups) >= 3:
                overall_reward_p = anova_oneway(reward_groups, use_var="unequal").pvalue
                overall_success_p = anova_oneway(success_groups, use_var="unequal").pvalue
                overall_violation_p = anova_oneway(
                    violation_groups, use_var="unequal"
                ).pvalue
                print("Welch ANOVA reward p-value:", overall_reward_p)
                print("Welch ANOVA success p-value:", overall_success_p)
                print("Welch ANOVA violation p-value:", overall_violation_p)
                if overall_reward_p < 0.05:
                    df_r = []
                    for g_name, data in metrics.items():
                        for val in flatten_metric(data["rewards"]):
                            df_r.append({"group": g_name, "value": val})
                    df_r = pd.DataFrame(df_r)
                    gh_r = pg.pairwise_gameshowell(dv="value", between="group", data=df_r)
                    print("Games-Howell post-hoc rewards:\n", gh_r)
                    for _, row in gh_r.iterrows():
                        if row["A"] == "PPO Only":
                            reward_posthoc[row["B"]] = row["pval"]
                        elif row["B"] == "PPO Only":
                            reward_posthoc[row["A"]] = row["pval"]
                if overall_success_p < 0.05:
                    df_s = []
                    for g_name, data in metrics.items():
                        for val in flatten_metric(data["success"]):
                            df_s.append({"group": g_name, "value": val})
                    df_s = pd.DataFrame(df_s)
                    gh_s = pg.pairwise_gameshowell(dv="value", between="group", data=df_s)
                    print("Games-Howell post-hoc success:\n", gh_s)
                    for _, row in gh_s.iterrows():
                        if row["A"] == "PPO Only":
                            success_posthoc[row["B"]] = row["pval"]
                        elif row["B"] == "PPO Only":
                            success_posthoc[row["A"]] = row["pval"]
                if overall_violation_p < 0.05:
                    df_v = []
                    for g_name, data in metrics.items():
                        for val in data["violation_flags"]:
                            df_v.append({"group": g_name, "value": val})
                    df_v = pd.DataFrame(df_v)
                    gh_v = pg.pairwise_gameshowell(dv="value", between="group", data=df_v)
                    print("Games-Howell post-hoc violations:\n", gh_v)
                    for _, row in gh_v.iterrows():
                        if row["A"] == "PPO Only":
                            violation_posthoc[row["B"]] = row["pval"]
                        elif row["B"] == "PPO Only":
                            violation_posthoc[row["A"]] = row["pval"]
        elif args.stat_test == "friedman":
            reward_groups = []
            success_groups = []
            violation_groups = []
            for data in metrics.values():
                rewards_flat = flatten_metric(data["rewards"])
                success_flat = flatten_metric(data["success"])
                violations = data["violation_flags"]
                if (
                    len(rewards_flat) == len(baseline_rewards)
                    and rewards_flat
                    and len(violations) == len(baseline_violations)
                ):
                    reward_groups.append(rewards_flat)
                    success_groups.append(success_flat)
                    violation_groups.append(violations)
            if len(reward_groups) >= 3:
                overall_reward_p = friedmanchisquare(*reward_groups).pvalue
                overall_success_p = friedmanchisquare(*success_groups).pvalue
                overall_violation_p = friedmanchisquare(*violation_groups).pvalue
                print("Friedman test reward p-value:", overall_reward_p)
                print("Friedman test success p-value:", overall_success_p)
                print("Friedman test violation p-value:", overall_violation_p)

        results = []
        reward_ps: list[float] = []
        reward_idx: list[int] = []
        success_ps: list[float] = []
        success_idx: list[int] = []
        violation_ps: list[float] = []
        violation_idx: list[int] = []
        for name, data in metrics.items():
            if args.stat_test in {"anova", "friedman"}:
                if name == "PPO Only":
                    p_reward = overall_reward_p
                    p_success = overall_success_p
                    p_violation = overall_violation_p
                    reward_effect = np.nan
                    success_effect = np.nan
                    violation_effect = np.nan
                else:
                    if args.stat_test == "anova" and overall_reward_p < 0.05:
                        p_reward = reward_posthoc.get(name, np.nan)
                    else:
                        p_reward = np.nan
                    if args.stat_test == "anova" and overall_success_p < 0.05:
                        p_success = success_posthoc.get(name, np.nan)
                    else:
                        p_success = np.nan
                    if args.stat_test == "anova" and overall_violation_p < 0.05:
                        p_violation = violation_posthoc.get(name, np.nan)
                    else:
                        p_violation = np.nan
                    reward_effect = compute_cohens_d(
                        baseline_rewards, flatten_metric(data["rewards"])
                    )
                    success_effect = compute_cohens_d(
                        baseline_success, flatten_metric(data["success"])
                    )
                    violation_effect = compute_cohens_d(
                        baseline_violations, np.asarray(data["violation_flags"])
                    )
            elif name == "PPO Only":
                p_reward = np.nan
                p_success = np.nan
                p_violation = np.nan
                reward_effect = np.nan
                success_effect = np.nan
                violation_effect = np.nan
            else:
                base_arr, meth_arr = get_paired_arrays(
                    metrics["PPO Only"]["rewards"], data["rewards"])
                base_succ, meth_succ = get_paired_arrays(
                    metrics["PPO Only"]["success"], data["success"])
                base_viol = np.asarray(metrics["PPO Only"]["violation_flags"])
                meth_viol = np.asarray(data["violation_flags"])
                if args.stat_test == "paired":
                    p_reward = ttest_rel(base_arr, meth_arr).pvalue
                    p_success = ttest_rel(base_succ, meth_succ).pvalue
                    p_violation = ttest_rel(base_viol, meth_viol).pvalue
                    reward_effect = compute_cohens_d(base_arr, meth_arr, paired=True)
                    success_effect = compute_cohens_d(base_succ, meth_succ, paired=True)
                    violation_effect = compute_cohens_d(
                        base_viol, meth_viol, paired=True
                    )
                elif args.stat_test == "welch":
                    meth_rewards = flatten_metric(data["rewards"])
                    meth_success = flatten_metric(data["success"])
                    p_reward = ttest_ind(
                        baseline_rewards, meth_rewards, equal_var=False
                    ).pvalue
                    p_success = ttest_ind(
                        baseline_success, meth_success, equal_var=False
                    ).pvalue
                    p_violation = ttest_ind(
                        baseline_violations, meth_viol, equal_var=False
                    ).pvalue
                    reward_effect = compute_cohens_d(
                        baseline_rewards, meth_rewards
                    )
                    success_effect = compute_cohens_d(
                        baseline_success, meth_success
                    )
                    violation_effect = compute_cohens_d(
                        baseline_violations, meth_viol
                    )
                else:  # mannwhitney
                    meth_rewards = flatten_metric(data["rewards"])
                    meth_success = flatten_metric(data["success"])
                    p_reward = mannwhitneyu(
                        baseline_rewards, meth_rewards, alternative="two-sided"
                    ).pvalue
                    p_success = mannwhitneyu(
                        baseline_success, meth_success, alternative="two-sided"
                    ).pvalue
                    p_violation = mannwhitneyu(
                        baseline_violations,
                        meth_viol,
                        alternative="two-sided",
                    ).pvalue
                    reward_effect = compute_cohens_d(
                        baseline_rewards, meth_rewards
                    )
                    success_effect = compute_cohens_d(
                        baseline_success, meth_success
                    )
                    violation_effect = compute_cohens_d(
                        baseline_violations, meth_viol
                    )
            reward_mean, reward_ci = mean_ci(flatten_metric(data["rewards"]))
            reward = f"{reward_mean:.2f} ± {reward_ci:.2f}"
            ood_mean, ood_ci = mean_ci(flatten_metric(data["ood_rewards"]))
            reward_ood = f"{ood_mean:.2f} ± {ood_ci:.2f}"
            if name != "PPO Only":
                check_reward_difference_ci(
                    flatten_metric(metrics["PPO Only"]["rewards"]),
                    flatten_metric(data["rewards"]))
            auc_reward = format_mean_ci(data["auc_reward"])
            success = format_bootstrap_ci(flatten_metric(data["success"]))
            planner = format_mean_ci(data["planner_pct"], scale=100)
            masked_action_rate = format_mean_ci(data["masked_action_rate"], scale=100)
            planner_adherence = format_mean_ci(
                data["planner_adherence_pct"], scale=100
            )
            unique_cells = format_mean_ci(data["unique_cells"])
            min_dist = format_mean_ci(data["min_dist"])
            episode_time = format_mean_ci(data["episode_time"])
            steps_per_sec = format_mean_ci(data["steps_per_sec"])
            wall_time = format_mean_ci(data["wall_time"])
            spikes = format_mean_ci(data["spikes"])
            train_cost = format_mean_ci(data["episode_costs"])
            violation = format_bootstrap_ci(data["violation_flags"])
            fve_mean, fve_ci = mean_ci(data["first_violation_episode"])
            fve_str = f"{fve_mean:.2f} ± {fve_ci:.2f}"

            results.append(
                {
                    "Budget": args.cost_limit,
                    "Setting": setting["name"],
                    "Model": name,
                    "Train Reward": reward,
                    "OOD Reward": reward_ood,
                    "Reward AUC": auc_reward,
                    "Success": success,
                    "Planner Usage %": planner,
                    "Masked Action Rate": masked_action_rate,
                    "Planner Adherence %": planner_adherence,
                    "Unique Cells": unique_cells,
                    "Min Enemy Dist": min_dist,
                    "Episode Time": episode_time,
                    "Steps/s": steps_per_sec,
                    "Total Time": wall_time,
                    "Intrinsic Spikes": spikes,
                    "Train Cost": train_cost,
                    "Pr[Jc > d]": violation,
                    "First Violation Episode": fve_str,
                    "Reward p-value": p_reward,
                    "Success p-value": p_success,
                    "Violation p-value": p_violation,
                    "Reward effect size": reward_effect,
                    "Success effect size": success_effect,
                    "Violation effect size": violation_effect,
                    "Reward p-adj": np.nan,
                    "Success p-adj": np.nan,
                    "Violation p-adj": np.nan,
                }
            )
            if name != "PPO Only" and not np.isnan(p_reward):
                reward_ps.append(p_reward)
                reward_idx.append(len(results) - 1)
            if name != "PPO Only" and not np.isnan(p_success):
                success_ps.append(p_success)
                success_idx.append(len(results) - 1)
            if name != "PPO Only" and not np.isnan(p_violation):
                violation_ps.append(p_violation)
                violation_idx.append(len(results) - 1)

        bench_results = []
        for name, vals in bench.items():
            ood_vals = bench_ood.get(name, [])
            if vals or ood_vals:
                entry = {
                    "Budget": args.cost_limit,
                    "Setting": setting["name"],
                    "Model": name,
                    "Benchmark Reward": format_mean_ci(vals) if vals else "N/A",
                }
                if ood_vals:
                    entry["OOD Reward"] = format_mean_ci(ood_vals)
                bench_results.append(entry)

        if reward_ps:
            _, padj, _, _ = multipletests(reward_ps, method="holm")
            for i, adj in zip(reward_idx, padj):
                results[i]["Reward p-adj"] = adj
        if success_ps:
            _, padj, _, _ = multipletests(success_ps, method="holm")
            for i, adj in zip(success_idx, padj):
                results[i]["Success p-adj"] = adj
        if violation_ps:
            _, padj, _, _ = multipletests(violation_ps, method="holm")
            for i, adj in zip(violation_idx, padj):
                results[i]["Violation p-adj"] = adj
        for name, data in metrics.items():
            if data["rewards"] and data["episode_costs"]:
                pareto_metrics[name]["rewards"].extend(
                    flatten_metric(data["rewards"])
                )
                pareto_metrics[name]["costs"].extend(data["episode_costs"])
                if data.get("violation_flags"):
                    pareto_metrics[name]["violations"].extend(
                        data["violation_flags"]
                    )

        all_results.extend(results)
        all_bench.extend(bench_results)

    pareto_rows = []
    for name, data in pareto_metrics.items():
        if data["rewards"] and data["costs"]:
            r_mean, r_ci = mean_ci(data["rewards"])
            c_mean, c_ci = mean_ci(data["costs"])
            pareto_rows.append(
                {
                    "Model": name,
                    "Reward Mean": r_mean,
                    "Reward CI": r_ci,
                    "Cost Mean": c_mean,
                    "Cost CI": c_ci,
                }
            )
    if pareto_rows:
        pareto_df = pd.DataFrame(pareto_rows)
        pareto_df["Model"] = pareto_df["Model"].replace(NAME_MAP)
        plot_pareto(
            pareto_df,
            args.cost_limit,
            os.path.join(figure_dir, "pareto_all.pdf"),
        )
    save_pareto_summaries(pareto_metrics, args.split)
    append_budget_sweep(pareto_metrics, args.cost_limit)
    save_violation_curves(curve_logs)

    df_train = pd.DataFrame(all_results)
    df_train["Model"] = df_train["Model"].replace(NAME_MAP)
    generate_results_table(
        df_train, os.path.join(result_dir, "training_results.html")
    )
    df_main = build_main_table(df_train)
    if not df_main.empty:
        generate_results_table(
            df_main, os.path.join(result_dir, "ablation_table.html")
        )
    if all_bench:
        df_bench = pd.DataFrame(all_bench)
        generate_results_table(
            df_bench, os.path.join(result_dir, "benchmark_results.html")
        )

    if args.postprocess:
        fig_out = Path("figures") / algo_name / seed_key
        table_out = Path("tables") / algo_name / seed_key
        fig_out.mkdir(parents=True, exist_ok=True)
        table_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "generate_figures.py", "--output-dir", str(fig_out)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "generate_tables.py", "--output-dir", str(table_out)],
            check=True,
        )

    if args.log_backend == "tensorboard" and logger is not None:
        logger.close()
    elif args.log_backend == "wandb" and logger is not None:
        logger.finish()


if __name__ == "__main__":
    cli_args = sys.argv[1:]
    base_args = parse_args(cli_args)

    def run_with_budgets(args: argparse.Namespace) -> None:
        for budget in [0.05, 0.10]:
            for dyn_risk in [False, True]:
                for dyn_cost in [False, True]:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.cost_limit = budget
                    run_args.dynamic_risk = dyn_risk
                    run_args.dynamic_cost = dyn_cost
                    run(run_args)

    if base_args.all_algos:
        import glob

        algo_files = sorted(
            glob.glob(os.path.join("configs", "algo", "*.yml"))
            + glob.glob(os.path.join("configs", "algo", "*.yaml"))
        )
        cleaned: list[str] = []
        skip = False
        for a in cli_args:
            if skip:
                skip = False
                continue
            if a == "--algo-config":
                skip = True
                continue
            if a.startswith("--algo-config=") or a == "--all-algos":
                continue
            cleaned.append(a)
        for algo_cfg in algo_files:
            algo_args = parse_args(cleaned + ["--algo-config", algo_cfg])
            run_with_budgets(algo_args)
    else:
        run_with_budgets(base_args)
