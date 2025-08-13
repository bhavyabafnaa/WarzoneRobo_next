import os
for d in ["videos", "results", "figures", "checkpoints"]:
    os.makedirs(d, exist_ok=True)
import argparse
import warnings
import yaml
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
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

from src.env import (
    GridWorldICM,
    export_benchmark_maps,
    visualize_paths_on_benchmark_maps,
    evaluate_on_benchmarks,
)
from src.visualization import (
    plot_training_curves,
    plot_pareto,
    plot_heatmap_with_path,
    generate_results_table,
    render_episode_video,
)
from src.icm import ICMModule
from src.rnd import RNDModule
from src.pseudocount import PseudoCountExploration
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent, get_beta_schedule
from src.utils import save_model, count_intrinsic_spikes


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
    """Return the filtered main results table with selected metrics."""

    cols = ["Model", "Train Reward", "Success", "Train Cost", "Pr[Jc > d]"]
    if "Model" not in df_train.columns:
        return pd.DataFrame(columns=cols)
    return df_train[df_train["Model"].isin(MAIN_METHODS)][cols].reset_index(drop=True)


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


def parse_args():
    """Parse command line arguments with optional YAML config defaults.

    The ``--config`` file is parsed first and any keys inside will set the
    parser defaults. Command line flags always take precedence, allowing quick
    overrides without editing the YAML.

    Example YAML snippet::

        grid_size: 8
        dynamic_risk: true
        add_noise: false
    """

    parser = argparse.ArgumentParser(
        description="Train or evaluate PPO agents")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
        default="configs/default.yaml",
    )
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--cost_weight", type=float, default=2.0)
    parser.add_argument("--risk_weight", type=float, default=3.0)
    parser.add_argument("--revisit_penalty", type=float, default=1.0)
    parser.add_argument(
        "--eta_lambda",
        type=float,
        choices=[0.01, 0.05],
        default=0.01,
        help="Learning rate for lambda update",
    )
    parser.add_argument(
        "--cost_limit",
        "--d",
        dest="cost_limit",
        type=float,
        default=1.0,
        help="Cost threshold for constraint",
    )
    parser.add_argument("--c1", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=0.5)
    parser.add_argument("--c3", type=float, default=0.01)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.6,
        help="Risk threshold for masking unsafe actions",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        choices=[2, 4],
        default=2.0,
        help="Soft risk penalty weight applied to policy logits",
    )
    parser.add_argument(
        "--disable-risk-penalty",
        action="store_true",
        help="Disable risk-based soft penalty on logits",
    )
    parser.add_argument(
        "--initial-beta",
        type=float,
        default=0.1,
        help="Starting weight for intrinsic reward",
    )
    parser.add_argument(
        "--final-beta",
        type=float,
        default=None,
        help="Final beta value after decay (defaults to initial value)",
    )
    parser.add_argument(
        "--dynamic_risk",
        action="store_true",
        help="Enable dynamic risk in env")
    parser.add_argument(
        "--dynamic_cost",
        action="store_true",
        help="Enable dynamic cost in env")
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
        default=8,
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
        "--stat-test",
        choices=["paired", "welch", "mannwhitney", "anova", "friedman"],
        default="paired",
        help="Statistical test for result comparisons",
    )

    # Parse once to read the config file path and load defaults from YAML. We
    # intentionally parse without removing any of the original command line
    # arguments so they can still override the config on the second pass.
    config_args, _ = parser.parse_known_args()
    if config_args.config and os.path.exists(config_args.config):
        with open(config_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)

    # Final parse with config defaults applied; command line flags take
    # precedence over YAML settings.
    return parser.parse_args()
def run(args):
    budget_str = f"budget_{args.cost_limit:.2f}"
    video_dir = os.path.join("videos", budget_str)
    result_dir = os.path.join("results", budget_str)
    figure_dir = os.path.join("figures", budget_str)
    checkpoint_dir = os.path.join("checkpoints", budget_str)
    plot_dir = None
    if args.plot_dir:
        plot_dir = os.path.join(args.plot_dir, budget_str)
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
            seeds = range(seeds[0])
    else:
        seeds = [args.seed]

    env = GridWorldICM(
        grid_size=grid_size,
        dynamic_risk=args.dynamic_risk,
        dynamic_cost=args.dynamic_cost,
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

    for setting in settings:
        args.disable_icm = setting["icm"]
        args.disable_rnd = setting["rnd"]
        args.disable_planner = setting["planner"]

        metrics = {
            "PPO Only": {
                "rewards": {},
                "success": {},
                "planner_pct": [],
                "mask_rate": [],
                "adherence_rate": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "coverage": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
            },
            "PPO + ICM": {
                "rewards": {},
                "success": {},
                "planner_pct": [],
                "mask_rate": [],
                "adherence_rate": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "coverage": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
            },
            "PPO + ICM + Planner": {
                "rewards": {},
                "success": {},
                "planner_pct": [],
                "mask_rate": [],
                "adherence_rate": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "coverage": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
            },
            "PPO + count": {
                "rewards": {},
                "success": {},
                "planner_pct": [],
                "mask_rate": [],
                "adherence_rate": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "coverage": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
            },
            "PPO + RND": {
                "rewards": {},
                "success": {},
                "planner_pct": [],
                "mask_rate": [],
                "adherence_rate": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "coverage": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
            },
            "PPO + PC": {
                "rewards": {},
                "success": {},
                "planner_pct": [],
                "mask_rate": [],
                "adherence_rate": [],
                "spikes": [],
                "episode_costs": [],
                "violation_flags": [],
                "first_violation_episode": [],
                "coverage": [],
                "min_dist": [],
                "episode_time": [],
                "steps_per_sec": [],
                "wall_time": [],
            },
        }
        bench = {
            "PPO Only": [],
            "PPO + ICM": [],
            "PPO + ICM + Planner": [],
            "PPO + count": [],
            "PPO + RND": [],
            "PPO + PC": [],
        }

        curve_logs = {
            "PPO Only": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
            },
            "PPO + ICM": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
            },
            "PPO + ICM + Planner": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
            },
            "PPO + count": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
            },
            "PPO + RND": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
            },
            "PPO + PC": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
                "episode_costs": [],
                "violation_flags": [],
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
            opt_ppo = optim.Adam(ppo_policy.parameters(), lr=3e-4)
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
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                c3=args.c3,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                imagination_k=args.K,
                world_model_lr=args.world_model_lr,
            )
            metrics["PPO Only"]["planner_pct"].append(
                float(np.mean(planner_rate_ppo_only)))
            metrics["PPO Only"]["mask_rate"].append(
                float(np.mean(mask_rates_ppo_only)))
            metrics["PPO Only"]["adherence_rate"].append(
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
            metrics["PPO Only"]["coverage"].append(
                float(np.mean(coverage_ppo_only)))
            metrics["PPO Only"]["episode_time"].append(
                float(np.mean(episode_times_ppo_only)))
            metrics["PPO Only"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_ppo_only)))
            metrics["PPO Only"]["wall_time"].append(
                float(wall_clock_times_ppo_only[-1]))
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
            render_episode_video(
                env,
                ppo_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_only_{run_seed}.gif"),
                H=args.H,
            )
            rewards_b, success_b = evaluate_policy_on_maps(
                env, ppo_policy, "test_maps", 5, H=args.H)
            metrics["PPO Only"]["rewards"][run_seed] = rewards_b
            metrics["PPO Only"]["success"][run_seed] = success_b
            bench["PPO Only"].append(float(np.mean(rewards_b)))

            # PPO + ICM
            if not args.disable_icm:
                print("Training PPO + ICM")
                ppo_icm_policy = PPOPolicy(input_dim, action_dim)
                opt_icm_policy = optim.Adam(
                    ppo_icm_policy.parameters(), lr=3e-4)
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
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                    eta_lambda=args.eta_lambda,
                    cost_limit=args.cost_limit,
                    c1=args.c1,
                    c2=args.c2,
                    c3=args.c3,
                    tau=args.tau,
                    kappa=args.kappa,
                    use_risk_penalty=not args.disable_risk_penalty,
                    H=args.H,
                    waypoint_bonus=args.waypoint_bonus,
                    imagination_k=args.K,
                    world_model_lr=args.world_model_lr,
                )
                metrics["PPO + ICM"]["planner_pct"].append(
                    float(np.mean(planner_rate_icm)))
                metrics["PPO + ICM"]["mask_rate"].append(
                    float(np.mean(mask_rates_icm)))
                metrics["PPO + ICM"]["adherence_rate"].append(
                    float(np.mean(adherence_rates_icm)))
                metrics["PPO + ICM"]["min_dist"].append(
                    float(np.mean(min_dists_icm)))
                metrics["PPO + ICM"]["spikes"].append(
                    count_intrinsic_spikes(intrinsic_icm))
                metrics["PPO + ICM"]["episode_costs"].append(
                    float(np.mean(episode_costs_icm)))
                metrics["PPO + ICM"]["violation_flags"].append(
                    float(np.mean(violation_flags_icm)))
                metrics["PPO + ICM"]["first_violation_episode"].append(
                    first_violation_episode_icm
                )
                metrics["PPO + ICM"]["coverage"].append(
                    float(np.mean(coverage_icm)))
                metrics["PPO + ICM"]["episode_time"].append(
                    float(np.mean(episode_times_icm)))
                metrics["PPO + ICM"]["steps_per_sec"].append(
                    float(np.mean(steps_per_sec_icm)))
                metrics["PPO + ICM"]["wall_time"].append(
                    float(wall_clock_times_icm[-1]))
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
                render_episode_video(
                    env,
                    ppo_icm_policy,
                    os.path.join(
                        video_dir, f"{safe_setting}_ppo_icm_{run_seed}.gif"),
                    H=args.H,
                )
                rewards_b, success_b = evaluate_policy_on_maps(
                    env, ppo_icm_policy, "test_maps", 5, H=args.H)
                metrics["PPO + ICM"]["rewards"][run_seed] = rewards_b
                metrics["PPO + ICM"]["success"][run_seed] = success_b
                bench["PPO + ICM"].append(float(np.mean(rewards_b)))

            # PPO + Pseudo-count exploration
            print("Training PPO + PC")
            ppo_pc_policy = PPOPolicy(input_dim, action_dim)
            opt_pc_policy = optim.Adam(ppo_pc_policy.parameters(), lr=3e-4)
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
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                c3=args.c3,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                imagination_k=args.K,
                world_model_lr=args.world_model_lr,
            )
            metrics["PPO + PC"]["planner_pct"].append(
                float(np.mean(planner_rate_pc)))
            metrics["PPO + PC"]["mask_rate"].append(
                float(np.mean(mask_rates_pc)))
            metrics["PPO + PC"]["adherence_rate"].append(
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
            metrics["PPO + PC"]["coverage"].append(
                float(np.mean(coverage_pc)))
            metrics["PPO + PC"]["episode_time"].append(
                float(np.mean(episode_times_pc)))
            metrics["PPO + PC"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_pc)))
            metrics["PPO + PC"]["wall_time"].append(
                float(wall_clock_times_pc[-1]))
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
            render_episode_video(
                env,
                ppo_pc_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_pc_{run_seed}.gif"),
                H=args.H,
            )
            rewards_b, success_b = evaluate_policy_on_maps(
                env, ppo_pc_policy, "test_maps", 5, H=args.H)
            metrics["PPO + PC"]["rewards"][run_seed] = rewards_b
            metrics["PPO + PC"]["success"][run_seed] = success_b
            bench["PPO + PC"].append(float(np.mean(rewards_b)))

            # PPO + ICM + Planner
            if not args.disable_icm and not args.disable_planner:
                print("Training PPO + ICM + Planner")
                ppo_icm_planner_policy = PPOPolicy(input_dim, action_dim)
                opt_plan_policy = optim.Adam(
                    ppo_icm_planner_policy.parameters(), lr=3e-4)
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
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                    eta_lambda=args.eta_lambda,
                    cost_limit=args.cost_limit,
                    c1=args.c1,
                    c2=args.c2,
                    c3=args.c3,
                    tau=args.tau,
                    kappa=args.kappa,
                    use_risk_penalty=not args.disable_risk_penalty,
                    H=args.H,
                    waypoint_bonus=args.waypoint_bonus,
                    imagination_k=args.K,
                    world_model_lr=args.world_model_lr,
                )
                metrics["PPO + ICM + Planner"]["planner_pct"].append(
                    float(np.mean(planner_rate_plan)))
                metrics["PPO + ICM + Planner"]["mask_rate"].append(
                    float(np.mean(mask_rates_icm_plan)))
                metrics["PPO + ICM + Planner"]["adherence_rate"].append(
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
                metrics["PPO + ICM + Planner"]["coverage"].append(
                    float(np.mean(coverage_icm_plan)))
                metrics["PPO + ICM + Planner"]["episode_time"].append(
                    float(np.mean(episode_times_icm_plan)))
                metrics["PPO + ICM + Planner"]["steps_per_sec"].append(
                    float(np.mean(steps_per_sec_icm_plan)))
                metrics["PPO + ICM + Planner"]["wall_time"].append(
                    float(wall_clock_times_icm_plan[-1]))
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
                render_episode_video(
                    env,
                    ppo_icm_planner_policy,
                    os.path.join(
                        video_dir, f"{safe_setting}_ppo_icm_planner_{run_seed}.gif"
                    ),
                    H=args.H,
                )
                rewards_b, success_b = evaluate_policy_on_maps(
                    env, ppo_icm_planner_policy, "test_maps", 5, H=args.H)
                metrics["PPO + ICM + Planner"]["rewards"][run_seed] = rewards_b
                metrics["PPO + ICM + Planner"]["success"][run_seed] = success_b
                bench["PPO + ICM + Planner"].append(float(np.mean(rewards_b)))
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
                ppo_count_policy.parameters(), lr=3e-4)
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
                eta_lambda=args.eta_lambda,
                cost_limit=args.cost_limit,
                c1=args.c1,
                c2=args.c2,
                c3=args.c3,
                tau=args.tau,
                kappa=args.kappa,
                use_risk_penalty=not args.disable_risk_penalty,
                H=args.H,
                waypoint_bonus=args.waypoint_bonus,
                imagination_k=args.K,
                world_model_lr=args.world_model_lr,
            )
            metrics["PPO + count"]["planner_pct"].append(
                float(np.mean(planner_rate_count)))
            metrics["PPO + count"]["mask_rate"].append(
                float(np.mean(mask_rates_count)))
            metrics["PPO + count"]["adherence_rate"].append(
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
            metrics["PPO + count"]["coverage"].append(
                float(np.mean(coverage_count)))
            metrics["PPO + count"]["episode_time"].append(
                float(np.mean(episode_times_count)))
            metrics["PPO + count"]["steps_per_sec"].append(
                float(np.mean(steps_per_sec_count)))
            metrics["PPO + count"]["wall_time"].append(
                float(wall_clock_times_count[-1]))
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
            render_episode_video(
                env,
                ppo_count_policy,
                os.path.join(
                    video_dir, f"{safe_setting}_ppo_count_{run_seed}.gif"),
                H=args.H,
            )
            rewards_b, success_b = evaluate_policy_on_maps(
                env, ppo_count_policy, "test_maps", 5, H=args.H)
            metrics["PPO + count"]["rewards"][run_seed] = rewards_b
            metrics["PPO + count"]["success"][run_seed] = success_b
            bench["PPO + count"].append(float(np.mean(rewards_b)))

            # RND exploration
            if not args.disable_rnd:
                print("Training PPO + RND")
                ppo_rnd_policy = PPOPolicy(input_dim, action_dim)
                opt_rnd_policy = optim.Adam(
                    ppo_rnd_policy.parameters(), lr=3e-4)
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
                    eta_lambda=args.eta_lambda,
                    cost_limit=args.cost_limit,
                    c1=args.c1,
                    c2=args.c2,
                    c3=args.c3,
                    tau=args.tau,
                    kappa=args.kappa,
                    use_risk_penalty=not args.disable_risk_penalty,
                    H=args.H,
                    waypoint_bonus=args.waypoint_bonus,
                    imagination_k=args.K,
                    world_model_lr=args.world_model_lr,
                )
                metrics["PPO + RND"]["planner_pct"].append(
                    float(np.mean(planner_rate_rnd)))
                metrics["PPO + RND"]["mask_rate"].append(
                    float(np.mean(mask_rates_rnd)))
                metrics["PPO + RND"]["adherence_rate"].append(
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
                metrics["PPO + RND"]["coverage"].append(
                    float(np.mean(coverage_rnd)))
                metrics["PPO + RND"]["episode_time"].append(
                    float(np.mean(episode_times_rnd)))
                metrics["PPO + RND"]["steps_per_sec"].append(
                    float(np.mean(steps_per_sec_rnd)))
                metrics["PPO + RND"]["wall_time"].append(
                    float(wall_clock_times_rnd[-1]))
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
                render_episode_video(
                    env,
                    ppo_rnd_policy,
                    os.path.join(
                        video_dir, f"{safe_setting}_ppo_rnd_{run_seed}.gif"),
                    H=args.H,
                )
                rewards_b, success_b = evaluate_policy_on_maps(
                    env, ppo_rnd_policy, "test_maps", 5, H=args.H)
                metrics["PPO + RND"]["rewards"][run_seed] = rewards_b
                metrics["PPO + RND"]["success"][run_seed] = success_b
                bench["PPO + RND"].append(float(np.mean(rewards_b)))

        # Plot aggregated curves across seeds for this setting
        for name, logs_dict in curve_logs.items():
            if logs_dict["rewards"]:
                out_file = None
                if plot_dir:
                    safe_setting = setting["name"].replace(" ", "_")
                    safe_name = name.replace(" ", "_").replace("+", "")
                    out_file = os.path.join(
                        plot_dir, f"{safe_setting}_{safe_name}.pdf"
                    )
                metrics_to_plot = {
                    "Reward": logs_dict["rewards"],
                    "Success": logs_dict["success"],
                }
                if logs_dict["intrinsic"]:
                    metrics_to_plot["Intrinsic Reward"] = logs_dict["intrinsic"]
                if logs_dict["episode_costs"]:
                    metrics_to_plot["Episode Cost"] = logs_dict["episode_costs"]
                if logs_dict["violation_flags"]:
                    metrics_to_plot["Constraint Violation"] = logs_dict["violation_flags"]
                plot_training_curves(metrics_to_plot, output_path=out_file)

        # Aggregate metrics across seeds for this setting
        baseline_rewards = np.array(
            flatten_metric(metrics["PPO Only"]["rewards"]))
        baseline_success = np.array(
            flatten_metric(metrics["PPO Only"]["success"]))

        overall_reward_p = np.nan
        overall_success_p = np.nan
        reward_posthoc: dict[str, float] = {}
        success_posthoc: dict[str, float] = {}
        if args.stat_test == "anova":
            reward_groups = []
            success_groups = []
            for data in metrics.values():
                rewards_flat = flatten_metric(data["rewards"])
                success_flat = flatten_metric(data["success"])
                if len(rewards_flat) == len(baseline_rewards) and rewards_flat:
                    reward_groups.append(rewards_flat)
                    success_groups.append(success_flat)
            if len(reward_groups) >= 3:
                overall_reward_p = anova_oneway(reward_groups, use_var="unequal").pvalue
                overall_success_p = anova_oneway(success_groups, use_var="unequal").pvalue
                print("Welch ANOVA reward p-value:", overall_reward_p)
                print("Welch ANOVA success p-value:", overall_success_p)
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
        elif args.stat_test == "friedman":
            reward_groups = []
            success_groups = []
            for data in metrics.values():
                rewards_flat = flatten_metric(data["rewards"])
                success_flat = flatten_metric(data["success"])
                if len(rewards_flat) == len(baseline_rewards) and rewards_flat:
                    reward_groups.append(rewards_flat)
                    success_groups.append(success_flat)
            if len(reward_groups) >= 3:
                overall_reward_p = friedmanchisquare(*reward_groups).pvalue
                overall_success_p = friedmanchisquare(*success_groups).pvalue
                print("Friedman test reward p-value:", overall_reward_p)
                print("Friedman test success p-value:", overall_success_p)

        results = []
        reward_ps: list[float] = []
        reward_idx: list[int] = []
        success_ps: list[float] = []
        success_idx: list[int] = []
        for name, data in metrics.items():
            if args.stat_test in {"anova", "friedman"}:
                if name == "PPO Only":
                    p_reward = overall_reward_p
                    p_success = overall_success_p
                    reward_effect = np.nan
                    success_effect = np.nan
                else:
                    if args.stat_test == "anova" and overall_reward_p < 0.05:
                        p_reward = reward_posthoc.get(name, np.nan)
                    else:
                        p_reward = np.nan
                    if args.stat_test == "anova" and overall_success_p < 0.05:
                        p_success = success_posthoc.get(name, np.nan)
                    else:
                        p_success = np.nan
                    reward_effect = compute_cohens_d(
                        baseline_rewards, flatten_metric(data["rewards"])
                    )
                    success_effect = compute_cohens_d(
                        baseline_success, flatten_metric(data["success"])
                    )
            elif name == "PPO Only":
                p_reward = np.nan
                p_success = np.nan
                reward_effect = np.nan
                success_effect = np.nan
            else:
                base_arr, meth_arr = get_paired_arrays(
                    metrics["PPO Only"]["rewards"], data["rewards"])
                base_succ, meth_succ = get_paired_arrays(
                    metrics["PPO Only"]["success"], data["success"])
                if args.stat_test == "paired":
                    p_reward = ttest_rel(base_arr, meth_arr).pvalue
                    p_success = ttest_rel(base_succ, meth_succ).pvalue
                    reward_effect = compute_cohens_d(base_arr, meth_arr, paired=True)
                    success_effect = compute_cohens_d(base_succ, meth_succ, paired=True)
                elif args.stat_test == "welch":
                    meth_rewards = flatten_metric(data["rewards"])
                    meth_success = flatten_metric(data["success"])
                    p_reward = ttest_ind(
                        baseline_rewards, meth_rewards, equal_var=False
                    ).pvalue
                    p_success = ttest_ind(
                        baseline_success, meth_success, equal_var=False
                    ).pvalue
                    reward_effect = compute_cohens_d(
                        baseline_rewards, meth_rewards
                    )
                    success_effect = compute_cohens_d(
                        baseline_success, meth_success
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
                    reward_effect = compute_cohens_d(
                        baseline_rewards, meth_rewards
                    )
                    success_effect = compute_cohens_d(
                        baseline_success, meth_success
                    )
            reward_mean, reward_ci = mean_ci(flatten_metric(data["rewards"]))
            reward = f"{reward_mean:.2f} ± {reward_ci:.2f}"
            if name != "PPO Only":
                check_reward_difference_ci(
                    flatten_metric(metrics["PPO Only"]["rewards"]),
                    flatten_metric(data["rewards"]))
            success = format_bootstrap_ci(flatten_metric(data["success"]))
            planner = format_mean_ci(data["planner_pct"], scale=100)
            mask_rate = format_mean_ci(data["mask_rate"])
            adherence = format_mean_ci(data["adherence_rate"])
            coverage = format_mean_ci(data["coverage"])
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
                    "Success": success,
                    "Planner Usage %": planner,
                    "Mask Rate": mask_rate,
                    "Adherence Rate": adherence,
                    "Coverage": coverage,
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
                    "Reward effect size": reward_effect,
                    "Success effect size": success_effect,
                    "Reward p-adj": np.nan,
                    "Success p-adj": np.nan,
                }
            )
            if name != "PPO Only" and not np.isnan(p_reward):
                reward_ps.append(p_reward)
                reward_idx.append(len(results) - 1)
            if name != "PPO Only" and not np.isnan(p_success):
                success_ps.append(p_success)
                success_idx.append(len(results) - 1)

        bench_results = []
        for name, vals in bench.items():
            if vals:
                bench_results.append(
                    {
                        "Budget": args.cost_limit,
                        "Setting": setting["name"],
                        "Model": name,
                        "Benchmark Reward": format_mean_ci(vals),
                    }
                )

        if reward_ps:
            _, padj, _, _ = multipletests(reward_ps, method="holm")
            for i, adj in zip(reward_idx, padj):
                results[i]["Reward p-adj"] = adj
        if success_ps:
            _, padj, _, _ = multipletests(success_ps, method="holm")
            for i, adj in zip(success_idx, padj):
                results[i]["Success p-adj"] = adj
        pareto_rows = []
        for name, data in metrics.items():
            if data["rewards"] and data["episode_costs"]:
                r_mean, r_ci = mean_ci(flatten_metric(data["rewards"]))
                c_mean, c_ci = mean_ci(data["episode_costs"])
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
            out_file = None
            if plot_dir:
                safe_setting = setting["name"].replace(" ", "_")
                out_file = os.path.join(plot_dir, f"{safe_setting}_pareto.pdf")
            plot_pareto(pareto_df, args.cost_limit, out_file)

        all_results.extend(results)
        all_bench.extend(bench_results)

    df_train = pd.DataFrame(all_results)
    df_train["Model"] = df_train["Model"].replace(NAME_MAP)
    generate_results_table(
        df_train, os.path.join(result_dir, "training_results.html")
    )
    df_main = build_main_table(df_train)
    if not df_main.empty:
        generate_results_table(
            df_main, os.path.join(result_dir, "main_table.html")
        )
    if all_bench:
        df_bench = pd.DataFrame(all_bench)
        generate_results_table(
            df_bench, os.path.join(result_dir, "benchmark_results.html")
        )

    if args.log_backend == "tensorboard" and logger is not None:
        logger.close()
    elif args.log_backend == "wandb" and logger is not None:
        logger.finish()


if __name__ == "__main__":
    base_args = parse_args()
    for budget in [0.05, 0.10]:
        args = argparse.Namespace(**vars(base_args))
        args.cost_limit = budget
        run(args)
