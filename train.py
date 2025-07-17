import os
import argparse
import yaml
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import (
    ttest_rel,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
)

from src.env import (
    GridWorldICM,
    export_benchmark_maps,
    visualize_paths_on_benchmark_maps,
    evaluate_on_benchmarks,
)
from src.visualization import (
    plot_training_curves,
    plot_heatmap_with_path,
    generate_results_table,
    render_episode_video,
)
from src.icm import ICMModule
from src.rnd import RNDModule
from src.pseudocount import PseudoCountExploration
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent
from src.utils import save_model, count_intrinsic_spikes


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
        default=None,
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
        choices=["paired", "welch", "mannwhitney", "anova"],
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


def main():
    args = parse_args()

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

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
    input_dim = 4 * grid_size * grid_size
    action_dim = 4

    if args.seeds:
        seeds = args.seeds
        if len(seeds) == 1:
            seeds = list(range(seeds[0]))
    else:
        seeds = [args.seed]

    env = GridWorldICM(
        grid_size=grid_size,
        dynamic_risk=args.dynamic_risk,
        dynamic_cost=args.dynamic_cost,
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

    export_benchmark_maps(env, num_train=15, num_test=5)

    policy_demo = PPOPolicy(input_dim, action_dim)
    visualize_paths_on_benchmark_maps(
        env, policy_demo, map_folder="train_maps/", num_maps=9
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
                "rewards": [],
                "success": [],
                "planner_pct": [],
                "spikes": [],
            },
            "PPO + ICM": {
                "rewards": [],
                "success": [],
                "planner_pct": [],
                "spikes": [],
            },
            "PPO + ICM + Planner": {
                "rewards": [],
                "success": [],
                "planner_pct": [],
                "spikes": [],
            },
            "PPO + count": {
                "rewards": [],
                "success": [],
                "planner_pct": [],
                "spikes": [],
            },
            "PPO + RND": {
                "rewards": [],
                "success": [],
                "planner_pct": [],
                "spikes": [],
            },
            "PPO + PC": {
                "rewards": [],
                "success": [],
                "planner_pct": [],
                "spikes": [],
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
            "PPO Only": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + ICM": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + ICM + Planner": {
                "rewards": [],
                "intrinsic": [],
                "success": [],
            },
            "PPO + count": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + RND": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + PC": {"rewards": [], "intrinsic": [], "success": []},
        }

        for run_seed in seeds:
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
            )
            metrics["PPO Only"]["rewards"].append(
                float(np.mean(rewards_ppo_only)))
            metrics["PPO Only"]["success"].append(
                float(
                    sum(success_ppo_only)) /
                len(success_ppo_only) if success_ppo_only else 0.0)
            metrics["PPO Only"]["planner_pct"].append(
                float(np.mean(planner_rate_ppo_only)))
            metrics["PPO Only"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_ppo_only))
            save_model(
                ppo_policy,
                os.path.join(
                    "checkpoints",
                    f"ppo_only_{run_seed}.pt"))
            curve_logs["PPO Only"]["rewards"].append(rewards_ppo_only)
            curve_logs["PPO Only"]["success"].append(success_ppo_only)
            render_episode_video(
                env, ppo_policy, os.path.join(
                    "videos", f"ppo_only_{run_seed}.gif"))
            mean_b, std_b = evaluate_on_benchmarks(
                env, ppo_policy, "test_maps", 5)
            bench["PPO Only"].append(mean_b)

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
                    beta=args.initial_beta,
                    final_beta=args.final_beta,
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                )
                metrics["PPO + ICM"]["rewards"].append(
                    float(np.mean(rewards_ppo_icm)))
                metrics["PPO + ICM"]["success"].append(
                    float(sum(success_icm)) / len(success_icm)
                    if success_icm
                    else 0.0
                )
                metrics["PPO + ICM"]["planner_pct"].append(
                    float(np.mean(planner_rate_icm)))
                metrics["PPO + ICM"]["spikes"].append(
                    count_intrinsic_spikes(intrinsic_icm))
                save_model(
                    ppo_icm_policy,
                    os.path.join(
                        "checkpoints",
                        f"ppo_icm_{run_seed}.pt"),
                    icm=icm)
                curve_logs["PPO + ICM"]["rewards"].append(rewards_ppo_icm)
                curve_logs["PPO + ICM"]["intrinsic"].append(intrinsic_icm)
                curve_logs["PPO + ICM"]["success"].append(success_icm)
                render_episode_video(
                    env, ppo_icm_policy, os.path.join(
                        "videos", f"ppo_icm_{run_seed}.gif"))
                mean_b, std_b = evaluate_on_benchmarks(
                    env, ppo_icm_policy, "test_maps", 5)
                bench["PPO + ICM"].append(mean_b)

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
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
            )
            metrics["PPO + PC"]["rewards"].append(float(np.mean(rewards_pc)))
            metrics["PPO + PC"]["success"].append(
                float(sum(success_pc)) / len(success_pc) if success_pc else 0.0
            )
            metrics["PPO + PC"]["planner_pct"].append(
                float(np.mean(planner_rate_pc)))
            metrics["PPO + PC"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_pc))
            save_model(
                ppo_pc_policy,
                os.path.join(
                    "checkpoints",
                    f"ppo_pc_{run_seed}.pt"))
            curve_logs["PPO + PC"]["rewards"].append(rewards_pc)
            curve_logs["PPO + PC"]["success"].append(success_pc)
            render_episode_video(
                env, ppo_pc_policy, os.path.join(
                    "videos", f"ppo_pc_{run_seed}.gif"))
            mean_b, std_b = evaluate_on_benchmarks(
                env, ppo_pc_policy, "test_maps", 5)
            bench["PPO + PC"].append(mean_b)

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
                    beta=args.initial_beta,
                    final_beta=args.final_beta,
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                )
                metrics["PPO + ICM + Planner"]["rewards"].append(
                    float(np.mean(rewards_ppo_icm_plan)))
                metrics["PPO + ICM + Planner"]["success"].append(
                    float(sum(success_plan)) / len(success_plan)
                    if success_plan
                    else 0.0
                )
                metrics["PPO + ICM + Planner"]["planner_pct"].append(
                    float(np.mean(planner_rate_plan)))
                metrics["PPO + ICM + Planner"]["spikes"].append(
                    count_intrinsic_spikes(intrinsic_plan))
                save_model(
                    ppo_icm_planner_policy,
                    os.path.join(
                        "checkpoints",
                        f"ppo_icm_planner_{run_seed}.pt"),
                    icm=icm,
                )
                curve_logs["PPO + ICM + Planner"]["rewards"].append(
                    rewards_ppo_icm_plan)
                curve_logs["PPO + ICM + Planner"]["intrinsic"].append(
                    intrinsic_plan)
                curve_logs["PPO + ICM + Planner"]["success"].append(
                    success_plan)
                render_episode_video(
                    env,
                    ppo_icm_planner_policy,
                    os.path.join("videos", f"ppo_icm_planner_{run_seed}.gif"),
                )
                mean_b, std_b = evaluate_on_benchmarks(
                    env, ppo_icm_planner_policy, "test_maps", 5)
                bench["PPO + ICM + Planner"].append(mean_b)
                if paths_plan:
                    heat_path = None
                    if args.plot_dir:
                        safe_setting = setting["name"].replace(" ", "_")
                        heat_filename = (
                            f"heatmap_{safe_setting}_{run_seed}.pdf"
                        )
                        heat_path = os.path.join(
                            args.plot_dir,
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
                beta=args.initial_beta,
                final_beta=args.final_beta,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
            )
            metrics["PPO + count"]["rewards"].append(
                float(np.mean(rewards_ppo_count))
            )
            metrics["PPO + count"]["success"].append(
                float(sum(success_count)) / len(success_count)
                if success_count
                else 0.0
            )
            metrics["PPO + count"]["planner_pct"].append(
                float(np.mean(planner_rate_count)))
            metrics["PPO + count"]["spikes"].append(
                count_intrinsic_spikes(intrinsic_count))
            save_model(
                ppo_count_policy,
                os.path.join(
                    "checkpoints",
                    f"ppo_count_{run_seed}.pt"))
            curve_logs["PPO + count"]["rewards"].append(rewards_ppo_count)
            curve_logs["PPO + count"]["success"].append(success_count)
            render_episode_video(
                env, ppo_count_policy, os.path.join(
                    "videos", f"ppo_count_{run_seed}.gif"))
            mean_b, std_b = evaluate_on_benchmarks(
                env, ppo_count_policy, "test_maps", 5)
            bench["PPO + count"].append(mean_b)

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
                    beta=args.initial_beta,
                    final_beta=args.final_beta,
                    planner_weights=planner_weights,
                    seed=run_seed,
                    add_noise=args.add_noise,
                    logger=logger,
                )
                metrics["PPO + RND"]["rewards"].append(
                    float(np.mean(rewards_ppo_rnd)))
                metrics["PPO + RND"]["success"].append(
                    float(sum(success_rnd)) / len(success_rnd)
                    if success_rnd
                    else 0.0
                )
                metrics["PPO + RND"]["planner_pct"].append(
                    float(np.mean(planner_rate_rnd)))
                metrics["PPO + RND"]["spikes"].append(
                    count_intrinsic_spikes(intrinsic_rnd))
                save_model(
                    ppo_rnd_policy,
                    os.path.join("checkpoints", f"ppo_rnd_{run_seed}.pt"),
                    rnd=rnd,
                )
                curve_logs["PPO + RND"]["rewards"].append(rewards_ppo_rnd)
                curve_logs["PPO + RND"]["success"].append(success_rnd)
                render_episode_video(
                    env, ppo_rnd_policy, os.path.join(
                        "videos", f"ppo_rnd_{run_seed}.gif"))
                mean_b, std_b = evaluate_on_benchmarks(
                    env, ppo_rnd_policy, "test_maps", 5)
            bench["PPO + RND"].append(mean_b)

        # Plot aggregated curves across seeds for this setting
        for name, logs_dict in curve_logs.items():
            if logs_dict["rewards"]:
                out_file = None
                if args.plot_dir:
                    safe_setting = setting["name"].replace(" ", "_")
                    safe_name = name.replace(" ", "_").replace("+", "")
                    out_file = os.path.join(
                        args.plot_dir, f"{safe_setting}_{safe_name}.pdf"
                    )
                plot_training_curves(
                    logs_dict["rewards"],
                    logs_dict["intrinsic"] if logs_dict["intrinsic"] else None,
                    logs_dict["success"],
                    output_path=out_file,
                )

        # Aggregate metrics across seeds for this setting
        baseline_rewards = np.array(metrics["PPO Only"]["rewards"])
        baseline_success = np.array(metrics["PPO Only"]["success"])

        anova_reward_p = np.nan
        anova_success_p = np.nan
        if args.stat_test == "anova":
            reward_groups = []
            success_groups = []
            for data in metrics.values():
                if len(data["rewards"]) == len(
                        baseline_rewards) and data["rewards"]:
                    reward_groups.append(data["rewards"])
                    success_groups.append(data["success"])
            if len(reward_groups) >= 3:
                anova_reward_p = f_oneway(*reward_groups).pvalue
                anova_success_p = f_oneway(*success_groups).pvalue

        results = []
        for name, data in metrics.items():
            if args.stat_test == "anova":
                p_reward = anova_reward_p
                p_success = anova_success_p
            elif name == "PPO Only" or len(baseline_rewards) != len(
                data["rewards"]
            ):
                p_reward = np.nan
                p_success = np.nan
            elif args.stat_test == "paired":
                p_reward = ttest_rel(baseline_rewards, data["rewards"]).pvalue
                p_success = ttest_rel(baseline_success, data["success"]).pvalue
            elif args.stat_test == "welch":
                p_reward = ttest_ind(
                    baseline_rewards,
                    data["rewards"],
                    equal_var=False).pvalue
                p_success = ttest_ind(
                    baseline_success,
                    data["success"],
                    equal_var=False).pvalue
            else:  # mannwhitney
                p_reward = mannwhitneyu(
                    baseline_rewards,
                    data["rewards"],
                    alternative="two-sided").pvalue
                p_success = mannwhitneyu(
                    baseline_success,
                    data["success"],
                    alternative="two-sided").pvalue

            results.append(
                {
                    "Setting": setting["name"],
                    "Model": name,
                    "Train Reward Mean": float(np.mean(data["rewards"])),
                    "Train Reward Std": float(np.std(data["rewards"])),
                    "Success Mean": float(np.mean(data["success"])),
                    "Success Std": float(np.std(data["success"])),
                    "Planner Usage %": float(
                        np.mean(data["planner_pct"])
                    ) * 100,
                    "Intrinsic Spikes": (
                        float(np.mean(data["spikes"]))
                        if data["spikes"]
                        else 0.0
                    ),
                    "Reward p-value": p_reward,
                    "Success p-value": p_success,
                }
            )

        bench_results = []
        for name, vals in bench.items():
            if vals:
                bench_results.append(
                    {
                        "Setting": setting["name"],
                        "Model": name,
                        "Benchmark Reward": float(np.mean(vals)),
                        "Benchmark Std": float(np.std(vals)),
                    }
                )

        all_results.extend(results)
        all_bench.extend(bench_results)

    df_train = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    generate_results_table(
        df_train, os.path.join("results", "training_results.html")
    )
    if all_bench:
        df_bench = pd.DataFrame(all_bench)
        generate_results_table(
            df_bench, os.path.join("results", "benchmark_results.html")
        )

    if args.log_backend == "tensorboard" and logger is not None:
        logger.close()
    elif args.log_backend == "wandb" and logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
