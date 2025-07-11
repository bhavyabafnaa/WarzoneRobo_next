import os
import argparse
import yaml
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

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
from src.utils import save_model


def parse_args():
    """Parse command line arguments.

    Any key found in a YAML file provided via ``--config`` will be mapped to an
    argument attribute. This allows specifying additional environment options
    such as ``dynamic_risk`` or ``add_noise`` purely in YAML:

    .. code-block:: yaml

       grid_size: 8
       dynamic_risk: true
       add_noise: false
    """

    parser = argparse.ArgumentParser(description="Train or evaluate PPO agents")
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
    parser.add_argument("--goal_weight", type=float, default=0.5)
    parser.add_argument("--revisit_penalty", type=float, default=1.0)
    parser.add_argument("--dynamic_risk", action="store_true", help="Enable dynamic risk in env")
    parser.add_argument("--add_noise", action="store_true", help="Add noise when resetting maps")
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

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
    input_dim = 5 * grid_size * grid_size
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
        seed=seeds[0],
    )
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(
        env.cost_map,
        env.risk_map,
        env.goal_pos,
        env.np_random,
        cost_weight=args.cost_weight,
        risk_weight=args.risk_weight,
        goal_weight=args.goal_weight,
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
        "goal_weight": args.goal_weight,
        "revisit_penalty": args.revisit_penalty,
    }

    settings = [
        {"name": "baseline", "icm": args.disable_icm, "rnd": args.disable_rnd, "planner": args.disable_planner}
    ]
    if args.ablation:
        settings = [
            {"name": "baseline", "icm": False, "rnd": False, "planner": False},
            {"name": "no_icm", "icm": True, "rnd": False, "planner": False},
            {"name": "no_rnd", "icm": False, "rnd": True, "planner": False},
            {"name": "no_planner", "icm": False, "rnd": False, "planner": True},
        ]

    all_results = []
    all_bench = []

    for setting in settings:
        args.disable_icm = setting["icm"]
        args.disable_rnd = setting["rnd"]
        args.disable_planner = setting["planner"]

        metrics = {
            "PPO Only": {"rewards": [], "success": []},
            "PPO + ICM": {"rewards": [], "success": []},
            "PPO + ICM + Planner": {"rewards": [], "success": []},
            "PPO + count": {"rewards": [], "success": []},
            "PPO + RND": {"rewards": [], "success": []},
            "PPO + PC": {"rewards": [], "success": []},
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
            "PPO + ICM + Planner": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + count": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + RND": {"rewards": [], "intrinsic": [], "success": []},
            "PPO + PC": {"rewards": [], "intrinsic": [], "success": []},
        }

        for run_seed in seeds:
            env.reset(seed=run_seed)
            icm = ICMModule(input_dim, action_dim)
            planner = SymbolicPlanner(
                env.cost_map,
                env.risk_map,
                env.goal_pos,
                env.np_random,
                cost_weight=args.cost_weight,
                risk_weight=args.risk_weight,
                goal_weight=args.goal_weight,
                revisit_penalty=args.revisit_penalty,
            )

        # PPO only
        ppo_policy = PPOPolicy(input_dim, action_dim)
        opt_ppo = optim.Adam(ppo_policy.parameters(), lr=3e-4)
        rewards_ppo_only, _, _, _, paths_ppo_only, _, success_ppo_only, _ = train_agent(
            env,
            ppo_policy,
            icm,
            planner,
            opt_ppo,
            opt_ppo,
            use_icm=False,
            use_planner=False,
            num_episodes=args.num_episodes,
            planner_weights=planner_weights,
            seed=run_seed,
            add_noise=args.add_noise,
            logger=logger,
        )
        metrics["PPO Only"]["rewards"].append(float(np.mean(rewards_ppo_only)))
        metrics["PPO Only"]["success"].append(
            float(sum(success_ppo_only)) / len(success_ppo_only) if success_ppo_only else 0.0
        )
        save_model(ppo_policy, os.path.join("checkpoints", f"ppo_only_{run_seed}.pt"))
        curve_logs["PPO Only"]["rewards"].append(rewards_ppo_only)
        curve_logs["PPO Only"]["success"].append(success_ppo_only)
        render_episode_video(env, ppo_policy, os.path.join("videos", f"ppo_only_{run_seed}.gif"))
        mean_b, std_b = evaluate_on_benchmarks(env, ppo_policy, "test_maps", 5)
        bench["PPO Only"].append(mean_b)

        # PPO + ICM
        if not args.disable_icm:
            ppo_icm_policy = PPOPolicy(input_dim, action_dim)
            opt_icm_policy = optim.Adam(ppo_icm_policy.parameters(), lr=3e-4)
            rewards_ppo_icm, intrinsic_icm, _, _, paths_icm, _, success_icm, _ = train_agent(
            env,
            ppo_icm_policy,
            icm,
            planner,
            opt_icm_policy,
            opt_icm_policy,
            use_icm=True,
            use_planner=False,
            num_episodes=args.num_episodes,
            planner_weights=planner_weights,
            seed=run_seed,
            add_noise=args.add_noise,
            logger=logger,
        )
            metrics["PPO + ICM"]["rewards"].append(float(np.mean(rewards_ppo_icm)))
            metrics["PPO + ICM"]["success"].append(
                float(sum(success_icm)) / len(success_icm) if success_icm else 0.0
            )
            save_model(ppo_icm_policy, os.path.join("checkpoints", f"ppo_icm_{run_seed}.pt"), icm=icm)
            curve_logs["PPO + ICM"]["rewards"].append(rewards_ppo_icm)
            curve_logs["PPO + ICM"]["intrinsic"].append(intrinsic_icm)
            curve_logs["PPO + ICM"]["success"].append(success_icm)
            render_episode_video(env, ppo_icm_policy, os.path.join("videos", f"ppo_icm_{run_seed}.gif"))
            mean_b, std_b = evaluate_on_benchmarks(env, ppo_icm_policy, "test_maps", 5)
            bench["PPO + ICM"].append(mean_b)

        # PPO + Pseudo-count exploration
        ppo_pc_policy = PPOPolicy(input_dim, action_dim)
        opt_pc_policy = optim.Adam(ppo_pc_policy.parameters(), lr=3e-4)
        pseudo = PseudoCountExploration()
        rewards_pc, _, _, _, paths_pc, _, success_pc, _ = train_agent(
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
            planner_weights=planner_weights,
            seed=run_seed,
            add_noise=args.add_noise,
            logger=logger,
        )
        metrics["PPO + PC"]["rewards"].append(float(np.mean(rewards_pc)))
        metrics["PPO + PC"]["success"].append(
            float(sum(success_pc)) / len(success_pc) if success_pc else 0.0
        )
        save_model(ppo_pc_policy, os.path.join("checkpoints", f"ppo_pc_{run_seed}.pt"))
        curve_logs["PPO + PC"]["rewards"].append(rewards_pc)
        curve_logs["PPO + PC"]["success"].append(success_pc)
        render_episode_video(env, ppo_pc_policy, os.path.join("videos", f"ppo_pc_{run_seed}.gif"))
        mean_b, std_b = evaluate_on_benchmarks(env, ppo_pc_policy, "test_maps", 5)
        bench["PPO + PC"].append(mean_b)

        # PPO + ICM + Planner
        if not args.disable_icm and not args.disable_planner:
            ppo_icm_planner_policy = PPOPolicy(input_dim, action_dim)
            opt_plan_policy = optim.Adam(ppo_icm_planner_policy.parameters(), lr=3e-4)
            rewards_ppo_icm_plan, intrinsic_plan, _, _, paths_plan, _, success_plan, _ = train_agent(
                env,
                ppo_icm_planner_policy,
                icm,
                planner,
                opt_plan_policy,
                opt_plan_policy,
                use_icm=True,
                use_planner=True,
                num_episodes=args.num_episodes,
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
            )
            metrics["PPO + ICM + Planner"]["rewards"].append(float(np.mean(rewards_ppo_icm_plan)))
            metrics["PPO + ICM + Planner"]["success"].append(
                float(sum(success_plan)) / len(success_plan) if success_plan else 0.0
            )
            save_model(
                ppo_icm_planner_policy,
                os.path.join("checkpoints", f"ppo_icm_planner_{run_seed}.pt"),
                icm=icm,
            )
            curve_logs["PPO + ICM + Planner"]["rewards"].append(rewards_ppo_icm_plan)
            curve_logs["PPO + ICM + Planner"]["intrinsic"].append(intrinsic_plan)
            curve_logs["PPO + ICM + Planner"]["success"].append(success_plan)
            render_episode_video(
                env,
                ppo_icm_planner_policy,
                os.path.join("videos", f"ppo_icm_planner_{run_seed}.gif"),
            )
            mean_b, std_b = evaluate_on_benchmarks(env, ppo_icm_planner_policy, "test_maps", 5)
            bench["PPO + ICM + Planner"].append(mean_b)
            if paths_plan:
                plot_heatmap_with_path(env, paths_plan[-1])

        # Count-based exploration
        ppo_count_policy = PPOPolicy(input_dim, action_dim)
        opt_count_policy = optim.Adam(ppo_count_policy.parameters(), lr=3e-4)
        rewards_ppo_count, _, _, _, paths_count, _, success_count, _ = train_agent(
            env,
            ppo_count_policy,
            icm,
            planner,
            opt_count_policy,
            opt_count_policy,
            use_icm="count",
            use_planner=False,
            num_episodes=args.num_episodes,
            planner_weights=planner_weights,
            seed=run_seed,
            add_noise=args.add_noise,
            logger=logger,
        )
        metrics["PPO + count"]["rewards"].append(float(np.mean(rewards_ppo_count)))
        metrics["PPO + count"]["success"].append(
            float(sum(success_count)) / len(success_count) if success_count else 0.0
        )
        save_model(ppo_count_policy, os.path.join("checkpoints", f"ppo_count_{run_seed}.pt"))
        curve_logs["PPO + count"]["rewards"].append(rewards_ppo_count)
        curve_logs["PPO + count"]["success"].append(success_count)
        render_episode_video(env, ppo_count_policy, os.path.join("videos", f"ppo_count_{run_seed}.gif"))
        mean_b, std_b = evaluate_on_benchmarks(env, ppo_count_policy, "test_maps", 5)
        bench["PPO + count"].append(mean_b)

        # RND exploration
        if not args.disable_rnd:
            ppo_rnd_policy = PPOPolicy(input_dim, action_dim)
            opt_rnd_policy = optim.Adam(ppo_rnd_policy.parameters(), lr=3e-4)
            rnd = RNDModule(input_dim)
            opt_rnd = optim.Adam(rnd.predictor.parameters(), lr=1e-3)
            rewards_ppo_rnd, _, _, _, paths_rnd, _, success_rnd, _ = train_agent(
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
                planner_weights=planner_weights,
                seed=run_seed,
                add_noise=args.add_noise,
                logger=logger,
            )
            metrics["PPO + RND"]["rewards"].append(float(np.mean(rewards_ppo_rnd)))
            metrics["PPO + RND"]["success"].append(
                float(sum(success_rnd)) / len(success_rnd) if success_rnd else 0.0
            )
            save_model(
                ppo_rnd_policy,
                os.path.join("checkpoints", f"ppo_rnd_{run_seed}.pt"),
                rnd=rnd,
            )
            curve_logs["PPO + RND"]["rewards"].append(rewards_ppo_rnd)
            curve_logs["PPO + RND"]["success"].append(success_rnd)
            render_episode_video(env, ppo_rnd_policy, os.path.join("videos", f"ppo_rnd_{run_seed}.gif"))
            mean_b, std_b = evaluate_on_benchmarks(env, ppo_rnd_policy, "test_maps", 5)
        bench["PPO + RND"].append(mean_b)

        # Plot aggregated curves across seeds for this setting
        for name, logs_dict in curve_logs.items():
            if logs_dict["rewards"]:
                plot_training_curves(
                    logs_dict["rewards"],
                    logs_dict["intrinsic"] if logs_dict["intrinsic"] else None,
                    logs_dict["success"],
                )

        # Aggregate metrics across seeds for this setting
        baseline_rewards = np.array(metrics["PPO Only"]["rewards"])
        baseline_success = np.array(metrics["PPO Only"]["success"])

        results = []
        for name, data in metrics.items():
            # paired t-tests relative to PPO Only baseline
            if name == "PPO Only" or len(baseline_rewards) != len(data["rewards"]):
                p_reward = np.nan
                p_success = np.nan
            else:
                p_reward = ttest_rel(baseline_rewards, data["rewards"]).pvalue
                p_success = ttest_rel(baseline_success, data["success"]).pvalue

            results.append(
                {
                    "Setting": setting["name"],
                    "Model": name,
                    "Train Reward Mean": float(np.mean(data["rewards"])),
                    "Train Reward Std": float(np.std(data["rewards"])),
                    "Success Mean": float(np.mean(data["success"])),
                    "Success Std": float(np.std(data["success"])),
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
generate_results_table(df_train, os.path.join("results", "training_results.html"))
if all_bench:
    df_bench = pd.DataFrame(all_bench)
    generate_results_table(df_bench, os.path.join("results", "benchmark_results.html"))

if args.log_backend == "tensorboard" and logger is not None:
    logger.close()
elif args.log_backend == "wandb" and logger is not None:
    logger.finish()


if __name__ == "__main__":
    main()
