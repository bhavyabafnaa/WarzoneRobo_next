import os
import argparse
import yaml
import torch
import torch.optim as optim

from src.env import (
    GridWorldICM,
    export_benchmark_maps,
    visualize_paths_on_benchmark_maps,
    plot_model_performance,
)
from src.visualization import plot_training_curves, plot_heatmap_with_path
from src.icm import ICMModule
from src.rnd import RNDModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent
from src.utils import save_model, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate PPO agents")
    parser.add_argument("--config", type=str, help="Path to YAML config file", default=None)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--cost_weight", type=float, default=2.0)
    parser.add_argument("--risk_weight", type=float, default=3.0)
    parser.add_argument("--goal_weight", type=float, default=0.5)
    parser.add_argument("--revisit_penalty", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    grid_size = args.grid_size
    input_dim = 5 * grid_size * grid_size
    action_dim = 4

    env = GridWorldICM(grid_size=grid_size)
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

    export_benchmark_maps(env, num_maps=10, folder="maps/")
    export_benchmark_maps(env, num_maps=5, folder="test_maps/")

    policy_demo = PPOPolicy(input_dim, action_dim)
    visualize_paths_on_benchmark_maps(env, policy_demo, map_folder="maps/", num_maps=9)

    planner_weights = {
        "cost_weight": args.cost_weight,
        "risk_weight": args.risk_weight,
        "goal_weight": args.goal_weight,
        "revisit_penalty": args.revisit_penalty,
    }

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
    )
    save_model(ppo_policy, os.path.join("checkpoints", "ppo_only.pt"))
    plot_training_curves(rewards_ppo_only, None, success_ppo_only)

    # PPO + ICM
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
    )
    save_model(ppo_icm_policy, os.path.join("checkpoints", "ppo_icm.pt"), icm=icm)
    plot_training_curves(rewards_ppo_icm, intrinsic_icm, success_icm)

    # PPO + ICM + Planner
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
    )
    save_model(
        ppo_icm_planner_policy,
        os.path.join("checkpoints", "ppo_icm_planner.pt"),
        icm=icm,
    )
    plot_training_curves(rewards_ppo_icm_plan, intrinsic_plan, success_plan)
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
    )
    save_model(ppo_count_policy, os.path.join("checkpoints", "ppo_count.pt"))
    plot_training_curves(rewards_ppo_count, None, success_count)

    # RND exploration
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
    )
    save_model(
        ppo_rnd_policy,
        os.path.join("checkpoints", "ppo_rnd.pt"),
        rnd=rnd,
    )
    plot_training_curves(rewards_ppo_rnd, None, success_rnd)

    # Load a saved model for evaluation on benchmark maps
    eval_policy, _, _ = load_model(
        PPOPolicy,
        input_dim,
        action_dim,
        os.path.join("checkpoints", "ppo_icm_planner.pt"),
        icm_class=ICMModule,
    )
    visualize_paths_on_benchmark_maps(env, eval_policy, map_folder="maps/", num_maps=3)

    models = [ppo_policy, ppo_icm_policy, ppo_icm_planner_policy, ppo_count_policy, ppo_rnd_policy]
    model_names = ['PPO Only', 'PPO + ICM', 'PPO + ICM + Planner', 'PPO + count', 'PPO + RND']
    plot_model_performance(models, model_names, env)


if __name__ == "__main__":
    main()
