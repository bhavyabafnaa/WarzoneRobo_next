import torch
import torch.optim as optim

from src.env import GridWorldICM, export_benchmark_maps, visualize_paths_on_benchmark_maps, plot_model_performance
from src.icm import ICMModule
from src.rnd import RNDModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent


def main():
    grid_size = 12
    input_dim = 5 * grid_size * grid_size
    action_dim = 4

    env = GridWorldICM(grid_size=grid_size)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.goal_pos, env.np_random)

    export_benchmark_maps(env, num_maps=10, folder="maps/")
    export_benchmark_maps(env, num_maps=5, folder="test_maps/")

    policy_demo = PPOPolicy(input_dim, action_dim)
    visualize_paths_on_benchmark_maps(env, policy_demo, map_folder="maps/", num_maps=9)

    # PPO only
    ppo_policy = PPOPolicy(input_dim, action_dim)
    opt_ppo = optim.Adam(ppo_policy.parameters(), lr=3e-4)
    rewards_ppo_only, *_ = train_agent(env, ppo_policy, icm, planner, opt_ppo, opt_ppo,
                                       use_icm=False, use_planner=False)

    # PPO + ICM
    ppo_icm_policy = PPOPolicy(input_dim, action_dim)
    opt_icm_policy = optim.Adam(ppo_icm_policy.parameters(), lr=3e-4)
    rewards_ppo_icm, *_ = train_agent(env, ppo_icm_policy, icm, planner, opt_icm_policy, opt_icm_policy,
                                      use_icm=True, use_planner=False)

    # PPO + ICM + Planner
    ppo_icm_planner_policy = PPOPolicy(input_dim, action_dim)
    opt_plan_policy = optim.Adam(ppo_icm_planner_policy.parameters(), lr=3e-4)
    rewards_ppo_icm_plan, *_ = train_agent(env, ppo_icm_planner_policy, icm, planner, opt_plan_policy, opt_plan_policy,
                                          use_icm=True, use_planner=True)

    # Count-based exploration
    ppo_count_policy = PPOPolicy(input_dim, action_dim)
    opt_count_policy = optim.Adam(ppo_count_policy.parameters(), lr=3e-4)
    rewards_ppo_count, *_ = train_agent(env, ppo_count_policy, icm, planner, opt_count_policy, opt_count_policy,
                                        use_icm="count", use_planner=False)

    # RND exploration
    ppo_rnd_policy = PPOPolicy(input_dim, action_dim)
    opt_rnd_policy = optim.Adam(ppo_rnd_policy.parameters(), lr=3e-4)
    rnd = RNDModule(input_dim)
    opt_rnd = optim.Adam(rnd.predictor.parameters(), lr=1e-3)
    rewards_ppo_rnd, *_ = train_agent(env, ppo_rnd_policy, icm, planner, opt_rnd_policy, opt_rnd,
                                      use_icm="rnd", use_planner=False, rnd=rnd)

    models = [ppo_policy, ppo_icm_policy, ppo_icm_planner_policy, ppo_count_policy, ppo_rnd_policy]
    model_names = ['PPO Only', 'PPO + ICM', 'PPO + ICM + Planner', 'PPO + count', 'PPO + RND']
    plot_model_performance(models, model_names, env)


if __name__ == "__main__":
    main()
