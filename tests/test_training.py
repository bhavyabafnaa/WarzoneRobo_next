import os
import torch
from torch import optim

from src.env import GridWorldICM
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent


def test_short_training_loop(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=10)
    os.makedirs("train_maps", exist_ok=True)
    env.save_map("train_maps/map_00.npz")

    input_dim = 5 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.goal_pos, env.np_random)
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    train_agent(
        env,
        policy,
        icm,
        planner,
        opt,
        opt,
        use_icm=False,
        use_planner=False,
        num_episodes=1,
    )
