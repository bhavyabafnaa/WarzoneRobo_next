import os
import os
from torch import optim

from src.env import GridWorldICM
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent
import yaml
import numpy as np


def test_short_training_loop(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
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
        eta_lambda=0.05,
        cost_limit=0.5,
        c1=1.0,
        c2=0.5,
        c3=0.01,

    )


def test_training_one_episode_metrics(tmp_path):
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = GridWorldICM(grid_size=cfg.get("grid_size", 4), max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    metrics = train_agent(
        env,
        policy,
        icm,
        planner,
        opt,
        opt,
        use_icm=False,
        use_planner=False,
        num_episodes=1,
        eta_lambda=cfg.get("eta_lambda", 0.01),
        cost_limit=cfg.get("cost_limit", 1.0),
        c1=cfg.get("c1", 1.0),
        c2=cfg.get("c2", 0.5),
        c3=cfg.get("c3", 0.01),
    )

    rewards, _, _, _, _, _, success_flags, _ = metrics
    assert len(rewards) == 1


def test_success_flag_survival(tmp_path):
    env = GridWorldICM(grid_size=2, max_steps=2)
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.enemy_positions = []
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    metrics = train_agent(
        env,
        policy,
        icm,
        planner,
        opt,
        opt,
        use_icm=False,
        use_planner=False,
        num_episodes=1,
        reset_env=False,
        eta_lambda=0.05,
        cost_limit=0.5,
        c1=1.0,
        c2=0.5,
        c3=0.01,
    )

    _, _, _, _, _, _, success_flags, _ = metrics
    assert success_flags == [1]
