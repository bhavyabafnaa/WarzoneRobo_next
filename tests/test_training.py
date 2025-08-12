import os
from torch import optim

import pytest
from src.env import GridWorldICM
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent
import yaml
import numpy as np


@pytest.mark.parametrize("budget", [0.05, 0.10])
def test_short_training_loop(tmp_path, budget):
    env = GridWorldICM(grid_size=4, max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4
    runs = []
    for run_seed in range(10):
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
            cost_limit=budget,
            c1=1.0,
            c2=0.5,
            c3=0.01,
            seed=run_seed,
        )
        runs.append(True)
    assert len(runs) == 10


@pytest.mark.parametrize("budget", [0.05, 0.10])
def test_training_one_episode_metrics(tmp_path, budget):
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = GridWorldICM(grid_size=cfg.get("grid_size", 4), max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4
    metrics_list = []
    for run_seed in range(10):
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
            cost_limit=budget,
            c1=cfg.get("c1", 1.0),
            c2=cfg.get("c2", 0.5),
            c3=cfg.get("c3", 0.01),
            seed=run_seed,
        )
        metrics_list.append(metrics)

    assert len(metrics_list) == 10
    metrics = metrics_list[0]
    if len(metrics) == 19:
        (
            rewards,
            _,
            _,
            _,
            _,
            _,
            success_flags,
            _,
            _,
            _,
            _,
            coverage_log,
            _,
            episode_costs,
            violation_flags,
            first_violation_episode,
            episode_times,
            steps_per_sec,
            wall_clock,
        ) = metrics
        lambda_log = [0.0] * len(rewards)
    else:
        (
            rewards,
            _,
            _,
            _,
            _,
            _,
            success_flags,
            _,
            _,
            _,
            _,
            coverage_log,
            _,
            episode_costs,
            violation_flags,
            lambda_log,
            first_violation_episode,
            episode_times,
            steps_per_sec,
            wall_clock,
        ) = metrics
    assert len(rewards) == 1
    assert len(coverage_log) == 1
    assert len(episode_costs) == 1
    assert len(violation_flags) == 1
    assert len(episode_times) == 1
    assert len(steps_per_sec) == 1
    assert len(wall_clock) == 1
    assert len(lambda_log) == 1
    assert isinstance(first_violation_episode, int)


@pytest.mark.parametrize("budget", [0.05, 0.10])
def test_success_flag_survival(tmp_path, budget):
    env = GridWorldICM(grid_size=2, max_steps=2)
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.enemy_positions = []
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4
    for run_seed in range(10):
        env.reset(seed=run_seed, load_map_path="maps/map_00.npz")
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
            cost_limit=budget,
            c1=1.0,
            c2=0.5,
            c3=0.01,
        )

        _, _, _, _, _, _, success_flags, *_rest = metrics
        assert success_flags == [1]
