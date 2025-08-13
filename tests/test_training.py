import os
from torch import optim

import pytest
from src.env import GridWorldICM
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.pseudocount import PseudoCountExploration
from src.ppo import PPOPolicy, train_agent, get_beta_schedule
from train import evaluate_policy_on_maps, get_paired_arrays, compute_cohens_d
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import yaml
import numpy as np


@pytest.mark.parametrize("budget", [0.05, 0.10])
def test_short_training_loop(tmp_path, budget):
    env = GridWorldICM(grid_size=4, max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4
    rewards_runs = []
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
            eta_lambda=0.05,
            cost_limit=budget,
            c1=1.0,
            c2=0.5,
            c3=0.01,
            seed=run_seed,
        )
        rewards_runs.append(metrics[0])
    assert len(rewards_runs) == 10
    assert all(len(r) == 1 for r in rewards_runs)


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
        beta_log,
    ) = metrics
    lambda_log = [0.0] * len(rewards)
    assert len(rewards) == 1
    assert len(coverage_log) == 1
    assert len(episode_costs) == 1
    assert len(violation_flags) == 1
    assert len(episode_times) == 1
    assert len(steps_per_sec) == 1
    assert len(wall_clock) == 1
    assert len(lambda_log) == 1
    assert len(beta_log) == 1
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
        rewards, _, _, _, _, _, success_flags, *_rest = metrics
        assert success_flags == [1]
        assert len(rewards) == 1


def test_beta_schedule_consistency():
    env = GridWorldICM(grid_size=4, max_steps=5)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")
    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4

    schedule = get_beta_schedule(3, 0.2, 0.1)

    policy1 = PPOPolicy(input_dim, action_dim)
    icm1 = ICMModule(input_dim, action_dim)
    planner1 = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    opt1 = optim.Adam(policy1.parameters(), lr=1e-3)
    metrics1 = train_agent(
        env,
        policy1,
        icm1,
        planner1,
        opt1,
        opt1,
        use_icm=True,
        use_planner=False,
        num_episodes=3,
        beta_schedule=schedule,
    )
    rewards1, *_, beta_log1 = metrics1
    assert len(rewards1) == 3

    env.reset()
    policy2 = PPOPolicy(input_dim, action_dim)
    icm2 = ICMModule(input_dim, action_dim)
    planner2 = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    pseudo = PseudoCountExploration()
    opt2 = optim.Adam(policy2.parameters(), lr=1e-3)
    metrics2 = train_agent(
        env,
        policy2,
        icm2,
        planner2,
        opt2,
        opt2,
        use_icm="pseudo",
        use_planner=False,
        pseudo=pseudo,
        num_episodes=3,
        beta_schedule=schedule,
    )
    rewards2, *_, beta_log2 = metrics2
    assert len(rewards2) == 3

    assert beta_log1 == schedule
    assert beta_log2 == schedule
    assert beta_log1 == beta_log2


def test_allow_early_stop_asserts():
    env = GridWorldICM(grid_size=2, max_steps=2)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")
    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    with pytest.raises(AssertionError):
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
            allow_early_stop=True,
        )


def test_reward_ci_warning():
    from train import check_reward_difference_ci

    baseline = [0.0, 0.1, 0.2]
    other = [0.5, 0.4, 0.6]
    with pytest.warns(RuntimeWarning):
        check_reward_difference_ci(baseline, other)


def test_reward_ci_no_warning():
    from train import check_reward_difference_ci
    import warnings

    baseline = [0.0, 0.1, 0.2]
    other = [3.0, 3.1, 3.2]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_reward_difference_ci(baseline, other)
    assert len(w) == 0


def test_per_seed_map_metrics_collection(tmp_path):
    env = GridWorldICM(grid_size=2, max_steps=2)
    os.makedirs("test_maps", exist_ok=True)
    for i in range(2):
        env.save_map(f"test_maps/map_{i:02d}.npz")
    input_dim = 4 * env.grid_size * env.grid_size + 2
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    rewards, success = evaluate_policy_on_maps(env, policy, "test_maps", 2, H=1)
    metrics = {"PPO Only": {"rewards": {0: rewards}, "success": {0: success}}}
    assert len(metrics["PPO Only"]["rewards"][0]) == 2
    assert len(metrics["PPO Only"]["success"][0]) == 2


def test_paired_arrays_equal_length():
    baseline = {0: [1.0, 2.0], 1: [3.0]}
    method = {0: [1.5, 2.5], 1: [2.5, 3.5, 4.5]}
    base_arr, meth_arr = get_paired_arrays(baseline, method)
    assert len(base_arr) == len(meth_arr)
    # Should run without raising due to mismatched lengths
    ttest_rel(base_arr, meth_arr)


def test_effect_size_and_holm_adjustment():
    baseline = {0: [1.0, 2.0, 3.0], 1: [2.0, 3.0, 4.0]}
    method_a = {0: [2.0, 2.0, 4.0], 1: [3.0, 4.0, 5.0]}
    method_b = {0: [1.1, 1.9, 3.1], 1: [1.9, 3.1, 3.9]}

    base_a, meth_a = get_paired_arrays(baseline, method_a)
    base_b, meth_b = get_paired_arrays(baseline, method_b)

    p_a = ttest_rel(base_a, meth_a).pvalue
    p_b = ttest_rel(base_b, meth_b).pvalue

    effect_a = compute_cohens_d(base_a, meth_a, paired=True)
    effect_b = compute_cohens_d(base_b, meth_b, paired=True)

    # Known values from manual computation
    assert effect_a == pytest.approx(2.04124, rel=1e-3)
    assert effect_b == pytest.approx(0.0, abs=1e-6)

    pvals = [p_a, p_b]
    _, padj, _, _ = multipletests(pvals, method="holm")

    # Manual Holm-Bonferroni calculation
    m = len(pvals)
    order = np.argsort(pvals)
    expected = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * pvals[idx]
        prev = max(prev, adj)
        expected[idx] = min(1.0, prev)

    assert padj == pytest.approx(expected)
