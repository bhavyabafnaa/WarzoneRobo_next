import numpy as np
from src.env import (
    GridWorldICM,
    export_benchmark_maps,
    evaluate_on_benchmarks,
)


def test_step_boundaries_and_rewards():
    env = GridWorldICM(grid_size=3, max_steps=10)
    env.reset()
    # starting position should lie within the grid
    assert 0 <= env.agent_pos[0] < env.grid_size
    assert 0 <= env.agent_pos[1] < env.grid_size
    # set deterministic maps and agent position
    env.agent_pos = [0, 0]
    env.cost_map = np.zeros((3, 3))
    env.risk_map = np.zeros((3, 3))
    env.mine_map = np.zeros((3, 3), dtype=bool)
    env.cost_map[0, 1] = 0.6
    env.risk_map[0, 1] = 0.4

    # attempt move outside grid
    env.step(0)
    assert env.agent_pos == [0, 0]

    # move right into cost/risk cell
    step_out = env.step(3)
    # handle old vs new step return signatures
    if len(step_out) == 6:
        _, reward, cost, *_ = step_out
        total_cost = env.episode_cost
    else:
        _, reward, cost, total_cost, *_ = step_out
    assert env.agent_pos == [0, 1]
    assert np.isclose(cost, 0.6)
    assert np.isclose(total_cost, 0.6)
    assert np.isclose(env.episode_cost, 0.6)
    assert reward <= -0.7


def test_randomized_mine_and_hazard_density():
    env = GridWorldICM(
        grid_size=5,
        mine_density_range=(0.03, 0.1),
        hazard_density_range=(0.1, 0.2),
    )
    env.reset(seed=0)
    assert 0.03 <= env.mine_density <= 0.1
    assert 0.1 <= env.hazard_density <= 0.2
    hazard_fraction = (env.risk_map > 0).mean()
    assert abs(hazard_fraction - env.hazard_density) <= 0.15


def test_dynamic_risk_alters_map_over_time():
    env_dyn = GridWorldICM(grid_size=3, dynamic_risk=True, max_steps=5, seed=0)
    env_dyn.reset(seed=0)
    env_dyn.risk_map = np.zeros((3, 3))
    env_dyn.enemy_positions = [[1, 1]]
    start = env_dyn.risk_map.copy()
    env_dyn.step(0)
    env_dyn.step(0)
    assert not np.array_equal(env_dyn.risk_map, start)

    env_static = GridWorldICM(grid_size=3, dynamic_risk=False, max_steps=5, seed=0)
    env_static.reset(seed=0)
    env_static.risk_map = np.zeros((3, 3))
    env_static.enemy_positions = [[1, 1]]
    start_static = env_static.risk_map.copy()
    env_static.step(0)
    env_static.step(0)
    assert np.array_equal(env_static.risk_map, start_static)


def test_dynamic_cost_alters_map_over_time():
    env_dyn = GridWorldICM(grid_size=3, dynamic_cost=True, max_steps=5, seed=0)
    env_dyn.reset(seed=0)
    env_dyn.cost_map = np.zeros((3, 3))
    env_dyn.cost_map[0, 0] = 1.0
    env_dyn.mine_map = np.zeros((3, 3), dtype=bool)
    env_dyn.mine_map[1, 1] = True
    start = env_dyn.cost_map.copy()
    env_dyn.step(1)
    env_dyn.step(1)
    assert not np.array_equal(env_dyn.cost_map, start)

    env_static = GridWorldICM(grid_size=3, dynamic_cost=False, max_steps=5, seed=0)
    env_static.reset(seed=0)
    env_static.cost_map = np.zeros((3, 3))
    env_static.cost_map[0, 0] = 1.0
    env_static.mine_map = np.zeros((3, 3), dtype=bool)
    env_static.mine_map[1, 1] = True
    start_static = env_static.cost_map.copy()
    env_static.step(1)
    env_static.step(1)
    assert np.array_equal(env_static.cost_map, start_static)


def test_survival_reward_positive():
    env = GridWorldICM(grid_size=2, max_steps=5, survival_reward=0.05)
    env.reset()
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.terrain_map = np.full((2, 2), "normal")
    env.enemy_positions = []
    _, reward, *_ = env.step(1)
    assert reward > 0


def test_export_benchmark_maps_counts(tmp_path):
    env = GridWorldICM(grid_size=4)
    export_benchmark_maps(
        env,
        train_folder=str(tmp_path / "train"),
        test_folder=str(tmp_path / "test"),
        ood_folder=str(tmp_path / "ood"),
    )
    assert len(list((tmp_path / "train").glob("*.npz"))) == 20
    assert len(list((tmp_path / "test").glob("*.npz"))) == 10
    assert len(list((tmp_path / "ood").glob("*.npz"))) == 10


def test_evaluate_on_benchmarks_with_ood(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=5)
    export_benchmark_maps(
        env,
        num_train=1,
        num_test=0,
        num_ood=1,
        train_folder=str(tmp_path / "train"),
        test_folder=str(tmp_path / "test"),
        ood_folder=str(tmp_path / "ood"),
    )

    class DummyPolicy:
        def act(self, state):
            return 0, None, None, None

    id_res, ood_res = evaluate_on_benchmarks(
        env,
        DummyPolicy(),
        map_folder=str(tmp_path / "train"),
        num_maps=1,
        H=1,
        ood_map_folder=str(tmp_path / "ood"),
        num_ood_maps=1,
    )
    assert isinstance(id_res, tuple) and isinstance(ood_res, tuple)
