import numpy as np
from src.env import (
    GridWorldICM,
    export_benchmark_maps,
    evaluate_on_benchmarks,
)


def test_step_boundaries_and_rewards():
    env = GridWorldICM(grid_size=3, max_steps=10)
    env.reset()
    env.cost_map = np.zeros((3, 3))
    env.risk_map = np.zeros((3, 3))
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


def test_dynamic_risk_updates_near_enemies():
    env = GridWorldICM(grid_size=3, dynamic_risk=True, max_steps=5, seed=0)
    env.reset(seed=0)
    env.risk_map = np.zeros((3, 3))
    env.enemy_positions = [[1, 1]]
    env.step(0)
    ex, ey = env.enemy_positions[0]
    assert env.risk_map[ex, ey] > 0


def test_dynamic_cost_updates_near_mines_and_decay():
    env = GridWorldICM(grid_size=3, dynamic_cost=True, max_steps=5, seed=0)
    env.reset(seed=0)
    env.cost_map = np.zeros((3, 3))
    env.cost_map[0, 0] = 1.0
    env.mine_map = np.zeros((3, 3), dtype=bool)
    env.mine_map[1, 1] = True
    env.step(1)  # move down, avoiding the mine
    # cost at previous high value should decay
    assert env.cost_map[0, 0] < 1.0
    # cost near the mine should increase
    assert env.cost_map[1, 1] > 0


def test_survival_reward_positive():
    env = GridWorldICM(grid_size=2, max_steps=5, survival_reward=0.05)
    env.reset()
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
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
