import numpy as np
from src.env import GridWorldICM


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
    _, reward, _, _, _ = env.step(3)
    assert env.agent_pos == [0, 1]
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
