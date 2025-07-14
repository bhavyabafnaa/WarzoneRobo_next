import numpy as np
from src.planner import SymbolicPlanner


def test_get_safe_subgoal_prefers_low_risk():
    cost = np.zeros((3, 3))
    risk = np.zeros((3, 3))
    risk[1, 0] = 1.0  # down from start is risky
    planner = SymbolicPlanner(cost, risk, np_random=np.random.RandomState(0))
    action = planner.get_safe_subgoal((0, 0))
    assert action in {0, 1, 2, 3}
    # best action should be right (3) as going down has risk
    assert action == 3


def test_get_safe_subgoal_prefers_low_cost_when_risk_equal():
    cost = np.zeros((3, 3))
    risk = np.zeros((3, 3))
    cost[0, 1] = 0.8  # right from start has higher cost
    cost[1, 0] = 0.2  # down from start has lower cost
    planner = SymbolicPlanner(cost, risk, np_random=np.random.RandomState(1))
    action = planner.get_safe_subgoal((0, 0))
    assert action in {0, 1, 2, 3}
    # best action should be down (1) due to lower cost
    assert action == 1


def test_planner_reset_clears_visited():
    cost = np.zeros((2, 2))
    risk = np.zeros((2, 2))
    planner = SymbolicPlanner(cost, risk, np_random=np.random.RandomState(2))
    planner.get_safe_subgoal((0, 0))
    assert planner.visited_map[0, 0]
    planner.reset()
    assert not planner.visited_map.any()
