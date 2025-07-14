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
