import numpy as np
from src.utils import count_intrinsic_spikes


def test_count_intrinsic_spikes_basic():
    values = [0, 1, 5, 2, 0, 7]
    # mean=2.5, spikes > 3.75 -> 5 and 7 -> 2 spikes
    assert count_intrinsic_spikes(values, threshold_factor=1.5) == 2


def test_count_intrinsic_spikes_empty():
    assert count_intrinsic_spikes([]) == 0

