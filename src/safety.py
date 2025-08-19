import os
from typing import Dict, List

import numpy as np
import pandas as pd


def _mean_ci95(values: List[float]):
    """Return mean and 95% CI bounds for ``values``.

    The confidence interval uses a normal approximation and returns
    ``(mean, lower, upper)``. For fewer than two samples the CI collapses
    to the mean value.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    sem = arr.std(ddof=1) / np.sqrt(arr.size)
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def save_pareto_summaries(
    method_metrics: Dict[str, Dict[str, List[float]]],
    split: str,
    out_dir: str = "results/safety",
) -> pd.DataFrame:
    """Write Pareto reward/cost summary for each method.

    Parameters
    ----------
    method_metrics: mapping from method name to dictionaries containing
        ``"rewards"`` and ``"costs"`` lists.
    split: dataset split name used in the output filename.
    out_dir: directory where the CSV is saved.
    """
    rows = []
    for method, data in method_metrics.items():
        rewards = data.get("rewards", [])
        costs = data.get("costs", [])
        if not rewards or not costs:
            continue
        r_mean, r_lo, r_hi = _mean_ci95(rewards)
        c_mean, c_lo, c_hi = _mean_ci95(costs)
        rows.append(
            {
                "method": method,
                "reward_mean": r_mean,
                "reward_ci95_lo": r_lo,
                "reward_ci95_hi": r_hi,
                "cost_mean": c_mean,
                "cost_ci95_lo": c_lo,
                "cost_ci95_hi": c_hi,
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pareto_{split}.csv")
    df.to_csv(path, index=False)
    return df


def save_violation_curves(
    curve_logs: Dict[str, Dict[str, List[List[float]]]],
    out_dir: str = "results/safety",
) -> None:
    """Compute and write running violation curves for each method.

    ``curve_logs`` should map method names to a dictionary containing a
    ``"violation_flags"`` key with one list of episode-wise flags per seed.
    """
    os.makedirs(out_dir, exist_ok=True)
    for method, logs_dict in curve_logs.items():
        flags = logs_dict.get("violation_flags", [])
        if not flags:
            continue
        rates = [np.cumsum(seed) / (np.arange(len(seed)) + 1) for seed in flags]
        min_len = min(len(r) for r in rates)
        arr = np.stack([r[:min_len] for r in rates])
        mean = arr.mean(axis=0)
        if arr.shape[0] > 1:
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ci = 1.96 * sem
        else:
            ci = np.zeros_like(mean)
        df = pd.DataFrame(
            {
                "episode": np.arange(1, min_len + 1),
                "violation_mean": mean,
                "violation_ci95_lo": mean - ci,
                "violation_ci95_hi": mean + ci,
            }
        )
        safe_method = method.replace(" ", "_").replace("+", "_")
        df.to_csv(
            os.path.join(out_dir, f"violations_over_training_{safe_method}.csv"),
            index=False,
        )


def append_budget_sweep(
    method_metrics: Dict[str, Dict[str, List[float]]],
    budget: float,
    out_dir: str = "results/safety",
) -> pd.DataFrame:
    """Append reward/cost/violation summaries for a given budget.

    The resulting CSV contains one row per method with the provided budget
    value. Existing data is preserved and new rows are appended.
    """
    rows = []
    for method, data in method_metrics.items():
        rewards = data.get("rewards", [])
        costs = data.get("costs", [])
        violations = data.get("violations", [])
        if not rewards or not costs:
            continue
        r_mean, r_lo, r_hi = _mean_ci95(rewards)
        c_mean, c_lo, c_hi = _mean_ci95(costs)
        v_mean, v_lo, v_hi = (
            _mean_ci95(violations) if violations else (0.0, 0.0, 0.0)
        )
        rows.append(
            {
                "method": method,
                "budget": budget,
                "reward_mean": r_mean,
                "reward_ci95_lo": r_lo,
                "reward_ci95_hi": r_hi,
                "cost_mean": c_mean,
                "cost_ci95_lo": c_lo,
                "cost_ci95_hi": c_hi,
                "violation_mean": v_mean,
                "violation_ci95_lo": v_lo,
                "violation_ci95_hi": v_hi,
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "budget_sweep.csv")
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df
