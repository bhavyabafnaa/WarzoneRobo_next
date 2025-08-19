import os
import pandas as pd

from src.safety import save_pareto_summaries, save_violation_curves, append_budget_sweep


def test_save_pareto_summaries(tmp_path):
    metrics = {
        "MethodA": {"rewards": [1.0, 2.0, 3.0], "costs": [0.1, 0.2, 0.3]},
        "MethodB": {"rewards": [2.0, 2.5, 3.5], "costs": [0.2, 0.3, 0.4]},
    }
    save_pareto_summaries(metrics, "train", out_dir=tmp_path)
    df = pd.read_csv(tmp_path / "pareto_train.csv")
    assert list(df.columns) == [
        "method",
        "reward_mean",
        "reward_ci95_lo",
        "reward_ci95_hi",
        "cost_mean",
        "cost_ci95_lo",
        "cost_ci95_hi",
    ]
    assert len(df) == 2


def test_save_violation_curves(tmp_path):
    logs = {"MethodA": {"violation_flags": [[0, 1, 0], [1, 0, 0]]}}
    save_violation_curves(logs, out_dir=tmp_path)
    out_file = tmp_path / "violations_over_training_MethodA.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert list(df.columns) == [
        "episode",
        "violation_mean",
        "violation_ci95_lo",
        "violation_ci95_hi",
    ]
    assert len(df) == 3


def test_append_budget_sweep(tmp_path):
    metrics = {
        "MethodA": {
            "rewards": [1.0, 2.0],
            "costs": [0.1, 0.2],
            "violations": [0.0, 1.0],
        }
    }
    append_budget_sweep(metrics, 0.05, out_dir=tmp_path)
    df = pd.read_csv(tmp_path / "budget_sweep.csv")
    assert list(df.columns) == [
        "method",
        "budget",
        "reward_mean",
        "reward_ci95_lo",
        "reward_ci95_hi",
        "cost_mean",
        "cost_ci95_lo",
        "cost_ci95_hi",
        "violation_mean",
        "violation_ci95_lo",
        "violation_ci95_hi",
    ]
    assert len(df) == 1
