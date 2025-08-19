import pandas as pd
from src.statistics import compare_to_ppo, bootstrap_tables


def build_data():
    return {
        "reward": {
            "PPO": [1.0, 1.2, 1.1, 0.9],
            "MethodA": [2.0, 2.1, 1.8, 2.2],
            "MethodB": [3.0, 2.9, 3.1, 3.2],
        }
    }


def test_compare_to_ppo_outputs(tmp_path):
    data = build_data()
    out_dir = tmp_path / "results" / "stats"
    df = compare_to_ppo(data, split="train", out_dir=str(out_dir))
    csv_path = out_dir / "comparisons_train.csv"
    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path)
    assert set(df_csv.columns) == {
        "metric",
        "method",
        "test",
        "p_raw",
        "p_holm",
        "effect_size",
        "n_pairs",
        "normality",
    }
    assert {"paired_t", "wilcoxon", "friedman", "welch_anova", "games_howell"}.issubset(
        set(df_csv["test"])
    )
    row = df_csv[(df_csv["method"] == "MethodA") & (df_csv["test"] == "paired_t")].iloc[0]
    assert row["p_raw"] < 0.05
    assert row["p_holm"] < 0.05


def test_bootstrap_tables(tmp_path):
    data = build_data()
    out_dir = tmp_path / "results" / "stats"
    tables = bootstrap_tables(data, split="train", n_boot=100, out_dir=str(out_dir))
    path = out_dir / "bootstrap_reward_train.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert set(df.columns) == {"model", "mean", "ci"}
    assert "PPO" in df["model"].values
