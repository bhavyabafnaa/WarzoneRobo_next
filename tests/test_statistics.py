import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from scipy.stats import friedmanchisquare
from train import MAIN_METHODS, NAME_MAP, build_main_table, compute_cohens_d
from src.visualization import generate_results_table


def test_welch_anova_gameshowell():
    baseline = np.array([1.0, 1.1, 0.9, 1.2])
    method1 = np.array([2.0, 2.1, 1.9, 2.2])
    method2 = np.array([3.0, 3.1, 2.9, 3.2])
    res = anova_oneway([baseline, method1, method2], use_var="unequal")
    assert res.pvalue < 0.05
    df = pd.DataFrame(
        {
            "score": np.concatenate([baseline, method1, method2]),
            "group": ["A"] * len(baseline)
            + ["B"] * len(method1)
            + ["C"] * len(method2),
        }
    )
    gh = pg.pairwise_gameshowell(dv="score", between="group", data=df)
    pvals = []
    for _, row in gh.iterrows():
        if "A" in (row["A"], row["B"]):
            pvals.append(row["pval"])
    _, padj, _, _ = multipletests(pvals, method="holm")
    assert all(p < 0.05 for p in padj)


def test_friedman_statistic():
    baseline = [1, 2, 3, 4, 5]
    method1 = [2, 3, 4, 5, 6]
    method2 = [3, 4, 5, 6, 7]
    stat, p = friedmanchisquare(baseline, method1, method2)
    assert p < 0.05


def test_main_results_table(tmp_path):
    raw_names = [
        "PPO Only",
        "PPO + ICM",
        "PPO + RND",
        "PPO + count",
        "PPO + PC",
        "PPO + ICM + Planner",
        "LPPO",
        "Shielded-PPO",
        "Planner-only",
        "Planner-Subgoal PPO",
        "Dyna-PPO(1)",
    ]

    data = [
        {
            "Model": name,
            "Train Reward": "1.00 ± 0.00",
            "Success": "1.00 ± 0.00",
            "Train Cost": "1.00 ± 0.00",
            "Pr[Jc > d]": "0.00 ± 0.00",
            "Adherence Rate": "1.00 ± 0.00",
            "Mask Rate": "0.00 ± 0.00",
            "Coverage": "1.00 ± 0.00",
            "Reward p-value": 0.05,
            "Violation p-value": 0.10,
        }
        for name in raw_names
    ]

    df_train = pd.DataFrame(data)
    df_train["Model"] = df_train["Model"].replace(NAME_MAP)
    df_main = build_main_table(df_train)

    assert list(df_main.columns) == [
        "Model",
        "Reward (±CI)",
        "Success (±CI)",
        "Avg Cost (±CI)",
        "Violations % (±CI)",
        "Planner Adherence %",
        "Masked %",
        "Coverage",
        "p_reward",
        "p_violation",
    ]
    assert len(df_main) == len(MAIN_METHODS)

    output_path = tmp_path / "main_table.html"
    generate_results_table(df_main, str(output_path))
    assert output_path.exists()


def test_violation_effect_size_column(tmp_path):
    baseline = np.array([0, 1, 0, 1])
    method = np.array([1, 1, 1, 1])
    effect = compute_cohens_d(baseline, method)
    assert np.isscalar(effect)
    df = pd.DataFrame(
        [
            {"Model": "PPO Only", "Violation effect size": np.nan},
            {"Model": "Method", "Violation effect size": effect},
        ]
    )
    output_path = tmp_path / "viol_table.html"
    generate_results_table(df, str(output_path))
    assert output_path.exists()
    csv_path = tmp_path / "viol_table.csv"
    df_out = pd.read_csv(csv_path)
    assert "Violation effect size" in df_out.columns
    assert df_out["Violation effect size"].shape == (2,)
