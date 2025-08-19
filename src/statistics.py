import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
import pingouin as pg

from train import compute_cohens_d, bootstrap_ci


def compare_to_ppo(
    metric_data: Dict[str, Dict[str, List[float]]],
    split: str,
    out_dir: str = "results/stats",
) -> pd.DataFrame:
    """Compute statistical comparisons versus PPO for each metric.

    Parameters
    ----------
    metric_data : dict
        Mapping from metric name to a mapping of model name -> list of values.
        Must include a "PPO" entry as the baseline for each metric.
    split : str
        Data split label (e.g., "train" or "eval").
    out_dir : str
        Directory to save comparison table.

    Returns
    -------
    pandas.DataFrame
        Table of statistical comparisons.
    """

    os.makedirs(out_dir, exist_ok=True)
    rows: List[tuple] = []
    for metric, models in metric_data.items():
        if "PPO" not in models:
            raise KeyError(f"PPO baseline missing for metric '{metric}'")
        baseline = np.asarray(models["PPO"], dtype=float)
        methods = {m: np.asarray(v, dtype=float) for m, v in models.items() if m != "PPO"}
        n_pairs = len(baseline)

        # Pairwise tests
        t_rows: List[tuple] = []
        w_rows: List[tuple] = []
        gh_rows: List[tuple] = []
        t_pvals: List[float] = []
        w_pvals: List[float] = []
        gh_pvals: List[float] = []
        normality: Dict[str, str] = {}
        effects: Dict[str, float] = {}
        for method, arr in methods.items():
            if arr.shape != baseline.shape:
                raise ValueError("All methods must have same number of observations as PPO")
            t_stat, t_p = ttest_rel(baseline, arr)
            try:
                w_stat, w_p = wilcoxon(baseline, arr)
            except ValueError:
                w_p = np.nan
            diff = arr - baseline
            if diff.size >= 3:
                norm_p = shapiro(diff).pvalue
                norm_note = "normal" if norm_p >= 0.05 else "non-normal"
            else:
                norm_note = "n<3"
            effect = compute_cohens_d(baseline, arr, paired=True)
            t_rows.append((method, t_p, effect, n_pairs, norm_note))
            w_rows.append((method, w_p, effect, n_pairs, norm_note))
            t_pvals.append(t_p)
            w_pvals.append(w_p if not np.isnan(w_p) else 1.0)
            normality[method] = norm_note
            effects[method] = effect

        if t_rows:
            _, t_adj, _, _ = multipletests(t_pvals, method="holm")
            for (method, p, effect, n, note), p_adj in zip(t_rows, t_adj):
                rows.append((metric, method, "paired_t", p, p_adj, effect, n, note))
        if w_rows:
            _, w_adj, _, _ = multipletests(w_pvals, method="holm")
            for (method, p, effect, n, note), p_adj in zip(w_rows, w_adj):
                rows.append((metric, method, "wilcoxon", p, p_adj, effect, n, note))

        arrays = [baseline] + [methods[m] for m in methods]
        if len(arrays) >= 2:
            welch = anova_oneway(arrays, use_var="unequal")
            rows.append((metric, "All", "welch_anova", welch.pvalue, welch.pvalue, np.nan, n_pairs, ""))
        if len(arrays) >= 3:
            fried_stat, fried_p = friedmanchisquare(*arrays)
            rows.append((metric, "All", "friedman", fried_p, fried_p, np.nan, n_pairs, ""))
            df = pd.DataFrame(
                {
                    "score": np.concatenate(arrays),
                    "group": np.concatenate(
                        [[name] * len(vals) for name, vals in [("PPO", baseline)] + list(methods.items())]
                    ),
                }
            )
            gh = pg.pairwise_gameshowell(dv="score", between="group", data=df)
            for method in methods:
                row = gh.query("(A == 'PPO' & B == @method) | (B == 'PPO' & A == @method)")
                if not row.empty:
                    p = float(row["pval"])  # type: ignore[index]
                else:
                    p = np.nan
                gh_rows.append((method, p, effects[method], n_pairs, normality[method]))
                gh_pvals.append(p if not np.isnan(p) else 1.0)
            if gh_rows:
                _, gh_adj, _, _ = multipletests(gh_pvals, method="holm")
                for (method, p, effect, n, note), p_adj in zip(gh_rows, gh_adj):
                    rows.append((metric, method, "games_howell", p, p_adj, effect, n, note))

    df = pd.DataFrame(
        rows,
        columns=[
            "metric",
            "method",
            "test",
            "p_raw",
            "p_holm",
            "effect_size",
            "n_pairs",
            "normality",
        ],
    )
    df.to_csv(os.path.join(out_dir, f"comparisons_{split}.csv"), index=False)
    return df


def bootstrap_tables(
    metric_data: Dict[str, Dict[str, List[float]]],
    split: str,
    n_boot: int = 10_000,
    out_dir: str = "results/stats",
) -> Dict[str, pd.DataFrame]:
    """Generate bootstrap mean/CI tables for each metric."""

    os.makedirs(out_dir, exist_ok=True)
    tables: Dict[str, pd.DataFrame] = {}
    for metric, models in metric_data.items():
        rows = []
        for model, values in models.items():
            mean, ci = bootstrap_ci(values, n_resamples=n_boot)
            rows.append({"model": model, "mean": mean, "ci": ci})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f"bootstrap_{metric}_{split}.csv"), index=False)
        tables[metric] = df
    return tables
