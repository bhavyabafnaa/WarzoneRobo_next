import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train import build_main_table, NAME_MAP, MAIN_METHODS
from src.visualization import generate_results_table


def _format_ci(mean: float, std: float, n: int) -> str:
    """Return a ``mean ± 95% CI`` string computed from ``std`` and ``n``."""
    if n <= 1:
        ci = 0.0
    else:
        ci = 1.96 * std / np.sqrt(n)
    return f"{mean:.2f} ± {ci:.2f}"


def _prep_df(df: pd.DataFrame, mean_col: str, std_col: str, n: int) -> pd.DataFrame:
    """Prepare minimal dataframe compatible with ``build_main_table``."""
    out = pd.DataFrame()
    out["Model"] = df["Model"].replace(NAME_MAP)
    out["Train Reward"] = df.apply(lambda r: _format_ci(r[mean_col], r[std_col], n), axis=1)
    # Placeholder columns expected by ``build_main_table``
    out["Reward AUC"] = "0.00 ± 0.00"
    out["Success"] = "0.00 ± 0.00"
    out["Train Cost"] = "0.00 ± 0.00"
    out["Pr[Jc > d]"] = "0.00 ± 0.00"
    out["Planner Adherence %"] = "0.00 ± 0.00"
    out["Masked Action Rate"] = "0.00 ± 0.00"
    out["Unique Cells"] = "0.00 ± 0.00"
    out["Reward p-value"] = np.nan
    out["Violation p-value"] = np.nan
    return build_main_table(out)


def main() -> None:
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    # === Main results table ===
    train_path = Path("results/training_results.csv")
    bench_path = Path("results/benchmark_results.csv")
    if train_path.exists() and bench_path.exists():
        df_train_raw = pd.read_csv(train_path)
        df_bench_raw = pd.read_csv(bench_path)

        # Baseline setting only
        df_train_base = df_train_raw[df_train_raw["Setting"] == "baseline"]
        df_bench_base = df_bench_raw[df_bench_raw["Setting"] == "baseline"]

        n_train = 200  # episodes
        n_bench = 10   # benchmark maps

        df_train = _prep_df(df_train_base, "Train Reward Mean", "Train Reward Std", n_train)
        df_test = _prep_df(df_bench_base, "Benchmark Reward", "Benchmark Std", n_bench)

        df_train = df_train.rename(columns={"Reward (±CI)": "Train Reward (±CI)"})
        df_test = df_test[["Model", "Reward (±CI)"]].rename(columns={"Reward (±CI)": "Test Reward (±CI)"})

        # OOD metrics not available; fill with NaN
        df_ood = df_train[["Model"]].copy()
        df_ood["OOD Reward (±CI)"] = np.nan

        df_main = df_train.merge(df_test, on="Model", how="outer").merge(df_ood, on="Model", how="left")
        generate_results_table(df_main, str(tables_dir / "main_results.tex"))

    # === Ablation table ===
    if train_path.exists():
        df_raw = pd.read_csv(train_path)
        df_raw["Model"] = df_raw["Model"].replace(NAME_MAP)
        baseline = df_raw[df_raw["Setting"] == "baseline"].set_index("Model")
        table = pd.DataFrame({"Model": MAIN_METHODS})
        for setting in ["no_icm", "no_rnd", "no_planner"]:
            ablated = df_raw[df_raw["Setting"] == setting].set_index("Model")
            delta = ablated["Train Reward Mean"] - baseline["Train Reward Mean"]
            table[f"Δ {setting}"] = table["Model"].map(delta)
        generate_results_table(table, str(tables_dir / "ablation.tex"))

    # === Hyperparameter table ===
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    hparams = pd.DataFrame(
        {
            "Parameter": [
                "ICM β initial",
                "ICM β final",
                "LPPO ηλ",
                "Shield τ",
                "Shield κ",
                "Planner horizon",
            ],
            "Value": [
                cfg.get("initial_beta"),
                cfg.get("final_beta"),
                cfg.get("eta_lambda"),
                cfg.get("tau"),
                cfg.get("kappa"),
                cfg.get("H"),
            ],
        }
    )
    generate_results_table(hparams, str(tables_dir / "hparams.tex"))


if __name__ == "__main__":
    main()
