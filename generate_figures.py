import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

HEADLESS = os.environ.get("DISPLAY", "") == ""
if HEADLESS:
    matplotlib.use("Agg")

from src.visualization import (
    plot_training_curves,
    plot_learning_panels,
    plot_violation_rate,
    plot_pareto,
)


def main(output_dir: str | Path = Path("figures")) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Training curves with confidence intervals
    reward_logs = [rng.normal(loc=0.0, scale=0.1, size=50).cumsum().tolist() for _ in range(3)]
    success_flags = [rng.integers(0, 2, size=50).tolist() for _ in range(3)]
    metrics = {"Reward": reward_logs, "Success": success_flags}
    plot_training_curves(
        metrics,
        str(output_dir / "training_curves_compact_CI.pdf"),
        show=not HEADLESS,
    )

    # Ablation curves across two mock methods
    logs = {
        "MethodA": {
            "Reward": [rng.normal(0.0, 0.1, size=50).cumsum().tolist() for _ in range(3)],
            "Success": [rng.integers(0, 2, size=50).tolist() for _ in range(3)],
        },
        "MethodB": {
            "Reward": [rng.normal(0.5, 0.1, size=50).cumsum().tolist() for _ in range(3)],
            "Success": [rng.integers(0, 2, size=50).tolist() for _ in range(3)],
        },
    }
    plot_learning_panels(
        logs,
        str(output_dir / "ablation_curves_compact_CI.pdf"),
        show=not HEADLESS,
    )

    # Violation rate over training
    violation_logs = [rng.integers(0, 2, size=50).tolist() for _ in range(3)]
    plot_violation_rate(
        violation_logs,
        str(output_dir / "violation_rate_over_training.pdf"),
        show=not HEADLESS,
    )

    # Pareto frontier between reward and cost
    df = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Reward Mean": [10, 12, 9],
            "Reward CI": [1, 1, 1],
            "Cost Mean": [1.0, 0.8, 1.2],
            "Cost CI": [0.1, 0.05, 0.15],
        }
    )
    plot_pareto(
        df,
        cost_limit=1.0,
        output_path=str(output_dir / "pareto_reward_vs_cost.pdf"),
        show=not HEADLESS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    main(args.output_dir)
