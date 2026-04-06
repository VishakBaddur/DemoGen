from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    """Heatmap: rows = task_type, columns = metric, values = max pairwise parity gap."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parity_by_task_csv",
        type=str,
        required=True,
        help="e.g. results/baseline_parity_gaps_by_task.csv",
    )
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.parity_by_task_csv)
    pivot = df.pivot(index="task_type", columns="metric", values="max_pairwise_gap")

    out = Path(args.output_path) if args.output_path else Path("results/figures") / (
        Path(args.parity_by_task_csv).stem + "_heatmap.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Max racial parity gap by task type and metric")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
