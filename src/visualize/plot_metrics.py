from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


GROUP_COL_MAP = {
    "race": "perceived_race",
    "gender": "perceived_gender",
    "ses": "perceived_ses",
}


def plot_mean_bar(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    condition: str,
    out_path: Path,
    order: List[str],
) -> None:
    """Plot a bar chart of mean metric by demographic group."""
    if metric_col not in df.columns:
        raise ValueError(f"Metric column not found in audit CSV: {metric_col}")

    # Force numeric conversion so plotting never silently uses non-numeric values.
    metric_series = pd.to_numeric(df[metric_col], errors="coerce")
    plot_df = df.copy()
    plot_df[metric_col] = metric_series
    plot_df = plot_df.dropna(subset=[metric_col, group_col])

    if plot_df.empty:
        raise ValueError(f"No plottable rows for metric '{metric_col}' after numeric coercion.")

    # Sanity-check readability scale to avoid plotting an unintended column.
    if metric_col == "readability":
        mean_val = float(plot_df[metric_col].mean())
        if mean_val < 1.0:
            raise ValueError(
                "Readability values look invalid (mean < 1). "
                "Expected FKGL values around double digits for this dataset."
            )

    plt.figure(figsize=(7, 4))
    sns.set(style="whitegrid")
    ax = sns.barplot(data=plot_df, x=group_col, y=metric_col, order=order, errorbar="sd")
    pretty_group = {
        "perceived_race": "Perceived race",
        "perceived_gender": "Perceived gender",
        "perceived_ses": "Perceived SES",
    }.get(group_col, group_col.replace("_", " ").title())
    ax.set_xlabel(pretty_group)
    if metric_col == "readability":
        ax.set_ylabel("Flesch-Kincaid grade level")
        pretty_metric = "Readability"
    elif metric_col == "llm_quality":
        ax.set_ylabel("LLM quality score")
        pretty_metric = "LLM quality"
    else:
        pretty_metric = metric_col.replace("_", " ").title()
        ax.set_ylabel(pretty_metric)

    pretty_condition = condition.replace("_", " ").title()
    ax.set_title(f"Mean {pretty_metric} by {pretty_group} ({pretty_condition})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    """Generate report-ready plots from an audit CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--audit_csv_path", type=str, default=None)
    parser.add_argument("--group_col", type=str, default="race", help="race | gender | ses")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["llm_quality", "length", "readability"],
    )
    args = parser.parse_args()

    condition = args.condition
    audit_csv_path = Path(args.audit_csv_path) if args.audit_csv_path else Path("results") / f"{condition}_audit.csv"
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Missing audit CSV: {audit_csv_path}")

    df = pd.read_csv(audit_csv_path)
    if args.group_col not in GROUP_COL_MAP:
        raise ValueError(f"Unsupported group_col: {args.group_col}")
    group_col_actual = GROUP_COL_MAP[args.group_col]

    # Preferred display order.
    if args.group_col == "race":
        order = ["White", "Black", "Asian"]
    else:
        order = sorted([x for x in df[group_col_actual].dropna().unique().tolist()])

    for metric in args.metrics:
        out_path = Path("results") / "figures" / f"{condition}_mean_{metric}_by_{args.group_col}.png"
        plot_mean_bar(
            df=df,
            metric_col=metric,
            group_col=group_col_actual,
            condition=condition,
            out_path=out_path,
            order=order,
        )

    print("Saved plots to results/figures/")


if __name__ == "__main__":
    main()

