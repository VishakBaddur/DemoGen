from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d using pooled standard deviation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    n1 = len(x)
    n2 = len(y)
    s1 = float(np.std(x, ddof=1))
    s2 = float(np.std(y, ddof=1))

    pooled_var_num = (n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)
    pooled_var_den = (n1 + n2 - 2)
    if pooled_var_den <= 0:
        return float("nan")
    pooled_sd = float(np.sqrt(pooled_var_num / pooled_var_den))
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def safe_anova(groups: List[np.ndarray]) -> float:
    """Run one-way ANOVA and return p-value, or NaN if invalid."""
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    try:
        _, p = stats.f_oneway(*groups)
        return float(p)
    except Exception:
        return float("nan")


def safe_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Run Welch t-test and return (t_stat, p_value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    try:
        t, p = stats.ttest_ind(x, y, equal_var=False)
        return float(t), float(p)
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    """Run significance tests and save results to CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True, help="baseline | persona_blind | reranked")
    parser.add_argument("--audit_csv_path", type=str, default=None)
    parser.add_argument("--group_col", type=str, default="perceived_race")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["llm_quality", "length", "readability"],
        help="Metrics for ANOVA and pairwise tests.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output CSV path (default: results/statistical_tests_<condition>.csv).",
    )
    args = parser.parse_args()

    condition = args.condition
    audit_csv_path = Path(args.audit_csv_path) if args.audit_csv_path else Path("results") / f"{condition}_audit.csv"
    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Missing audit CSV: {audit_csv_path}")

    df = pd.read_csv(audit_csv_path)

    metric_cols = list(args.metrics)
    groups = sorted([g for g in df[args.group_col].dropna().unique().tolist()])

    out_rows: List[Dict[str, Any]] = []

    # One-way ANOVA across all racial groups.
    for metric in metric_cols:
        group_arrays = [df[df[args.group_col] == g][metric].values for g in groups]
        anova_p = safe_anova(group_arrays)
        out_rows.append(
            {
                "condition": condition,
                "metric": metric,
                "test_type": "one_way_anova",
                "group_col": args.group_col,
                "group1": "",
                "group2": "",
                "p_value": anova_p,
                "p_value_bonferroni": anova_p,
                "cohens_d": "",
            }
        )

    # Pairwise t-tests: White vs Black and White vs Asian.
    white_label = "White"
    pair_targets = [("White", "Black"), ("White", "Asian")]
    bonf_m = len(pair_targets)

    for metric in metric_cols:
        for g1, g2 in pair_targets:
            x = df[df[args.group_col] == g1][metric].values
            y = df[df[args.group_col] == g2][metric].values
            t_stat, p = safe_ttest(x, y)
            d = cohen_d(x, y)
            if np.isfinite(p):
                p_adj = min(float(p) * bonf_m, 1.0)
            else:
                p_adj = float("nan")
            out_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "test_type": "pairwise_ttest",
                    "group_col": args.group_col,
                    "group1": g1,
                    "group2": g2,
                    "p_value": p,
                    "p_value_bonferroni": p_adj,
                    "cohens_d": d,
                }
            )

    out_path = Path(args.output_path) if args.output_path else Path("results") / f"statistical_tests_{condition}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"Wrote statistical tests: {out_path}")


if __name__ == "__main__":
    main()

