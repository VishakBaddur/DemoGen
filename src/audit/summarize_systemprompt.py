from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    s1 = float(np.std(x, ddof=1))
    s2 = float(np.std(y, ddof=1))
    denom = len(x) + len(y) - 2
    if denom <= 0:
        return float("nan")
    pooled = math.sqrt(((len(x) - 1) * s1 * s1 + (len(y) - 1) * s2 * s2) / denom)
    if pooled == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def main() -> None:
    audit_path = Path("results/systemprompt_audit.csv")
    parity_path = Path("results/systemprompt_parity_gaps.csv")
    out_path = Path("results/systemprompt_summary.csv")
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    if not parity_path.exists():
        raise FileNotFoundError(parity_path)

    df = pd.read_csv(audit_path)
    parity = pd.read_csv(parity_path)
    for metric in ["llm_quality", "length", "readability"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    rows: List[Dict[str, object]] = []
    rows.append({"section": "sample", "stat": "n_rows", "value": len(df)})
    rows.append({"section": "sample", "stat": "n_valid_llm_quality", "value": int(df["llm_quality"].notna().sum())})

    for metric in ["llm_quality", "length", "readability"]:
        rows.append(
            {
                "section": "overall_mean",
                "metric": metric,
                "stat": "mean",
                "value": float(df[metric].mean()),
            }
        )
        for race, sub in df.groupby("perceived_race"):
            rows.append(
                {
                    "section": "mean_by_race",
                    "metric": metric,
                    "race": race,
                    "stat": "mean",
                    "value": float(sub[metric].mean()),
                    "n": int(sub[metric].notna().sum()),
                }
            )
        metric_gaps = parity[parity["metric"] == metric]
        rows.append(
            {
                "section": "max_pairwise_gap",
                "metric": metric,
                "stat": "max_gap",
                "value": float(metric_gaps["gap"].max()),
            }
        )

    groups = [
        df.loc[df["perceived_race"] == race, "llm_quality"].dropna().values
        for race in sorted(df["perceived_race"].dropna().unique().tolist())
    ]
    _, anova_p = stats.f_oneway(*groups)
    rows.append(
        {
            "section": "anova",
            "metric": "llm_quality",
            "stat": "one_way_anova_p",
            "value": float(anova_p),
        }
    )

    for g1, g2 in [("White", "Black"), ("White", "Asian"), ("Asian", "Black")]:
        x = df.loc[df["perceived_race"] == g1, "llm_quality"].dropna().values
        y = df.loc[df["perceived_race"] == g2, "llm_quality"].dropna().values
        t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
        rows.append(
            {
                "section": "pairwise_ttest",
                "metric": "llm_quality",
                "group1": g1,
                "group2": g2,
                "mean_group1": float(np.mean(x)),
                "mean_group2": float(np.mean(y)),
                "gap_abs": float(abs(np.mean(x) - np.mean(y))),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d_group1_minus_group2": cohen_d(x, y),
                "n_group1": int(len(x)),
                "n_group2": int(len(y)),
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
