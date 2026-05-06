"""Per-task one-way ANOVA on llm_quality across perceived_race (baseline).

Writes results CSV for reporting (e.g., Table 9) and optional LaTeX rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from scipy import stats


TASK_LABELS = {
    "cover_letter": "Cover Letter",
    "concept_explanation": "Concept Explanation",
    "advice_giving": "Advice Giving",
    "problem_solving": "Problem Solving",
    "recommendation_letter": "Recommendation Letter",
}


def per_task_anova_pvalues(df: pd.DataFrame, metric: str, group_col: str) -> List[Tuple[str, float, int]]:
    """Return list of (task_type, anova_p, n_valid) sorted by task name."""
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric, group_col, "task_type"])
    rows: List[Tuple[str, float, int]] = []
    for task in sorted(df["task_type"].unique()):
        sub = df[df["task_type"] == task]
        races = sorted(sub[group_col].dropna().unique().tolist())
        groups = [sub[sub[group_col] == r][metric].values for r in races]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            p = float("nan")
        else:
            try:
                _, p = stats.f_oneway(*groups)
            except Exception:
                p = float("nan")
        rows.append((task, float(p), int(len(sub))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-task ANOVA baseline llm_quality by race.")
    parser.add_argument("--audit_csv", type=str, default="results/baseline_audit.csv")
    parser.add_argument("--condition", type=str, default="baseline", help="label in output CSV")
    parser.add_argument("--metric", type=str, default="llm_quality")
    parser.add_argument("--group_col", type=str, default="perceived_race")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/baseline_per_task_anova_llm_quality.csv",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    path = Path(args.audit_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    results = per_task_anova_pvalues(df, args.metric, args.group_col)
    out_rows = []
    for task, p, n in results:
        out_rows.append(
            {
                "condition": args.condition,
                "task_type": task,
                "task_label": TASK_LABELS.get(task, task),
                "metric": args.metric,
                "group_col": args.group_col,
                "n_responses": n,
                "anova_p": p,
                "significant_0.05": bool(p < args.alpha) if pd.notna(p) else False,
            }
        )
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    for r in out_rows:
        sig = "Yes" if r["significant_0.05"] else "No"
        print(f"  {r['task_label']}: p={r['anova_p']:.6g} n={r['n_responses']} significant={sig}")


if __name__ == "__main__":
    main()
