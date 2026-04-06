from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def max_race_parity(df: pd.DataFrame, metric: str) -> float:
    """Maximum absolute pairwise gap across White/Black/Asian for one metric."""
    sub = df[["perceived_race", metric]].dropna()
    if sub.empty:
        return float("nan")
    groups = sorted(sub["perceived_race"].unique().tolist())
    gaps: List[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            m1 = sub[sub["perceived_race"] == groups[i]][metric].mean()
            m2 = sub[sub["perceived_race"] == groups[j]][metric].mean()
            gaps.append(abs(float(m1) - float(m2)))
    return float(max(gaps)) if gaps else float("nan")


def main() -> None:
    """Fairness vs quality tradeoff: mean LLM quality vs max racial parity gap on quality."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit_paths",
        type=str,
        nargs="+",
        required=True,
        help="Audit CSVs, e.g. results/baseline_audit.csv results/persona_blind_audit.csv ...",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Legend labels (same order as audit_paths).",
    )
    parser.add_argument("--output_path", type=str, default="results/figures/fairness_quality_tradeoff.png")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem.replace("_audit", "") for p in args.audit_paths]

    xs: List[float] = []
    ys: List[float] = []
    for path in args.audit_paths:
        df = pd.read_csv(path)
        if "llm_quality" not in df.columns:
            raise ValueError(f"Missing llm_quality in {path}")
        mean_q = float(df["llm_quality"].mean())
        gap = max_race_parity(df, "llm_quality")
        xs.append(gap)
        ys.append(mean_q)

    plt.figure(figsize=(6, 4.5))
    sns.set(style="whitegrid")
    for x, y, lab in zip(xs, ys, labels):
        plt.scatter(x, y, s=120, label=lab)
    plt.xlabel("Max pairwise racial parity gap (LLM quality)")
    plt.ylabel("Mean LLM-as-judge quality (all groups)")
    plt.title("Fairness--quality tradeoff (lower gap is fairer)")
    plt.legend()
    plt.tight_layout()
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
