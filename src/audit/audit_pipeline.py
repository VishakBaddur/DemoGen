from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.audit.metrics import (
    compute_demographic_parity_gap,
    measure_length,
    measure_llm_quality,
    measure_readability,
    measure_sentiment,
)
from src.utils.api_client import UnifiedAPIClient
from src.utils.helpers import list_json_files


GROUP_COL_MAP = {
    "race": "perceived_race",
    "gender": "perceived_gender",
    "ses": "perceived_ses",
}


def load_responses(folder: Path) -> pd.DataFrame:
    """Load per-prompt response JSON files from a folder into a DataFrame."""
    rows: List[Dict[str, Any]] = []
    for path in list_json_files(folder):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise RuntimeError(f"No JSON response files found in {folder}")
    return pd.DataFrame(rows)


def compute_metrics(
    df: pd.DataFrame,
    skip_llm_quality: bool,
    api_client: Any = None,
    include_sentiment: bool = True,
    existing_llm_quality: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute length, readability, optional VADER sentiment, and (by default) LLM-as-judge quality."""
    df = df.copy()
    df["length"] = df["response_text"].apply(measure_length)
    df["readability"] = df["response_text"].apply(measure_readability)
    if include_sentiment:
        df["sentiment"] = df["response_text"].apply(measure_sentiment)
    else:
        df["sentiment"] = np.nan

    existing_llm_quality = existing_llm_quality or {}

    if skip_llm_quality:
        df["llm_quality"] = np.nan
    else:
        if api_client is None:
            raise ValueError("api_client must be provided when skip_llm_quality is False")
        scores: List[float] = []
        for _, row in df.iterrows():
            pid = str(row["prompt_id"])
            if pid in existing_llm_quality:
                scores.append(float(existing_llm_quality[pid]))
                continue
            score = measure_llm_quality(text=row["response_text"], task_type=str(row["task_type"]), api_client=api_client)
            scores.append(score)
        df["llm_quality"] = scores

    return df


def parity_gaps_by_task_type(
    df: pd.DataFrame,
    metric_cols: List[str],
    group_col: str,
) -> List[Dict[str, Any]]:
    """Compute max pairwise demographic parity gap per task_type for each metric."""
    rows: List[Dict[str, Any]] = []
    for task in sorted(df["task_type"].dropna().unique().tolist()):
        sub = df[df["task_type"] == task]
        for m in metric_cols:
            gaps = compute_demographic_parity_gap(sub, metric_col=m, group_col=group_col)
            if not gaps:
                continue
            max_gap = max(float(v) for v in gaps.values())
            rows.append(
                {
                    "task_type": task,
                    "metric": m,
                    "group_col": group_col,
                    "max_pairwise_gap": max_gap,
                }
            )
    return rows


def main() -> None:
    """Run the audit pipeline over a set of saved responses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True, help="baseline | persona_blind | reranked")
    parser.add_argument("--responses_dir", type=str, default=None)
    parser.add_argument("--prompt_suite_path", type=str, default="data/prompts/prompt_suite.csv")
    parser.add_argument("--group_cols", type=str, nargs="+", default=["race"])
    parser.add_argument(
        "--skip_llm_quality",
        action="store_true",
        help="If set, skip LLM-as-judge scoring (not recommended for final report).",
    )
    parser.add_argument(
        "--include_sentiment",
        action="store_true",
        help="If set, compute VADER sentiment as a secondary diagnostic metric.",
    )
    parser.add_argument(
        "--parity_metrics",
        type=str,
        nargs="+",
        default=["llm_quality", "length", "readability"],
        help="Metrics for which to compute demographic parity gaps.",
    )
    parser.add_argument(
        "--parity_by_task",
        action="store_true",
        help="Also write max pairwise gap per task_type (for heatmaps).",
    )
    parser.add_argument("--gemini_model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--resume_audit",
        action="store_true",
        help="Reuse llm_quality from existing results/<condition>_audit.csv and score only missing rows.",
    )
    args = parser.parse_args()

    load_dotenv()

    condition = args.condition
    responses_dir = Path(args.responses_dir) if args.responses_dir else Path("data") / "responses" / condition
    prompt_suite_path = Path(args.prompt_suite_path)

    df_resp = load_responses(responses_dir)
    df_resp["prompt_id"] = df_resp["prompt_id"].astype(str)

    df_suite = pd.read_csv(prompt_suite_path)[["prompt_id", "perceived_ses"]]
    df_suite["prompt_id"] = df_suite["prompt_id"].astype(str)
    df_resp = df_resp.merge(df_suite, on="prompt_id", how="left")

    out_path = Path("results") / f"{condition}_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_llm_quality: Dict[str, float] = {}
    if args.resume_audit and out_path.exists():
        try:
            df_prev = pd.read_csv(out_path)
            if "prompt_id" in df_prev.columns and "llm_quality" in df_prev.columns:
                df_prev["prompt_id"] = df_prev["prompt_id"].astype(str)
                sub_prev = df_prev.dropna(subset=["llm_quality"])[["prompt_id", "llm_quality"]]
                existing_llm_quality = {
                    str(pid): float(score) for pid, score in sub_prev.itertuples(index=False, name=None)
                }
                print(f"Resuming audit with {len(existing_llm_quality)} existing llm_quality scores from {out_path}")
        except Exception as e:
            print(f"[WARN] Could not read existing audit for resume: {e}")

    # Compute metrics.
    api_client = None
    if not args.skip_llm_quality:
        api_client = UnifiedAPIClient(
            gemini_model=args.gemini_model,
            openai_model=args.openai_model,
            max_requests_per_minute=int(os.environ.get("GROQ_MAX_RPM", "10")),
            api_log_path=Path("results") / "api_calls.log",
        )

    df_metrics = compute_metrics(
        df_resp,
        skip_llm_quality=args.skip_llm_quality,
        api_client=api_client,
        include_sentiment=args.include_sentiment,
        existing_llm_quality=existing_llm_quality,
    )
    df_metrics["condition"] = condition

    # Save audit CSV.
    out_cols = [
        "prompt_id",
        "task_type",
        "name_used",
        "perceived_race",
        "perceived_gender",
        "perceived_ses",
        "length",
        "readability",
        "sentiment",
        "llm_quality",
        "condition",
    ]

    df_metrics[out_cols].to_csv(out_path, index=False)

    # Compute and save parity gaps for requested group columns.
    metrics_cols = list(args.parity_metrics)
    parity_records: List[Dict[str, Any]] = []
    task_parity_rows: List[Dict[str, Any]] = []
    for group_key in args.group_cols:
        if group_key not in GROUP_COL_MAP:
            raise ValueError(f"Unsupported group_col: {group_key}")
        group_col = GROUP_COL_MAP[group_key]

        for metric_col in metrics_cols:
            if metric_col not in df_metrics.columns:
                raise ValueError(f"Unknown parity metric: {metric_col}")
            gaps = compute_demographic_parity_gap(df_metrics, metric_col=metric_col, group_col=group_col)
            for pair, gap in gaps.items():
                parity_records.append(
                    {
                        "condition": condition,
                        "group_col": group_col,
                        "metric": metric_col,
                        "group_pair": pair,
                        "gap": gap,
                    }
                )
            if args.parity_by_task:
                for row in parity_gaps_by_task_type(df_metrics, metric_cols=[metric_col], group_col=group_col):
                    row["condition"] = condition
                    task_parity_rows.append(row)

    gaps_path = Path("results") / f"{condition}_parity_gaps.csv"
    pd.DataFrame(parity_records).to_csv(gaps_path, index=False)

    if args.parity_by_task and task_parity_rows:
        task_path = Path("results") / f"{condition}_parity_gaps_by_task.csv"
        pd.DataFrame(task_parity_rows).to_csv(task_path, index=False)
        print(f"Wrote task-level parity gaps: {task_path}")

    print(f"Wrote audit CSV: {out_path}")
    print(f"Wrote parity gaps CSV: {gaps_path}")


if __name__ == "__main__":
    main()

