from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.audit.metrics import measure_llm_quality  # noqa: E402
from src.utils.api_client import UnifiedAPIClient  # noqa: E402


def utc_iso() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def load_task_quality_targets(baseline_audit_path: Path) -> Dict[str, float]:
    """Mean LLM-as-judge quality per task_type from baseline audit (fairness reference)."""
    df = pd.read_csv(baseline_audit_path)
    if "llm_quality" not in df.columns or df["llm_quality"].isna().all():
        raise RuntimeError(
            "baseline_audit.csv must contain computed llm_quality. "
            "Run: python src/audit/audit_pipeline.py --condition baseline"
        )
    sub = df.dropna(subset=["llm_quality"])
    return sub.groupby("task_type")["llm_quality"].mean().to_dict()


def _json_safe(val: Any) -> Any:
    """Convert NaN to None for JSON serialization."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def pick_closest_to_target(scores: List[float], target: float) -> int:
    """Return index of score closest to target."""
    best_i = 0
    best_d = abs(scores[0] - target)
    for i, s in enumerate(scores[1:], start=1):
        d = abs(s - target)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def main() -> None:
    """Self-consistency reranking: k samples, LLM-judge scores, pick fairest vs task mean quality."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--prompt_suite_path", type=str, default="data/prompts/prompt_suite.csv")
    parser.add_argument("--baseline_audit_path", type=str, default="results/baseline_audit.csv")
    parser.add_argument("--condition", type=str, default="reranked")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()

    suite = pd.read_csv(Path(args.prompt_suite_path))
    targets = load_task_quality_targets(Path(args.baseline_audit_path))

    rpm = int(os.environ.get("GROQ_MAX_RPM", "10"))
    # One client so Groq rate limits apply to generation + judging together.
    client = UnifiedAPIClient(
        max_requests_per_minute=rpm,
        api_log_path=Path("results") / "api_calls.log",
    )

    out_dir = Path(args.data_dir) / "responses" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in out_dir.glob("*.json") if p.is_file()}

    for _, row in tqdm(suite.iterrows(), total=len(suite), desc="Reranked"):
        prompt_id = str(row["prompt_id"])
        if prompt_id in existing:
            continue

        prompt_text = str(row["prompt_text"])
        task_type = str(row["task_type"])
        target_q = float(targets.get(task_type, 3.0))

        candidates: List[str] = []
        scores: List[float] = []
        for _ in range(args.k):
            try:
                res = client.generate(
                    prompt_text,
                    prompt_id=prompt_id,
                    temperature=args.temperature,
                    model_preference=["gemini", "openai"],
                )
                text = res.text
                candidates.append(text)
                q = measure_llm_quality(text, task_type=task_type, api_client=client)
                scores.append(float(q) if q == q else float("nan"))
            except Exception as e:
                print(f"\n[ERROR] prompt_id={prompt_id} sample failed: {e}")
                candidates.append("")
                scores.append(float("nan"))

        valid = [(i, s) for i, s in enumerate(scores) if s == s]
        if not valid:
            print(f"\n[WARN] No valid judge scores for prompt_id={prompt_id}, skipping.")
            continue

        best_local = min(valid, key=lambda t: abs(t[1] - target_q))[0]

        payload: Dict[str, object] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "response_text": candidates[best_local],
            "model": "reranked_self_consistency",
            "timestamp": utc_iso(),
            "task_type": task_type,
            "name_used": str(row["name_used"]),
            "perceived_race": str(row["perceived_race"]),
            "perceived_gender": str(row["perceived_gender"]),
            "k": args.k,
            "temperature": args.temperature,
            "target_task_mean_quality": target_q,
            "candidate_scores": [_json_safe(s) for s in scores],
            "selected_index": best_local,
        }

        with (out_dir / f"{prompt_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
            f.flush()


if __name__ == "__main__":
    main()
