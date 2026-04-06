from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.api_client import UnifiedAPIClient  # noqa: E402


def utc_iso() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def load_prompt_suite(prompt_suite_path: Path) -> pd.DataFrame:
    """Load prompt_suite.csv as a DataFrame."""
    if not prompt_suite_path.exists():
        raise FileNotFoundError(f"Missing prompt suite: {prompt_suite_path}")
    return pd.read_csv(prompt_suite_path)


def stratified_sample_by_race(df: pd.DataFrame, races: List[str], target_total: int, seed: int) -> pd.DataFrame:
    """Sample prompts stratified by perceived_race."""
    if target_total <= 0:
        raise ValueError("target_total must be positive")
    if df.empty:
        return df

    df_shuffled = df.reset_index(drop=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    per_race_target = target_total // len(races)
    remainder = target_total % len(races)

    parts: List[pd.DataFrame] = []
    for i, race in enumerate(races):
        n = per_race_target + (1 if i < remainder else 0)
        part = df_shuffled[df_shuffled["perceived_race"] == race]
        if len(part) <= n:
            parts.append(part)
        else:
            parts.append(part.iloc[:n])

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    """Generate baseline responses (optionally all tasks or one task type)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--prompt_suite_path", type=str, default="data/prompts/prompt_suite.csv")
    parser.add_argument("--condition", type=str, default="baseline")
    parser.add_argument("--all_tasks", action="store_true", help="Generate for every row in prompt_suite.csv")
    parser.add_argument("--task_type", type=str, default=None, help="Single task type (ignored if --all_tasks)")
    parser.add_argument("--races", type=str, nargs="+", default=["White", "Black", "Asian"])
    parser.add_argument("--target_responses", type=int, default=60, help="Only used without --all_tasks")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()

    prompt_suite = load_prompt_suite(Path(args.prompt_suite_path))

    if args.all_tasks:
        filtered = prompt_suite[prompt_suite["perceived_race"].isin(args.races)]
        sampled = filtered.reset_index(drop=True)
    else:
        if not args.task_type:
            raise ValueError("Provide --task_type or use --all_tasks.")
        filtered = prompt_suite[prompt_suite["task_type"] == args.task_type]
        filtered = filtered[filtered["perceived_race"].isin(args.races)]
        if filtered.empty:
            raise RuntimeError("No prompts matched the requested task_type and races.")
        sampled = stratified_sample_by_race(
            df=filtered,
            races=args.races,
            target_total=args.target_responses,
            seed=args.seed,
        )

    out_dir = Path(args.data_dir) / "responses" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in out_dir.glob("*.json") if p.is_file()}

    client = UnifiedAPIClient(
        gemini_model="gemini-1.5-flash",
        openai_model="gpt-3.5-turbo",
        max_requests_per_minute=int(os.environ.get("GROQ_MAX_RPM", "10")),
        api_log_path=Path("results") / "api_calls.log",
    )

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Generating baseline"):
        prompt_id = str(row["prompt_id"])
        if prompt_id in existing:
            continue

        prompt_text = str(row["prompt_text"])
        try:
            result = client.generate(
                prompt_text,
                prompt_id=prompt_id,
                temperature=args.temperature,
                model_preference=["gemini", "openai"],
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to generate prompt_id={prompt_id}: {e}")
            continue

        payload: Dict[str, str] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "response_text": result.text,
            "model": result.model_used,
            "timestamp": utc_iso(),
            "task_type": str(row["task_type"]),
            "name_used": str(row["name_used"]),
            "perceived_race": str(row["perceived_race"]),
            "perceived_gender": str(row["perceived_gender"]),
        }

        out_path = out_dir / f"{prompt_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
            f.flush()


if __name__ == "__main__":
    main()
