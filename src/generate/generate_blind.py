from __future__ import annotations

import argparse
import json
import os
import re
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


def load_names(data_dir: Path) -> List[str]:
    """Load all person names used in prompts for replacement."""
    path = data_dir / "names" / "names_race.csv"
    df = pd.read_csv(path)
    return sorted([str(x) for x in df["name"].tolist()], key=len, reverse=True)


def anonymize_prompt(prompt_text: str, names: List[str]) -> str:
    """Replace known names with neutral phrasing; normalize pronoun-heavy templates."""
    text = str(prompt_text)
    for name in names:
        if name in text:
            text = text.replace(name, "the applicant")
    text = re.sub(r"\bThe applicant's\b", "The applicant's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe applicant's\b", "the applicant's", text)
    return text


def main() -> None:
    """Generate persona-blind responses (demographic names removed)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--prompt_suite_path", type=str, default="data/prompts/prompt_suite.csv")
    parser.add_argument("--condition", type=str, default="persona_blind")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv()

    suite = pd.read_csv(Path(args.prompt_suite_path))
    names = load_names(Path(args.data_dir))

    out_dir = Path(args.data_dir) / "responses" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in out_dir.glob("*.json") if p.is_file()}

    client = UnifiedAPIClient(
        max_requests_per_minute=int(os.environ.get("GROQ_MAX_RPM", "10")),
        api_log_path=Path("results") / "api_calls.log",
    )

    for _, row in tqdm(suite.iterrows(), total=len(suite), desc="Persona-blind"):
        prompt_id = str(row["prompt_id"])
        if prompt_id in existing:
            continue

        blind_text = anonymize_prompt(str(row["prompt_text"]), names)
        try:
            result = client.generate(
                blind_text,
                prompt_id=prompt_id,
                temperature=args.temperature,
                model_preference=["gemini", "openai"],
            )
        except Exception as e:
            print(f"\n[ERROR] prompt_id={prompt_id}: {e}")
            continue

        payload: Dict[str, str] = {
            "prompt_id": prompt_id,
            "prompt_text": blind_text,
            "response_text": result.text,
            "model": result.model_used,
            "timestamp": utc_iso(),
            "task_type": str(row["task_type"]),
            "name_used": str(row["name_used"]),
            "perceived_race": str(row["perceived_race"]),
            "perceived_gender": str(row["perceived_gender"]),
        }

        with (out_dir / f"{prompt_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
            f.flush()


if __name__ == "__main__":
    main()
