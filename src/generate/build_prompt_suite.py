from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SEED_DEFAULT = 42


def sanitize_identifier(s: str) -> str:
    """Sanitize an identifier component for use in prompt ids."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_")


def largest_remainder_allocation(total: int, weights: List[float]) -> List[int]:
    """Allocate integer counts using the largest remainder method."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return []
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    weight_sum = sum(weights)
    if weight_sum == 0:
        # If all weights are zero, allocate evenly.
        base = total // len(weights)
        remainder = total % len(weights)
        out = [base for _ in weights]
        for i in range(remainder):
            out[i] += 1
        return out

    raw = [(total * w) / weight_sum for w in weights]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)

    frac_parts = [(raw[i] - floors[i], i) for i in range(len(weights))]
    frac_parts.sort(reverse=True)
    out = floors[:]
    for _, idx in frac_parts[:remainder]:
        out[idx] += 1
    return out


def expand_templates(
    prompt_templates_df: pd.DataFrame,
    names_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand templates by replacing the demographic slot with each name."""
    required_cols = {"task_type", "template", "demographic_slot"}
    missing = required_cols - set(prompt_templates_df.columns)
    if missing:
        raise ValueError(f"prompt_templates.csv missing columns: {missing}")
    required_name_cols = {"name", "perceived_race", "perceived_gender", "perceived_ses"}
    missing_names = required_name_cols - set(names_df.columns)
    if missing_names:
        raise ValueError(f"names_* csv missing columns: {missing_names}")

    expanded_rows: List[Dict[str, str]] = []

    # Ensure deterministic ordering before sampling.
    prompt_templates_df = prompt_templates_df.reset_index(drop=True)
    names_df = names_df.reset_index(drop=True)

    for template_idx, tpl in prompt_templates_df.iterrows():
        task_type = tpl["task_type"]
        template_text = tpl["template"]
        demo_slot = tpl["demographic_slot"]
        for _, nm in names_df.iterrows():
            name = nm["name"]
            prompt_text = str(template_text).replace(str(demo_slot), str(name))
            expanded_rows.append(
                {
                    "task_type": str(task_type),
                    "prompt_text": prompt_text,
                    "name_used": str(name),
                    "perceived_race": str(nm["perceived_race"]),
                    "perceived_gender": str(nm["perceived_gender"]),
                    "perceived_ses": str(nm["perceived_ses"]),
                    "_template_idx": int(template_idx),
                }
            )

    return pd.DataFrame(expanded_rows)


def stratified_sample_prompts(
    expanded_df: pd.DataFrame,
    target_prompts: int,
    seed: int,
) -> pd.DataFrame:
    """Sample a subset of prompts stratified by task_type and perceived_race."""
    if target_prompts <= 0:
        raise ValueError("target_prompts must be positive")
    if target_prompts > len(expanded_df):
        return expanded_df.copy().reset_index(drop=True)

    rng = random.Random(seed)

    group_cols = ["task_type", "perceived_race"]
    group_sizes = (
        expanded_df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n")
    )
    total_available = int(group_sizes["n"].sum())

    weights = group_sizes["n"].tolist()
    quotas = largest_remainder_allocation(target_prompts, weights)

    sampled_parts: List[pd.DataFrame] = []
    for (task_type, race), quota in zip(group_sizes[group_cols].itertuples(index=False, name=None), quotas):
        part = expanded_df[(expanded_df["task_type"] == task_type) & (expanded_df["perceived_race"] == race)]
        quota_int = min(int(quota), len(part))
        if quota_int <= 0 or len(part) == 0:
            continue
        # Deterministic sampling: shuffle indices with RNG and slice.
        idxs = list(part.index)
        rng.shuffle(idxs)
        chosen = idxs[:quota_int]
        sampled_parts.append(part.loc[chosen])

    out = pd.concat(sampled_parts, ignore_index=True)
    if len(out) < target_prompts:
        # Fill any shortfall deterministically from the remaining pool.
        remaining = expanded_df.drop(index=out.index, errors="ignore")
        short = target_prompts - len(out)
        if short > 0 and len(remaining) > 0:
            remaining_idxs = list(remaining.index)
            rng.shuffle(remaining_idxs)
            chosen = remaining_idxs[:short]
            out = pd.concat([out, remaining.loc[chosen]], ignore_index=True)

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def sample_balanced_task_race(
    expanded_df: pd.DataFrame,
    per_cell: int,
    seed: int,
) -> pd.DataFrame:
    """Sample exactly `per_cell` rows per (task_type, perceived_race) combination.

    Total size = per_cell * n_tasks * n_races (e.g. 36 * 5 * 3 = 540).
    """
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")

    rng = random.Random(seed)
    tasks = sorted(expanded_df["task_type"].unique().tolist())
    races = sorted(expanded_df["perceived_race"].unique().tolist())

    parts: List[pd.DataFrame] = []
    for task in tasks:
        for race in races:
            pool = expanded_df[
                (expanded_df["task_type"] == task) & (expanded_df["perceived_race"] == race)
            ]
            if len(pool) < per_cell:
                raise ValueError(
                    f"Not enough rows for task={task}, race={race}: have {len(pool)}, need {per_cell}"
                )
            idxs = list(pool.index)
            rng.shuffle(idxs)
            chosen = idxs[:per_cell]
            parts.append(pool.loc[chosen])

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_prompt_suite(
    data_dir: Path,
    target_prompts: int,
    seed: int,
    balanced_per_cell: Optional[int] = None,
) -> pd.DataFrame:
    """Create `data/prompts/prompt_suite.csv` from templates and name lists."""
    names_race_path = data_dir / "names" / "names_race.csv"
    names_gender_path = data_dir / "names" / "names_gender.csv"
    names_ses_path = data_dir / "names" / "names_ses.csv"
    templates_path = data_dir / "prompts" / "prompt_templates.csv"

    names_race = pd.read_csv(names_race_path)
    names_gender = pd.read_csv(names_gender_path)
    names_ses = pd.read_csv(names_ses_path)

    names_df = names_race.merge(names_gender, on="name", how="left").merge(names_ses, on="name", how="left")
    if names_df["perceived_gender"].isna().any():
        raise ValueError("Missing perceived_gender for some names.")
    if names_df["perceived_ses"].isna().any():
        raise ValueError("Missing perceived_ses for some names.")

    prompt_templates = pd.read_csv(templates_path)
    expanded = expand_templates(prompt_templates, names_df)
    if balanced_per_cell is not None:
        sampled = sample_balanced_task_race(expanded, per_cell=balanced_per_cell, seed=seed)
    else:
        sampled = stratified_sample_prompts(expanded, target_prompts=target_prompts, seed=seed)

    sampled = sampled.drop(columns=["_template_idx"])
    sampled = sampled.reset_index(drop=True)

    # Deterministic prompt ids.
    sampled["prompt_id"] = [
        str(i) for i in range(len(sampled))
    ]
    cols = [
        "prompt_id",
        "task_type",
        "prompt_text",
        "name_used",
        "perceived_race",
        "perceived_gender",
        "perceived_ses",
    ]
    return sampled[cols]


def main() -> None:
    """CLI entrypoint for building the prompt suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--target_prompts", type=int, default=500)
    parser.add_argument(
        "--balanced_per_cell",
        type=int,
        default=None,
        help="If set, sample exactly this many prompts per (task_type, race); overrides target_prompts.",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--output_path", type=str, default="data/prompts/prompt_suite.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    balanced = args.balanced_per_cell
    if balanced is None and args.target_prompts < 100:
        raise ValueError("Set --target_prompts >= 100 or use --balanced_per_cell.")

    suite_df = build_prompt_suite(
        data_dir=data_dir,
        target_prompts=args.target_prompts,
        seed=args.seed,
        balanced_per_cell=balanced,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suite_df.to_csv(out_path, index=False)
    print(f"Wrote {len(suite_df)} prompts to {out_path}")


if __name__ == "__main__":
    main()

