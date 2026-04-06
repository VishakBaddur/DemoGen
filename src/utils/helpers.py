from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return it as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dict payload as JSON to disk (overwrite if exists)."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float if possible, otherwise return None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def list_json_files(folder: Path) -> List[Path]:
    """Return sorted JSON file paths under a folder."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix == ".json"])


def unique_values(values: Iterable[Any]) -> List[Any]:
    """Return unique values while preserving first-seen order."""
    seen = set()
    out: List[Any] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out

