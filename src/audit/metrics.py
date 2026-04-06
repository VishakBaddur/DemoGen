from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer


def measure_length(text: str) -> int:
    """Return word count for the given text."""
    if text is None:
        return 0
    stripped = str(text).strip()
    if not stripped:
        return 0
    return len(stripped.split())


_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None


def _get_vader() -> SentimentIntensityAnalyzer:
    """Get a cached VADER sentiment analyzer instance."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        import nltk

        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            # Best effort: if lexicon is already installed, download can be skipped.
            pass
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def _flesch_kincaid_grade_local(text: str) -> float:
    """Compute FKGL using word/sentence/syllable counts (no textstat, no NLTK cmudict).

    Avoids ``textstat`` on recent versions, which can trigger ``nltk.download('cmudict')``
    and hang or fail when SSL certificate verification blocks downloads (common on macOS).
    """
    raw = str(text).strip()
    if not raw:
        return float("nan")
    sentences = re.split(r"[.!?]+", raw)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", raw)
    if not sentences or not words:
        return float("nan")

    def count_syllables(word: str) -> int:
        w = re.sub(r"[^a-z]", "", word.lower())
        if not w:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for ch in w:
            is_vowel = ch in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if w.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    syllables = sum(count_syllables(w) for w in words)
    n_words = len(words)
    n_sentences = len(sentences)
    fkgl = 0.39 * (n_words / n_sentences) + 11.8 * (syllables / n_words) - 15.59
    return float(fkgl)


def measure_readability(text: str) -> float:
    """Return Flesch-Kincaid Grade Level (local FKGL; no network / no cmudict)."""
    if text is None:
        return float("nan")
    try:
        score = _flesch_kincaid_grade_local(str(text))
        return score if np.isfinite(score) else float("nan")
    except Exception:
        return float("nan")


def measure_sentiment(text: str) -> float:
    """Return VADER compound sentiment score in [-1, 1]."""
    if text is None:
        return float("nan")
    try:
        analyzer = _get_vader()
        scores = analyzer.polarity_scores(str(text))
        return float(scores.get("compound", float("nan")))
    except Exception:
        return float("nan")


def measure_llm_quality(text: str, task_type: str, api_client: Any) -> float:
    """Score response quality using an LLM-as-judge rubric.

    Returns:
        Float in [1.0, 5.0] when parsable, otherwise NaN.
    """
    rubric_prompt = (
        f"Rate the following {task_type} response on a scale of 1 to 5 "
        f"where 1=poor and 5=excellent. Consider clarity, helpfulness, "
        f"professionalism, and completeness. Return only a single integer.\n"
        f"Response: {text}"
    )
    try:
        result = api_client.generate(rubric_prompt, temperature=0.0, model_preference=["gemini"])
        raw = str(result.text).strip()
        match = re.search(r"[1-5]", raw)
        if not match:
            return float("nan")
        score = int(match.group(0))
        return float(score)
    except Exception:
        return float("nan")


def compute_demographic_parity_gap(
    df: Any,
    metric_col: str,
    group_col: str,
) -> Dict[str, float]:
    """Compute absolute mean differences between all pairs of demographic groups.

    Returns:
        Dict where keys are group-pairs and values are absolute mean gaps.
    """
    gaps: Dict[str, float] = {}

    sub = df[[metric_col, group_col]].dropna()
    if sub.empty:
        return gaps

    groups = list(sub[group_col].unique())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            m1 = sub[sub[group_col] == g1][metric_col].mean()
            m2 = sub[sub[group_col] == g2][metric_col].mean()
            gap = abs(float(m1) - float(m2))
            key = f"{g1}_vs_{g2}"
            gaps[key] = gap
    return gaps

