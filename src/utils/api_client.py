from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


@dataclass
class APIResult:
    """Container for an LLM generation result."""

    text: str
    model_used: str


class UnifiedAPIClient:
    """Unified API client for Groq (primary) with OpenAI (fallback)."""

    def __init__(
        self,
        gemini_model: str = "gemini-1.5-flash",
        openai_model: str = "gpt-3.5-turbo",
        max_requests_per_minute: int = 10,
        min_interval_seconds: Optional[float] = None,
        api_log_path: Optional[Path] = None,
    ) -> None:
        """Initialize the client and internal rate limiting state.

        Args:
            gemini_model: Kept for compatibility; primary provider is Groq.
            openai_model: OpenAI model name.
            max_requests_per_minute: Max primary-provider requests per minute.
            min_interval_seconds: Minimum seconds between Groq calls (reduces 429 bursts).
                Default: env ``GROQ_MIN_INTERVAL`` or ``6.0`` (~10/min effective cap).
            api_log_path: JSONL log file path for all API calls.
        """
        self.gemini_model = gemini_model
        self.groq_model = "llama-3.1-8b-instant"
        self.openai_model = openai_model
        self.max_requests_per_minute = max_requests_per_minute
        if min_interval_seconds is not None:
            self.min_interval_seconds = float(min_interval_seconds)
        else:
            self.min_interval_seconds = float(os.environ.get("GROQ_MIN_INTERVAL", "6.0"))
        self._last_groq_call_time: float = 0.0
        self._gemini_request_times: Deque[float] = deque()
        self._api_log_path = api_log_path

        # Lazy-imports in generate() to surface clear errors.
        self._gemini_available = None
        self._openai_available = None

    def _now_iso(self) -> str:
        """Return current UTC timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()

    def _append_api_log(self, record: Dict[str, Any]) -> None:
        """Append one JSON record to the API log (flush immediately)."""
        if self._api_log_path is None:
            return
        self._api_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._api_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _enforce_gemini_rate_limit(self) -> None:
        """Ensure primary-provider request rate stays under the configured max."""
        now = time.time()
        # Spacing between calls avoids bursting the Groq free tier (reduces 429).
        if self.min_interval_seconds > 0 and self._last_groq_call_time > 0:
            wait = self.min_interval_seconds - (now - self._last_groq_call_time)
            if wait > 0:
                time.sleep(wait)
                now = time.time()

        if self.max_requests_per_minute <= 0:
            return

        while self._gemini_request_times and (now - self._gemini_request_times[0]) > 60.0:
            self._gemini_request_times.popleft()

        if len(self._gemini_request_times) >= self.max_requests_per_minute:
            oldest = self._gemini_request_times[0]
            sleep_for = max(0.0, 60.0 - (now - oldest))
            time.sleep(sleep_for + 0.01)
            now = time.time()

        self._gemini_request_times.append(now)
        self._last_groq_call_time = now

    def _is_rate_limit_error(self, err: Exception) -> bool:
        """Heuristically detect rate limit / quota errors."""
        try:
            import httpx

            if isinstance(err, httpx.HTTPStatusError) and err.response is not None:
                if err.response.status_code == 429:
                    return True
        except ImportError:
            pass

        msg = str(err).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return True
        if "resource exhausted" in msg:
            return True

        for attr in ("status", "status_code", "code"):
            v = getattr(err, attr, None)
            if v == 429:
                return True
        return False

    def _retry_after_seconds(self, err: Exception) -> Optional[float]:
        """Parse Retry-After header from httpx HTTPStatusError if present."""
        try:
            import httpx

            if isinstance(err, httpx.HTTPStatusError) and err.response is not None:
                ra = err.response.headers.get("retry-after")
                if ra is not None:
                    return float(ra)
        except (ImportError, TypeError, ValueError):
            pass
        return None

    def _gemini_available_check(self) -> bool:
        """Return whether groq is importable."""
        if self._gemini_available is not None:
            return self._gemini_available
        try:
            import groq  # noqa: F401

            self._gemini_available = True
        except Exception:
            self._gemini_available = False
        return self._gemini_available

    def _openai_available_check(self) -> bool:
        """Return whether openai is importable."""
        if self._openai_available is not None:
            return self._openai_available
        try:
            import openai  # noqa: F401

            self._openai_available = True
        except Exception:
            self._openai_available = False
        return self._openai_available

    def generate(
        self,
        prompt: str,
        prompt_id: Optional[str] = None,
        temperature: float = 0.0,
        model_preference: Optional[List[str]] = None,
        max_retries: int = 6,
    ) -> APIResult:
        """Generate text using Groq first, with OpenAI fallback.

        Args:
            prompt: Prompt text to send to the model.
            prompt_id: Optional prompt id for logging.
            temperature: Sampling temperature.
            model_preference: Ordering of models to try. Values: ["gemini", "openai"].
            max_retries: Retries on rate limit errors.
        """
        pref = model_preference or ["gemini", "openai"]
        last_err: Optional[Exception] = None

        for model_choice in pref:
            if model_choice == "gemini":
                if not self._gemini_available_check():
                    last_err = RuntimeError("groq not available")
                    continue
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    last_err = RuntimeError("GROQ_API_KEY is not set in .env")
                    continue
                result = self._generate_gemini(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                return result
            if model_choice == "openai":
                if not self._openai_available_check():
                    last_err = RuntimeError("openai not available")
                    continue
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    last_err = RuntimeError("OPENAI_API_KEY is not set in .env")
                    continue
                result = self._generate_openai(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    temperature=temperature,
                )
                return result
            last_err = RuntimeError(f"Unknown model choice: {model_choice}")

        raise RuntimeError(f"All model providers failed. Last error: {last_err}")

    def _generate_gemini(
        self,
        prompt: str,
        prompt_id: Optional[str],
        temperature: float,
        max_retries: int,
    ) -> APIResult:
        """Generate using Groq with retry logic for rate limit errors.

        Note: method name is retained for backward compatibility.
        """
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        for attempt in range(max_retries):
            try:
                self._enforce_gemini_rate_limit()
                completion = client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                text = completion.choices[0].message.content or ""
                record = {
                    "timestamp": self._now_iso(),
                    "prompt_id": prompt_id,
                    "model_used": self.groq_model,
                }
                self._append_api_log(record)
                return APIResult(text=text, model_used=self.groq_model)
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < (max_retries - 1):
                    ra = self._retry_after_seconds(e)
                    if ra is not None and ra > 0:
                        wait = min(ra + random.random() * 0.5, 120.0)
                    else:
                        backoff = (2**attempt) * 2.0
                        jitter = random.random() * 0.5
                        wait = backoff + jitter
                    time.sleep(wait)
                    continue
                raise

        # Should be unreachable because we either return or raise.
        raise RuntimeError("Groq generation failed after retries")

    def _generate_openai(
        self,
        prompt: str,
        prompt_id: Optional[str],
        temperature: float,
    ) -> APIResult:
        """Generate using OpenAI as a fallback provider."""
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = completion.choices[0].message.content or ""
        record = {
            "timestamp": self._now_iso(),
            "prompt_id": prompt_id,
            "model_used": self.openai_model,
        }
        self._append_api_log(record)
        return APIResult(text=text, model_used=self.openai_model)

