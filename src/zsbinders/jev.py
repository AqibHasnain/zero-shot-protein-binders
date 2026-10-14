"""Client for the Featherless Simple Jev classifier API."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import requests

DEMO_API_URL = "https://simple-jev-demo-api.featherless.ai/v1/classifier"
DEMO_MODELS_URL = "https://simple-jev-demo-api.featherless.ai/v1/models"
PROD_API_URL = "https://api.featherless.ai/v1/classifier"

DEFAULT_MODEL = "featherless-ai/Qwen3.8-27B-classifier"
DEMO_RATE_LIMIT_RPS = 4.0

QUESTION = {
    "type": "noul",
    "instructions": "Does the candidate protein bind the target protein?",
    "criteria": {
        "true": "The candidate binds the target protein.",
        "false": "The candidate does not bind the target protein.",
    },
}

# The API rejects requests whose User-Agent looks like a bare stdlib client.
HEADERS = {"Content-Type": "application/json", "User-Agent": "zsbinders/0.1"}

NON_FEATURE_COLUMNS = {"uuid", "y_true"}


class RateLimiter:
    """Token-bucket limiter shared across worker threads."""

    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            time.sleep(wait)


@dataclass
class JevClient:
    model: str = DEFAULT_MODEL
    api_url: str = DEMO_API_URL
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 4

    def build_request(self, row: pd.Series) -> dict[str, Any]:
        """Request body for one design: every feature column becomes a state field."""
        state = {k: v for k, v in row.items() if k not in NON_FEATURE_COLUMNS}
        return {"model": self.model, "state": state, "questions": {"binding": QUESTION}}

    def score_one(self, row: pd.Series, session: requests.Session) -> dict[str, Any]:
        """p(binder) for one design. Failures are returned, not raised, so a sweep continues."""
        payload = self.build_request(row)
        headers = dict(HEADERS)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            try:
                response = session.post(
                    self.api_url, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                p_binder = response.json()["answers"]["binding"]["noul"]
                return {
                    "uuid": row["uuid"],
                    "p_binder": p_binder,
                    "y_pred": "binder" if p_binder >= 0.5 else "non_binder",
                    "error": None,
                }
            except requests.HTTPError as exc:
                retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
                if not retryable or attempt == self.max_retries:
                    return _failure(row, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}")
                time.sleep(float(exc.response.headers.get("Retry-After", 2.0**attempt)))
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    return _failure(row, str(exc))
                time.sleep(2.0**attempt)
        return _failure(row, "exhausted retries")


def _failure(row: pd.Series, message: str) -> dict[str, Any]:
    return {"uuid": row["uuid"], "p_binder": None, "y_pred": None, "error": message}


def score_dataframe(
    df: pd.DataFrame,
    client: JevClient,
    rps: float = DEMO_RATE_LIMIT_RPS,
    workers: int = 8,
    progress_every: int = 50,
    already_scored: Iterable[str] = (),
) -> pd.DataFrame:
    """Score every design, returning the input frame plus `p_binder`, `y_pred`, `error`.

    Requests overlap across `workers` threads while the shared limiter holds the aggregate
    rate at `rps`, so throughput is set by the rate limit rather than by per-call latency.
    """
    pending = df[~df["uuid"].isin(set(already_scored))].reset_index(drop=True)
    limiter = RateLimiter(rps)
    done = 0
    lock = threading.Lock()

    def task(row: pd.Series) -> dict[str, Any]:
        nonlocal done
        limiter.acquire()
        with requests.Session() as session:
            result = client.score_one(row, session)
        with lock:
            done += 1
            if done % progress_every == 0 or done == len(pending):
                print(f"  scored {done}/{len(pending)}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, (row for _, row in pending.iterrows())))

    return pending.merge(pd.DataFrame(results), on="uuid", how="left")


def list_models(models_url: str = DEMO_MODELS_URL) -> list[str]:
    response = requests.get(models_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]
