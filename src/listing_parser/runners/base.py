"""Provider-agnostic runner protocol and the `RunnerResult` shape.

A `Runner` is any callable object that, given a description and a
listing_type, returns a `RunnerResult`. That's the entire contract —
everything else (retries, concurrency, disk I/O, progress) lives in
`pipeline.py` so each concrete provider stays minimal.

Why a Protocol + dataclass rather than a base class:

* The two teacher / student providers we care about in the near term
  are implemented very differently: `BedrockHaikuRunner` wraps boto3
  + prompt caching, a future `OllamaRunner` will open an HTTP stream
  to localhost. Forcing both into a `class BaseRunner` hierarchy would
  couple them more than the "same shape" they actually share.
* A Protocol keeps type-checking honest without the runtime ceremony
  (no MRO, no `super().__init__()`, no "inheritance means subsumption"
  promises we don't want to make).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

ListingType = Literal["Rent", "Sale", "Room"]


@dataclass(frozen=True)
class RunnerResult:
    """The atomic output of a single runner call.

    * `pred` — already-parsed JSON (the runner is responsible for fence
      stripping + `json.loads`). `None` when parsing failed; the scorer
      treats `None` predictions as `parse_ok=False` rather than crashing.
    * `raw` — the model's untouched text. Always written to disk so we
      can re-score with a stricter cleaner later without re-calling the
      model. Keep it even on success — it's cheap storage and priceless
      for post-mortem.
    * `error` — short human-readable string when something went wrong
      (parse failure, HTTP 500, auth). Optional; `None` on happy path.
    * `usage` — provider-specific token / latency / cache counters. Kept
      as an untyped dict because every provider surfaces a different
      shape and the scorer doesn't care. Useful for per-run cost
      reporting only.
    """

    pred: dict[str, Any] | None
    raw: str
    error: str | None = None
    usage: dict[str, Any] | None = None


class Runner(Protocol):
    """Provider-agnostic prediction interface.

    Async because every real-world provider benefits from concurrent
    in-flight requests (Bedrock with a semaphore, Ollama with an
    HTTP connection pool). Synchronous runners are free to wrap a
    thread-pool call; we don't try to accommodate them at the Protocol
    level because that would push asyncio glue into every implementer.
    """

    name: str
    """Short slug identifying the runner in reports / sidecars.

    Not used for routing — just for the report header and the on-disk
    log so a human can tell runs apart at a glance. E.g.
    `"haiku-4.5-teacher"`, `"ollama-llama31-8b-q6"`.
    """

    async def predict(
        self,
        description: str,
        listing_type: ListingType,
    ) -> RunnerResult:
        ...
