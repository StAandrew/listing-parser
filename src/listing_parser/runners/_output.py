"""Shared output handling for runners: parse-and-retry, error formatting.

Every runner eventually has to go from "raw model text" to "structured
JSON or an error". The rules are the same across providers:

* Strip ```json ... ``` fences and tolerate trailing prose (some models
  append "Hope this helps!" after the closing brace).
* If parsing fails, build a correction user-turn and retry up to
  `max_parse_retries` times.
* Do NOT retry on schema-validity failure. The scorer's `schema_rate`
  metric exists to catch this — retrying would hide regressions that
  a production inference stack would have to live with.

Runners import `parse_with_retry()` and hand it a provider-specific
`call()` coroutine. The helper owns the retry bookkeeping; the caller
owns the LLM connection.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from listing_parser.cleaning import parse_output_json
from listing_parser.prompting import build_correction_message


@dataclass(frozen=True)
class ParseAttempt:
    """One call's result. Internal to `parse_with_retry`."""

    raw: str
    pred: dict[str, Any] | None
    usage: dict[str, Any] | None


async def parse_with_retry(
    call: Callable[[tuple[str, ...]], Awaitable[ParseAttempt]],
    user_message: str,
    *,
    max_parse_retries: int = 1,
) -> tuple[dict[str, Any] | None, str, str | None, dict[str, Any] | None]:
    """Drive a runner's LLM call with one-shot parse-failure retry.

    `call(turns)` should:
      * invoke the provider with the given user turns + the runner's
        system prompt;
      * return a `ParseAttempt` with the raw response and an attempted
        parse of it (runners generally share `cleaning.parse_output_json`
        via this module's convenience re-export, but they can pre-parse
        in whatever way makes sense for their wire format).

    Returns `(pred, raw, error, usage)` — the raw/usage fields are taken
    from the LAST attempt so the on-disk record matches what the scorer
    saw, not a discarded intermediate.

    `max_parse_retries=1` is the default because Haiku's empirical parse
    failure rate is <1%, most of those recover on the first retry, and
    each retry roughly doubles cost. Set to 0 to disable retry and
    measure the model's first-pass rate honestly.
    """
    turns: tuple[str, ...] = (user_message,)
    last_raw = ""
    last_usage: dict[str, Any] | None = None

    for attempt in range(max_parse_retries + 1):
        try:
            result = await call(turns)
        except Exception as e:
            # Non-retryable runtime error — surface and stop trying.
            # The caller gets raw="" rather than an out-of-date retry,
            # which is the honest signal "we never got text back".
            return None, last_raw, f"{type(e).__name__}: {e}", last_usage

        last_raw = result.raw
        last_usage = result.usage
        if result.pred is not None:
            return result.pred, result.raw, None, result.usage

        if attempt < max_parse_retries:
            turns = (
                user_message,
                build_correction_message(
                    result.raw,
                    "model output was not a parseable JSON object",
                ),
            )
            continue

    return None, last_raw, "parse failed after retries", last_usage


# Re-export so runners only have to import from this module for output handling.
# Keeping the import local rather than `from cleaning import *` so the public
# surface of this helper module stays obvious.
__all__ = [
    "ParseAttempt",
    "parse_output_json",
    "parse_with_retry",
]
