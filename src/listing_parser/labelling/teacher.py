"""Bedrock teacher client — async-friendly wrapper around the Converse API.

The pipeline calls this once or twice per row (first attempt + optional
correction retry). A single module owns:

  * the boto3 client lifecycle;
  * prompt-caching directives on the system message (expensive to send
    fresh every time given ~11k tokens of closed-vocab reinforcement);
  * exponential backoff on ThrottlingException;
  * a semaphore that bounds concurrency across the whole run.

We expose a *function*, not a class, because the pipeline treats the
teacher as an uninterpreted "string in, string out" operation and we
don't need per-instance state.

Concurrency model
-----------------
`boto3` is synchronous. We get asyncio friendliness via
`asyncio.to_thread(...)` — each call parks on a worker thread, which
is fine for I/O-bound Bedrock requests. The semaphore gates how many
threads can be in flight at once; the bound matches the `--concurrency`
CLI flag and defaults to 8.

We do NOT use `aioboto3`. It'd be more "correct" but would add a
dependency and the synchronous SDK + thread-pool is measurably faster
up to ~32 concurrent requests.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Awaitable, Callable

DEFAULT_REGION = "eu-west-2"
# Inference profile selection matters: the `eu.*` profile does NOT support
# prompt caching (confirmed 2026-04 against Haiku 4.5 in eu-west-2), while
# `global.*` does. Caching is the difference between ~35M and ~3.5M input
# tokens over a 3.2k-row run so we default to `global.*`. If a workload
# strictly must keep data within EU regions, override via --model-id at
# the cost of significantly higher token bills.
DEFAULT_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


# ---------------------------------------------------------------------------
# Client construction. Cached per (profile, region) because boto3 clients
# are thread-safe and relatively expensive to build.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _bedrock_client(profile: str | None, region: str):
    """Build (or reuse) a bedrock-runtime client for the given profile+region.

    Separate from the Bedrock control-plane client; we only need Converse,
    which lives on `bedrock-runtime`.
    """
    import boto3  # noqa: PLC0415 — deferred so tests that don't touch the
                  # teacher don't pay the boto3 import cost.

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("bedrock-runtime", region_name=region)


# ---------------------------------------------------------------------------
# Message shape helpers.
#
# Bedrock's Converse API takes a `system` field (list of content blocks)
# and a `messages` field (same). We enable prompt caching on the system
# message so the 11k-token closed-vocab prompt is only billed as
# "cacheWriteInputTokens" once and as "cacheReadInputTokens" (~10% the
# cost) on every subsequent call in the session.
# ---------------------------------------------------------------------------


def _system_with_cache(system_prompt: str) -> list[dict[str, Any]]:
    """Wrap the system prompt as a cached content block.

    `cachePoint` tells Bedrock "cache everything up to this marker".
    The marker is positional; we put it after the full system text so
    the whole thing is cached. The 1-hour TTL is the longest Bedrock
    supports for Haiku 4.5 and covers essentially any labelling run we
    care about (the slowest 3.2k-row run is <30 min even at N=4
    concurrency).
    """
    return [
        {"text": system_prompt},
        {"cachePoint": {"type": "default", "ttl": "1h"}},
    ]


def _system_plain(system_prompt: str) -> list[dict[str, Any]]:
    """Plain (non-cached) system block.

    Bedrock Llama models don't accept `cachePoint` — the Converse API
    rejects the directive with a ValidationException. This is the
    fallback shape for providers without prompt caching.
    """
    return [{"text": system_prompt}]


def _user_messages(*user_turns: str) -> list[dict[str, Any]]:
    """Build the `messages` list from one or more user turns.

    We only ever send user-role turns from this client — the assistant's
    response on retry is baked into our correction prompt rather than
    threaded into the conversation, because Bedrock's cache eviction
    rules around assistant messages are fiddly and we don't benefit
    from the assistant context anyway.
    """
    return [
        {"role": "user", "content": [{"text": turn}]} for turn in user_turns
    ]


# ---------------------------------------------------------------------------
# Core call.
# ---------------------------------------------------------------------------


@dataclass
class TeacherResponse:
    """What the labelling pipeline needs from one Bedrock call."""

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: int


class _Throttled(Exception):
    """Internal marker for the retry loop — caller never sees it."""


def _sync_converse(
    client: Any,
    model_id: str,
    system_prompt: str,
    user_turns: tuple[str, ...],
    max_tokens: int,
    temperature: float,
    use_cache: bool,
) -> TeacherResponse:
    """One synchronous Bedrock call. No retries here — that's the job of
    the async wrapper so it can share the semaphore with sleeping tasks.

    Raises `_Throttled` on ThrottlingException so the wrapper knows to
    back off; everything else propagates.

    `use_cache` toggles the `cachePoint` directive on the system block.
    Anthropic + Nova models support it; Meta Llama does not and returns
    ValidationException if the directive is present. Callers pass
    `use_cache=False` for Llama; default-True preserves the original
    labelling behaviour.
    """
    from botocore.exceptions import ClientError  # noqa: PLC0415

    started = time.monotonic()
    system_block = _system_with_cache(system_prompt) if use_cache else _system_plain(
        system_prompt
    )
    try:
        resp = client.converse(
            modelId=model_id,
            system=system_block,
            messages=_user_messages(*user_turns),
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("ThrottlingException", "TooManyRequestsException"):
            raise _Throttled() from e
        raise

    elapsed_ms = int((time.monotonic() - started) * 1000)
    # `content` is a list of blocks; for a non-tool-using response it's
    # one `{"text": ...}` block. Concatenate defensively in case Bedrock
    # ever returns multiple blocks.
    blocks = resp["output"]["message"]["content"]
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))

    usage = resp.get("usage", {})
    return TeacherResponse(
        text=text,
        input_tokens=usage.get("inputTokens", 0),
        output_tokens=usage.get("outputTokens", 0),
        cache_read_tokens=usage.get("cacheReadInputTokens", 0),
        cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
        latency_ms=elapsed_ms,
    )


async def call_teacher(
    system_prompt: str,
    user_turns: tuple[str, ...],
    *,
    semaphore: asyncio.Semaphore,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    max_backoff_attempts: int = 6,
    on_throttle: Callable[[], Awaitable[None]] | None = None,
    use_cache: bool = True,
) -> TeacherResponse:
    """Async, semaphore-bounded, backoff-retried Bedrock call.

    The semaphore is passed in (rather than owned by this module) so
    the pipeline can share one across many concurrent tasks. That way
    "--concurrency 8" really means "8 Bedrock calls in flight", not
    "8 per teacher instance".

    Backoff: exponential with jitter. We cap at `max_backoff_attempts`
    throttles so a persistent quota problem surfaces quickly rather
    than silently holding a row for minutes.

    If `on_throttle` is provided, it's awaited each time a
    ThrottlingException is caught (before backing off). Lets the caller
    surface throttle pressure in the UI without this module having to
    know about the progress reporter.

    `use_cache` wraps the system block in a `cachePoint` directive. On
    for Anthropic / Nova (the default, matching the teacher's original
    behaviour); off for Meta Llama which rejects the directive.
    """
    client = _bedrock_client(profile, region)

    async with semaphore:
        for attempt in range(max_backoff_attempts):
            try:
                return await asyncio.to_thread(
                    _sync_converse,
                    client,
                    model_id,
                    system_prompt,
                    user_turns,
                    max_tokens,
                    temperature,
                    use_cache,
                )
            except _Throttled:
                if on_throttle is not None:
                    await on_throttle()
                # Exponential backoff with a full-jitter scheme: sleep a
                # random value in [0, base * 2**attempt]. Matches AWS's
                # own recommendation for retry-with-jitter.
                sleep_s = random.uniform(0, min(60.0, 0.5 * (2**attempt)))
                await asyncio.sleep(sleep_s)
        raise RuntimeError(
            f"gave up after {max_backoff_attempts} throttles calling {model_id}"
        )
