"""Bedrock InvokeModel client for CMI-imported Llama models.

Custom Model Import (CMI) doesn't support the Converse API — it's
InvokeModel-only, with a provider-specific prompt format. For a
Llama 3.1 8B import, that means:

  * Bedrock expects the full Llama-3.1 chat template rendered as a
    plain string (`<|begin_of_text|>...<|eot_id|>` tokens) in the
    request's `prompt` field. There's no `system`/`messages`
    structure — we build it ourselves.
  * Response shape is `{"generation": "...", "prompt_token_count":
    N, "generation_token_count": M, "stop_reason": "..."}`. No
    `output.message.content` wrapping.
  * Prompt caching directives aren't available — CMI bills every
    token on every call. Our ~5.5k-token system prompt is resent in
    full each time, which is fine for eval (60 rows) but would be
    expensive for sustained production traffic. See
    `docs/serving-economics.md` or the CLAUDE.md cost notes.

Why this lives in `labelling/` rather than `runners/`
----------------------------------------------------
It's a sibling of `teacher.py` — both own a low-level Bedrock
protocol for the runner layer to delegate to. Putting them next to
each other keeps "Bedrock wire protocols" in one place so a future
third protocol (e.g. Bedrock Invoke for Anthropic on-demand) has an
obvious home.

We return the same `TeacherResponse` dataclass defined in
`teacher.py` so the runner layer's dispatch is a one-line switch
rather than a shape adapter.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from listing_parser.labelling.teacher import (
    TeacherResponse,
    _bedrock_client,
    _Throttled,
)

# Region for CMI-imported Llama. `eu-west-2` (London) doesn't offer
# Bedrock CMI at time of writing; Frankfurt is the nearest region
# with CMI support. Caller can override, same as `teacher.py`.
DEFAULT_IMPORT_REGION = "eu-central-1"


# ---------------------------------------------------------------------------
# Llama-3.1 chat template rendering.
#
# We render the template manually rather than calling
# `transformers.AutoTokenizer.apply_chat_template` because:
#   * The runner shouldn't pull in transformers for inference — it's
#     a 2GB dependency for a string formatting operation.
#   * The Llama-3.1 template is stable and well-documented. Hand-
#     rendering keeps the dependency story honest.
#
# This MUST match how we rendered the prompt at fine-tune time. The
# training notebook patched `{% generation %}` markers into the chat
# template but the surrounding scaffold is identical to the one here.
# ---------------------------------------------------------------------------


def _render_llama31_prompt(system_prompt: str, user_turns: tuple[str, ...]) -> str:
    """Render a system + user-turns conversation in Llama 3.1 chat format.

    The assistant hasn't responded yet; we append the assistant-start
    header so the model knows to begin its turn.

    Multi-turn user-only retries (the correction prompt path) are
    rendered as consecutive user messages with no intervening
    assistant turn. Llama tolerates that; it sees back-to-back user
    headers as "the user clarified".
    """
    parts = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>\n\n",
        system_prompt,
        "<|eot_id|>",
    ]
    for turn in user_turns:
        parts.extend([
            "<|start_header_id|>user<|end_header_id|>\n\n",
            turn,
            "<|eot_id|>",
        ])
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Core call.
# ---------------------------------------------------------------------------


def _sync_invoke(
    client: Any,
    model_id: str,
    system_prompt: str,
    user_turns: tuple[str, ...],
    max_tokens: int,
    temperature: float,
) -> TeacherResponse:
    """One synchronous InvokeModel call. Mirrors `_sync_converse`'s
    role in `teacher.py` — no retries here; the async wrapper handles
    throttle backoff.

    Raises `_Throttled` on ThrottlingException so the wrapper knows
    to back off. Any other `ClientError` propagates.
    """
    from botocore.exceptions import ClientError  # noqa: PLC0415

    started = time.monotonic()
    prompt = _render_llama31_prompt(system_prompt, user_turns)

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
    })

    try:
        resp = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("ThrottlingException", "TooManyRequestsException"):
            raise _Throttled() from e
        # CMI has its own cold-start error (`ModelNotReadyException`)
        # that deserves a retry rather than a hard failure. Treat it
        # as a throttle so the existing backoff picks it up.
        if code == "ModelNotReadyException":
            raise _Throttled() from e
        raise

    elapsed_ms = int((time.monotonic() - started) * 1000)
    # Bedrock returns a StreamingBody; .read() yields bytes; then JSON.
    payload = json.loads(resp["body"].read())

    # `generation` is the only text field — no content blocks to merge.
    text = payload.get("generation", "")
    return TeacherResponse(
        text=text,
        input_tokens=int(payload.get("prompt_token_count", 0) or 0),
        output_tokens=int(payload.get("generation_token_count", 0) or 0),
        # CMI doesn't support prompt caching, so these are always 0.
        # Kept in the dataclass shape so the runner-level code can be
        # schema-identical across Converse and InvokeModel paths.
        cache_read_tokens=0,
        cache_write_tokens=0,
        latency_ms=elapsed_ms,
    )


async def call_imported(
    system_prompt: str,
    user_turns: tuple[str, ...],
    *,
    semaphore: asyncio.Semaphore,
    model_id: str,
    profile: str | None = None,
    region: str = DEFAULT_IMPORT_REGION,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_backoff_attempts: int = 6,
    on_throttle: Callable[[], Awaitable[None]] | None = None,
) -> TeacherResponse:
    """Async, semaphore-bounded, backoff-retried Bedrock InvokeModel call.

    Interface mirrors `teacher.call_teacher` exactly so the runner
    layer's dispatch between Converse and InvokeModel is a one-line
    branch. Differences:

      * `model_id` is required (no default) — CMI ARNs are account-
        specific; there's no sensible fallback.
      * Backoff catches both `ThrottlingException` AND
        `ModelNotReadyException`. The latter fires on cold start and
        is resolved by waiting — the first call of a run after >5min
        idle can surface one or two `ModelNotReadyException`s before
        the CMU copy is warm, which isn't a failure mode the user
        should see.
      * No `use_cache` parameter — CMI doesn't support caching.
    """
    client = _bedrock_client(profile, region)

    async with semaphore:
        for attempt in range(max_backoff_attempts):
            try:
                return await asyncio.to_thread(
                    _sync_invoke,
                    client,
                    model_id,
                    system_prompt,
                    user_turns,
                    max_tokens,
                    temperature,
                )
            except _Throttled:
                if on_throttle is not None:
                    await on_throttle()
                # Exponential backoff with full jitter. Same shape as
                # `teacher.call_teacher`, but we start from a longer
                # base (1.0s vs 0.5s) because CMI cold-start warmups
                # want a more patient first retry — spamming a model
                # that's still loading wastes retry budget.
                sleep_s = random.uniform(0, min(60.0, 1.0 * (2**attempt)))
                await asyncio.sleep(sleep_s)
        raise RuntimeError(
            f"gave up after {max_backoff_attempts} throttles/cold-starts "
            f"calling {model_id}"
        )
