"""Bedrock-backed runners — today that's Haiku 4.5 (the teacher).

We reuse `listing_parser.labelling.teacher` verbatim: the Converse-API
client, the semaphore-guarded async wrapper, the throttle backoff, and
the 1h prompt-caching directive all apply identically in the runner
context. The one functional difference is what we do with the model's
output:

    labelling → clean + validate + retry on either failure
    runners   → parse only; no cleaning, no schema retry

Cleaning would lie to the scorer about what the model produced;
schema-retry would mask exactly the regression `schema_rate` is there
to detect. So the runner is deliberately thinner than the labeller.

A future "Bedrock base-model Llama" runner will share this module;
we'll factor a `_BedrockRunnerBase` out when that second caller lands
rather than speculating on the abstraction now.
"""

from __future__ import annotations

import asyncio
from typing import Any

from listing_parser.labelling.teacher import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    call_teacher,
)
from listing_parser.prompting import build_system_prompt, build_user_message
from listing_parser.runners._output import ParseAttempt, parse_output_json, parse_with_retry
from listing_parser.runners.base import ListingType, RunnerResult


class BedrockHaikuRunner:
    """Runner that calls Haiku 4.5 on Bedrock (the teacher model).

    Baseline for everything else: scoring a fine-tune against the
    teacher tells us whether we've caught up to the training signal
    we paid for. The teacher's own number is not 1.0 on the gold set —
    the gold was labelled at temperature 0.2 and we re-infer with the
    same temperature, so sampling noise alone drops it a few percent.
    That's fine; it's a meaningful ceiling, not an artefact.

    Concurrency is bounded by the semaphore we create at construction
    time; pass `--concurrency` to the CLI to control it. We default
    to 4 which empirically avoids Bedrock throttles on default quotas.
    """

    name: str

    def __init__(
        self,
        *,
        name: str = "haiku-4.5-teacher",
        profile: str | None = None,
        region: str = DEFAULT_REGION,
        model_id: str = DEFAULT_MODEL_ID,
        concurrency: int = 4,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_parse_retries: int = 1,
        on_throttle: Any = None,
    ) -> None:
        self.name = name
        self._profile = profile
        self._region = region
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_parse_retries = max_parse_retries
        self._on_throttle = on_throttle
        # Built lazily so tests that never call `predict` don't need to
        # import asyncio loops.
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_size = concurrency
        # Cache the system prompt once per runner instance — it's ~5.5k
        # tokens and `build_system_prompt` reads files + renders vocab
        # blocks. Bedrock still caches across the wire via `cachePoint`;
        # this cache is just in-process so we don't re-render per call.
        self._system_prompt: str | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        # Defer construction — `asyncio.Semaphore` binds to the current
        # event loop at creation time, and creating it in `__init__`
        # would bind to whatever loop happens to be current then (often
        # none, raising RuntimeError on first use).
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._semaphore_size)
        return self._semaphore

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt()
        return self._system_prompt

    async def predict(
        self,
        description: str,
        listing_type: ListingType,
    ) -> RunnerResult:
        user_msg = build_user_message(description, listing_type)
        system_prompt = self._get_system_prompt()
        semaphore = self._get_semaphore()

        async def _call(turns: tuple[str, ...]) -> ParseAttempt:
            # Hand the current turns (first attempt or correction retry)
            # to the teacher client. Usage counters are captured per-
            # call and merged into the last ParseAttempt we return.
            resp = await call_teacher(
                system_prompt,
                turns,
                semaphore=semaphore,
                profile=self._profile,
                region=self._region,
                model_id=self._model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                on_throttle=self._on_throttle,
            )
            return ParseAttempt(
                raw=resp.text,
                pred=parse_output_json(resp.text),
                usage={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cache_read_tokens": resp.cache_read_tokens,
                    "cache_write_tokens": resp.cache_write_tokens,
                    "latency_ms": resp.latency_ms,
                },
            )

        pred, raw, error, usage = await parse_with_retry(
            _call,
            user_msg,
            max_parse_retries=self._max_parse_retries,
        )
        return RunnerResult(pred=pred, raw=raw, error=error, usage=usage)
