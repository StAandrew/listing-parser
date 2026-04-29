"""Bedrock-backed runners.

Two providers today, both going through the same Converse-API client:

* `BedrockHaikuRunner` — Haiku 4.5, the teacher baseline. What
  downstream runs are compared against. Uses `cachePoint` on the
  system block so the 5.5k-token prompt is billed at cache-read rates
  after the first call in a run.
* `BedrockLlamaRunner`  — Llama 3.1 8B Instruct, the student base
  model baseline. Matches what a fine-tune will be served as via
  Bedrock Custom Model Import. Does NOT use `cachePoint` — Meta
  models on Bedrock reject the directive with ValidationException.

Both reuse `labelling.teacher.call_teacher` for connection, throttle
backoff, and semaphore-bounded concurrency; the shared base in this
module adds only the runner-facing concerns (parse-and-retry, prompt
assembly, `RunnerResult` shaping).

When a third Bedrock provider lands (e.g. the merged fine-tune through
Bedrock CMI) it should subclass `_BedrockRunnerBase` with its own
`model_id` default, temperature, and `use_cache` flag; no other
changes should be needed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from listing_parser.labelling.bedrock_invoke import call_imported
from listing_parser.labelling.teacher import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    call_teacher,
)
from listing_parser.prompting import build_system_prompt, build_user_message
from listing_parser.runners._output import ParseAttempt, parse_output_json, parse_with_retry
from listing_parser.runners.base import ListingType, RunnerResult


class _BedrockRunnerBase:
    """Shared plumbing for any Bedrock-backed runner.

    Subclasses only need to set class-level defaults (model id,
    temperature, cache flag) and optionally override the runner
    `name`. The async `predict` method is identical across providers —
    the Haiku vs Llama distinction lives entirely in constructor
    defaults plus the `use_cache` wire-level toggle.

    Attributes populated from kwargs (all overridable per instance):
      * `_model_id`   Bedrock model id or inference profile
      * `_region`     AWS region
      * `_profile`    AWS profile (from env by default)
      * `_temperature` / `_max_tokens` inference config
      * `_use_cache`  whether to wrap the system block in a cachePoint

    Cached state:
      * `_system_prompt`  rendered once per instance (~5.5k tokens;
                          file reads + vocab-block rendering aren't
                          free and the output is deterministic)
      * `_semaphore`      built lazily on first predict() call so the
                          runner can be constructed outside an event
                          loop without tripping "no current loop"
    """

    # --- defaults; subclasses override ------------------------------------
    _DEFAULT_NAME: str = "bedrock"
    _DEFAULT_MODEL_ID: str = DEFAULT_MODEL_ID
    _DEFAULT_REGION: str = DEFAULT_REGION
    _DEFAULT_TEMPERATURE: float = 0.0
    _DEFAULT_USE_CACHE: bool = False
    # Dispatch flag: Converse for foundation models (Anthropic, Meta
    # on-demand); InvokeModel for CMI-imported custom models. CMI
    # imports flip this in the subclass. The two protocols are wire-
    # level incompatible — Converse takes structured messages + an
    # optional cachePoint, InvokeModel takes a pre-rendered prompt
    # string and the provider's native response format — so this
    # isn't just a convenience flag, it selects between two mutually-
    # exclusive code paths.
    _USES_INVOKE_MODEL: bool = False

    def __init__(
        self,
        *,
        name: str | None = None,
        profile: str | None = None,
        region: str | None = None,
        model_id: str | None = None,
        concurrency: int = 4,
        max_tokens: int = 2048,
        temperature: float | None = None,
        use_cache: bool | None = None,
        max_parse_retries: int = 1,
        on_throttle: Any = None,
    ) -> None:
        self.name = name or self._DEFAULT_NAME
        self._profile = profile
        self._region = region or self._DEFAULT_REGION
        self._model_id = model_id or self._DEFAULT_MODEL_ID
        self._max_tokens = max_tokens
        self._temperature = (
            self._DEFAULT_TEMPERATURE if temperature is None else temperature
        )
        self._use_cache = (
            self._DEFAULT_USE_CACHE if use_cache is None else use_cache
        )
        self._max_parse_retries = max_parse_retries
        self._on_throttle = on_throttle
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_size = concurrency
        self._system_prompt: str | None = None

    # --- lazy getters -----------------------------------------------------

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

    # --- the Runner Protocol method --------------------------------------

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
            # to the teacher client. Usage counters are captured per
            # call and merged into the last ParseAttempt we return.
            # Dispatch on protocol: Converse (default) vs InvokeModel
            # (CMI-imported models). The two low-level functions return
            # the same `TeacherResponse` shape, so downstream parsing
            # and retry logic is protocol-agnostic.
            if self._USES_INVOKE_MODEL:
                resp = await call_imported(
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
            else:
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
                    use_cache=self._use_cache,
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


class BedrockHaikuRunner(_BedrockRunnerBase):
    """Runner that calls Haiku 4.5 on Bedrock (the teacher model).

    Baseline for everything else: scoring a fine-tune against the
    teacher tells us whether we've caught up to the training signal
    we paid for. The teacher's own number is not 1.0 on the gold set —
    the gold was labelled at temperature 0.2 and we re-infer with the
    same temperature, so sampling noise alone drops it a few percent.
    That's fine; it's a meaningful ceiling, not an artefact.

    Defaults:
      * model = `global.anthropic.claude-haiku-4-5-20251001-v1:0`
        (the only inference profile supporting prompt caching).
      * temperature = 0.2 — matches labelling so teacher-vs-gold is a
        like-for-like comparison.
      * use_cache = True — the Converse API accepts `cachePoint` for
        Anthropic models.

    Concurrency default is 4; Bedrock throttles kick in around 6–8 on
    default quotas.
    """

    _DEFAULT_NAME = "haiku-4.5-teacher"
    _DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
    _DEFAULT_REGION = DEFAULT_REGION  # eu-west-2
    _DEFAULT_TEMPERATURE = 0.2
    _DEFAULT_USE_CACHE = True


class BedrockLlamaRunner(_BedrockRunnerBase):
    """Runner that calls Llama 3.1 8B Instruct on Bedrock — the student
    base-model baseline.

    Why this is the right baseline (rather than Ollama locally):
      * The fine-tune target is Llama 3.1 8B served at fp16/bf16 via
        Bedrock Custom Model Import. A Q4_K_M Ollama baseline would
        under-report the base model's capability and make the
        fine-tune look artificially better.
      * Bedrock model ids are versioned and stable, so numbers in
        `report.md` are reproducible 6 months from now — an HF GGUF
        digest is not, in practice.

    Defaults:
      * model = `meta.llama3-1-8b-instruct-v1:0` (ON_DEMAND in
        us-west-2; switch to `us.meta.llama3-1-8b-instruct-v1:0` if
        quota routing becomes useful).
      * region = `us-west-2` — where the model's ON_DEMAND inference
        is available. The Haiku teacher runs in eu-west-2; these two
        runs therefore hit different regions, which is fine — neither
        the scorer nor predictions.jsonl cares.
      * temperature = 0.0 — honest greedy decoding gives the most
        reproducible "what does this base model actually produce"
        signal. Re-running should reproduce.
      * use_cache = False — Bedrock's Llama integration returns
        ValidationException on `cachePoint`. With our ~5.5k-token
        system prompt + 60 rows that's roughly 330k input tokens
        non-cached, or about $0.05 at Llama 3.1 8B on-demand pricing.
        Cheap enough to not bother.

    Expectations, so there are no surprises:
      * parse_rate will likely be 50-85%. Llama 3.1 8B follows closed-
        vocab JSON instructions much worse than Haiku. That's the gap
        fine-tuning is meant to close; don't inflate it with extra
        parse retries.
      * macro_accuracy will be dominated by "field absent in gold,
        absent in prediction" agreement, masking the real regression.
        The per-field breakdown in `report.md` is what matters.
    """

    _DEFAULT_NAME = "llama-3.1-8b-base"
    _DEFAULT_MODEL_ID = "meta.llama3-1-8b-instruct-v1:0"
    _DEFAULT_REGION = "us-west-2"
    _DEFAULT_TEMPERATURE = 0.0
    _DEFAULT_USE_CACHE = False


class BedrockFineTuneRunner(_BedrockRunnerBase):
    """Runner for a Bedrock-CMI-hosted fine-tune of Llama 3.1 8B.

    CMI imports produce an account-specific `imported-model/<id>` ARN
    rather than a shared, human-readable model id. Two consequences:

      * `_DEFAULT_MODEL_ID = None` — the ARN MUST be passed explicitly.
        Defaulting it would silently bind the runner to whichever ARN
        was current when this module was written, which would go stale
        the next time you reimport.
      * `_DEFAULT_NAME` is the slug of the underlying fine-tune (e.g.
        "llama-3.1-8b-ft-v1-full"); the on-disk report directory
        takes this slug.

    Other defaults deliberately mirror `BedrockLlamaRunner`:
      * region = `eu-central-1` — where the CMI import lives. Frankfurt
        rather than London because CMI isn't offered in eu-west-2 at
        time of writing. Override with `--region` if that changes.
      * temperature = 0.0 — greedy decode; the fine-tune was trained
        at temp 0 during SFT, so matching at inference is honest.
      * use_cache = False — CMI-served Llama rejects the `cachePoint`
        directive, same as vanilla Bedrock Llama.

    Operational gotchas that bite on CMI (not on foundation models):
      * Cold start: 60-120s on the first call after >5 min idle. The
        runner's existing throttle backoff won't hide this — the first
        scoring row in a run just takes a long time.
      * On-demand quota: default is low (~1 req/s). At `--concurrency
        2`+ you'll throttle constantly. Start with `--concurrency 1`
        unless you've got a quota increase from AWS service quotas.
    """

    _DEFAULT_NAME = "llama-3.1-8b-ft"
    _DEFAULT_MODEL_ID = None  # must be passed explicitly
    _DEFAULT_REGION = "eu-central-1"
    _DEFAULT_TEMPERATURE = 0.0
    _DEFAULT_USE_CACHE = False
    _USES_INVOKE_MODEL = True  # CMI rejects Converse; InvokeModel only

    def __init__(self, *, model_id: str | None = None, **kwargs) -> None:
        if not model_id:
            raise ValueError(
                "BedrockFineTuneRunner requires --model-id "
                "(the CMI imported-model ARN, e.g. "
                "arn:aws:bedrock:eu-central-1:ACCOUNT:imported-model/abc)"
            )
        super().__init__(model_id=model_id, **kwargs)
