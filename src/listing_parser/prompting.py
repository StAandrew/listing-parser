"""Compose the prompts sent to the teacher at label-writing time.

The canonical system prompt and output schema live in `prompt.md` at
the repo root so they can be read and edited by humans. This module
loads that markdown, extracts the fenced sections the model actually
needs, and renders the final strings in the shape the teacher API
expects.

Three entry points:

  * `build_system_prompt()` — the system message. Combines the
    human-authored rules block from `prompt.md` with a
    machine-authored "closed-vocabulary reinforcement" section that
    lists every controlled vocab as a JSON array. The reinforcement
    block lives in code (not `prompt.md`) because the vocabs live in
    `schema.py` — having the prompt regenerated from `schema.py` on
    every call means adding a new amenity in one place updates the
    prompt automatically, no manual sync.

  * `build_user_message(description, listing_type)` — the per-row
    message. Follows the `listing_type: X\n---\n<description>`
    convention documented at the bottom of `prompt.md`.

  * `build_correction_message(previous_raw, error)` — used by the
    retry path in the labelling pipeline. When the teacher's first
    attempt fails parsing or Pydantic validation, we feed the raw
    output and the error back in as a follow-up user message asking
    for a corrected JSON response.

All three functions are pure: same inputs → same output bytes. The
labelling pipeline relies on this for resumability (identical prompts
for the same row across runs) and the tests rely on it to assert the
prompt surface stays stable.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Literal

from listing_parser.schema import (
    AMENITIES_APPLIANCES,
    AMENITIES_FACILITIES,
    AMENITIES_INTERIOR,
    AMENITIES_OUTDOOR,
    BILLS_INCLUDED,
    COMPASS_POINTS,
    FURNISH_TYPES,
    LET_TYPES,
    PARKING_TYPES,
    PROPERTY_TYPES,
    PURCHASE_SCHEMES,
    ROOM_TYPES,
    TENURE_TYPES,
)

ListingType = Literal["Rent", "Sale", "Room"]

__all__ = [
    "ListingType",
    "build_correction_message",
    "build_system_prompt",
    "build_user_message",
    "load_prompt_markdown",
]


# Default search order for prompt.md — one of these must exist. The
# package is editable-installed so the repo root is always discoverable
# from the source tree; we fall back to the cwd for people running
# scripts from elsewhere.
_PROMPT_CANDIDATES = (
    Path(__file__).resolve().parent / "_assets" / "prompt.md",       # packaged
    Path(__file__).resolve().parents[2] / "prompt.md",                # repo root
    Path.cwd() / "prompt.md",                                          # fallback
)
_EXAMPLES_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "examples.json",
    Path.cwd() / "examples.json",
)


@cache
def load_prompt_markdown(path: Path | None = None) -> str:
    """Return the raw text of `prompt.md`.

    Cached because the labelling pipeline calls `build_system_prompt`
    once per row and re-reading the file each time is wasteful. Pass an
    explicit `path` to bypass the cache in tests.
    """
    if path is not None:
        return path.read_text(encoding="utf-8")
    for candidate in _PROMPT_CANDIDATES:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"could not find prompt.md in any of: {[str(p) for p in _PROMPT_CANDIDATES]}"
    )


# Fenced-block extractor. `prompt.md` has three top-level fenced blocks:
# the system rules, the JSON schema, and a user-message template. We
# pull them out by header + first-fence-after-header.
_FENCE_AFTER_HEADER_RE = re.compile(
    r"^##\s+(?P<title>.+?)\s*\n+```(?:\w+)?\n(?P<body>.*?)\n```",
    re.DOTALL | re.MULTILINE,
)


def _extract_fenced_sections(markdown: str) -> dict[str, str]:
    """Map each `## Section` to the text of the first fenced block below it.

    Returns the body of each block only (fences stripped). Sections
    without a fenced block right under them are omitted — this is fine
    for our use case where only "System prompt" and "Output schema"
    have fenced blocks directly below them.
    """
    sections: dict[str, str] = {}
    for m in _FENCE_AFTER_HEADER_RE.finditer(markdown):
        sections[m.group("title").strip()] = m.group("body")
    return sections


def _render_vocab_block() -> str:
    """Return a deterministic "closed-vocabulary reinforcement" block.

    The human-authored rules in `prompt.md` say "match the controlled
    vocabulary exactly". Haiku's smoke-test output showed it ignores
    that rule for amenities about 3% of the time ("Electric heating",
    "Oven", "Fully fitted kitchen"). Appending the vocabularies as
    explicit JSON arrays plus an unambiguous "discard-don't-invent"
    rule closes most of that gap without touching the prose.

    Regenerated from `schema.py` on every call, so adding a new amenity
    in one place propagates to the prompt automatically.
    """
    vocabs: dict[str, tuple[str, ...] | frozenset[str]] = {
        "property_type": PROPERTY_TYPES,
        "amenities_interior": AMENITIES_INTERIOR,
        "amenities_outdoor": AMENITIES_OUTDOOR,
        "amenities_facilities": AMENITIES_FACILITIES,
        "amenities_appliances": AMENITIES_APPLIANCES,
        "parking": PARKING_TYPES,
        "rent.furnish_type": FURNISH_TYPES,
        "rent.let_type": LET_TYPES,
        "rent.bills_included": BILLS_INCLUDED,
        "sale.tenure_type": TENURE_TYPES,
        "sale.purchase_scheme": PURCHASE_SCHEMES,
        "room.room_type": ROOM_TYPES,
        "window_orientation": tuple(sorted(COMPASS_POINTS)),
    }

    lines = [
        "## Closed-vocabulary enforcement",
        "",
        "Every field below has a fixed vocabulary. If you see a feature in the "
        "description that isn't in the relevant list, DROP it — do not invent "
        "a near-match, do not paraphrase. Extra or mistyped values are "
        "discarded by the downstream validator, so a wrong value is strictly "
        "worse than omitting the field.",
        "",
    ]
    for field, values in vocabs.items():
        lines.append(f"- `{field}`: {json.dumps(list(values), ensure_ascii=False)}")
    lines.extend([
        "",
        "## Listing-type gating",
        "",
        "The user message always starts with `listing_type: Rent | Sale | Room`. "
        "Use it to decide which sub-block(s) apply:",
        "",
        "- `listing_type: Rent` → populate `rent` only; do NOT emit `sale` or `room`.",
        "- `listing_type: Sale` → populate `sale` only; do NOT emit `rent` or `room`.",
        "- `listing_type: Room` → populate `room`; `rent` is OK if the listing "
        "specifies rent-adjacent facts (deposit, bills, furnishings), otherwise "
        "omit it. Do NOT emit `sale`.",
        "",
        "Cross-block keys (e.g. `room.room_type` on a Rent listing) are "
        "discarded by the validator.",
    ])
    return "\n".join(lines)


@cache
def _load_examples() -> list[dict]:
    """Return the worked examples from `examples.json`, or [] if missing.

    Cached because the labelling pipeline calls `build_system_prompt`
    once per row. Used to produce a few-shot block in the system
    prompt — both for quality (Haiku labels better with examples) and
    as intentional stable content that pushes the system prompt past
    Haiku 4.5's prompt-caching minimum (measured ~4096 tokens; below
    that the `cachePoint` directive is silently ignored).
    """
    for candidate in _EXAMPLES_CANDIDATES:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8")).get("examples", [])
    return []


def _render_few_shot_block() -> str:
    """Render the worked examples as a deterministic few-shot block.

    Each example is shown in the same shape the teacher will produce:
    user-message-style prefix, then the JSON answer. Keeps the prompt
    internally consistent — the model doesn't have to infer the output
    format from the schema alone.
    """
    examples = _load_examples()
    if not examples:
        return ""
    lines = [
        "## Worked examples",
        "",
        "Three reference labellings follow. Match their level of care: extract "
        "only what is stated, use the evidence map, omit keys for unmentioned "
        "facts.",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lt = ex.get("listing_type", "")
        desc = ex.get("input", "")
        out = ex.get("output", {})
        lines.extend([
            f"### Example {i} — listing_type: {lt}",
            "",
            "User message:",
            "",
            "```",
            f"listing_type: {lt}",
            "---",
            desc,
            "```",
            "",
            "Expected JSON output:",
            "",
            "```json",
            json.dumps(out, ensure_ascii=False, indent=2),
            "```",
            "",
        ])
    return "\n".join(lines)


def build_system_prompt(prompt_path: Path | None = None) -> str:
    """Compose the system message Haiku sees on every labelling call.

    Layout:

        <rules block from prompt.md>

        <output schema block from prompt.md>

        <machine-rendered closed-vocab reinforcement, from schema.py>

        <few-shot worked examples, from examples.json>

    The "What to leave out" bullets and the user-message template stay
    in `prompt.md` but aren't copied into the system message — the
    first because Haiku tends to include them verbatim in its output
    ("not extracting price"), the second because the user message is
    built separately by `build_user_message`.

    Total size is typically ~5.5k tokens, comfortably above Haiku 4.5's
    ~4k-token minimum for prompt caching. Keeping it that size is
    deliberate: the marginal token cost is negligible because the
    system block is cached across every row in a run.
    """
    markdown = load_prompt_markdown(prompt_path)
    sections = _extract_fenced_sections(markdown)

    rules = sections.get("System prompt", "").strip()
    schema = sections.get("Output schema", "").strip()
    if not rules or not schema:
        raise ValueError(
            "prompt.md is missing required sections: need fenced blocks under "
            "`## System prompt` and `## Output schema`"
        )

    parts = [
        rules,
        "## Output schema (reproduced from prompt.md)",
        "",
        "```jsonc",
        schema,
        "```",
        _render_vocab_block(),
    ]
    few_shot = _render_few_shot_block()
    if few_shot:
        parts.append(few_shot)
    return "\n\n".join(parts)


def build_user_message(description: str, listing_type: ListingType) -> str:
    """Per-row user message.

    Shape is fixed by the `prompt.md` template at the bottom of the
    file. Keeping this concrete rather than reading the template from
    `prompt.md` because the template is trivial (two lines) and
    extracting it as a third fenced block would add parser complexity
    for negligible benefit.
    """
    if listing_type not in ("Rent", "Sale", "Room"):
        raise ValueError(
            f"listing_type must be 'Rent', 'Sale', or 'Room'; got {listing_type!r}"
        )
    return f"listing_type: {listing_type}\n---\n{description}"


def build_correction_message(previous_raw: str, error: str) -> str:
    """User message for the retry loop.

    Called when the teacher's first attempt fails parsing or Pydantic
    validation. We re-send the original user message on retry (the
    labelling pipeline is responsible for that) and append this
    correction note as a second user turn, with the previous raw
    output quoted and the validator error surfaced verbatim.

    Keeping the error string raw (rather than summarising) because
    Pydantic's messages are already terse and Haiku is trained on
    them — compressing loses the "which field" signal.
    """
    return (
        "Your previous response was invalid. The extracted JSON failed "
        "validation with the following error:\n\n"
        f"{error.strip()}\n\n"
        "For reference, your previous response was:\n\n"
        f"{previous_raw.strip()}\n\n"
        "Please emit a corrected JSON object. Apply the original rules "
        "(omit unknown keys, stick to the controlled vocabulary, no "
        "prose or fences) and fix the specific issue called out above."
    )
