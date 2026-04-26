"""Student / teacher runners that turn a gold JSONL into predictions.

A runner owns the model-call side of benchmarking:

    gold JSONL  ──► runner.predict(description, listing_type)  ──► predictions.jsonl
                                  │
                                  └─ provider-specific: Bedrock / Ollama / local fine-tune

Runners share one contract — given a description + listing_type, they
return a `RunnerResult` with `{pred, raw, error}`. The shared driver in
`pipeline.py` reads the gold file, runs them under bounded concurrency,
and writes JSONL rows the scorer can consume.

The scorer is deliberately decoupled from all of this: it reads a
completed `predictions.jsonl` and has zero dependencies on how the
predictions were produced. That split lets us add new providers without
touching eval logic.
"""

from listing_parser.runners.base import Runner, RunnerResult
from listing_parser.runners.bedrock import BedrockHaikuRunner, BedrockLlamaRunner
from listing_parser.runners.pipeline import RunStats, run_predictions

__all__ = [
    "BedrockHaikuRunner",
    "BedrockLlamaRunner",
    "RunStats",
    "Runner",
    "RunnerResult",
    "run_predictions",
]
