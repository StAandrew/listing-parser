"""End-to-end Haiku labelling pipeline.

Turns a stream of `(row_id, description, listing_type)` inputs into
cleaned, validated extractions on disk. Owns the orchestration; the
individual steps live in sibling modules:

    sources  -> read input listings
    teacher  -> call Bedrock/Claude
    sinks    -> write labelled rows + regenerate the index
    pipeline -> glue (this package's __init__ exports `run`)

The rest of the codebase treats `data/labelled/*.json` as the
authoritative label store; this package is how those files are produced.
"""

from listing_parser.labelling.pipeline import run

__all__ = ["run"]
