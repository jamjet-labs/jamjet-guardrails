"""The audit record every action writes to the rail context.

What goes in is provenance and location: which detector decided, at which
version, what it decided, which finding types it reported and where they were.
What never goes in is the content, in any form, on any path.

That is not a policy statement, it is the same rule `jamjet_guardrails.chain`
already enforces one layer down and for the same reason: a value a detector
CHOSE after seeing the content can BE the content. The chain bounds a finding's
type for that reason and refuses to interpolate anything else a detector
returned into a message. This module inherits the problem, because a rail
context is written into NeMo's event log, and an event log is forwarded,
persisted and rendered by things that were never told they were holding a
credential.

So the record carries:

- the digest of what was inspected, never the text of it;
- finding types, which the core already bounds, and spans, which are integers;
- an error as the exception's TYPE name only. An exception's MESSAGE routinely
  quotes the value that caused it, which is exactly the string the run was
  checking. `jamjet_guardrails.chain._bounded` makes the same choice.
"""

from __future__ import annotations

import json
from typing import Any

from jamjet_guardrails import ChainResult, Direction

# The same bound `jamjet_guardrails.chain` puts on any caller-supplied string it
# is willing to record. Repeated rather than imported: it is private there, and
# an adapter reaching into another distribution's private name is a break
# waiting for a patch release.
_ERROR_TYPE_LIMIT = 200


def _bounded(exc: BaseException) -> str:
    """An exception's type name, bounded, and never its message.

    A `ValueError` raised deep inside a detector says what it choked on, and
    what it choked on is the content. The type name is chosen by whoever wrote
    the class, before any content existed, which is the property that makes it
    safe to record.
    """
    return type(exc).__name__[:_ERROR_TYPE_LIMIT]


def record_for(direction: Direction, result: ChainResult) -> str:
    """The serialised audit record for one completed chain run.

    `saw` is written once, at the top, rather than once per verdict. That is not
    a shortening: `GuardrailChain` hashes the content it was handed exactly once
    and stamps that digest onto every verdict it builds, so the per-verdict
    values cannot differ. `tests/test_audit.py` asserts that equality against a
    real run rather than trusting the docstring, which is what makes flattening
    it safe to read.
    """
    return json.dumps(
        {
            "direction": direction,
            "decision": result.decision,
            "saw": result.verdicts[0].saw if result.verdicts else None,
            "verdicts": [
                {
                    "detector": verdict.provenance.detector,
                    "version": verdict.provenance.version,
                    "kind": verdict.provenance.kind,
                    "decision": verdict.decision,
                    "types": [finding.type for finding in verdict.findings],
                    # Parallel to `types` by INDEX, with a null where a finding
                    # carries no span. Filtering the spans instead would shorten
                    # this list without shortening the other, so a reader lining
                    # them up would attribute every span after the gap to the
                    # wrong type. A classifier's findings legitimately have
                    # `span=None`, so the gap is a real case and not a defensive
                    # one.
                    "spans": [
                        None if finding.span is None else list(finding.span)
                        for finding in verdict.findings
                    ],
                    **({"error": verdict.error} if verdict.error is not None else {}),
                }
                for verdict in result.verdicts
            ],
        },
        sort_keys=True,
    )


def record_for_failure(direction: Direction, digest: str | None, exc: BaseException) -> str:
    """The record for a run that never produced a `ChainResult`.

    `GuardrailChain.run` raises in one case its own docstring names, and the
    library's instruction to a caller is unambiguous: treat any exception out of
    `run` as a deny. There is no audit record from the chain in that case, so
    this writes the one the host still needs, carrying the same digest the chain
    would have stamped and no verdicts, because none were produced.

    A record that omitted the failure entirely would be the worse half of the
    same problem: the rail denies and the log says nothing about why.

    `digest` is None where the failure happened before there was content to
    hash. A hash of the empty string would be a lie about what was inspected,
    and a reader replaying the record from it would find no match and no reason.
    """
    return json.dumps(
        {
            "direction": direction,
            "decision": "deny",
            "saw": digest,
            "verdicts": [],
            "error": _bounded(exc),
        },
        sort_keys=True,
    )


def parse(record: str) -> dict[str, Any]:
    """Read a record back, for tests and for a host that wants the fields.

    Exists so that nothing outside this module has to know the record is JSON,
    and so a change of encoding is a change in one file.
    """
    loaded: dict[str, Any] = json.loads(record)
    return loaded
