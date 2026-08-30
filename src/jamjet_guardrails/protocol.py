"""The Guardrail contract, and the content hash every verdict carries."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from jamjet_guardrails.types import Context, Direction, Kind, Verdict


def saw(content: str) -> str:
    """SHA-256 over the exact inspected string, UTF-8, lowercase hex.

    Exact means exact: no case folding, no stripping, no Unicode normalisation.
    A chain replays from these hashes, so two different pieces of content must
    never produce the same one.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@runtime_checkable
class Guardrail(Protocol):
    """One check over one piece of content.

    Implementations decide and may rewrite. They never call a model, never
    retry, and never decide what happens next.

    Every guardrail declares its own ``kind`` so the chain can stamp provenance
    onto a synthesised verdict without guessing when ``check`` raises (Task 5).

    ``runtime_checkable`` buys presence, not correctness: ``isinstance`` sees only
    that these members exist, never their types or signatures. Static typing is
    what holds implementations to ``check``'s shape.
    """

    name: str
    version: str
    kind: Kind
    directions: frozenset[Direction]

    def check(self, content: str, context: Context) -> Verdict: ...
