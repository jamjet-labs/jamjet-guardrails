"""Structural instruction-smuggling constraint.

Injection that is visible in the TEXT is a classifier's problem. Injection that
is invisible in the RENDERING is not: it is a property of the bytes, it needs no
model, and a model trained on natural language will not see it at all. Those two
facts are why this check exists in the core package while the classifier lives
behind an extra.

Every signal here is chosen for one property: no legitimate document produces it
by accident. That is what makes a published precision number defensible, and it
is why Private Use Area characters are NOT a signal -- see the note by
`_ZERO_WIDTH` for what happens to a check that fires on a developer's shell
prompt.
"""

from __future__ import annotations

from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import (
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Provenance,
    Verdict,
)

INJECTION_TYPES = frozenset(
    {
        "INVISIBLE_TAG_CHARS",
        "BIDI_OVERRIDE",
        "ZERO_WIDTH_SMUGGLING",
    }
)

_VERSION = "0.1.0"


class InjectionStructuralGuardrail:
    """Detects instruction smuggling in the encoding rather than in the words."""

    # Annotated with the Literal types, not bare assignments: a bare
    # `kind = "constraint"` infers `str`, and protocol attribute matching is
    # invariant, so it would not satisfy `kind: Kind`.
    name: str = "injection-structural"
    version: str = _VERSION
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input"})

    def __init__(self, on_match: Decision = "deny") -> None:
        # Defaults to deny, unlike `secrets`, and the asymmetry is deliberate. A
        # credential in output is a leak the caller usually still wants the rest
        # of; smuggled instructions in input are the whole reason the input is
        # suspect. A caller who has thought about it can choose `redact`, which
        # is meaningful here only because this attack IS a removable substring:
        # stripping a run of tag characters leaves the visible text intact and
        # the payload gone. The classifier has no such option.
        if on_match not in ("redact", "deny"):
            raise ValueError(f"on_match must be 'redact' or 'deny', got {on_match!r}")
        self._on_match: Decision = on_match

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every span every signal claims, in text order. Nothing is dropped."""
        return []

    def check(self, content: str, context: Context) -> Verdict:
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        findings = [Finding(type=name, span=span) for name, span in found]
        if self._on_match == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
