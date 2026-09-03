"""Value types for guardrail decisions.

Every guardrail returns a Verdict: what was decided, what the content was
rewritten to, what was found, WHICH KIND of check decided it, and the hash of
the exact content that check inspected.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["input", "output"]
Decision = Literal["allow", "redact", "deny"]
Origin = Literal["user", "retrieved", "model", "tool"]
Kind = Literal["constraint", "classifier"]

# Restrictive order: index is severity, combining takes the max.
_SEVERITY: tuple[Decision, ...] = ("allow", "redact", "deny")

_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")

# The directions a Context may carry, and the origins. NAMED rather than written
# inline in `__post_init__`, where they were, because this is the copy whose
# drift fails OPEN: growing it alone lets a Context carry a direction no
# guardrail declares, so `chain.run` skips every guardrail and reports allow over
# content nothing checked. Four other modules declare the same direction set and
# each refuses something against it, and none of them could see a literal buried
# in a method body. `tests/test_chain_identity.py` now holds all five to
# `get_args(Direction)` and to each other.
#
# Still listed literally and still NOT derived from `get_args(Direction)`, for
# the reason given for Kind below: the domain is designed to grow, and deriving
# the check would auto-accept a new member before anything could handle it. The
# test compares the two; the module does not.
_DIRECTIONS: tuple[str, ...] = ("input", "output")
_ORIGINS: tuple[str, ...] = ("user", "retrieved", "model", "tool")


def combine(a: Decision, b: Decision) -> Decision:
    """Restrictive combination: deny > redact > allow. Never weakens."""
    return _SEVERITY[max(_SEVERITY.index(a), _SEVERITY.index(b))]


@dataclass(frozen=True, slots=True)
class Provenance:
    """Which kind of check decided, and what is needed to reproduce it."""

    kind: Kind
    detector: str
    version: str
    model: str | None = None
    revision: str | None = None
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class Finding:
    """One detection. `confidence` is None if and only if a constraint found it."""

    type: str
    span: tuple[int, int] | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Context:
    """What is being checked and where it came from. Guardrails never mutate it."""

    direction: Direction
    origin: Origin
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A direction outside the domain silenced the ENTIRE chain and returned
        # allow. `chain.run` skips a guardrail whose `directions` do not contain
        # the context's, so `direction="out"`, `"Output"` or `""` skipped every
        # guardrail: allow, zero verdicts, content untouched, over content full
        # of credentials.
        #
        # mypy catches a literal typo here and NOT a config-sourced one:
        # `Context(direction="Output", ...)` errors under --strict, and
        # `Context(direction=config["direction"], ...)` does not. Config is
        # exactly the provenance `build_chain` writes its five refusals for.
        #
        # This is the direct mirror of a case `detectors.build` already refuses:
        # a guardrail declaring no direction it can run in is refused because
        # "every context would skip it, so it would be configured and silent".
        # A context carrying an unknown direction skips every GUARDRAIL, which
        # is the same silence reached from the other end, and it was unguarded.
        # `corpus.Case` validates its `direction` and `expect_decision` against
        # the same domains, so the argument was already made twice in this
        # package before it was made here.
        #
        # Listed literally rather than derived from get_args(Direction), for the
        # reason `Verdict` gives below and `detectors._RUNNABLE_DIRECTIONS`
        # gives again: these domains are designed to grow, and deriving the
        # check would auto-accept a new member before anything could handle it.
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"unknown direction {self.direction!r}; expected 'input' or 'output'. "
                "A chain runs only the guardrails whose directions contain this one, "
                "so an unknown value skips every guardrail and allows the content"
            )
        # `origin` is checked for the same reason one rung quieter. Nothing in
        # this package branches on it today, which is precisely why a typo would
        # never surface: it travels into the audit record, and a caller filtering
        # their own log for origin="retrieved" silently sees none of the rows
        # written as "retreived".
        if self.origin not in _ORIGINS:
            raise ValueError(
                f"unknown origin {self.origin!r}; expected one of "
                "'user', 'retrieved', 'model', 'tool'"
            )


@dataclass(frozen=True, slots=True)
class Verdict:
    """One guardrail's decision about one piece of content."""

    decision: Decision
    content: str | None
    findings: Sequence[Finding]
    provenance: Provenance
    saw: str
    error: str | None = None
    """Set only on a verdict the chain synthesised because ``check`` raised.

    It carries no findings: an error is not a detection, and a classifier
    finding would need a confidence that does not exist. Deny plus a reason.
    """

    def __post_init__(self) -> None:
        # Freeze FIRST, then validate the frozen copy. The protection this was
        # written for is unchanged (a caller who keeps the list it passed cannot
        # reach the tuple stored here), and copying first buys a second one: the
        # checks below read `findings` twice, so an iterator would be truthy on
        # the first read, exhausted by the second, and stored EMPTY. A verdict
        # that quietly dropped every finding would still pass every check here.
        # `corpus.Case` already copies before it validates, for the same reason.
        object.__setattr__(self, "findings", tuple(self.findings))
        # The decision itself, checked FIRST and for the same argument the kind
        # check below makes. Every other rule here is a positive match on a known
        # decision, so an unrecognised one walks past all of them: the error rule
        # tests `!= "deny"`, the content rule tests `== "redact"`. A verdict
        # carrying decision="allowed" therefore constructed cleanly, and a caller
        # branching `if decision == "deny"` ... `elif decision == "redact"` read
        # it as an allow. Inside a chain it reached `combine`, which raised a
        # bare `ValueError: tuple.index(x): x not in tuple` naming neither the
        # detector nor the value.
        #
        # Listed literally rather than derived from get_args(Decision), for the
        # reason the kind check gives.
        if self.decision not in ("allow", "redact", "deny"):
            raise ValueError(
                f"unknown decision {self.decision!r}; expected 'allow', 'redact' or 'deny'"
            )
        if self.error is not None and self.decision != "deny":
            raise ValueError("a verdict carrying an error must deny")
        if self.error is not None and self.findings:
            raise ValueError(
                "a verdict carrying an error must not carry findings; an error is not a detection"
            )
        if not _SHA256_HEX.match(self.saw):
            raise ValueError(f"saw must be a lowercase sha256 hex digest, got {self.saw!r}")
        if self.decision == "redact" and self.content is None:
            raise ValueError("a redact verdict must supply the rewritten content")
        # Checked once, OUTSIDE the loop: a bogus kind carrying zero findings would
        # never be looked at otherwise. Both branches below are positive matches, so
        # without this an unrecognised kind disables the library's central invariant
        # silently and fails open.
        #
        # The handled kinds are listed literally and deliberately NOT derived from
        # get_args(Kind). Kind is designed to grow; deriving this would auto-accept a
        # new kind and reopen the hole. Listing them means adding to the Literal breaks
        # here until someone writes that kind's confidence rule.
        if self.provenance.kind not in ("constraint", "classifier"):
            raise ValueError(
                f"unknown provenance kind {self.provenance.kind!r}; every kind must "
                "declare whether its findings carry confidence"
            )
        for f in self.findings:
            if self.provenance.kind == "constraint" and f.confidence is not None:
                raise ValueError(
                    f"constraint finding {f.type!r} carries confidence {f.confidence!r}; "
                    "a constraint matches or it does not"
                )
            if self.provenance.kind == "classifier" and f.confidence is None:
                raise ValueError(f"classifier finding {f.type!r} must carry a confidence")


@dataclass(frozen=True, slots=True)
class ChainResult:
    """The audit record of one chain run."""

    decision: Decision
    content: str
    verdicts: Sequence[Verdict]

    def __post_init__(self) -> None:
        # The audit record of a run must not change after the run reported it.
        object.__setattr__(self, "verdicts", tuple(self.verdicts))
