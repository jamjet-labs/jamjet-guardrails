"""Turn one `ChainResult` into one Guardrails AI `ValidationResult`.

Every validator in this package, generated or composite, goes through here, so
the mapping from a decision to a result is written once. Two validators that
mapped `deny` differently would be a policy difference nobody chose.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from guardrails.classes.validation.validation_result import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
)
from jamjet_guardrails import ChainResult, Context, Direction, GuardrailChain, Origin, saw

# The bound `jamjet_guardrails.chain` puts on any caller-supplied string it is
# willing to record, repeated here because it is private there and an adapter
# reaching into another distribution's private name is a break waiting for a
# patch release.
_ERROR_TYPE_LIMIT = 200


def _bounded(exc: BaseException) -> str:
    """An exception's type name, bounded, and never its message.

    An exception raised inside a check says what it choked on, and what it choked
    on is the content being validated. An `error_message` is rendered into a
    caller's logs and, under `on_fail="exception"`, into a `ValidationError` that
    may reach a user.
    """
    return type(exc).__name__[:_ERROR_TYPE_LIMIT]


def _spans(result: ChainResult) -> list[ErrorSpan]:
    """One `ErrorSpan` per located finding, over the value that was validated.

    `ErrorSpan.start` and `.end` are documented as indices into the validated
    chunk, which is exactly what a `Finding.span` indexes: the chain hands every
    guardrail the same original string, so no translation is needed and none is
    attempted. A finding with no span contributes nothing, because there is no
    honest index to give it.

    `reason` is required and is a string a caller renders. It names the detector
    and the finding type and nothing else. A finding's TYPE is the one
    detector-chosen string the core already lets reach a redaction placeholder,
    and it is bounded there; the content is never in it.
    """
    return [
        ErrorSpan(
            start=finding.span[0],
            end=finding.span[1],
            reason=f"{verdict.provenance.detector}:{finding.type}",
        )
        for verdict in result.verdicts
        for finding in verdict.findings
        if finding.span is not None
    ]


def _message(label: str, result: ChainResult) -> str:
    """What failed, in the words of the checks that failed it.

    Names the validator, the decision and the finding types, and quotes nothing
    from the content. Sorted, so the same failure reads the same way twice.
    """
    fired = sorted(
        {
            f"{verdict.provenance.detector}:{finding.type}"
            for verdict in result.verdicts
            for finding in verdict.findings
        }
    )
    errors = sorted(
        {
            f"{verdict.provenance.detector} failed: {verdict.error}"
            for verdict in result.verdicts
            if verdict.error is not None
        }
    )
    detail = ", ".join(fired + errors) or "no findings"
    return f"{label}: {result.decision} ({detail})"


def outcome(
    chain: GuardrailChain,
    value: str,
    metadata: dict[str, Any] | None,
    direction: Direction,
    origin: Origin,
    label: str,
) -> ValidationResult:
    """Run the chain over one value and answer in Guardrails AI's vocabulary.

    - `allow` is a `PassResult`.
    - `redact` is a `FailResult` carrying the rewritten content as `fix_value`,
      so `on_fail="fix"` produces the redacted string.
    - `deny` is a `FailResult` carrying NO `fix_value`. There is no fix for a
      deny, and the chain's merged content is explicitly not safe to send:
      `GuardrailChain`'s own docstring says that string exists for the audit
      record. Measured on guardrails-ai 0.11.0, `on_fail="fix"` over a
      `FailResult` with no `fix_value` yields `validation_passed=False` and
      `validated_output=None`, which is the right answer and a quiet one, so the
      README says deny-class checks belong under `on_fail="exception"`.
    - An exception out of `run` is a `FailResult` with no `fix_value` either. The
      library's instruction is that any exception out of `run` is a deny and that
      catching it and carrying on converts the library from fail-closed to
      fail-open in one line.
    """
    context = Context(direction=direction, origin=origin, metadata=metadata or {})
    try:
        result = chain.run(value, context)
    except Exception as exc:  # noqa: BLE001 - the library's instruction, followed
        return FailResult(
            error_message=f"{label}: deny (check failed: {_bounded(exc)}; saw {saw(value)})",
            error_spans=[],
        )
    if result.decision == "allow":
        return PassResult()
    spans: Sequence[ErrorSpan] = _spans(result)
    message = _message(label, result)
    if result.decision == "redact":
        return FailResult(error_message=message, fix_value=result.content, error_spans=list(spans))
    return FailResult(error_message=message, error_spans=list(spans))
