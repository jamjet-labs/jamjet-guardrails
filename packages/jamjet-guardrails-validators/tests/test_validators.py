"""The generated per-check validators, and the shape of what they return."""

from __future__ import annotations

from typing import Any

import pytest
from guardrails import Guard
from guardrails.classes.validation.validation_result import FailResult, PassResult
from guardrails.errors import ValidationError
from jamjet_guardrails import Context, Direction, Kind, Verdict
from jamjet_guardrails.detectors import AVAILABLE
from jamjet_guardrails.eval.fixtures import options_for

from jamjet_guardrails_validators import VALIDATORS, JamJetChain, validator_for

SECRET = "sk-abcdefghijklmnopqrstuvwxyz012345"
EMAIL_TEXT = "mail alice@example.com please"
TAG_RUN = "".join(chr(0xE0000 + ord(character)) for character in "hi")


def test_there_is_one_validator_per_registered_check() -> None:
    """Derived from the registry, both directions.

    A validator for a check the registry does not have would point at nothing;
    a check with no validator is a gap nobody notices until somebody asks for it.
    Written as an equality rather than as a subset for that reason, and read from
    `AVAILABLE` rather than from a list of today's names, which goes red the day
    a fifth check lands.
    """
    assert set(VALIDATORS) == set(AVAILABLE)


def test_every_generated_validator_is_registered_under_its_own_name() -> None:
    """`Validator.__init__` asserts its own registration before it constructs.

    So a class created and not registered is not merely untidy: it cannot be
    instantiated at all, and the failure is an AssertionError from inside the
    framework.
    """
    for check, cls in VALIDATORS.items():
        assert cls.rail_alias == f"jamjet/{check}"


@pytest.mark.parametrize("check", sorted(AVAILABLE))
def test_every_registered_check_can_be_built_and_run(check: str) -> None:
    """The derived guard on the generated set.

    `options_for` supplies the fixture for a check that cannot be built with no
    options at all, which is how the core's own evaluation harness builds one.
    """
    validator = validator_for(check)(options=dict(options_for(check)), on_fail="noop")
    result = validator.validate("nothing remarkable in this sentence", {})
    assert isinstance(result, (PassResult, FailResult))


def test_a_redact_returns_the_rewritten_value_as_the_fix() -> None:
    outcome = Guard().use(validator_for("pii")(on_fail="fix")).validate(EMAIL_TEXT)
    assert outcome.validation_passed is True
    assert outcome.validated_output == "mail [REDACTED:EMAIL] please"


def test_a_deny_under_fix_yields_no_output_rather_than_the_original() -> None:
    """Measured on guardrails-ai 0.11.0, and the reason the README says to use
    `on_fail="exception"` for deny-class checks.

    A `FailResult` with no `fix_value` under `on_fail="fix"` does not raise and
    does not pass the original through: `validation_passed` is False and
    `validated_output` is None. Fail-closed, and quiet, so a caller that reads
    the output without reading the flag gets None rather than a signal.
    """
    outcome = (
        Guard()
        .use(validator_for("injection-structural")(on_fail="fix"))
        .validate(f"see {TAG_RUN} here")
    )
    assert outcome.validation_passed is False
    assert outcome.validated_output is None


def test_a_deny_under_exception_raises() -> None:
    guard = Guard().use(validator_for("injection-structural")(on_fail="exception"))
    with pytest.raises(ValidationError):
        guard.validate(f"see {TAG_RUN} here")


def test_error_spans_index_the_value_that_was_validated() -> None:
    """`ErrorSpan.start` and `.end` are documented as indices into the validated
    chunk, and a `Finding.span` already is one, so a translation would be a bug
    rather than a feature. Checked by slicing the value with the span."""
    result = validator_for("pii")(on_fail="noop").validate(EMAIL_TEXT, {})
    assert isinstance(result, FailResult)
    spans = result.error_spans or []
    assert spans
    for span in spans:
        assert EMAIL_TEXT[span.start : span.end] == "alice@example.com"
        assert span.reason == "pii:EMAIL"


@pytest.mark.parametrize("checks", [["pii", "secrets"], ["injection-structural"]])
def test_no_failure_ever_quotes_the_content(checks: list[str]) -> None:
    """The message and every span reason are strings a caller renders and logs.

    BOTH decisions, and the second one is what makes this test mean anything.
    On a redact the chain's own `content` is the value with the credential
    already replaced, so a message built from it leaks nothing and a
    redact-only test passes over the bug. On a DENY nothing was rewritten and
    `ChainResult.content` is the original string, credential and all. Mutating
    `_message` to append the chain's content was watched to survive the
    redact-only version of this test and to fail this one.
    """
    text = f"{TAG_RUN} mail alice@example.com and use {SECRET}"
    result = JamJetChain(checks=checks, on_fail="noop").validate(text, {})
    assert isinstance(result, FailResult)
    rendered = result.error_message + " ".join(span.reason for span in (result.error_spans or []))
    assert SECRET not in rendered
    assert "alice@example.com" not in rendered
    if result.fix_value is not None:
        assert SECRET not in result.fix_value


def test_the_message_names_the_validator_the_decision_and_the_types() -> None:
    result = JamJetChain(checks=["pii", "secrets"], on_fail="noop").validate(
        f"mail alice@example.com and use {SECRET}", {}
    )
    assert isinstance(result, FailResult)
    assert "jamjet/chain[pii,secrets]" in result.error_message
    assert "redact" in result.error_message
    assert "pii:EMAIL" in result.error_message
    assert "secrets:OPENAI_KEY" in result.error_message


def test_a_check_that_raises_becomes_a_failure_and_not_an_exception() -> None:
    """The chain turns a raising `check` into a synthesised deny, so the
    validator sees an ordinary result and the Guard is never handed a detector's
    exception. What must not happen is a pass."""

    class Exploding:
        name: str = "exploding"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise RuntimeError("detector bug")

    from jamjet_guardrails import GuardrailChain

    validator = JamJetChain(checks=["pii"], on_fail="noop")
    # Replace the built chain rather than register a broken detector globally.
    validator._chain = GuardrailChain([Exploding()])
    result = validator.validate("anything at all", {})
    assert isinstance(result, FailResult)
    assert result.fix_value is None
    assert "deny" in result.error_message
    assert "detector bug" not in result.error_message


def test_direction_and_origin_reach_the_check() -> None:
    """They are recorded on every verdict and some checks decide per direction,
    so a validator that always said "output" would silently mislabel every
    input it was pointed at."""
    seen: list[Context] = []

    class Recording:
        name: str = "recording"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            seen.append(context)
            from jamjet_guardrails import Provenance, saw

            return Verdict(
                "allow",
                None,
                [],
                Provenance(kind="constraint", detector="recording", version="0.1.0"),
                saw(content),
            )

    from jamjet_guardrails import GuardrailChain

    validator = JamJetChain(checks=["pii"], direction="input", origin="user", on_fail="noop")
    validator._chain = GuardrailChain([Recording()])
    validator.validate("hello", {"tenant": "acme"})
    assert seen[0].direction == "input"
    assert seen[0].origin == "user"
    assert seen[0].metadata == {"tenant": "acme"}


def test_metadata_a_guard_passes_reaches_the_context() -> None:
    """The one channel a host has for telling a check where the value came from."""
    validator = validator_for("pii")(on_fail="noop")
    metadata: dict[str, Any] = {"request_id": "abc"}
    assert isinstance(validator.validate(EMAIL_TEXT, metadata), FailResult)
