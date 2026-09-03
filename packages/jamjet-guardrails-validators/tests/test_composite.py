"""The composite, against the composition case the core chain's docstring names.

These are the tests the package exists for. Each one runs the leak first, so the
assertion about `JamJetChain` is measured against a failure that actually
happens on the installed version rather than against a claim in a docstring.
"""

from __future__ import annotations

import pytest
from guardrails import Guard
from guardrails.classes.validation.validation_result import FailResult, PassResult
from jamjet_guardrails import Context, GuardrailChain, build

from jamjet_guardrails_validators import JamJetChain, validator_for

# The shape from `GuardrailChain`'s docstring, with a body that says so.
# Canonical Slack bot-token shape: `xoxb-`, a 10-digit team id, a 13-digit
# bot-user id and a 24-character secret. The middle segment is Luhn-valid and
# starts with a 2, so `pii` matches it as a card. The secret carries
# `EXAMPLEONLY` and `notarealtoken` in its own 24 characters, so nothing here
# reads as a live credential to a scanner or to a person.
SLACK_TOKEN = "SLACK_BOT_TOKEN=xoxb-0000000000-2000000000008-EXAMPLEONLYnotarealtoken"
SLACK_SECRET_TAIL = "EXAMPLEONLYnotarealtoken"
STUMP = "xoxb-0000000000-"

MERGED = "SLACK_BOT_TOKEN=[REDACTED:CREDIT_CARD+SLACK_TOKEN]"

TAG_RUN = "".join(chr(0xE0000 + ord(character)) for character in "hi")
INJECTION_AND_EMAIL = f"hello alice@example.com {TAG_RUN} bye"


def _stacked(*checks: str, on_fail: str = "fix") -> Guard:
    """Two per-check validators in ONE Guard.

    Variadic `use`, not `use(...).use(...)`: guardrails-ai documents that calling
    `Guard.use` twice for the same target OVERWRITES the earlier validators, so
    the chained spelling silently runs only the last one. A comparison against a
    guard that ran one validator would prove nothing about composing two.
    """
    guard = Guard().use(*[validator_for(check)(on_fail=on_fail) for check in checks])
    assert len(guard.get_validators("output")) == len(checks), (
        "the guard did not keep both validators, so this comparison would prove nothing"
    )
    return guard


def test_stacking_two_validators_leaves_the_stump_and_the_composite_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak, then the fix, in one test, on the same input.

    `GUARDRAILS_RUN_SYNC=true` selects guardrails-ai's sequential validator
    service, which feeds each validator the previous one's `fix_value`. That is
    the sequential-rewrite path `GuardrailChain` was rebuilt to close, and the
    output below is character for character the string the core's docstring
    predicts: `pii` redacts the Luhn-valid middle segment first, the placeholder
    splits the token, `secrets` matches only the 16-character prefix, and the
    24-character tail survives into a value the guard reports as valid.
    """
    monkeypatch.setenv("GUARDRAILS_RUN_SYNC", "true")

    leaked = _stacked("pii", "secrets").validate(SLACK_TOKEN)
    assert leaked.validation_passed is True, "the leak needs the guard to report success"
    assert leaked.validated_output == (
        "SLACK_BOT_TOKEN=[REDACTED:SLACK_TOKEN][REDACTED:CREDIT_CARD]-EXAMPLEONLYnotarealtoken"
    )
    assert SLACK_SECRET_TAIL in (leaked.validated_output or "")

    fixed = Guard().use(JamJetChain(checks=["pii", "secrets"], on_fail="fix")).validate(SLACK_TOKEN)
    assert fixed.validation_passed is True
    # The tail asserted ABSENT rather than the placeholder asserted present: the
    # failure is the tail SURVIVING, and a test that only checked the replacement
    # text would also pass on output that carried both.
    assert SLACK_SECRET_TAIL not in (fixed.validated_output or "")
    assert STUMP not in (fixed.validated_output or "")
    assert fixed.validated_output == MERGED


@pytest.mark.parametrize("run_sync", ["true", "false"])
def test_the_composite_leaves_no_stump_on_either_service(
    monkeypatch: pytest.MonkeyPatch, run_sync: str
) -> None:
    """One validator, so which service guardrails picks cannot change the answer.

    That is the property rather than the instance. The leak above is a property
    of composing fixes, and `JamJetChain` composes none: it hands one
    `GuardrailChain` the value once and returns one merged rewrite.
    """
    monkeypatch.setenv("GUARDRAILS_RUN_SYNC", run_sync)
    outcome = (
        Guard().use(JamJetChain(checks=["pii", "secrets"], on_fail="fix")).validate(SLACK_TOKEN)
    )
    assert outcome.validated_output == MERGED
    assert SLACK_SECRET_TAIL not in (outcome.validated_output or "")


def test_stacking_can_lose_a_deny_and_the_composite_cannot() -> None:
    """The other shape of the same problem, on the service that runs by default.

    The async service runs every validator over the ORIGINAL value and merges the
    fixes. A deny is a `FailResult` with no `fix_value`, so it has nothing to
    contribute to that merge and is outvoted by a validator that does. The result
    is order-dependent: one order refuses the value, the other returns it with
    the injection payload intact and `validation_passed` True.
    """
    denied_first = _stacked("injection-structural", "pii").validate(INJECTION_AND_EMAIL)
    redacted_first = _stacked("pii", "injection-structural").validate(INJECTION_AND_EMAIL)

    assert denied_first.validation_passed is False
    assert redacted_first.validation_passed is True
    assert TAG_RUN in (redacted_first.validated_output or "")

    for on_fail, expected in (("fix", False), ("noop", False)):
        outcome = (
            Guard()
            .use(JamJetChain(checks=["pii", "injection-structural"], on_fail=on_fail))
            .validate(INJECTION_AND_EMAIL)
        )
        assert outcome.validation_passed is expected, on_fail
    # And in the other order, because order-dependence is what the leak WAS.
    outcome = (
        Guard()
        .use(JamJetChain(checks=["injection-structural", "pii"], on_fail="fix"))
        .validate(INJECTION_AND_EMAIL)
    )
    assert outcome.validation_passed is False


def test_the_two_orders_of_the_composite_agree() -> None:
    """The property the leak was an instance of.

    Any rule under which one check can consume evidence another needs shows up
    here as two different answers. The core asserts the same thing over its own
    chain; this asserts it survives the trip through a Guard.
    """
    for text in (
        SLACK_TOKEN,
        "mail alice@example.com and use sk-abcdefghijklmnopqrstuvwxyz012345",
        "id 4111111111111111 mail bob@example.com card 4012 8888 8888 1881",
    ):
        forward = Guard().use(JamJetChain(checks=["pii", "secrets"], on_fail="fix")).validate(text)
        backward = Guard().use(JamJetChain(checks=["secrets", "pii"], on_fail="fix")).validate(text)
        assert forward.validated_output == backward.validated_output, text
        assert forward.validation_passed == backward.validation_passed, text


def test_the_composite_fix_is_what_the_core_chain_produces() -> None:
    """The adapter must not have its own opinion about the rewrite.

    Everything this package adds is translation. If the value a Guard returns
    ever stops equalling the value `GuardrailChain` produced, the translation has
    started deciding something.
    """
    expected = GuardrailChain([build("pii"), build("secrets")]).run(
        SLACK_TOKEN, Context(direction="output", origin="model")
    )
    outcome = (
        Guard().use(JamJetChain(checks=["pii", "secrets"], on_fail="fix")).validate(SLACK_TOKEN)
    )
    assert outcome.validated_output == expected.content


def test_a_guard_that_kept_only_one_validator_would_be_caught() -> None:
    """The guard on `_stacked`, which every leak test above depends on.

    `Guard.use` overwrites rather than appends when called twice for the same
    target, so a leak test written with the chained spelling compares a
    two-validator composite against a ONE-validator guard and passes for the
    wrong reason. Asserted here against the chained spelling directly, so the
    behaviour is pinned rather than remembered.
    """
    chained = Guard().use(validator_for("pii")()).use(validator_for("secrets")())
    assert len(chained.get_validators("output")) == 1
    assert len(_stacked("pii", "secrets").get_validators("output")) == 2


def test_a_deny_carries_no_fix_value() -> None:
    """There is no fix for a deny, and offering one would be offering the content.

    `GuardrailChain`'s merged content on a deny is explicitly not safe to send.
    Asserted on the `FailResult` itself rather than through a Guard, because
    `on_fail` decides what a Guard does with it and this is about what is handed
    over.
    """
    validator = JamJetChain(checks=["injection-structural"], on_fail="noop")
    result = validator.validate(f"see {TAG_RUN} here", {})
    assert isinstance(result, FailResult)
    assert result.fix_value is None
    assert result.error_spans


def test_an_allow_is_a_pass_result() -> None:
    validator = JamJetChain(checks=["pii", "secrets"], on_fail="fix")
    assert isinstance(validator.validate("nothing to see here", {}), PassResult)
