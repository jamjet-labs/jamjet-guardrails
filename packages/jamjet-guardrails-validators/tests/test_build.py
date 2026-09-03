"""Construction refusals, all of them before any value exists."""

from __future__ import annotations

import pytest
from jamjet_guardrails import GuardrailUnavailableError, build_chain

from jamjet_guardrails_validators import JamJetChain, validator_for

MALFORMED: list[object] = [
    # `checks="pii"` rather than `checks=["pii"]`. A string is an iterable of its
    # own characters, so unguarded this asks the registry for a check called "p".
    "pii",
    b"pii",
    None,
    42,
    # One bracket too many.
    [["pii"]],
    [1],
    # Nothing at all: a validator that runs no check passes every value.
    [],
]


@pytest.mark.parametrize("value", MALFORMED)
def test_the_composite_refuses_the_same_shapes_the_core_does(value: object) -> None:
    """The anti-drift guard for the one place this package does not call
    `build_chain`.

    `chain_for` uses `build_chain` when no listed check carries options, and
    `build(name, **options)` when one does, because `build_chain` constructs
    every check with no options and some registered checks refuse to exist
    without them. Two paths is two chances to refuse different things, so this
    asserts the core's own front door and this package's construction reject the
    same set.
    """
    core_refused = False
    try:
        build_chain(value)  # type: ignore[arg-type]
    except GuardrailUnavailableError:
        core_refused = True
    assert core_refused, f"build_chain accepted {value!r}; this comparison would prove nothing"
    with pytest.raises(GuardrailUnavailableError):
        JamJetChain(checks=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", MALFORMED)
def test_the_options_path_refuses_the_same_shapes(value: object) -> None:
    """The same set again with an option present, which is the path that does
    not go through `build_chain` at all."""
    with pytest.raises(GuardrailUnavailableError):
        JamJetChain(checks=value, options={"pii": {}})  # type: ignore[arg-type]


def test_an_empty_check_list_is_refused_in_this_package_s_own_words() -> None:
    """The message, not just the refusal, and the difference is the whole test.

    `build_chain([])` refuses an empty list too, so deleting the check in
    `check_names` still produced a `GuardrailUnavailableError` and the shape test
    above stayed green over it: a guard nothing can be watched to break is not a
    guard. What the core cannot say is what an empty list means HERE, which is a
    validator that runs nothing and therefore passes every value.
    """
    with pytest.raises(GuardrailUnavailableError, match="passes every value"):
        JamJetChain(checks=[])


def test_a_check_that_is_not_installed_is_refused_at_construction() -> None:
    """Not at first value. A validator handed back for a check nobody installed
    would look configured and fail on the request that trips over it."""
    with pytest.raises(GuardrailUnavailableError, match="not available"):
        JamJetChain(checks=["not-a-check"])


def test_options_for_a_check_the_validator_does_not_run_are_refused() -> None:
    """The configuration reads as though `secrets` were tuned, and `secrets` is
    not in the chain at all."""
    with pytest.raises(GuardrailUnavailableError, match="does not run"):
        JamJetChain(checks=["pii"], options={"secrets": {"on_match": "deny"}})


def test_options_reach_the_check_they_name() -> None:
    """Proved by a decision that only the option produces.

    `secrets` redacts by default; `on_match: deny` is the only reason this
    fails without a fix. Asserting the validator constructs would not have
    caught options silently dropped on the way through.
    """
    from guardrails.classes.validation.validation_result import FailResult

    validator = JamJetChain(
        checks=["secrets"], options={"secrets": {"on_match": "deny"}}, on_fail="noop"
    )
    result = validator.validate("key sk-abcdefghijklmnopqrstuvwxyz012345 here", {})
    assert isinstance(result, FailResult)
    assert result.fix_value is None, "on_match=deny should leave no fix"


def test_validator_for_refuses_an_unknown_name_and_says_what_exists() -> None:
    with pytest.raises(KeyError, match="no validator for"):
        validator_for("not-a-check")
