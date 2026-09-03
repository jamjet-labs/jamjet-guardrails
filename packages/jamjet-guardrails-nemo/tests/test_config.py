"""Configuration refusals, all of them at load and none of them at first message."""

from __future__ import annotations

from typing import Any

import pytest
from jamjet_guardrails import GuardrailUnavailableError
from jamjet_guardrails.detectors import AVAILABLE
from jamjet_guardrails.eval.fixtures import options_for

from jamjet_guardrails_nemo import _config

MALFORMED_LISTS: list[object] = [
    # `input: pii` rather than `input: [pii]`. A string is an iterable of its own
    # characters, so unguarded the registry is asked for a check called "p".
    "pii",
    b"pii",
    # `input:` with nothing after it.
    None,
    42,
    # One bracket too many.
    [["pii"]],
    [1],
]


def test_a_config_with_no_block_refuses_and_says_where_it_goes() -> None:
    """Measured on nemoguardrails 0.24.0: `RailsConfig` has the pydantic default
    `extra="ignore"`, so a TOP-LEVEL `jamjet_guardrails:` key in config.yml is
    parsed, discarded, and unreachable from the loaded config. A user who writes
    it there arrives here with an empty `custom_data`, and the message has to
    name the one place the key can live.
    """
    with pytest.raises(GuardrailUnavailableError) as excinfo:
        _config.read({})
    assert "custom_data" in str(excinfo.value)
    assert "jamjet_guardrails" in str(excinfo.value)


def test_the_refusal_names_checks_that_are_actually_installed() -> None:
    """The example in the message is derived, not written out.

    A hard-coded example goes stale the release a check is added or renamed, and
    an example naming a check the installed version cannot build is worse than
    no example: it sends the reader to configure something that will refuse.
    """
    with pytest.raises(GuardrailUnavailableError) as excinfo:
        _config.read({})
    message = str(excinfo.value)
    named = [name for name in AVAILABLE if name in message]
    assert named, f"the refusal names no installed check: {message}"


def test_a_misspelled_direction_key_refuses() -> None:
    """`inputs:` leaves `input` empty, so that rail checks nothing while the
    config reads as though it does. Refusing the unknown key is the only way
    that is ever noticed."""
    with pytest.raises(GuardrailUnavailableError, match="keys this adapter does not know"):
        _config.read({"jamjet_guardrails": {"inputs": ["pii"]}})


def test_a_config_that_names_no_checks_refuses() -> None:
    with pytest.raises(GuardrailUnavailableError, match="names no checks"):
        _config.read({"jamjet_guardrails": {"input": [], "output": []}})


def test_options_for_a_check_no_direction_runs_refuses() -> None:
    """The config reads as though `secrets` were tuned, and `secrets` is not in
    either chain."""
    with pytest.raises(GuardrailUnavailableError, match="neither direction"):
        _config.read(
            {"jamjet_guardrails": {"input": ["pii"], "options": {"secrets": {"on_match": "deny"}}}}
        )


@pytest.mark.parametrize("value", MALFORMED_LISTS)
def test_a_malformed_direction_list_refuses(value: object) -> None:
    with pytest.raises(GuardrailUnavailableError):
        _config.read({"jamjet_guardrails": {"input": value}})


@pytest.mark.parametrize("value", MALFORMED_LISTS)
def test_the_options_path_refuses_the_same_shapes_the_core_does(value: object) -> None:
    """The anti-drift guard for the one place this adapter does not call
    `build_chain`.

    `chain_for` uses `build_chain` when no listed check carries options, and
    `build(name, **options)` when one does, because `build_chain` constructs
    every check with no options and some registered checks refuse to exist
    without them. Two paths is two chances to refuse different things, so this
    asserts the core's own front door and this adapter's validation reject the
    same set.
    """
    from jamjet_guardrails import build_chain

    core_refused = False
    try:
        build_chain(value)  # type: ignore[arg-type]
    except GuardrailUnavailableError:
        core_refused = True
    assert core_refused, f"build_chain accepted {value!r}; this comparison would prove nothing"
    with pytest.raises(GuardrailUnavailableError):
        _config.read({"jamjet_guardrails": {"input": value}})


def test_a_named_check_that_is_not_installed_refuses_at_load() -> None:
    configured = _config.read({"jamjet_guardrails": {"input": ["not-a-check"]}})
    with pytest.raises(GuardrailUnavailableError, match="not available"):
        _config.chain_for(configured.input, configured.options)


def test_options_reach_the_check_they_name() -> None:
    """The options path, proved by a decision that only the option produces.

    `secrets` redacts by default; `on_match: deny` is the only reason this run
    denies. Asserting the chain builds would not have caught options silently
    dropped on the way through.
    """
    from jamjet_guardrails import Context

    configured = _config.read(
        {"jamjet_guardrails": {"input": ["secrets"], "options": {"secrets": {"on_match": "deny"}}}}
    )
    chain = _config.chain_for(configured.input, configured.options)
    result = chain.run(
        "key sk-abcdefghijklmnopqrstuvwxyz012345 here",
        Context(direction="input", origin="user"),
    )
    assert result.decision == "deny"


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_every_registered_check_can_be_configured_through_this_adapter(name: str) -> None:
    """Derived from the registry, never from a list of names.

    A test naming today's checks goes red the day a fifth lands, and an adapter
    that cannot carry a check the core registers is a gap nobody would notice
    until somebody configured it. `options_for` supplies the fixture for a check
    that cannot be built with no options at all.
    """
    options: dict[str, Any] = {name: dict(options_for(name))} if options_for(name) else {}
    configured = _config.read({"jamjet_guardrails": {"input": [name], "options": options}})
    chain = _config.chain_for(configured.input, configured.options)
    assert chain is not None
