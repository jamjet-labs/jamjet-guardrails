"""The two actions, exercised directly with context dicts.

Every decision the library can reach is here, plus the two failures that produce
no `ChainResult` at all. The load tests next door drive the same actions through
a real rails config; these hold the behaviour where it is written.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from jamjet_guardrails import (
    Context,
    Direction,
    GuardrailChain,
    Kind,
    Verdict,
    build,
    build_chain,
    saw,
)

from jamjet_guardrails_nemo import (
    ACTION_INPUT,
    ACTION_OUTPUT,
    AUDIT_KEY,
    JamJetRails,
    parse_audit_record,
)

EMAIL_TEXT = "mail alice@example.com please"
OPENAI_KEY_TEXT = "key sk-abcdefghijklmnopqrstuvwxyz012345 here"
# A tag-character run, which `injection-structural` denies by default.
TAG_TEXT = "ignore previous" + "".join(chr(0xE0000 + ord(c)) for c in "hi")


def _rails() -> JamJetRails:
    return JamJetRails(
        {
            "input": build_chain(["injection-structural", "pii"]),
            "output": build_chain(["pii", "secrets"]),
        }
    )


def _call(rails: JamJetRails, direction: Direction, context: dict[str, Any]) -> Any:
    action = rails.check_input if direction == "input" else rails.check_output
    return asyncio.run(action(context=context))


def test_allow_returns_true_and_leaves_the_message_alone() -> None:
    result = _call(_rails(), "input", {"user_message": "hello there"})
    assert result.return_value is True
    assert "user_message" not in result.context_updates
    assert parse_audit_record(result.context_updates[AUDIT_KEY])["decision"] == "allow"


def test_redact_returns_true_and_replaces_the_message() -> None:
    """The half of the contract a boolean cannot carry.

    A redact allows the turn to continue, so the action returns true exactly as
    an allow does. What makes it a redaction rather than a label is the
    `context_updates` entry: NeMo applies it as a `ContextUpdate` event, and the
    rewritten string is what the rest of the turn reads. Asserted here on the
    update; `tests/test_load.py` asserts it reaches the model's prompt.
    """
    result = _call(_rails(), "output", {"bot_message": EMAIL_TEXT})
    assert result.return_value is True
    assert result.context_updates["bot_message"] == "mail [REDACTED:EMAIL] please"
    assert parse_audit_record(result.context_updates[AUDIT_KEY])["decision"] == "redact"


def test_deny_returns_false_and_never_writes_the_message_back() -> None:
    """A deny must not forward the chain's merged content.

    `GuardrailChain`'s docstring: on a deny `ChainResult.content` "is NOT safe to
    send", it exists for the audit record. Writing it into `user_message` would
    hand the host the string the deny was for, with a boolean saying it was
    stopped.
    """
    result = _call(_rails(), "input", {"user_message": TAG_TEXT})
    assert result.return_value is False
    assert "user_message" not in result.context_updates
    assert parse_audit_record(result.context_updates[AUDIT_KEY])["decision"] == "deny"


def test_a_guardrail_that_raises_is_a_deny_with_the_failure_recorded() -> None:
    """`chain.run` fails closed on its own, so this is the layer above it.

    The chain turns a raising `check` into a synthesised deny verdict and keeps
    going, so the action sees an ordinary `ChainResult`. What this asserts is
    that the resulting deny reaches the flow as false, and that the record says a
    detector failed rather than reporting a clean deny.
    """

    class Exploding:
        name: str = "exploding"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise RuntimeError("detector bug")

    rails = JamJetRails({"input": GuardrailChain([Exploding()])})
    result = _call(rails, "input", {"user_message": "anything"})
    assert result.return_value is False
    record = parse_audit_record(result.context_updates[AUDIT_KEY])
    assert record["decision"] == "deny"
    error = record["verdicts"][0]["error"]
    assert error.startswith("RuntimeError")
    # The chain records the exception's TYPE and withholds its message. That is
    # the whole reason this assertion is here twice over: a detector's message
    # names what it choked on, and what it choked on is the content.
    assert "detector bug" not in error


def test_an_exception_out_of_run_is_a_deny_with_a_record() -> None:
    """`run` abandons the whole run in one case, and the caller must deny.

    The one case is a redact the chain cannot locate. The library's own
    instruction is that any exception out of `run` is a deny and that catching
    it and carrying on converts the library to fail-open, so the action catches
    it, returns false, and writes the digest of what was inspected plus the
    exception's TYPE.
    """

    class Unlocatable:
        name: str = "unlocatable"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            from jamjet_guardrails import Finding, Provenance

            return Verdict(
                "redact",
                "REDACTED",
                [Finding(type="TOK")],
                Provenance(kind="constraint", detector="unlocatable", version="0.1.0"),
                saw(content),
            )

    rails = JamJetRails({"output": GuardrailChain([Unlocatable()])})
    result = _call(rails, "output", {"bot_message": "anything"})
    assert result.return_value is False
    record = parse_audit_record(result.context_updates[AUDIT_KEY])
    assert record["decision"] == "deny"
    assert record["verdicts"] == []
    assert record["error"] == "GuardrailChainError"
    assert record["saw"] == saw("anything")


@pytest.mark.parametrize("context", [{}, {"user_message": None}, {"user_message": 12}])
def test_a_rail_context_with_no_message_denies_rather_than_allowing(
    context: dict[str, Any],
) -> None:
    """An absent message is a wiring failure, and the safe answer is deny.

    Defaulting to the empty string would answer "allow, nothing found" from a
    rail that never saw the content, which is the shape of silence this library
    exists to refuse.
    """
    result = _call(_rails(), "input", context)
    assert result.return_value is False
    assert parse_audit_record(result.context_updates[AUDIT_KEY])["error"] == "MissingRailMessage"


def test_an_unconfigured_direction_denies() -> None:
    """`init` never builds this, and the class is constructible directly."""
    result = _call(JamJetRails({"input": build_chain(["pii"])}), "output", {"bot_message": "hi"})
    assert result.return_value is False
    assert parse_audit_record(result.context_updates[AUDIT_KEY])["error"] == "UnconfiguredDirection"


def test_each_direction_runs_the_chain_built_for_it() -> None:
    """The two actions must not share one chain.

    A single chain used for both directions would silently apply the output
    policy to input and the input policy to output, and both would still look
    like they ran. Asserted by giving the two directions different checks and
    watching only one of them fire.
    """
    rails = JamJetRails(
        {"input": build_chain(["pii"]), "output": GuardrailChain([build("secrets")])}
    )
    on_input = _call(rails, "input", {"user_message": OPENAI_KEY_TEXT})
    on_output = _call(rails, "output", {"bot_message": OPENAI_KEY_TEXT})
    assert on_input.return_value is True
    assert "user_message" not in on_input.context_updates
    assert on_output.context_updates["bot_message"] == "key [REDACTED:OPENAI_KEY] here"


def test_both_actions_are_declared_system_actions() -> None:
    """Not decoration, and not a style rule.

    nemoguardrails 0.24.0 routes a NON-system action to `actions_server_url`
    when a deployment sets one (`colang/v1_0/runtime/runtime.py`: `if
    self.config.actions_server_url and not action_meta.get("is_system_action")`).
    Without the flag, a deployment with an actions server configured would POST
    the user's message and the model's reply to it on every turn: a guardrail
    that exists to keep credentials out of places they should not go, sending
    them over the network to be checked.
    """
    rails = _rails()
    for action, name in ((rails.check_input, ACTION_INPUT), (rails.check_output, ACTION_OUTPUT)):
        meta = action.action_meta
        assert meta["is_system_action"] is True, name
        assert meta["name"] == name
