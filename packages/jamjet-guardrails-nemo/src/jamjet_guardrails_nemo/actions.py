"""The two system actions the shipped flows execute.

One object holds both chains and both actions, rather than two module-level
functions reading a module-level chain. A rails application is a value a host
can hold several of (a test suite holds one per case), and a module global
shared between them would let one application's configuration decide another
one's checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jamjet_guardrails import Context, Direction, GuardrailChain, Origin, saw
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

from jamjet_guardrails_nemo._audit import record_for, record_for_failure

# The rail-context key every call writes its audit record under.
AUDIT_KEY = "jamjet_guardrails"

ACTION_INPUT = "jamjet_guardrails_check_input"
ACTION_OUTPUT = "jamjet_guardrails_check_output"

# Which rail-context variable each direction inspects, and what origin the
# content has. `origin` is recorded on the Context and nothing in this library
# branches on it today, which is exactly why it has to be right: it travels
# into a caller's own audit trail, and a caller filtering theirs for
# origin="model" sees none of the rows written under the wrong one.
_MESSAGE_KEY: Mapping[Direction, str] = {"input": "user_message", "output": "bot_message"}
_ORIGIN: Mapping[Direction, Origin] = {"input": "user", "output": "model"}


class UnconfiguredDirection(Exception):
    """The action ran for a direction no chain was built for.

    Its TYPE name is what reaches the audit record, so it is named to be read
    there. See `JamJetRails._run`.
    """


class MissingRailMessage(Exception):
    """The rail context carried no string to check.

    Its TYPE name is what reaches the audit record, so it is named to be read
    there. Raised rather than defaulted to the empty string: an absent
    `user_message` means the action ran somewhere the adapter was not wired for,
    and answering "allow, nothing to check" would report a check that inspected
    nothing.
    """


class JamJetRails:
    """Holds one chain per configured direction and the actions that run them.

    Constructed by `jamjet_guardrails_nemo.init` at rails load, from chains that
    were already built, so an instance that exists is one whose every configured
    check was resolvable.
    """

    def __init__(self, chains: Mapping[Direction, GuardrailChain]) -> None:
        self._chains = dict(chains)

    def chain(self, direction: Direction) -> GuardrailChain | None:
        """The chain for a direction, or None where that direction is unconfigured."""
        return self._chains.get(direction)

    # `is_system_action=True` is not decoration. nemoguardrails 0.24.0 routes a
    # NON-system action to `actions_server_url` when a deployment sets one
    # (colang/v1_0/runtime/runtime.py: `if self.config.actions_server_url and not
    # action_meta.get("is_system_action")`). Without this flag, a deployment
    # that has an actions server configured would POST the user's message and
    # the model's reply to it on every turn: a guardrail that exists to keep
    # credentials out of places they should not go, sending them over the
    # network to be checked.
    # The ignore is for `untyped-decorator`, which is what `@action` is: the
    # framework ships no py.typed marker, so mypy reads every symbol from it as
    # Any and refuses to let one wrap a checked function under --strict. Narrow
    # to that one code, so a genuine typing error in the method below is still
    # reported.
    @action(is_system_action=True, name=ACTION_INPUT)  # type: ignore[untyped-decorator]
    async def check_input(self, context: dict[str, Any] | None = None) -> ActionResult:
        """Run the input chain over `user_message`."""
        return self._run("input", context)

    @action(is_system_action=True, name=ACTION_OUTPUT)  # type: ignore[untyped-decorator]
    async def check_output(self, context: dict[str, Any] | None = None) -> ActionResult:
        """Run the output chain over `bot_message`."""
        return self._run("output", context)

    def _run(self, direction: Direction, context: dict[str, Any] | None) -> ActionResult:
        """One direction's whole behaviour: allow, redact, deny, and failure.

        Returns rather than raises on every path. An exception out of an action
        is turned into a bot message by the runtime, which is neither a deny nor
        an allow that anything downstream can read; returning False is the deny
        the flow is written to act on, and the audit record is what says why.
        """
        chain = self._chains.get(direction)
        if chain is None:
            # Unreachable through `init`, which only registers an action for a
            # direction it built a chain for. Kept because this class is
            # constructible directly: answering True here would be an allow from
            # a rail that has no checks at all.
            return ActionResult(
                return_value=False,
                context_updates={
                    AUDIT_KEY: record_for_failure(direction, None, UnconfiguredDirection())
                },
            )
        key = _MESSAGE_KEY[direction]
        content = (context or {}).get(key)
        if not isinstance(content, str):
            return ActionResult(
                return_value=False,
                context_updates={
                    AUDIT_KEY: record_for_failure(direction, None, MissingRailMessage())
                },
            )
        try:
            result = chain.run(content, Context(direction=direction, origin=_ORIGIN[direction]))
        except Exception as exc:  # noqa: BLE001 - the library's instruction, followed
            # `GuardrailChain.run`'s own docstring: "A caller must treat any
            # exception out of `run` as a deny. A caller that catches and carries
            # on has converted this library from fail-closed to fail-open in one
            # line." The digest is computed here rather than taken from a result
            # that does not exist, so the record still says what was inspected.
            return ActionResult(
                return_value=False,
                context_updates={AUDIT_KEY: record_for_failure(direction, saw(content), exc)},
            )
        updates: dict[str, Any] = {AUDIT_KEY: record_for(direction, result)}
        if result.decision == "deny":
            # `result.content` is NOT forwarded. The chain's docstring is
            # explicit that on a deny the merged content is there for the audit
            # record and is not safe to send, so writing it back over
            # `user_message` or `bot_message` would hand the host the exact
            # string the deny exists to stop.
            return ActionResult(return_value=False, context_updates=updates)
        if result.decision == "redact":
            # Measured on nemoguardrails 0.24.0: a `context_updates` entry for
            # `user_message` reaches the prompt the model is given (the original
            # substring is absent from it and the placeholder is present), and
            # one for `bot_message` replaces the content `generate` returns.
            # That is what makes redact a real rewrite here rather than a label.
            updates[key] = result.content
        return ActionResult(return_value=True, context_updates=updates)
