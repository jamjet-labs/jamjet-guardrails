"""The composite validator, and one generated validator per registered check."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from guardrails.classes.validation.validation_result import ValidationResult
from guardrails.validator_base import Validator, register_validator
from jamjet_guardrails import Direction, Origin
from jamjet_guardrails.detectors import AVAILABLE

from jamjet_guardrails_validators._build import chain_for, check_names
from jamjet_guardrails_validators._outcome import outcome


# The ignore is `disallow_subclassing_any` under --strict: guardrails-ai ships
# no py.typed marker, so `Validator` reads as `Any` and mypy refuses to let a
# checked class inherit from it. Narrow to that one code, so the members below
# are still checked.
class JamJetChain(Validator):  # type: ignore[misc]
    """One validator that runs several jamjet-guardrails checks and returns ONE fix.

    **This is the one to use.** Stacking two per-check validators is not
    equivalent, and the difference is a leak rather than a preference.

    Measured on guardrails-ai 0.11.0, which has two validator services and picks
    between them without asking:

    *The sequential service* (`GUARDRAILS_RUN_SYNC=true`) feeds each validator
    the previous validator's `fix_value`. That is exactly the sequential-rewrite
    leak `GuardrailChain` was rebuilt to close. With `jamjet/pii` before
    `jamjet/secrets` over::

        SLACK_BOT_TOKEN=xoxb-0000000000-2000000000008-EXAMPLEONLYnotarealtoken

    the guard returns `validation_passed=True` and::

        SLACK_BOT_TOKEN=[REDACTED:SLACK_TOKEN][REDACTED:CREDIT_CARD]-EXAMPLEONLYnotarealtoken

    The 13-digit segment is Luhn-valid and starts with a 2, so `pii` reads it as
    a card and redacts it first; the placeholder splits the token; `secrets` then
    matches only the 16-character prefix; and the 24-character secret tail
    survives into a value the guard calls valid.

    *The default async service* runs every validator over the ORIGINAL value and
    combines the fixes with a three-way merge. That closes the stump and opens
    something worse: a `FailResult` with no `fix_value`, which is what a DENY
    is, is outvoted by one that has a fix. Stacking `jamjet/injection-structural`
    with `jamjet/pii` under `on_fail="fix"` over a string carrying both an
    address and a tag-character injection payload returns
    `validation_passed=False` in one order and `validation_passed=True`, with the
    payload intact, in the other. Order-dependence is the tell, and it is the
    same tell the core chain's docstring names.

    `JamJetChain` has neither failure because it does not compose fixes at all.
    It runs one `GuardrailChain`, which hands every check the same original
    string, merges every redaction in a single pass, and combines decisions
    restrictively so a deny cannot be talked back down. The host gets one
    `FailResult` with one `fix_value`.

    ``checks`` is refused at construction if it is empty, is a bare string, holds
    a non-string, or names a check that is not installed. A validator that
    silently runs nothing passes every value.
    """

    # Set by the decorator below; declared so mypy knows the attribute exists.
    rail_alias: ClassVar[str]

    def __init__(
        self,
        checks: Iterable[str],
        *,
        on_fail: Any = None,
        direction: Direction = "output",
        origin: Origin = "model",
        options: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # Names normalised and the chain built BEFORE `super().__init__`, which
        # is not an ordering preference. `Validator.__init__` stores whatever it
        # is handed, so `list(checks)` inline there raised a bare `TypeError`
        # from inside the framework for `checks=42` instead of the
        # `GuardrailUnavailableError` this seam promises, and the caller wrapping
        # the configuration seam in one `except GuardrailUnavailableError` never
        # saw it.
        #
        # Built HERE, not on first value, for the same reason: a check that is
        # named and not installed is a configuration mistake, and the cheapest
        # place to find one is before any content exists. This is the posture
        # `build_chain` takes and the one the NeMo adapter takes at rails load.
        listed = check_names(checks)
        chain = chain_for(listed, options)
        super().__init__(
            on_fail=on_fail,
            checks=listed,
            direction=direction,
            origin=origin,
            options=dict(options or {}),
            **kwargs,
        )
        self._checks = listed
        self._direction: Direction = direction
        self._origin: Origin = origin
        self._chain = chain

    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        return outcome(
            self._chain,
            str(value),
            metadata,
            self._direction,
            self._origin,
            f"jamjet/chain[{','.join(self._checks)}]",
        )


# `register_validator` is what sets `rail_alias` and puts the class in the
# guardrails registry, and `Validator.__init__` asserts that registration before
# it will construct anything. Applied as a call rather than as a decorator on the
# class above so the same line can be applied to the generated classes below,
# with one explanation instead of two.
JamJetChain = register_validator(name="jamjet/chain", data_type="string")(  # type: ignore[misc]
    JamJetChain
)


class _JamJetCheck(Validator):  # type: ignore[misc]
    """Base for the generated per-check validators. Never registered itself.

    Exists for one check, so a caller composing a single check does not have to
    reach for the composite. Two or more of these in one `Guard` is the
    composition `JamJetChain` exists to replace; see its docstring for the
    measurement.
    """

    check: ClassVar[str]
    rail_alias: ClassVar[str]

    def __init__(
        self,
        *,
        on_fail: Any = None,
        direction: Direction = "output",
        origin: Origin = "model",
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            on_fail=on_fail,
            direction=direction,
            origin=origin,
            options=dict(options or {}),
            **kwargs,
        )
        self._direction: Direction = direction
        self._origin: Origin = origin
        # A chain of one, rather than the bare guardrail, so a single check gets
        # the same treatment the composite does: every verdict rebuilt from what
        # the chain itself passed to `check`, a raising detector turned into a
        # deny rather than an exception out of the validator, and spans that have
        # been range-checked against the value before they become `ErrorSpan`s.
        check = type(self).check
        # `None` rather than an empty mapping when nothing is configured, so a
        # check with no options goes through `build_chain`, the mechanism the
        # rest of this package documents. `{check: {}}` would take the
        # option-carrying branch to reach the identical result, which is a second
        # path for no reason and a second place to diverge.
        self._chain = chain_for([check], {check: dict(options)} if options else None)

    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        return outcome(
            self._chain,
            str(value),
            metadata,
            self._direction,
            self._origin,
            f"jamjet/{type(self).check}",
        )


def _class_name(check: str) -> str:
    """`injection-structural` becomes `JamJetInjectionStructural`."""
    return "JamJet" + "".join(part.capitalize() for part in check.split("-"))


def _generate() -> dict[str, type[Validator]]:
    """One registered validator class per name in `AVAILABLE`, at import.

    Derived from the registry rather than written out, and that is the whole
    design: a check added to the core library is a validator here on the next
    release with no edit, and a check renamed cannot leave a validator behind
    pointing at a name the registry no longer knows.
    """
    generated: dict[str, type[Validator]] = {}
    for check in sorted(AVAILABLE):
        namespace: dict[str, Any] = {
            "check": check,
            "__doc__": (
                f"Runs the jamjet-guardrails {check!r} check over a string.\n\n"
                f"Generated from `jamjet_guardrails.detectors.AVAILABLE` at import.\n"
                f"Use `JamJetChain` when you want more than one check: two of these "
                f"in one Guard compose badly, and its docstring says how."
            ),
        }
        cls = type(_class_name(check), (_JamJetCheck,), namespace)
        generated[check] = register_validator(name=f"jamjet/{check}", data_type="string")(cls)
    return generated


VALIDATORS: dict[str, type[Validator]] = _generate()


def validator_for(check: str) -> type[Validator]:
    """The validator class for one registered check, or a refusal naming what exists.

    A function rather than a module attribute per check, because the set is
    derived at import: a name that resolves through this lookup is a name the
    lookup can refuse, and `AttributeError` on a dynamically created module
    attribute would name nothing useful.
    """
    try:
        return VALIDATORS[check]
    except KeyError:
        raise KeyError(
            f"no validator for {check!r}; jamjet-guardrails registers {sorted(VALIDATORS)}"
        ) from None
