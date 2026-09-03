"""Build a chain from names and per-check options, refusing at construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jamjet_guardrails import GuardrailChain, GuardrailUnavailableError, build, build_chain


def check_names(value: object) -> list[str]:
    """The names a caller asked for, refusing every shape that is not a list of them.

    Separate from `chain_for`, and called FIRST by every constructor, because
    `Validator.__init__` has to be given the normalised list and `list(value)`
    on a non-iterable raises a bare `TypeError` from inside the framework's
    constructor rather than the `GuardrailUnavailableError` this seam promises.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise GuardrailUnavailableError(
            f"checks takes a list of check names, not the {type(value).__name__} "
            f"{value!r}; pass a list such as ['pii', 'secrets']"
        )
    if not isinstance(value, Iterable):
        cause = (
            " A configuration key present with no value arrives as None." if value is None else ""
        )
        raise GuardrailUnavailableError(
            f"checks takes a list of check names; a {type(value).__name__} is not iterable.{cause}"
        )
    listed = list(value)
    not_strings = [entry for entry in listed if not isinstance(entry, str)]
    if not_strings:
        raise GuardrailUnavailableError(
            f"checks takes check names as strings; these entries are not: {not_strings!r}"
        )
    if not listed:
        raise GuardrailUnavailableError(
            "checks names no check. A validator that runs nothing passes every value, "
            "which is the one answer a guardrail must never give by accident"
        )
    return listed


def chain_for(
    names: Sequence[str],
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> GuardrailChain:
    """One chain, built eagerly, or `GuardrailUnavailableError` from the constructor.

    `build_chain` is the mechanism whenever no listed check carries options: it
    is the core's own front door and it refuses all five shapes of a
    configuration that would check less than it names, including a bare string
    (`checks="pii"` reads as three one-character names), a non-iterable, a
    non-string entry, an unknown name and an empty list.

    It cannot be the mechanism when a check DOES carry options, and the reason is
    not stylistic. `build_chain` constructs every guardrail with no options at
    all, and some registered checks refuse to exist without them: `rules` with no
    patterns, no banned substrings and no limits checks nothing, so the core
    refuses to build it. Routing an option-carrying configuration through
    `build_chain` would therefore reject a configuration that is valid. The
    option-carrying path uses `build(name, **options)` per check, which is the
    same registry front door with the same refusals per name, plus the container
    checks below.

    `tests/test_build.py` asserts both paths refuse the same shapes, because two
    paths is two chances to refuse different things.
    """
    supplied = dict(options or {})
    listed = check_names(names)
    # An option for a check this validator does not run configures nothing, and
    # the configuration reads as though it did.
    orphaned = sorted(set(supplied) - set(listed))
    if orphaned:
        raise GuardrailUnavailableError(
            f"options configures checks this validator does not run: {orphaned}"
        )
    if any(name in supplied for name in listed):
        return GuardrailChain([build(name, **supplied.get(name, {})) for name in listed])
    return build_chain(listed)
