"""What a chain refuses about a guardrail's declared identity, at construction.

`tests/test_chain.py` covers what a chain does with what a guardrail RETURNS.
This file covers what it does with what a guardrail DECLARES, which is read once
in `GuardrailChain.__init__` and used to stamp every verdict that guardrail ever
produces. Two refusals here were made by `detectors.build` on the registry door
and by nothing at all on the door `GuardrailChain` is: direct construction is a
supported path, because the chain's own docstring tells a caller who wants no
checks to write `GuardrailChain([])` themselves.
"""

from __future__ import annotations

from typing import cast

import pytest

from jamjet_guardrails.chain import _CALLER_STRING_LIMIT, _RUNNABLE_DIRECTIONS, GuardrailChain
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import Context, Direction, Kind, Provenance, Verdict

OUT = Context(direction="output", origin="model")
IN = Context(direction="input", origin="user")


class _Declared:
    """A double whose declared identity is whatever the test hands it.

    One class rather than a local class per test, because the refusals below
    differ only in which declaration is wrong, and writing them out separately
    is how two of them end up checking slightly different things.
    """

    kind: Kind = "constraint"

    def __init__(
        self,
        *,
        name: str = "declared",
        version: str = "0.1.0",
        directions: frozenset[str] = frozenset({"input", "output"}),
    ) -> None:
        self.name = name
        self.version = version
        self.directions = cast("frozenset[Direction]", directions)

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind="constraint", detector=self.name, version=self.version),
            saw(content),
        )


@pytest.mark.parametrize("field", ["name", "version"])
def test_a_declared_name_or_version_above_the_ceiling_refuses_the_chain(field: str) -> None:
    """`Provenance.detector` was bounded on the way into a MESSAGE and nowhere else.

    `_Identity.named` is the bounded copy and it is used only in error text, so a
    guardrail declaring a two-million-character name cost one truncated error
    string and an unbounded `provenance.detector` in every verdict of every run
    it took part in. That is the larger of the two by the number of verdicts,
    and every log, trace and database column downstream of a `ChainResult` pays
    for it once per verdict rather than once per failure.

    Refused rather than truncated. A truncated name is a record that does not say
    what the guardrail declared, and `_verified` grades a returned
    `provenance.detector` against this copy, so under truncation the honest
    guardrail returning its own full name is the one recorded as lying.

    Mutation-checked: deleting the length loop in `_identity_of` makes the first
    assertion fail.
    """
    over = "x" * (_CALLER_STRING_LIMIT + 1)
    # Spelled out rather than passed as `**{field: over}`: mypy cannot see which
    # keyword a dict unpack names, so the strict gate reads it as passing a str
    # where the third parameter wants a frozenset and fails the build.
    guardrail = _Declared(name=over) if field == "name" else _Declared(version=over)
    with pytest.raises(GuardrailUnavailableError, match="characters"):
        GuardrailChain([guardrail])


@pytest.mark.parametrize("field", ["name", "version"])
def test_a_declared_name_or_version_at_the_ceiling_is_still_recorded_whole(field: str) -> None:
    """The false-reject control, and the assertion that the RECORD is the point.

    A guard that refused every name would pass the test above. This one holds
    the boundary at exactly the ceiling and then reads the value back out of the
    provenance the chain stamped, which is where the unbounded copy actually
    lived.
    """
    at_limit = "x" * _CALLER_STRING_LIMIT
    guardrail = _Declared(name=at_limit) if field == "name" else _Declared(version=at_limit)
    result = GuardrailChain([guardrail]).run("content", OUT)
    (verdict,) = result.verdicts
    recorded = verdict.provenance.detector if field == "name" else verdict.provenance.version
    assert recorded == at_limit


@pytest.mark.parametrize(
    "directions",
    [frozenset(), frozenset({"inptu"}), frozenset({"stream"}), frozenset({"INPUT", "Output"})],
)
def test_a_guardrail_that_no_context_can_run_refuses_the_chain(directions: frozenset[str]) -> None:
    """`detectors.build` refuses this on the registry door. `GuardrailChain` did not.

    Such a guardrail is inert: `run` skips it in every context, so it is
    configured, silent, and indistinguishable from a working check in every
    artifact this library emits. Alone it produces exactly the output
    `build_chain` refuses to build from an empty name list, which is allow,
    content untouched, no verdicts; beside a live detector it makes the chain
    quieter than the caller asked for while raising nothing to say so.

    Parametrised over a typo and a wrong-case pair as well as the empty set,
    because the intersection test is what catches those and an emptiness test is
    not: `{"inptu"}` is a non-empty set no `Context` can ever match.

    Mutation-checked: replacing the intersection test in `_identity_of` with
    `if not directions` lets the three non-empty cases through, and deleting it
    lets all four through.
    """
    with pytest.raises(GuardrailUnavailableError, match="no direction it can run in"):
        GuardrailChain([_Declared(directions=directions)])


def test_a_guardrail_declaring_one_runnable_direction_is_still_built() -> None:
    """The false-reject control for the refusal above.

    A guard that fired on a guardrail declaring `{"output"}` beside a direction
    nobody recognises would deny a legitimate configuration, and the test above
    cannot tell that apart from a correct refusal on its own.
    """
    chain = GuardrailChain([_Declared(directions=frozenset({"output", "stream"}))])
    assert chain.run("content", OUT).decision == "allow"
    # Skipped, not run, in the direction it does not declare. Still no verdicts,
    # which is the shape the refusal above keeps out when it is ALL a chain can
    # do rather than one direction of what it does.
    assert chain.run("content", IN).verdicts == ()


def test_the_two_declared_copies_of_the_runnable_directions_agree() -> None:
    """One value, declared in three modules, and each looks right alone.

    `types.Context.__post_init__` refuses a direction outside its own literal
    list, `detectors._RUNNABLE_DIRECTIONS` refuses a guardrail declaring none of
    its own, and `chain._RUNNABLE_DIRECTIONS` now does the same. The chain cannot
    import the registry's copy, because `detectors/__init__.py` imports the
    chain, so the duplication is structural rather than careless. Each is listed
    literally rather than derived from `get_args(Direction)` for the reason
    `types.py` records: the domain is designed to grow, and deriving the check
    would auto-accept a new member before anything could handle it.

    A re-declared value drifts. Add a fourth direction to `Direction` and to two
    of these three, and the third refuses every guardrail that declares only the
    new one, or skips it in silence; both halves read correctly in their own
    file. This holds them to each other through BEHAVIOUR rather than through
    the literals, so a copy spelled differently but behaving the same passes and
    one that behaves differently cannot.

    Mutation-checked: adding "stream" to `chain._RUNNABLE_DIRECTIONS` alone fails
    on the `Context` half, and removing "input" from it fails on the registry
    half.
    """
    from jamjet_guardrails.detectors import _RUNNABLE_DIRECTIONS as REGISTRY_DIRECTIONS

    assert _RUNNABLE_DIRECTIONS == set(REGISTRY_DIRECTIONS)
    for direction in sorted(_RUNNABLE_DIRECTIONS):
        # Every direction the chain will run must be one a Context can carry, or
        # the chain accepts a guardrail nothing can ever reach.
        assert Context(direction=cast(Direction, direction), origin="user").direction == direction
    for outside in ("stream", "tool", "Input", "", "both"):
        # And nothing outside it may construct a Context, or a guardrail the
        # chain refused as unreachable would in fact have been reachable.
        assert outside not in _RUNNABLE_DIRECTIONS
        with pytest.raises(ValueError, match="unknown direction"):
            Context(direction=cast(Direction, outside), origin="user")
