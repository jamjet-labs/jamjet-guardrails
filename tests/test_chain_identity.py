"""What a chain refuses about a guardrail's declared identity, at construction.

`tests/test_chain.py` covers what a chain does with what a guardrail RETURNS.
This file covers what it does with what a guardrail DECLARES, which is read once
in `GuardrailChain.__init__` and used to stamp every verdict that guardrail ever
produces.

One of the two refusals added here, the inert `directions`, is the fifth refusal
`detectors.build` already made on the registry door and nothing made on the door
`GuardrailChain` is: direct construction is a supported path, because the chain's
own docstring tells a caller who wants no checks to write `GuardrailChain([])`
themselves. The other, the length ceiling, has NO registry parity and this file
used to claim it did.
"""

from __future__ import annotations

from typing import cast, get_args

import pytest

from jamjet_guardrails.chain import _CALLER_STRING_LIMIT, _RUNNABLE_DIRECTIONS, GuardrailChain
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail, saw
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
    with pytest.raises(GuardrailUnavailableError, match="none of which it can run in"):
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


def test_all_five_declared_copies_of_the_runnable_directions_agree() -> None:
    """One value, declared in five modules, and each looks right alone.

    CHANGED from `test_the_two_declared_copies_...`, which compared the chain's
    copy to the registry's, walked every member into a `Context`, and then
    asserted that five hard-coded strings build no `Context`. It missed the copy
    whose drift FAILS OPEN. Growing `types._DIRECTIONS` alone lets a `Context`
    carry a direction no guardrail declares; `chain.run` then skips every
    guardrail and reports `allow`, no verdicts, content untouched. That mutation
    was made and the entire suite stayed green over a payload carrying an AWS
    key, an email address and an SSN. A denylist of five strings cannot catch a
    sixth.

    Five copies, none of which can import another without closing a cycle:

        chain._RUNNABLE_DIRECTIONS
        detectors._RUNNABLE_DIRECTIONS
        types._DIRECTIONS
        authoring._RUNNABLE
        eval.corpus._DIRECTIONS

    Each is listed literally rather than derived from `get_args(Direction)`, for
    the reason `types.py` records: the domain is designed to grow, and deriving
    the check would auto-accept a new member before anything could handle it.
    That argument is about the MODULES. A test is the right place to compare
    them, and comparing every copy to `get_args(Direction)` closes the drift in
    both directions at once: a copy that grows is caught, and so is one that
    does not grow when the Literal does.

    Mutation-checked: adding "stream" to any one copy fails, removing "input"
    from any one copy fails, and adding "system" to `types._DIRECTIONS` alone,
    the mutation the old test missed, fails here.
    """
    from jamjet_guardrails.authoring import _RUNNABLE as AUTHORING_DIRECTIONS
    from jamjet_guardrails.detectors import _RUNNABLE_DIRECTIONS as REGISTRY_DIRECTIONS
    from jamjet_guardrails.eval.corpus import _DIRECTIONS as CORPUS_DIRECTIONS
    from jamjet_guardrails.types import _DIRECTIONS as CONTEXT_DIRECTIONS

    declared = set(get_args(Direction))
    assert declared, "Direction declares no members; every comparison below would be vacuous"
    for label, copy in (
        ("chain", _RUNNABLE_DIRECTIONS),
        ("detectors", REGISTRY_DIRECTIONS),
        ("types", CONTEXT_DIRECTIONS),
        ("authoring", AUTHORING_DIRECTIONS),
        ("eval.corpus", CORPUS_DIRECTIONS),
    ):
        assert set(copy) == declared, (
            f"{label} declares {sorted(set(copy))} and Direction declares "
            f"{sorted(declared)}; a copy that drifts is a refusal that means "
            "something different from the one beside it"
        )

    # And the behaviour, because equal literals are not the same claim as equal
    # behaviour: a copy could agree and its consumer could still test membership
    # against something else.
    for direction in sorted(declared):
        assert Context(direction=cast(Direction, direction), origin="user").direction == direction
    for outside in ("stream", "system", "tool", "Input", "", "both"):
        assert outside not in declared
        with pytest.raises(ValueError, match="unknown direction"):
            Context(direction=cast(Direction, outside), origin="user")


def test_the_caller_string_ceiling_is_the_same_at_every_door() -> None:
    """Three modules refuse an over-long identity and none can import another.

    `chain` holds the ceiling for a guardrail entering a chain, `authoring` holds
    it where a check is authored, and `detectors` holds it as a message bound on
    the name it reports. `detectors` imports `chain`, and `authoring` is imported
    by `detectors`, so the constant cannot live in one place without a cycle.

    Two ceilings that disagree are worse than one that is wrong: a check would
    construct, pass the registry, and be refused by the chain, which is the
    detonates-later shape this repository refuses everywhere else.
    """
    from jamjet_guardrails.authoring import _CALLER_STRING_LIMIT as AUTHORING_LIMIT
    from jamjet_guardrails.detectors import _NAME_LIMIT as REGISTRY_LIMIT

    assert _CALLER_STRING_LIMIT == AUTHORING_LIMIT == REGISTRY_LIMIT


def test_a_refusal_never_quotes_the_directions_it_refuses() -> None:
    """The message bound, on the refusal that was written without one.

    `directions` is the guardrail's own declared data, read from caller code
    that runs after the caller has content in hand, so a declared direction can
    BE the content. The first draft of the inert-guardrail refusal interpolated
    `sorted(directions)` whole: a guardrail declaring one two-million-character
    direction produced a two-million-character refusal, and one declaring a
    direction that was an AWS key and an email address put both into the message
    that a caller wrapping the configuration seam writes to a log.

    That is the defect `_bounded` exists to close, reached from a third side,
    and `_refusal`'s own docstring already forbade it: the guardrail is named by
    POSITION because what is being refused is its account of itself.

    Mutation-checked: restoring `sorted(directions)` in either refusal fails
    here.
    """
    secret = "AKIAIOSFODNN7EXAMPLE alice@example.com"
    huge = "D" * 500_000

    for directions in (frozenset({secret}), frozenset({huge}), frozenset({secret, huge})):
        with pytest.raises(GuardrailUnavailableError) as raised:
            GuardrailChain([_Declared(directions=directions)])
        message = str(raised.value)
        assert secret not in message, message[:200]
        assert huge not in message
        assert len(message) < 500, len(message)
        # The count is what replaces the values, and it has to be there or the
        # message says nothing about what was wrong.
        assert f"{len(directions)} direction(s)" in message

    # The registry door carries the same message and had the same hole. Reached
    # through `build` rather than restated, because the two messages are written
    # separately in two modules that cannot import each other's constant.
    from jamjet_guardrails.detectors import AVAILABLE, build

    def factory(**_options: object) -> Guardrail:
        return cast("Guardrail", _Declared(directions=frozenset({secret})))

    AVAILABLE["__probe__"] = factory
    try:
        with pytest.raises(GuardrailUnavailableError) as raised:
            build("__probe__")
    finally:
        del AVAILABLE["__probe__"]
    assert secret not in str(raised.value)
    assert len(str(raised.value)) < 500
