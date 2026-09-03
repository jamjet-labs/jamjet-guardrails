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

import ast
from pathlib import Path
from typing import cast, get_args

import pytest

from jamjet_guardrails.chain import _CALLER_STRING_LIMIT, _RUNNABLE_DIRECTIONS, GuardrailChain
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import Context, Direction, Kind, Provenance, Verdict

OUT = Context(direction="output", origin="model")
IN = Context(direction="input", origin="user")

SOURCE = Path(__file__).resolve().parent.parent / "src" / "jamjet_guardrails"


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


def test_every_declared_copy_of_the_runnable_directions_agrees() -> None:
    """One value, declared in seven places, and each looks right alone.

    CHANGED from `test_the_two_declared_copies_...`, which compared the chain's
    copy to the registry's, walked every member into a `Context`, and then
    asserted that five hard-coded strings build no `Context`. It missed the copy
    whose drift FAILS OPEN. Growing `types._DIRECTIONS` alone lets a `Context`
    carry a direction no guardrail declares; `chain.run` then skips every
    guardrail and reports `allow`, no verdicts, content untouched. That mutation
    was made and the entire suite stayed green over a payload carrying an AWS
    key, an email address and an SSN. A denylist of five strings cannot catch a
    sixth.

    Seven copies, none of which can import another without closing a cycle:

        chain._RUNNABLE_DIRECTIONS
        detectors._RUNNABLE_DIRECTIONS
        types._DIRECTIONS
        authoring._RUNNABLE
        eval.corpus._DIRECTIONS
        url_exfiltration._DEFAULT_ON_DETECT           its KEYS
        template_integrity._DEFAULT_ON_DETECT         its KEYS

    CHANGED again, and the reason is the sentence this docstring already
    ended with. Two checks that shipped after it added copies six and seven,
    a default-decision mapping each, whose KEYS encode this same domain and
    which nothing here could see. `url_exfiltration` indexed its mapping with
    every direction a caller declared, so a `Direction` that grew into
    `_RUNNABLE` and not into the mapping threw a bare `KeyError` out of a
    constructor whose contract is `GuardrailUnavailableError`; that module now
    DERIVES its `_RUNNABLE` from these keys, which is why only the mapping is
    listed for it. A list is still a list, so
    `test_no_source_file_declares_an_unguarded_copy_of_the_runnable_directions`
    below finds the eighth by scanning for it.

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
    from jamjet_guardrails.detectors.template_integrity import (
        _DEFAULT_ON_DETECT as TEMPLATE_DEFAULTS,
    )
    from jamjet_guardrails.detectors.url_exfiltration import (
        _DEFAULT_ON_DETECT as URL_DEFAULTS,
    )
    from jamjet_guardrails.detectors.url_exfiltration import _RUNNABLE as URL_DIRECTIONS
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
        ("url_exfiltration on_detect keys", URL_DEFAULTS.keys()),
        ("url_exfiltration directions", URL_DIRECTIONS),
        ("template_integrity on_detect keys", TEMPLATE_DEFAULTS.keys()),
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


def _declared_string_domain(node: ast.expr) -> frozenset[str] | None:
    """The set of string literals a module-level value spells out, or None.

    Four shapes, because they are the four this package writes a domain in: a
    set, list or tuple display of strings, a `frozenset(...)`/`set(...)` around
    one, and a dict display whose KEYS are strings. A value assembled any other
    way is not a literal and cannot be read off the source, which is the limit
    of this guard and is why it is a floor under the named list above rather
    than a replacement for it.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("frozenset", "set") and len(node.args) == 1:
            return _declared_string_domain(node.args[0])
        return None
    if isinstance(node, ast.Dict):
        keys = node.keys
    elif isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        keys = list(node.elts)
    else:
        return None
    if not keys or not all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key in keys
    ):
        return None
    return frozenset(key.value for key in keys)  # type: ignore[union-attr]


def test_no_source_file_declares_an_unguarded_copy_of_the_runnable_directions() -> None:
    """The list above is a list, and the test above says what a list cannot do.

    It said it in its own docstring -- "a denylist of five strings cannot catch
    a sixth" -- and then two checks shipped with a sixth and a seventh and it
    caught neither, because each was a `_DEFAULT_ON_DETECT` mapping whose KEYS
    are this domain and which no module names to any other. One of them was
    indexed unguarded and would have thrown a bare `KeyError` out of a
    constructor.

    So this one finds the copies instead of listing them: every module-level
    assignment under `src/` whose literal value spells exactly the members of
    `Direction`, in any of the four shapes this package writes a domain in, is a
    copy of it, and every copy has to be one the test above holds. The rule is
    derived from `get_args(Direction)` rather than from a set of names somebody
    remembered, which is the same reason `docs/conformance.md` gives for
    deriving an exemption from a property.

    A false positive here is a constant that happens to spell `{"input",
    "output"}` for an unrelated reason. There is no such constant, and if one
    ever appears it is still the same two strings and still worth a maintainer
    looking at the two together.

    Mutation watched: `url_exfiltration._DEFAULT_ON_DETECT` removed from
    `_GUARDED` below, which is exactly the state the package shipped in. FAILS,
    naming the file and the constant.
    """
    declared = frozenset(get_args(Direction))
    assert declared, "Direction declares no members; the scan below would be vacuous"

    guarded = {
        ("chain.py", "_RUNNABLE_DIRECTIONS"),
        ("types.py", "_DIRECTIONS"),
        ("authoring.py", "_RUNNABLE"),
        ("eval/corpus.py", "_DIRECTIONS"),
        ("detectors/__init__.py", "_RUNNABLE_DIRECTIONS"),
        ("detectors/url_exfiltration.py", "_DEFAULT_ON_DETECT"),
        ("detectors/template_integrity.py", "_DEFAULT_ON_DETECT"),
    }

    found: set[tuple[str, str]] = set()
    for path in sorted(SOURCE.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.AnnAssign):
                targets: list[ast.expr] = [node.target]
                value = node.value
            elif isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            else:
                continue
            if value is None or _declared_string_domain(value) != declared:
                continue
            where = path.relative_to(SOURCE).as_posix()
            found.update((where, target.id) for target in targets if isinstance(target, ast.Name))

    assert found == guarded, (
        f"the runnable-direction domain is declared at {sorted(found - guarded)} "
        f"with nothing holding it to Direction, and expected at "
        f"{sorted(guarded - found)}; every copy has to be in "
        "test_every_declared_copy_of_the_runnable_directions_agrees"
    )


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
