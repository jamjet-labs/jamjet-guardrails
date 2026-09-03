"""The registry seam, whose whole job is to refuse to skip a check.

Every test here is about one failure direction: a guardrail that is named in
configuration but does not run. Absent, empty, misspelt and mis-shaped inputs
all have to arrive as an exception at construction, because the alternative --
a chain that reports `allow` over content nobody checked -- is indistinguishable
from a working deployment right up until it matters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType
from typing import cast

import pytest

import jamjet_guardrails

# Imported the way the documented front door reads. This line is also the check
# mypy performs: under no_implicit_reexport a name imported into __init__.py but
# left out of __all__ is not importable from the root, and this file is inside
# `[tool.mypy] files`, so the gate fails rather than the surface silently
# shrinking.
from jamjet_guardrails import build_chain as build_chain_from_the_root
from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors import AVAILABLE, PiiGuardrail, build, build_chain
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import Context, Direction, Kind, Provenance, Verdict

OUT = Context(direction="output", origin="model")


def test_every_bundled_detector_is_registered() -> None:
    assert set(AVAILABLE) == {
        "injection-structural",
        "pii",
        "rules",
        "secrets",
        "url-exfiltration",
    }


def test_build_returns_a_working_guardrail() -> None:
    guardrail = build("pii")
    assert guardrail.name == "pii"
    assert guardrail.check("alice@example.com", OUT).decision == "redact"


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_build_returns_the_guardrail_asked_for_and_not_a_default(name: str) -> None:
    """Every entry, both directions.

    `build("pii").name == "pii"` on its own cannot tell a real lookup from a
    hardcoded fallback to PiiGuardrail, and the registry key is the one constant
    such a fallback would coincide with. Running it over every registered name
    means no single hardcoded return value can satisfy the whole set. It also
    pins the key to the detector's own name, which is what `provenance.detector`
    carries into the audit record: a key that disagreed with it would make the
    audit trail name a guardrail the config never asked for.

    `**options_for(name)` because `rules` refuses to build with no options at
    all; every other entry's fixture is empty, so this is a no-op for them.
    """
    assert build(name, **options_for(name)).name == name


def test_build_passes_options_through() -> None:
    guardrail = build("secrets", on_match="deny")
    assert guardrail.check("AKIAIOSFODNN7EXAMPLE", OUT).decision == "deny"
    # The default is "redact", so the assertion above cannot pass by accident:
    # an implementation that dropped **options would produce this instead.
    assert build("secrets").check("AKIAIOSFODNN7EXAMPLE", OUT).decision == "redact"


def test_an_unknown_name_raises_and_names_what_is_available() -> None:
    with pytest.raises(GuardrailUnavailableError) as exc:
        build("injection")
    message = str(exc.value)
    assert "injection" in message
    assert "pii" in message and "secrets" in message


def test_build_chain_fails_at_construction_not_at_run() -> None:
    """The whole point: a configured-but-absent check must never silently skip."""
    with pytest.raises(GuardrailUnavailableError):
        build_chain(["pii", "injection"])


def test_an_empty_chain_is_refused_rather_than_silently_allowing_everything() -> None:
    """An empty chain returns allow over anything. That must not be the default.

    Verified against a real payload: GuardrailChain([]).run(...) reports allow,
    leaves the content untouched and records no verdicts, so a config that lists
    no guardrails would disable the library while looking configured.
    """
    with pytest.raises(GuardrailUnavailableError, match="at least one"):
        build_chain([])


def test_an_empty_iterable_that_is_not_a_sequence_is_refused_too() -> None:
    """`not names` interrogates the input; the refusal has to interrogate the result.

    Measured before this test existed: `build_chain(n for n in [])` returned a
    chain that reported allow over an AWS key, an email address and an SSN with
    the content untouched and no verdicts recorded -- byte for byte the
    fail-open the empty-list refusal exists to prevent, reached through a door
    an input-shaped check does not cover, because every generator is truthy.

    A config loader that yields names rather than listing them is a supported
    caller -- the signature is `Iterable[str]` -- so this passes the generator
    straight in. It needed a `cast` while the parameter said `Sequence`, and a
    test that must cast to reach supported behaviour is the signature telling on
    itself.
    """
    no_names: list[str] = []
    with pytest.raises(GuardrailUnavailableError, match="at least one"):
        build_chain(name for name in no_names)


def test_the_empty_chain_this_refusal_exists_to_prevent_really_does_allow_everything() -> None:
    """The measurement the refusal above rests on, kept executable.

    If GuardrailChain([]) ever stopped allowing everything, the refusal would be
    guarding nothing and this test says so rather than leaving the reason to a
    docstring nobody re-runs.
    """
    payload = "key AKIAIOSFODNN7EXAMPLE mail alice@example.com ssn 123-45-6789"
    result = GuardrailChain([]).run(payload, OUT)
    assert result.decision == "allow"
    assert result.content == payload
    assert result.verdicts == ()


class BadKind:
    """A guardrail declaring a kind this library does not know.

    The cast is on `kind` and nowhere else, because that is precisely the lie:
    the class satisfies the Guardrail protocol in every other respect, which is
    exactly the shape a third-party detector would have. Static typing cannot
    stop it, so the registry has to.

    Its name and version differ from both bundled detectors' ("pii"/"secrets",
    both at 0.1.0), so nothing here can pass by coinciding with a constant the
    registry could have hardcoded.
    """

    name: str = "bad"
    version: str = "7.3.1"
    kind: Kind = cast(Kind, "heuristic")
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
        raise AssertionError("must not run")


class Inert:
    """A guardrail that declares no direction, so no context can ever run it.

    Not the degenerate case of "a guardrail declaring one direction runs in that
    direction" -- that property is about a guardrail that runs somewhere. This
    one runs nowhere, by construction.

    Name and version differ from both bundled detectors and from every other
    double here, so no assertion about it can coincide with a hardcodable value.
    """

    name: str = "inert"
    version: str = "4.2.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset()

    def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
        raise AssertionError("must not run")


class OnlyUnrunnableDirection:
    """Declares a direction no Context can carry. Inert by the same mechanism.

    `Context.direction` is Literal["input", "output"], so a guardrail declaring
    only "sideways" is skipped by every context exactly as an empty frozenset
    is. Refusing the empty set but not this one would guard the spelling rather
    than the property.
    """

    name: str = "sideways"
    version: str = "5.5.5"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({cast(Direction, "sideways")})

    def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
        raise AssertionError("must not run")


class Half:
    """Has `kind`, lacks `directions`: the shape that aborts a run outright.

    Deliberately does NOT satisfy the Guardrail protocol, which is why it needs
    a cast to be registered at all. Member presence is exactly what is missing,
    and exactly what a runtime protocol check sees.
    """

    name: str = "half"
    version: str = "9.9.9"
    kind: Kind = "constraint"

    def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
        raise AssertionError("must not run")


class ListDirections:
    """Declares `directions` as a list. The chain runs it perfectly well.

    `chain.run` tests `context.direction not in guardrail.directions`, which
    every container supports, and `isinstance(x, Guardrail)` sees only that the
    member is present. So this is a working detector, and a check that predicts
    the chain has to accept it. Set intersection did not: `&` needs both
    operands to be sets, so this raised a raw TypeError -- outside the contract
    the neighbouring input checks exist to enforce.

    Unlike the other doubles here its `check` really runs, because the point is
    that the chain reaches it.
    """

    name: str = "listdirs"
    version: str = "3.1.4"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = cast("frozenset[Direction]", ["output"])

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind="constraint", detector=self.name, version=self.version),
            saw(content),
        )


class NoneDirections:
    """Declares `directions = None`.

    Passes `isinstance(x, Guardrail)` because that is a presence check and the
    member IS present -- as None. Everything that then operates on it raises.
    """

    name: str = "nodirs"
    version: str = "6.6.6"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = cast("frozenset[Direction]", None)

    def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
        raise AssertionError("must not run")


def test_a_guardrail_whose_directions_cannot_be_tested_for_membership_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The third instance of one shape: a bare TypeError out of the seam.

    `any(d in None for d in ...)` raises, and `directions = None` clears the
    protocol check because presence is all that check buys. So a malformed
    registration walked past `except GuardrailUnavailableError`.
    """
    monkeypatch.setitem(AVAILABLE, "nodirs", NoneDirections)
    with pytest.raises(GuardrailUnavailableError, match="membership"):
        build("nodirs")


def test_the_chain_refuses_the_directions_that_refusal_predicts() -> None:
    """CHANGED from `test_the_chain_raises_the_typeerror_that_refusal_predicts`,
    which asserted CPython's own `TypeError: argument of type 'NoneType' is not
    iterable` out of `run`.

    The refusal predicts the chain here too, not just for the container shapes,
    and this is kept executable so the two cannot drift: `build` refuses exactly
    the `directions` the chain refuses. What changed is the shape of the chain's
    half. The direction test used to read `guardrail.directions` on every run,
    outside the try that makes a broken detector fail closed, so a malformed one
    surfaced as a bare `TypeError` mid-run and walked straight past a caller's
    `except GuardrailUnavailableError`. The chain now reads `directions` once
    when it is built and refuses what it cannot read, in the contract's own
    error type, before any content has been checked.

    Nothing here depends on how CPython words its message any more, which was
    its own maintenance problem: 3.14 says "is not a container or iterable"
    where 3.10 through 3.13 say "is not iterable", and matching the longer
    phrase once made both CI legs red.
    """
    with pytest.raises(GuardrailUnavailableError, match="position 0 in the chain"):
        GuardrailChain([cast(Guardrail, NoneDirections())])


def test_a_name_that_is_not_a_string_is_refused_by_name() -> None:
    """`build` is public and takes untyped callers too, same as `build_chain`.

    An unhashable name reached the registry lookup and raised
    `TypeError: unhashable type: 'list'`.
    """
    with pytest.raises(GuardrailUnavailableError, match="not the list"):
        build(cast(str, ["pii"]))


def test_a_registration_that_is_not_callable_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed registration must produce the contract error, not `'int' object
    is not callable`."""
    monkeypatch.setitem(AVAILABLE, "notcallable", cast(Callable[..., Guardrail], 5))
    with pytest.raises(GuardrailUnavailableError, match="not callable"):
        build("notcallable")


def test_a_detectors_own_constructor_error_is_not_disguised_as_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the callable check is a check and not a `try`.

    Wrapping the construction call in `except TypeError` would close the
    non-callable hole and swallow a genuine TypeError raised INSIDE a detector's
    __init__ along with it. A detector bug must stay a detector bug and keep its
    own message; only the registration itself is this seam's to judge.
    """

    def exploding_init(**options: object) -> Guardrail:
        raise TypeError("detector __init__ is broken")

    monkeypatch.setitem(AVAILABLE, "boom", exploding_init)
    with pytest.raises(TypeError, match="detector __init__ is broken"):
        build("boom")

    # Same boundary, real detector: PiiGuardrail's own validation is Task 6's
    # documented contract and must reach the caller unwrapped, not be relabelled
    # as an availability problem.
    with pytest.raises(ValueError, match="unknown PII types"):
        build("pii", types=frozenset({"BOGUS"}))


def test_the_availability_message_survives_a_registry_key_that_is_not_a_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error PATH must not raise. This one did.

    `sorted(AVAILABLE)` over mixed key types raised
    `TypeError: '<' not supported between instances of 'int' and 'str'` -- while
    building the very message that reports a guardrail is unavailable. The
    contract error could not be constructed, so a bare TypeError came out of the
    seam instead. Sorting by repr is total over any key.
    """
    monkeypatch.setitem(AVAILABLE, cast(str, 5), PiiGuardrail)
    with pytest.raises(GuardrailUnavailableError, match="not available"):
        build("injection")


def test_a_guardrail_whose_directions_are_not_a_set_is_built_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must predict the chain, not approximate it.

    Measured against the set-intersection form: list, tuple, str and None
    directions each raised `TypeError: unsupported operand type(s) for &`, while
    this same guardrail returned `allow` and recorded its verdict when the chain
    ran it. A check that refuses what the chain runs is wrong twice over -- it
    rejects a working detector, and it does so with an error type this seam
    promises callers never have to catch.
    """
    monkeypatch.setitem(AVAILABLE, "listdirs", ListDirections)

    assert build("listdirs").name == "listdirs"

    result = build_chain(["listdirs"]).run("anything", OUT)
    assert [v.provenance.detector for v in result.verdicts] == ["listdirs"]


def test_a_guardrail_declaring_an_unknown_kind_is_rejected_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 5 leaves this failing deep in the error path; catch it at the seam."""
    monkeypatch.setitem(AVAILABLE, "bad", BadKind)
    with pytest.raises(GuardrailUnavailableError, match="unknown kind"):
        build_chain(["bad"])
    assert isinstance(GuardrailChain([]), GuardrailChain)


def test_build_rejects_an_unknown_kind_too_not_only_build_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build` is public, so a guardrail it hands back must already be usable.

    Returning an object that only detonates on first use would put the failure a
    long way from the mistake, which is the same distance the chain-level check
    exists to close.
    """
    monkeypatch.setitem(AVAILABLE, "bad", BadKind)
    with pytest.raises(GuardrailUnavailableError, match="unknown kind"):
        build("bad")


def test_a_guardrail_that_can_never_run_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inert guardrail is a configured check that does not check."""
    monkeypatch.setitem(AVAILABLE, "inert", Inert)
    with pytest.raises(GuardrailUnavailableError, match="none of which it can run in"):
        build("inert")
    with pytest.raises(GuardrailUnavailableError, match="none of which it can run in"):
        build_chain(["inert"])


def test_a_guardrail_whose_only_direction_no_context_carries_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same property, different spelling: it can never run either."""
    monkeypatch.setitem(AVAILABLE, "sideways", OnlyUnrunnableDirection)
    with pytest.raises(GuardrailUnavailableError, match="none of which it can run in"):
        build("sideways")


def test_the_chain_now_refuses_the_inert_guardrail_it_used_to_run_silently() -> None:
    """CHANGED from `test_the_chain_an_inert_guardrail_produces_is_the_one_build_
    chain_refuses`, which RAN an inert guardrail through `GuardrailChain` and
    asserted the silence it produced.

    That test documented the hole rather than closing it. `build` and
    `build_chain` refused an inert guardrail on the registry door; `GuardrailChain`
    took it, and the output was byte for byte the `GuardrailChain([])` output the
    emptiness refusal exists to prevent: allow, content untouched, no verdicts.
    Direct construction is a supported door, not a back one, because the chain's
    own docstring tells a caller who genuinely wants no checks to write
    `GuardrailChain([])` themselves.

    `_identity_of` now makes the same refusal the registry makes, so the shape
    below cannot be built at all. The mixed case is the one that mattered: an
    inert guardrail beside a live detector made the chain quieter than the
    caller asked for and raised nothing to say so, and a count of the guardrails
    built could not see it because two were built.

    Mutation-checked: deleting the intersection test in `_identity_of` restores
    both silences and fails both halves of this test.
    """
    with pytest.raises(GuardrailUnavailableError, match="none of which it can run in"):
        GuardrailChain([Inert()])

    mixed: list[Guardrail] = [PiiGuardrail(), Inert()]
    with pytest.raises(GuardrailUnavailableError, match="position 1 in the chain"):
        GuardrailChain(mixed)


def test_the_chain_refuses_a_half_built_guardrail_before_it_checks_anything() -> None:
    """CHANGED from `test_the_chain_does_not_fail_closed_on_a_half_built_guardrail`,
    which asserted a bare `AttributeError` out of `run`.

    Why the protocol check is not cosmetic, kept executable. A guardrail missing
    `directions` used to abort a whole run with `AttributeError`, from a
    direction test that sat outside the try, and nothing downstream of it ran
    either: an error type outside the contract, at the worst possible moment,
    with content already in hand.

    The chain now reads `name`, `version`, `kind` and `directions` once, when it
    is built, and a guardrail that cannot supply them is refused there with
    `GuardrailUnavailableError` -- the same error `build` raises for the same
    object, which is what makes one `except GuardrailUnavailableError` cover
    both doors into this room. `build`'s refusal is still the one a caller meets
    first; this is the backstop for a guardrail that never went through it.
    """
    with pytest.raises(GuardrailUnavailableError, match="position 0 in the chain"):
        GuardrailChain([cast(Guardrail, Half())])


def test_a_factory_returning_a_half_built_object_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(AVAILABLE, "half", cast(Callable[..., Guardrail], Half))
    with pytest.raises(GuardrailUnavailableError, match="does not implement"):
        build("half")


def test_a_factory_returning_none_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same check subsumes None and any other non-guardrail object."""
    monkeypatch.setitem(AVAILABLE, "null", cast(Callable[..., Guardrail], lambda: None))
    with pytest.raises(GuardrailUnavailableError, match="does not implement"):
        build("null")


def test_the_availability_message_is_read_from_the_registry_not_hardcoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both names appear in a hardcoded "installed: ['pii', 'secrets']" too.

    That is the whole of what the message test above proves, so it proves nothing
    about where the list came from -- and where it came from is the point: in
    Phase 2 the injection detector arrives through an optional extra, and an
    operator diagnosing a refusal needs the message to name what is *actually*
    installed in front of them. A third name only this test registers is the
    fixture no literal can satisfy.
    """
    monkeypatch.setitem(AVAILABLE, "bad", BadKind)
    with pytest.raises(GuardrailUnavailableError) as exc:
        build("injection")
    assert "bad" in str(exc.value), str(exc.value)


def test_build_chain_runs_the_named_guardrails_in_order() -> None:
    result = build_chain(["pii", "secrets"]).run(
        "mail alice@example.com key AKIAIOSFODNN7EXAMPLE", OUT
    )
    assert result.decision == "redact"
    assert "[REDACTED:EMAIL]" in result.content
    assert "[REDACTED:AWS_ACCESS_KEY]" in result.content
    assert [v.provenance.detector for v in result.verdicts] == ["pii", "secrets"]


def test_build_chain_takes_its_order_from_the_caller_not_from_the_registry() -> None:
    """["pii", "secrets"] is also alphabetical order AND the registry's own order.

    A build_chain that iterated AVAILABLE and ignored `names` entirely would pass
    the test above. The reversed list is the fixture that separates the three.
    """
    result = build_chain(["secrets", "pii"]).run(
        "mail alice@example.com key AKIAIOSFODNN7EXAMPLE", OUT
    )
    assert [v.provenance.detector for v in result.verdicts] == ["secrets", "pii"]


def test_a_bare_string_is_not_read_as_a_sequence_of_one_name() -> None:
    """`str` is an Iterable[str], so `guardrails: pii` in YAML lands here.

    Without the guard the characters "p", "i", "i" are each looked up and the
    first raises, which is loud but names the wrong mistake.
    """
    with pytest.raises(GuardrailUnavailableError) as exc:
        build_chain("pii")
    assert "iterable of guardrail names" in str(exc.value)
    assert "'pii'" in str(exc.value)


def test_bytes_are_refused_the_same_way_a_string_is() -> None:
    """b"pii" iterates into integers, so every lookup would raise the wrong thing."""
    with pytest.raises(GuardrailUnavailableError) as exc:
        build_chain(cast(Iterable[str], b"pii"))
    assert "bytes" in str(exc.value)


def test_a_config_key_present_but_null_is_refused_by_name() -> None:
    """`guardrails:` with no value is the misparsed key the refusals exist for.

    It arrived as `TypeError: 'NoneType' object is not iterable`, which a config
    loader wrapping `except GuardrailUnavailableError` does not catch -- the one
    error type this seam promises is the one it did not raise.
    """
    with pytest.raises(GuardrailUnavailableError, match="is not iterable"):
        build_chain(cast(Iterable[str], None))


def test_the_not_iterable_message_names_a_null_config_only_when_it_is_null() -> None:
    """The one message that could name a cause it has not established.

    Both halves are checked, not just the interesting one: appended
    unconditionally, the null-config explanation told `build_chain(42)` it was a
    config key present but empty, which is a different mistake entirely.
    """
    with pytest.raises(GuardrailUnavailableError) as null_exc:
        build_chain(cast(Iterable[str], None))
    assert "arrives here as None" in str(null_exc.value)

    with pytest.raises(GuardrailUnavailableError) as int_exc:
        build_chain(cast(Iterable[str], 42))
    assert "int is not iterable" in str(int_exc.value)
    assert "None" not in str(int_exc.value), str(int_exc.value)


def test_names_that_are_not_strings_are_refused_by_name() -> None:
    """`guardrails: [[pii]]` -- one bracket too many -- reached AVAILABLE itself.

    An unhashable name raised `TypeError: unhashable type: 'list'` from the dict
    lookup: same defect class as the bare string, one layer further in.
    """
    with pytest.raises(GuardrailUnavailableError, match="not strings"):
        build_chain(cast(Iterable[str], [["pii"]]))


def test_build_chain_returns_a_chain() -> None:
    assert isinstance(build_chain(["pii"]), GuardrailChain)


# --- the package's public surface -------------------------------------------


def test_every_exported_name_is_importable_from_the_package_root() -> None:
    """`__all__` naming something absent breaks `import *` at the consumer."""
    missing = [name for name in jamjet_guardrails.__all__ if not hasattr(jamjet_guardrails, name)]
    assert missing == [], f"listed in __all__ but not importable: {missing}"


def test_every_public_name_at_the_package_root_is_exported() -> None:
    """The other direction: an import added without wiring __all__ is half-done.

    Under mypy's no_implicit_reexport such a name is invisible to consumers
    while looking present to anyone reading the module, so the two halves have
    to be checked against each other rather than either against a literal list.
    Submodules are excluded: they are attributes of the package by virtue of
    being imported, not part of the surface this list curates.
    """
    public = {
        name
        for name, value in vars(jamjet_guardrails).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    }
    assert public, "found no public names at the package root; this guard proves nothing"
    assert public == set(jamjet_guardrails.__all__)


def test_dunder_all_is_sorted() -> None:
    """Sorted, so a later addition has one correct place and cannot hide."""
    assert list(jamjet_guardrails.__all__) == sorted(jamjet_guardrails.__all__)


def test_the_root_names_are_the_registry_functions_themselves() -> None:
    """Re-exported, not reimplemented: a second copy would drift from this one."""
    assert jamjet_guardrails.build is build
    assert jamjet_guardrails.build_chain is build_chain
    assert build_chain_from_the_root is build_chain


def test_a_detector_configured_to_check_nothing_is_refused_through_build() -> None:
    """The fifth costume, reached the way a configuration reaches it.

    `build("pii", types=frozenset())` returned a guardrail that allowed all of
    an email, an SSN, a card and an AWS key at once. That is the output
    `build_chain` refuses to build from an empty list of names, arriving through
    a detector option instead of through the list.

    Asserted through `build` and not only on the detector, because the error
    type is the point: a caller wrapping this seam in
    `except GuardrailUnavailableError` is exactly the caller a malformed config
    reaches, and this refusal has to be inside that net with the other four.
    """
    with pytest.raises(GuardrailUnavailableError, match="empty set of types"):
        build("pii", types=frozenset())


def test_a_detector_configured_with_some_types_is_still_built() -> None:
    """The false-reject control: the refusal above must be about EMPTY, not about types=."""
    guardrail = build("pii", types=frozenset({"EMAIL"}))
    assert guardrail.check("mail alice@example.com", OUT).decision == "redact"


def test_every_registered_check_declares_the_types_it_can_report() -> None:
    """Both directions. A check in AVAILABLE with no TYPES entry has a README
    row nothing checks and a corpus whose labels nothing constrains; a TYPES
    entry for a check nothing registers is a table describing something that
    does not exist."""
    from jamjet_guardrails.detectors import TYPES

    assert set(TYPES) == set(AVAILABLE)


def test_no_two_checks_claim_the_same_finding_type() -> None:
    """Disjointness across checks, which generalises the pairwise test the
    three structural signals already carry. Two checks reporting one type name
    make a merged placeholder ambiguous about which check fired and make a
    per-type row in the published table the sum of two different measurements."""
    from jamjet_guardrails.detectors import TYPES

    seen: dict[str, str] = {}
    collisions: list[str] = []
    for check, types in sorted(TYPES.items()):
        for type_name in sorted(types):
            if type_name in seen:
                collisions.append(f"{type_name} claimed by {seen[type_name]} and {check}")
            seen[type_name] = check
    assert collisions == [], collisions
