"""Bundled detectors, and the registry that refuses to skip a missing one."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors.injection_structural import (
    INJECTION_TYPES,
    InjectionStructuralGuardrail,
)
from jamjet_guardrails.detectors.pii import PII_TYPES, PiiGuardrail
from jamjet_guardrails.detectors.rules import RULES_TYPES, build_rules
from jamjet_guardrails.detectors.secrets import SECRET_TYPES, SecretsGuardrail
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail
from jamjet_guardrails.types import Direction

AVAILABLE: dict[str, Callable[..., Guardrail]] = {
    "injection-structural": InjectionStructuralGuardrail,
    "pii": PiiGuardrail,
    "rules": build_rules,
    "secrets": SecretsGuardrail,
}

# The finding types each registered check can report, keyed by registry name.
#
# Public, and beside AVAILABLE rather than inside the Guardrail protocol,
# because a port is held to the verdicts it produces and not to a table it
# exposes: adding a protocol member for this would put a requirement into the
# conformance contract that has nothing to do with conformance.
#
# It exists because three things need to know a check's types before running
# it: the README row test, the corpus completeness test, and the scaffold that
# writes a new check's files. It lived in tests/test_readme.py, where none of
# the other two could reach it.
#
# For a check whose types are configured rather than fixed, this is the set the
# PUBLISHED ROW is measured under, which is the fixture in
# jamjet_guardrails.eval.fixtures. A user's own types at runtime are outside
# this table and outside every claim made about it.
TYPES: dict[str, frozenset[str]] = {
    "injection-structural": INJECTION_TYPES,
    "pii": PII_TYPES,
    "rules": RULES_TYPES,
    "secrets": SECRET_TYPES,
}

# The directions a Context can actually carry, listed literally and deliberately
# NOT derived from get_args(Direction) -- the reason types.py gives for handling
# Kind the same way. Direction is designed to grow, and deriving this would
# auto-accept a new direction before anything could produce a Context carrying
# it, which is the hole in the other direction. Adding to the Literal instead
# refuses a guardrail declaring only the new direction, loudly and at
# construction, until someone has taught the chain to run it.
_RUNNABLE_DIRECTIONS: frozenset[Direction] = frozenset({"input", "output"})

# The same ceiling `chain._CALLER_STRING_LIMIT` holds, spelled here because
# `detectors` imports `chain` and not the other way round, and because this is a
# message bound rather than a refusal: `build` reports which guardrail it
# refused, and a name is the only way to say that.
_NAME_LIMIT = 200


def build(name: str, **options: object) -> Guardrail:
    """Construct one guardrail by name, or refuse to hand one back at all.

    Raises ``GuardrailUnavailableError`` in five cases, all at construction.
    They are one mistake in five costumes: a check that is configured and
    would not check. A caller holding a guardrail built by this function may
    still raise this error from ``PatternGuardrail.check`` if they call it
    with a direction the guardrail does not declare, but a chain does not:
    it filters directions before calling check.

    - **The name is not registered.** The message names what IS installed, read
      from ``AVAILABLE`` rather than written out, so a detector living behind an
      optional extra appears in the message on the machine that has it.
    - **The factory did not return a Guardrail.** ``None``, or an object missing
      a member, is refused here rather than at first use. Not cosmetic: an
      object carrying ``kind`` but no ``directions`` raises ``AttributeError``
      from the chain's direction test, which sits OUTSIDE the try that makes a
      broken detector fail closed, so it abandons the whole run instead of
      denying one guardrail, and nothing after it in the chain runs either. A
      runtime protocol check buys member presence and nothing else, which is
      precisely what is absent in that shape.
    - **The guardrail declares an unknown kind.** Task 5 made the chain
      synthesise a deny carrying the dead guardrail's own provenance when
      ``check`` raises, so an unknown kind AND a raise makes
      ``Verdict.__post_init__`` reject the synthesised verdict and ``run``
      propagate a ``ValueError`` from deep inside the error path.
    - **The guardrail declares no direction it can ever run in.** An empty
      ``directions``, or one holding only directions no ``Context`` carries,
      makes a guardrail inert by construction: the chain skips it in every
      context. One inert guardrail produces exactly the output ``build_chain``
      refuses to build from an empty list -- allow, content untouched, no
      verdicts -- and alongside a live detector it makes the chain quieter than
      the caller asked for while raising nothing to say so.
    - **The detector's own options select nothing.** ``build("pii",
      types=frozenset())`` returned a guardrail that allowed an email, an SSN,
      a card and an AWS key in one string, from a configuration a caller can
      write: ``guardrails: {pii: {types: []}}``. This one is raised BY the
      detector, in its ``__init__``, because only the detector knows what its
      own options mean; it is listed here because it is the same mistake as the
      four above and it reaches a caller through this function, with the same
      error type, so one ``except GuardrailUnavailableError`` around the
      configuration seam catches all five.

      An option value that is out of DOMAIN is a different thing and keeps the
      detector's own ``ValueError``: ``types={"PASSPORT"}`` is a bad argument,
      not an unavailable check, and this function deliberately does not wrap a
      detector's constructor.

    This all sits in ``build`` rather than in ``build_chain`` because ``build``
    is public too. A factory that hands back an object which only detonates on
    first use has moved the failure away from the mistake; an object that goes
    quiet instead of detonating has moved it away for good.
    """
    if not isinstance(name, str):
        # `build` is public and reached by untyped callers exactly as
        # `build_chain` is. An unhashable name reached the lookup below and
        # raised `TypeError: unhashable type: 'list'`.
        raise GuardrailUnavailableError(
            f"build() takes a guardrail name as a string, not the {type(name).__name__} {name!r}"
        )
    try:
        factory = AVAILABLE[name]
    except KeyError:
        # Sorted by repr, which is total over any key. Plain `sorted` compares
        # the keys against each other, so a single non-string key made the
        # message raise `TypeError: '<' not supported between instances of
        # 'int' and 'str'` -- while building the error that reports a guardrail
        # is unavailable. An error path that cannot construct its own error
        # emits something outside the contract instead.
        raise GuardrailUnavailableError(
            f"guardrail {name!r} is not available; installed: {sorted(AVAILABLE, key=repr)}"
        ) from None
    if not callable(factory):
        # Checked rather than caught. Wrapping the call below in `except
        # TypeError` would close this hole and swallow a genuine TypeError
        # raised INSIDE a detector's __init__ along with it, which is a
        # detector bug that must stay visible.
        raise GuardrailUnavailableError(
            f"the object registered as {name!r} is not callable: {factory!r}"
        )
    guardrail = factory(**options)
    if not isinstance(guardrail, Guardrail):
        raise GuardrailUnavailableError(
            f"the factory registered as {name!r} returned {guardrail!r}, which does not "
            "implement Guardrail; expected name, version, kind, directions and check"
        )
    if guardrail.kind not in ("constraint", "classifier"):
        raise GuardrailUnavailableError(
            f"guardrail {guardrail.name!r} declares unknown kind "
            f"{guardrail.kind!r}; expected 'constraint' or 'classifier'"
        )
    # Membership, one direction at a time, and NOT set intersection. `&` needs
    # both operands to be sets, and `directions` is a protocol member a detector
    # supplies: a guardrail declaring `directions = ["output"]` satisfies the
    # protocol, runs correctly in the chain, and was refused here with a raw
    # TypeError -- the error type outside this seam's contract that the input
    # checks below exist to eliminate. Tuple, list and str directions all failed
    # the same way. `chain.run` tests membership, which every container
    # supports, so testing membership here makes this check PREDICT the chain
    # rather than approximate it.
    try:
        runnable = any(d in guardrail.directions for d in _RUNNABLE_DIRECTIONS)
    except TypeError as exc:
        # The try wraps ONLY the membership test, the same discipline chain.py
        # applies to its `check` call. `directions = None` clears the protocol
        # check above -- presence is all that check buys, and the member IS
        # present, as None -- and then every operation on it raises. The chain
        # raises the identical TypeError from its direction test, which sits
        # outside the try that makes a broken detector fail closed, so it would
        # abandon the whole run.
        raise GuardrailUnavailableError(
            f"guardrail {guardrail.name!r} declares directions "
            f"{guardrail.directions!r}, which cannot be tested for membership ({exc}); "
            f"the chain would raise the same TypeError mid-run"
        ) from exc
    if not runnable:
        # Neither the name nor the declared directions are quoted. Both are
        # values the guardrail chose, `directions` is read from caller code that
        # can run after the caller has content in hand, and `chain._refusal`
        # records what a message that quotes either one costs: a
        # two-million-character declaration produced a message of the same size,
        # and a declaration that IS the content puts the content into whatever
        # log wraps the configuration seam. The name is bounded because it is
        # what tells a reader WHICH check was refused; the directions are
        # counted, because their values add nothing a count does not.
        raise GuardrailUnavailableError(
            f"guardrail {str(guardrail.name)[:_NAME_LIMIT]!r} declares "
            f"{len(list(guardrail.directions))} direction(s), none of which it can run in; "
            f"every context would skip it, so it would be configured and silent. "
            f"Expected at least one of {sorted(_RUNNABLE_DIRECTIONS)}"
        )
    return guardrail


def build_chain(names: Iterable[str]) -> GuardrailChain:
    """Construct a chain, refusing anything that would check less than was asked.

    Raises ``GuardrailUnavailableError``, at construction, for every shape of
    "this configuration does not mean what it looks like":

    - **Any named guardrail is absent**, or ``build`` refuses it for any of the
      reasons above.
    - **``names`` is empty.** An empty chain allows everything: it returns
      ``allow`` over content full of credentials and records no verdicts. A
      config listing no guardrails is a mistake -- a typo, a misparsed key, an
      empty list -- not a request to disable the library. Callers who genuinely
      want no checks construct ``GuardrailChain([])`` themselves, which is
      explicit at the call site.
    - **``names`` is a string or bytes.** Both are iterables of their own parts,
      so a scalar config value (``guardrails: pii`` rather than
      ``guardrails: [pii]``) would otherwise be read as three one-character
      names.
    - **``names`` is not iterable at all.** A config key present but empty
      (``guardrails:`` with no value) arrives as ``None``.
    - **``names`` holds anything that is not a string.** One bracket too many
      (``guardrails: [[pii]]``) reached the registry lookup itself.

    The last three raise ``GuardrailUnavailableError`` rather than the
    ``TypeError`` they used to, because a caller wrapping this seam in
    ``except GuardrailUnavailableError`` is exactly the caller a malformed
    config reaches, and an error type outside the contract walks straight past.
    """
    if isinstance(names, (str, bytes, bytearray)):
        raise GuardrailUnavailableError(
            f"build_chain() takes an iterable of guardrail names, not the "
            f"{type(names).__name__} value {names!r}; pass a list such as ['pii', 'secrets']"
        )
    if not isinstance(names, Iterable):
        # The null-config explanation is attached only when the value IS null.
        # Appended unconditionally it told `build_chain(42)` it was a config key
        # present but empty, which is a cause the message has not established --
        # the one message here that named something it had not checked.
        cause = (
            " A config key that is present but empty -- `guardrails:` with no value --"
            " arrives here as None."
            if names is None
            else ""
        )
        raise GuardrailUnavailableError(
            f"build_chain() takes an iterable of guardrail names; a "
            f"{type(names).__name__} is not iterable.{cause}"
        )
    requested = list(names)
    not_strings = [entry for entry in requested if not isinstance(entry, str)]
    if not_strings:
        raise GuardrailUnavailableError(
            f"build_chain() takes guardrail names as strings; these entries are "
            f"not strings: {not_strings!r}"
        )
    # Built eagerly and in the caller's order. Every name is resolved before the
    # chain exists, so a chain that was constructed at all has every guardrail
    # its configuration named, and every one of them can run somewhere.
    guardrails = [build(name) for name in requested]
    if not guardrails:
        # The emptiness question is asked of the RESULT, not of `names`. An
        # earlier form asked `if not names`, which reads the same and is not:
        # every generator is truthy, so `build_chain(n for n in [])` walked
        # straight past it and handed back a chain that reported allow over an
        # AWS key, an email address and an SSN with the content untouched.
        # Counting what was built cannot be fooled by the container, nor by a
        # Sequence subclass that lies about its own length.
        raise GuardrailUnavailableError(
            "build_chain() needs at least one guardrail name; an empty chain "
            "allows all content. Pass GuardrailChain([]) directly if that is "
            "what you want."
        )
    return GuardrailChain(guardrails)


__all__ = [
    "AVAILABLE",
    "TYPES",
    "InjectionStructuralGuardrail",
    "PiiGuardrail",
    "SecretsGuardrail",
    "build",
    "build_chain",
    "build_rules",
]
