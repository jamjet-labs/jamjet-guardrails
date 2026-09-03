"""Run guardrails over one piece of content and combine their verdicts restrictively."""

from __future__ import annotations

import hmac
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.errors import GuardrailChainError, GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import (
    ChainResult,
    Context,
    Decision,
    Finding,
    Kind,
    Provenance,
    Verdict,
    combine,
)

# How much of any single caller-supplied string reaches the audit record. First
# written for an exception's class name, and used below for the one other
# caller-supplied string an error message is allowed to carry: the guardrail's
# own declared name, which is what tells a reader WHICH check misbehaved.
#
# Nothing a `check` RETURNS is interpolated into a message any more, at any
# length. A returned claim is not merely unbounded, it is chosen after the
# detector has seen the content: a class name or a finding type can BE the
# content, so bounding one only shortens the leak. See `_verified`.
_ERROR_TYPE_LIMIT = 200

# The two kinds and the three decisions, as this module's OWN string objects.
# Every comparison against a caller-supplied value puts one of these on the left
# and stores THIS object rather than the one that matched it, so no `str`
# subclass with a lying `__eq__` reaches the audit record even if it managed to
# compare equal on the way in.
#
# Listed literally rather than derived from get_args(Kind)/get_args(Decision),
# for the reason `types.py` gives at each of its own copies: both domains are
# designed to grow, and deriving these would auto-accept a new member before
# anything could handle it.
_KINDS: tuple[Kind, ...] = ("constraint", "classifier")
_DECISIONS: tuple[Decision, ...] = ("allow", "redact", "deny")


def _bounded_str(value: str) -> str:
    """Truncate one caller-supplied string to `_ERROR_TYPE_LIMIT`.

    Shared by `_bounded`, directly below, and by `_Identity`: an exception's
    type name and a guardrail's declared name are two strings with the same
    property. Each is chosen by code this library did not write, so each gets
    the same ceiling. A 2,000,000-character name interpolated four times into
    one message produced a 4,000,073-character `error`, which every log, trace
    and database column downstream of a `ChainResult` then pays for.
    """
    return value[:_ERROR_TYPE_LIMIT]


def _bounded(exc: BaseException) -> str:
    """A detector failure, described without quoting the content it failed on.

    The detector's own message is NOT copied. `f"{type(exc).__name__}: {exc}"`
    put whatever the detector chose to say into the verdict, and the shape a
    detector naturally writes is the one that leaks:

        raise ValueError(f"could not parse: {content!r}")
            ->  error = "ValueError: could not parse: 'AKIA... sk-...'"

    `secrets.check` guarantees "the credential is not in the audit record
    either" and `corpus._reject` states that a rejected value is never echoed,
    "a loader that prints the offending value undoes that on the first malformed
    line". `error` was the one path in this library that broke both, and the only
    place any detector-derived string entered a `Verdict` at all.

    The exception TYPE is kept, because it is the detector's class name and not
    its data, and it is what tells a reader whether they are looking at a regex
    that will not compile or a file that will not open.

    `str(exc)` is dropped whole rather than filtered. No filter can know which of
    a detector's substrings came from the content, a filter that is wrong once
    has leaked the thing it was written to hold back, and truncating instead
    would leak the first N characters of whatever the detector interpolated. The
    traceback still carries the message to whoever is debugging the detector,
    which is where that detail belongs and where it is not persisted.
    """
    return (
        f"{_bounded_str(type(exc).__name__)} raised; the message is withheld "
        "because a detector's message may quote the content"
    )


@dataclass(frozen=True, slots=True)
class _Identity:
    """The chain's own copy of one guardrail's declared identity.

    Built once, in ``GuardrailChain.__init__``, from a single read of each
    attribute, and used for every verdict that guardrail ever produces. The
    guardrail's own attributes are never read again.

    That is the whole of the defence against a mutating attribute. ``name``
    read as a property can return "honest-detector" on the read that validates
    it and "pii" on the read that records it, and both reads look correct at
    their own call site; the record is then a lie no downstream reader can
    detect. Reading once and keeping the copy removes the second read rather
    than trying to make it agree with the first.

    Every field here is a plain ``str`` this module owns: ``name`` and
    ``version`` are exact-``str`` values coerced with ``str(...)`` after their
    type was checked, and ``kind`` is one of ``_KINDS``, this module's own
    object, not whatever compared equal to it. ``named`` is ``name`` bounded for
    use in an error message, which is the only place a caller-supplied string is
    interpolated at all.
    """

    name: str
    version: str
    kind: Kind
    directions: frozenset[str]
    named: str


def _refusal(position: int, clause: str) -> str:
    """Why a chain will not be built, naming the position and nothing else.

    The guardrail is named by its POSITION, which is the chain's own knowledge,
    because the thing being refused is precisely the guardrail's account of
    itself: a name that is not a string cannot be quoted, and one that is a
    2,000,000-character string, or that IS the content the caller is about to
    check, must not be either.
    """
    return (
        f"the guardrail at position {position} in the chain {clause}. A chain built "
        "from it could not describe its own decisions: every verdict it records, "
        "including the deny the chain synthesises when the guardrail fails or lies, "
        "is stamped with the identity declared here"
    )


def _identity_of(guardrail: Guardrail, position: int) -> _Identity:
    """Read one guardrail's declared identity ONCE, or refuse to build the chain.

    Refuses with ``GuardrailUnavailableError``, at construction, mirroring
    ``detectors.build``: a guardrail that cannot say what it is, is a guardrail
    whose decisions cannot be described, and a configuration that cannot
    describe its own decisions is the failure this library exists to prevent.
    Doing it here rather than per verdict is what makes the synthesised deny
    ALWAYS constructible: `Verdict` rejects an unknown provenance kind, so a
    guardrail declaring ``kind="heuristic"`` used to turn the chain's own
    fail-closed path into a ``ValueError`` out of ``run``, abandoning the run
    and its audit record at exactly the moment a detector had misbehaved.

    `type(name) is str`, not `isinstance(name, str)`. `isinstance` consults
    `obj.__class__`, which is an ordinary writable attribute, and a `str`
    subclass gets reflected priority on `==` and `!=` no matter which side it
    is written on, so a subclass with a lying `__eq__` walks past every
    comparison an `isinstance` check admits. A subclass is not acceptable here
    for the same reason: the whole attack class is subclasses with lying
    dunders, and the exact type is the only property that cannot be faked.
    """
    try:
        # One read each, into a local. Every check and every stored copy below
        # is made against THESE objects, so a property that answers differently
        # on a second read has no second read to answer.
        name = guardrail.name
        version = guardrail.version
        kind = guardrail.kind
        declared_directions = list(guardrail.directions)
    except Exception as exc:
        # The exception's own message is not copied, for `_bounded`'s reason:
        # a property that raises is caller code, and its message is caller
        # prose. `raise ... from exc` keeps it in the traceback, where it
        # reaches whoever is debugging the guardrail and is not persisted.
        raise GuardrailUnavailableError(
            _refusal(position, "raised while declaring its name, version, kind or directions")
        ) from exc
    if type(name) is not str:
        raise GuardrailUnavailableError(_refusal(position, "declares a name that is not a str"))
    if type(version) is not str:
        raise GuardrailUnavailableError(_refusal(position, "declares a version that is not a str"))
    # `str.__eq__(ours, theirs)`, not `ours == theirs`. The unbound method is
    # `str`'s own comparison, so a subclass's `__eq__` is never consulted and
    # reflected priority never applies; it returns NotImplemented rather than
    # True for a non-string, which `is True` rejects. The matching member of
    # `_KINDS` is what gets stored, so what lands in the audit record is this
    # module's object and never the one that compared equal to it.
    checked_kind: Kind | None = None
    for known in _KINDS:
        if str.__eq__(known, kind) is True:
            checked_kind = known
            break
    if checked_kind is None:
        raise GuardrailUnavailableError(
            _refusal(position, f"declares a kind that is not one of {list(_KINDS)}")
        )
    return _Identity(
        name=str(name),
        version=str(version),
        kind=checked_kind,
        # Frozen here, over exact strings only. A `directions` that is a
        # generator, that changes between construction and run, or whose
        # `__contains__` answers for itself is no longer consulted at all: the
        # chain runs the guardrails its configuration named, decided against a
        # set this module built. A member that is not an exact `str` is dropped
        # rather than kept, because it could not name a direction a `Context`
        # carries and a lying one would claim to name all of them.
        directions=frozenset(str(d) for d in declared_directions if type(d) is str),
        named=_bounded_str(str(name)),
    )


def _synthesised_deny(identity: _Identity, digest: str, error: str) -> Verdict:
    """A ``deny`` built from the guardrail's OWN declarations, never its return value.

    Both callers in ``run`` reach this because ``guardrail.check`` produced
    something the chain will not pass on: an exception, or a ``Verdict`` whose
    account of itself does not match what the chain independently knows. Either
    way, ``provenance`` here is stamped from the identity read at construction
    -- the identity the guardrail declared by being in the chain at all -- and
    never from anything the untrustworthy return value claimed about itself. A
    classifier that misbehaves must not be recorded as a constraint, whether it
    misbehaved by raising or by lying about what it is.

    This cannot fail. ``kind`` was checked against ``_KINDS`` before the chain
    existed, so the ``Verdict`` below always constructs, which is what makes
    the fail-closed path reachable for every guardrail a chain holds.
    """
    return Verdict(
        decision="deny",
        content=None,
        findings=[],
        provenance=Provenance(kind=identity.kind, detector=identity.name, version=identity.version),
        saw=digest,
        error=error,
    )


def _verified(returned: object, identity: _Identity, digest: str, content: str) -> Verdict:
    """A verdict this library BUILT from a returned one, or a synthesised deny.

    Validating a returned object and then continuing to use that same object is
    the defect this function exists to avoid, so nothing it was handed reaches
    the caller. Every field is read exactly once into a local, checked against
    what the chain independently knows, and then a fresh ``Verdict``, fresh
    ``Provenance`` and fresh ``Finding`` objects are constructed from those
    locals. What reaches ``ChainResult.verdicts``, ``combine`` and the rewrite
    is therefore always an object this library made.

    The reason is that a check is not a guarantee about a mutable, subclassable
    object. A ``str`` subclass whose ``__eq__`` returns True passes any
    comparison and then reports something else to the audit record. An object
    with ``__class__`` set to ``Verdict`` passes ``isinstance`` and can answer a
    property honestly on the read that validates it and falsely on every read
    after. An ``int`` subclass passes ``0 <= start < end <= len(content)`` and
    then indexes as ``-3``, and ``content[cursor:-3]`` emits a long prefix of
    the ORIGINAL content in front of the placeholder: a returned span can
    UN-redact, which is worse than the mislabelling it looks like. None of
    those is fixed by checking harder. They are fixed by not keeping the object.

    ``type(x) is T`` throughout, never ``isinstance``, which will look like the
    obvious choice to the next reader and is not: ``isinstance`` consults
    ``obj.__class__``, an ordinary writable attribute, and it admits subclasses,
    which is the entire attack class. After ``type(x) is str`` a plain ``==`` is
    already safe, because an exact ``str`` has no ``__eq__`` of its own to lie
    with and cannot take reflected priority.

    The chain's own values win wherever the two disagree at all: ``saw`` is set
    to the chain's digest and ``provenance`` to the chain's copy of the declared
    identity, rather than to whatever the verdict claimed and happened to match.

    Every message names the CLAUSE and nothing else. Not the claimed provenance,
    not a finding's type, not ``type(returned).__name__``, not a span. Those are
    strings a detector chooses after seeing the content, and a detector whose
    class name or finding type IS the content puts the content straight into the
    audit record: exactly what `_bounded` refuses for an exception's message,
    reached from the return side instead. The guardrail is named from the
    chain's own bounded copy of the name it declared at construction.
    """
    # `type(...) is Verdict`: `isinstance` here was defeated by a plain object
    # with `__class__ = Verdict` whose `saw` and `provenance` were properties
    # returning the honest value on the read that validated them and a lie on
    # every read after.
    if type(returned) is not Verdict:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned something that is not a Verdict",
        )
    # One read each. `Verdict` is a frozen slots dataclass and the type is
    # exact, so these are slot reads and cannot run caller code -- but the
    # VALUES in those slots are still the detector's, and each is graded below.
    decision = returned.decision
    claimed_content = returned.content
    findings = returned.findings
    provenance = returned.provenance
    claimed_saw = returned.saw
    claimed_error = returned.error

    if type(decision) is not str or decision not in _DECISIONS:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a decision that is not "
            f"one of {list(_DECISIONS)}",
        )
    # Three refusals of one claim, and the third is not redundant. The type
    # check makes the second safe (an exact `str` cannot lie about equality);
    # `hmac.compare_digest` is a second, INDEPENDENT refusal of a lying type,
    # because it accepts only real `str` or `bytes` and compares the characters
    # themselves rather than asking the object whether it is equal. Standard
    # library, so the zero-dependency promise holds.
    if (
        type(claimed_saw) is not str
        or claimed_saw != digest
        or not hmac.compare_digest(claimed_saw, digest)
    ):
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a saw that is not the digest the "
            "chain computed over the content it gave check()",
        )
    if type(provenance) is not Provenance:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a provenance that is not a Provenance",
        )
    claimed_kind = provenance.kind
    claimed_detector = provenance.detector
    claimed_version = provenance.version
    if type(claimed_kind) is not str or claimed_kind != identity.kind:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a provenance.kind that is not "
            "the kind it declares",
        )
    if type(claimed_detector) is not str or claimed_detector != identity.name:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a provenance.detector that is not its own name",
        )
    if type(claimed_version) is not str or claimed_version != identity.version:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a provenance.version that is not "
            "its own version",
        )
    # `model`, `revision` and `threshold` are the three provenance fields the
    # chain CANNOT grade: it knows what the guardrail is called and what kind of
    # check it is, and it has no way to know which model weights answered. They
    # are still copied through rather than dropped, for the reason a finding's
    # type is: a classifier's audit record without the model and revision that
    # decided is not a provenance record at all, and this library exists to make
    # that record complete. Copied means REBUILT -- checked for exact type and
    # coerced -- so what lands in the record is a value this module constructed,
    # holding the detector's claim about itself and nothing that can lie about
    # its own type later.
    claimed_model = provenance.model
    claimed_revision = provenance.revision
    claimed_threshold = provenance.threshold
    if (claimed_model is not None and type(claimed_model) is not str) or (
        claimed_revision is not None and type(claimed_revision) is not str
    ):
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a provenance.model or "
            "provenance.revision that is neither a str nor None",
        )
    if claimed_threshold is not None and (
        type(claimed_threshold) not in (float, int) or not math.isfinite(claimed_threshold)
    ):
        # `math.isfinite`, not another comparison. NaN fails every comparison,
        # `<` and `>` included, so a bound spelled as one more comparison would
        # let NaN walk straight through it exactly as an unchecked `!=` let a
        # lying `str` subclass through before `type(x) is str` was checked
        # first here. `float` and `int` are both always safe to call it on,
        # which is why the type check runs first and stays an `or`, not a
        # second `if`: calling `math.isfinite` on a value that failed it would
        # be calling it on caller data of an unknown shape.
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a provenance.threshold that is "
            "neither a finite number nor None",
        )
    if claimed_content is not None and type(claimed_content) is not str:
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned content that is neither a str nor None",
        )
    if claimed_error is not None:
        # `error` is the chain's field: it says "this verdict was synthesised
        # because the guardrail failed". A detector setting it is either
        # confused or writing its own prose into the audit record, which is the
        # leak `_bounded` exists to close.
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned an error, which only the chain sets",
        )
    if decision == "redact" and claimed_content is None:
        # `Verdict` forbids this state, so it arrives only from a detector that
        # reached around its own type with `object.__setattr__`. It is a deny
        # rather than a raise BECAUSE of which way the alternatives fail:
        # keeping the decision at `redact` over a string nothing rewrote tells
        # the caller to forward un-redacted content, and raising loses the whole
        # run's audit record over one detector. A deny loses neither.
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a redact carrying no content",
        )

    rebuilt: list[Finding] = []
    for finding in findings:
        if type(finding) is not Finding:
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding that is not a Finding",
            )
        finding_type = finding.type
        span = finding.span
        confidence = finding.confidence
        if type(finding_type) is not str:
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding whose type is not a str",
            )
        if len(finding_type) > _ERROR_TYPE_LIMIT:
            # `finding.type` is the one detector-chosen string this library lets
            # reach the rewrite AT ALL, by design: a redaction placeholder has to
            # NAME what claimed the region, so `[REDACTED:EMAIL]` is `EMAIL` read
            # straight out of the finding that redacted it. Unbounded, that design
            # choice is a channel rather than a label. `Finding(type=content,
            # span=(0, len(content)))` on a `redact` makes the placeholder BE the
            # content, reproducing it verbatim inside a string this library calls
            # safe to forward, and a finding whose type runs to millions of
            # characters inflates every audit record downstream of a
            # `ChainResult` by the same amount for one finding. The bound does
            # not and cannot stop a SHORT type from equalling a short secret --
            # see `docs/conformance.md` -- only the unbounded case; it is the
            # same ceiling `_bounded_str` already holds every other caller-chosen
            # string to, and it is generous next to a real type name, which is a
            # short constant such as `INVISIBLE_TAG_CHARS`.
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding whose type "
                f"exceeds {_ERROR_TYPE_LIMIT} characters",
            )
        if confidence is not None and (
            type(confidence) not in (float, int) or not math.isfinite(confidence)
        ):
            # Same reasoning as `provenance.threshold` above: NaN and infinity
            # are both `float`, so the type check alone admits them, and neither
            # is a confidence a caller could threshold against.
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding whose confidence is "
                "neither a finite number nor None",
            )
        # Coerced once, above the branch, so both shapes of finding carry the
        # same rebuilt value and neither can be changed without the other.
        rebuilt_confidence = None if confidence is None else float(confidence)
        if span is None:
            rebuilt.append(
                Finding(type=str(finding_type), span=None, confidence=rebuilt_confidence)
            )
            continue
        # The shape is checked before the bound, and both are checked before
        # anything is read out of the pair. A span of `(1, 2, 3)`, `("a", "b")`
        # or `5` used to raise out of `run` -- unpacking or comparing, either
        # way abandoning the whole run and its audit record over one detector's
        # malformed finding. It is a contract violation like any other now.
        if type(span) is not tuple or len(span) != 2:
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding whose span is not a pair",
            )
        start, end = span
        # `type(...) is int` rejects an `int` SUBCLASS, which is the point: one
        # with lying comparison dunders passed `0 <= start < end <= len(content)`
        # by reflected priority, passed the second copy of the same bound in
        # `_spans_of`, and reached `_rewrite` still lying. It rejects `bool` too,
        # which is not a span offset by any reading.
        if type(start) is not int or type(end) is not int:
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding whose span is not a "
                "pair of plain ints",
            )
        if not 0 <= start < end <= len(content):
            return _synthesised_deny(
                identity,
                digest,
                f"guardrail {identity.named!r} returned a finding whose span does not "
                "index into the content check() was given",
            )
        rebuilt.append(
            Finding(
                type=str(finding_type),
                span=(int(start), int(end)),
                confidence=rebuilt_confidence,
            )
        )

    try:
        return Verdict(
            # `str(...)` of a value already known to be an exact `str` returns
            # that same object, so these coercions cost nothing and say what the
            # code means: from here down, every string in this verdict is one
            # this library produced. `cast` is for mypy alone; `decision` was
            # checked against `_DECISIONS` above.
            decision=cast(Decision, str(decision)),
            content=None if claimed_content is None else str(claimed_content),
            findings=rebuilt,
            # The chain's identity, not the verdict's claim about it, even
            # though the two just compared equal: what is recorded is the value
            # the chain holds. The three fields it cannot hold are rebuilt from
            # the checked reads above.
            provenance=Provenance(
                kind=identity.kind,
                detector=identity.name,
                version=identity.version,
                model=None if claimed_model is None else str(claimed_model),
                revision=None if claimed_revision is None else str(claimed_revision),
                threshold=None if claimed_threshold is None else float(claimed_threshold),
            ),
            # The chain's digest, for the same reason.
            saw=digest,
        )
    except ValueError:
        # Every `Verdict` invariant is either checked above or guaranteed by a
        # value the chain owns, with one exception: the rule that a constraint's
        # findings carry no confidence and a classifier's all do. A detector can
        # break that only by reaching around its own frozen type with
        # `object.__setattr__`, and the pairing is `types.py`'s to define, so it
        # is not restated here. Named rather than left to `run`'s generic
        # handler, whose message would report that a detector raised when what
        # actually happened is that the chain refused to rebuild.
        return _synthesised_deny(
            identity,
            digest,
            f"guardrail {identity.named!r} returned a verdict this library cannot rebuild",
        )


class GuardrailChain:
    """Runs each guardrail whose directions include the context's direction.

    **Every guardrail inspects the content the chain was given.** No guardrail
    ever sees a string another guardrail has already rewritten, so every
    ``Verdict`` in a run carries the same ``saw`` and every span in the run
    indexes into the same string.

    That is not a simplification, it is a leak fix. Rewriting sequentially let a
    redaction by one detector cut a match another detector was about to make:

        SLACK_BOT_TOKEN=xoxb-0000000000-2000000000008-EXAMPLEONLYnotarealtoken

    ``pii`` redacts the 13-digit segment, which is Luhn-valid and starts with a
    2, so it is a card as far as that detector can tell. The placeholder splits
    the token, ``secrets`` then matches only the 16-character prefix, and the
    24-character secret tail survives into content the chain returns as
    ``redact`` with a ``SLACK_TOKEN`` finding: an audit record saying the
    credential was handled, over a string that still contains it. Measured at
    4.87% of canonical Slack bot tokens with a random 13-digit second segment.
    ``["secrets", "pii"]`` leaked none of them, which is the tell: the defect was
    order-dependence, so reordering the README would have hidden it rather than
    fixed it.

    **Redactions are collected and applied once.** Every ``redact`` verdict
    contributes its findings' spans; they are merged exactly as two patterns
    inside one detector are merged, and the whole rewrite happens in a single
    pass over the original content. Overlapping regions collapse into one
    placeholder naming every type that claimed any part of it, sorted, so the
    Slack token above comes back as ``[REDACTED:CREDIT_CARD+SLACK_TOKEN]`` and a
    reader can still see which checks fired over which bytes. Per-detector
    detail is not lost either: each ``Verdict`` keeps its own findings, its own
    spans and its own ``content``, which is that detector's view of the same
    input.

    **A finding's ``type`` is the one detector-chosen string this library lets
    reach the rewrite, by design.** A placeholder has to name what claimed the
    region it replaces, so the type is read straight out of the finding that
    redacted it -- that is what ``EMAIL`` in ``[REDACTED:EMAIL]`` is. What is
    bounded is its length, not its content: a type longer than
    ``_ERROR_TYPE_LIMIT`` characters is a contract violation, refused like any
    other over-long caller string in this module, because unbounded it stops
    being a label and becomes a channel -- a finding whose type IS the content
    it redacted reproduces that content, verbatim, inside a string this
    library calls safe to forward. The bound does not stop a SHORT type from
    equalling a short secret; a real type name is a short constant such as
    ``INVISIBLE_TAG_CHARS``, and the bound is generous next to one. See
    ``_verified``.

    A guardrail's own ``Verdict.content`` is therefore NOT what the chain
    returns. It is that one guardrail's rewrite of the input, correct on its own
    and useful to a caller holding a single guardrail. The chain owns the
    composed rewrite because only the chain can see all the spans at once.

    Decisions combine restrictively (deny > redact > allow) and never weaken, so
    a later guardrail cannot talk an earlier deny back down to allow.

    ``ChainResult.content`` is the merged rewrite, or the original where nothing
    redacted. It is always a real string. **On a deny it is NOT safe to send.** A
    deny means the content must not go through at all; the field is there for the
    audit record, not for delivery. Callers branch on ``decision`` first and only
    ever forward ``content`` on an allow or a redact. Only a redact contributes
    to the rewrite: content returned alongside any other decision is ignored.

    **The identity is read once, at construction.** ``name``, ``version``,
    ``kind`` and ``directions`` are read a single time per guardrail in
    ``__init__``, checked strictly, and kept as this module's own copies; the
    guardrail's attributes are never read again. A guardrail that cannot declare
    a usable identity is refused there and then with
    ``GuardrailUnavailableError``, mirroring ``detectors.build``, because a
    chain built from it could not describe its own decisions. Reading once is
    what closes a ``name`` that answers honestly while it is being validated and
    differently while it is being recorded; checking ``kind`` once is what makes
    the synthesised deny below always constructible.

    Fails closed on a broken detector: if ``check`` raises, the chain records a
    synthesised ``deny`` carrying the failure's TYPE and the guardrail's own
    declared provenance, then keeps going. The detector's own message is not
    recorded; see ``_bounded``. A later guardrail may deny too and its verdict
    belongs in the audit record; the decision cannot weaken, so continuing is
    safe. ``KeyboardInterrupt`` and ``SystemExit`` are not caught: fail-closed
    covers detector bugs, not the operator pressing ctrl-c.

    Fails closed on a returned verdict too, not only a raised one, and does it
    by REBUILDING rather than by approving. ``check`` returning without raising
    says only that the guardrail finished; it says nothing about whether what it
    returned is TRUE. ``saw`` might be the hash of text nobody sent it,
    ``provenance`` might name a different detector, kind or version than the one
    that ran, and a finding's span might index past the content it claims to
    describe or lie about its own value. None of those is distinguishable from
    an honest verdict by its shape, only by checking it against what the chain
    itself passed to ``check``: the content, the digest of that exact content,
    and the guardrail's own declared identity. The chain is the only party
    holding all three, which makes it the only party able to grade the verdict's
    account of itself -- a guardrail attesting to its own provenance is marking
    its own homework, and the mark matters precisely because a false record and
    a true one read identically to everyone downstream.

    Grading alone is not enough, which is why nothing a detector returns is kept.
    A verdict that passes is not appended; a fresh one is built from the checked
    reads, stamped with the chain's own digest and the chain's own copy of the
    identity. A verdict that fails is replaced exactly as a raised exception is:
    a synthesised ``deny`` carrying the guardrail's DECLARED provenance, never
    the false one the verdict returned, and an ``error`` naming which clause
    broke and nothing else. See ``_verified``.

    ``run`` raises in one case: ``GuardrailChainError`` if a guardrail returns a
    redact the chain cannot LOCATE, meaning no findings or a finding without a
    span. The chain rewrites from spans, so an unlocatable redact would leave
    the decision at ``redact`` over a string nothing rewrote, telling the caller
    to forward content that was never rewritten. Nothing else abandons a run. A
    malformed or out-of-range span does not reach it, because ``_verified``
    checks every finding's span over every decision first; nor does a redact
    carrying no content, which is a rebuild ``_verified`` refuses and therefore
    a synthesised deny like any other.

    **A caller must treat any exception out of ``run`` as a deny.** That case
    abandons the run, so there is no ``ChainResult`` and no audit record, which
    is acceptable only because nothing was allowed through. A caller that
    catches and carries on has converted this library from fail-closed to
    fail-open in one line.

    Raises ``GuardrailUnavailableError`` from ``__init__``, not from ``run``,
    for a guardrail that cannot declare a usable ``name``, ``version`` or
    ``kind``, or whose ``directions`` cannot be read. Nothing has been checked
    at that point and nothing has been allowed, so refusing to build is the
    cheapest possible failure and the only one that arrives before a caller has
    content in hand.
    """

    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        self._guardrails = list(guardrails)
        # Read and checked here, once, for every guardrail -- including one that
        # no context will ever run. A chain that exists is a chain that can
        # describe every decision it will ever record.
        self._identities = [
            _identity_of(guardrail, position) for position, guardrail in enumerate(self._guardrails)
        ]

    def run(self, content: str, context: Context) -> ChainResult:
        decision: Decision = "allow"
        verdicts: list[Verdict] = []
        found: list[tuple[str, tuple[int, int]]] = []
        digest = saw(content)

        for guardrail, identity in zip(self._guardrails, self._identities, strict=True):
            # Against the chain's OWN frozen copy of the declared directions, so
            # a `directions` object cannot decide per call whether it is in this
            # run.
            if context.direction not in identity.directions:
                continue
            # The call AND the grading of what it returned are both inside the
            # try. They used to be split, with the grading in an `else:`, and
            # that split was the bug: a malformed span raised out of `run` from
            # code written to keep the run alive, abandoning the audit record
            # over exactly the sort of detector this check exists to catch. Any
            # exception from any part of it is now the same synthesised deny.
            #
            # `_spans_of` below is deliberately still OUTSIDE: its
            # GuardrailChainError is an assertion about this library's own
            # consistency, not a detector failure, and swallowing it would bury
            # a library bug in a verdict.
            try:
                # `content`, never a working value. This argument is the whole of
                # the fix in the docstring: a detector that is handed a rewritten
                # string can match a truncated credential and report success over
                # a tail that is still there.
                returned = guardrail.check(content, context)
                verdict = _verified(returned, identity, digest, content)
            except Exception as exc:  # noqa: BLE001 - fail closed on any detector bug
                verdict = _synthesised_deny(identity, digest, _bounded(exc))
            verdicts.append(verdict)
            decision = combine(decision, verdict.decision)
            if verdict.decision == "redact":
                found += self._spans_of(verdict, identity, content)

        # `found` is sorted here rather than by each detector, because spans from
        # different guardrails interleave and `_merge` requires text order over
        # the whole set. A tie on the start offset puts the shorter span first;
        # `sorted` is stable, so equal spans keep the order their guardrails ran
        # in and the result is total either way.
        rewritten = _rewrite(content, sorted(found, key=lambda pair: pair[1])) if found else content
        return ChainResult(decision=decision, content=rewritten, verdicts=verdicts)

    @staticmethod
    def _spans_of(
        verdict: Verdict, identity: _Identity, content: str
    ) -> list[tuple[str, tuple[int, int]]]:
        """The spans one redact contributes, or a refusal to apply it at all.

        Every refusal here has the same shape: a redact the chain cannot place is
        a redact the chain cannot perform, and performing it partially or not at
        all while still reporting ``redact`` tells the caller to forward a string
        that was not rewritten.

        The bounds check is not defensive padding. Spans arrive from a
        ``Guardrail`` implementation, which is caller-supplied code, and
        ``_rewrite`` indexes with them. A negative start makes ``content[cursor
        : start]`` a slice measured from the END of the string, which emits a
        long prefix of the ORIGINAL, un-redacted content in front of the
        placeholder. So an out-of-range span does not merely mislabel a region,
        it un-redacts one.

        Repeated here rather than trusted from upstream. `_verified` already runs
        this same bound over every finding of every decision before `run` ever
        calls this method, and every `Finding` reaching here was built by
        `_verified` from plain ints, so a `redact` arriving today always has
        spans already known to be in range and unable to lie about it. This
        method's own check is not redundant for that reason: it is what makes
        `_spans_of` correct on its own terms, callable directly against any
        `Verdict` a test or a future caller constructs, rather than correct only
        in combination with a precondition enforced somewhere else in the file.

        The messages name the guardrail from the chain's own bounded copy of its
        declared name and never quote a finding's type, which is a string the
        detector chose after seeing the content.
        """
        if verdict.content is None:
            # Unreachable from `run` twice over: `Verdict` rejects a redact with
            # no content, and `_verified` refuses to rebuild one, so a redact
            # arriving here through a chain always carries content. mypy still
            # needs the narrowing, and this method is callable directly. It
            # raises rather than falling through because of which way the
            # fallback fails: leaving the content alone would keep the decision
            # at "redact" over content nothing rewrote, telling the caller to
            # forward an un-redacted string. Raising costs nothing and fails
            # closed.
            raise GuardrailChainError(
                f"guardrail {identity.named!r} returned a redact with no content"
            )
        if not verdict.findings:
            raise GuardrailChainError(
                f"guardrail {identity.named!r} returned a redact with no findings; the "
                "chain rewrites from spans, so it has nothing to redact and would "
                "report a redact over content nothing changed"
            )
        spans: list[tuple[str, tuple[int, int]]] = []
        for finding in verdict.findings:
            if finding.span is None:
                raise GuardrailChainError(
                    f"guardrail {identity.named!r} returned a redact whose finding "
                    "carries no span; the chain rewrites from spans, so it cannot "
                    "apply this redaction"
                )
            start, end = finding.span
            if not 0 <= start < end <= len(content):
                raise GuardrailChainError(
                    f"guardrail {identity.named!r} returned a redact whose finding "
                    f"spans a region that does not index into the {len(content)} "
                    "characters it was given"
                )
            spans.append((finding.type, (start, end)))
        return spans
