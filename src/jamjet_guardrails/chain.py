"""Run guardrails over one piece of content and combine their verdicts restrictively."""

from __future__ import annotations

from collections.abc import Sequence

from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.errors import GuardrailChainError
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import (
    ChainResult,
    Context,
    Decision,
    Provenance,
    Verdict,
    combine,
)

# How much of any single detector-chosen string reaches the audit record. First
# written for an exception's class name and reused below by `_contract_violation`
# for a returned verdict's own claims about itself -- its provenance and its
# findings' types -- because the argument is identical in every case: none of
# these strings is ours, each is caller-supplied and can be any length, and an
# unbounded one in a verdict is a cost paid by every log, trace and database
# column downstream of a ChainResult.
_ERROR_TYPE_LIMIT = 200


def _bounded_str(value: str) -> str:
    """Truncate one caller-supplied string to `_ERROR_TYPE_LIMIT`.

    Shared by `_bounded`, directly below, and by `_contract_violation` further
    down: an exception's type name, a verdict's claimed provenance and a
    finding's type are four different strings with the same property. Each is
    chosen by code this library did not write, so each gets the same ceiling.
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

    Fails closed on a broken detector: if ``check`` raises, the chain records a
    synthesised ``deny`` carrying the failure's TYPE and the guardrail's own
    provenance, then keeps going. The detector's own message is not recorded;
    see ``_bounded``. A later guardrail may deny too and its verdict
    belongs in the audit record; the decision cannot weaken, so continuing is
    safe. ``KeyboardInterrupt`` and ``SystemExit`` are not caught: fail-closed
    covers detector bugs, not the operator pressing ctrl-c.

    Fails closed on a returned verdict too, not only a raised one. ``check``
    returning without raising says only that the guardrail finished; it says
    nothing about whether what it returned is TRUE. ``saw`` might be the hash of
    text nobody sent it, ``provenance`` might name a different detector, kind or
    version than the one that ran, and a finding's span might index past the
    content it claims to describe. None of those is distinguishable from an
    honest verdict by its shape, only by checking it against what the chain
    itself passed to ``check``: the content, the digest of that exact content,
    and the guardrail's own declared identity. The chain is the only party
    holding all three, which makes it the only party able to grade the
    verdict's account of itself -- a guardrail attesting to its own provenance
    is marking its own homework, and the mark matters precisely because a false
    record and a true one read identically to everyone downstream. A verdict
    that fails this check is replaced exactly as a raised exception is: a
    synthesised ``deny`` carrying the guardrail's own DECLARED provenance, never
    the false one the verdict returned, and an ``error`` naming which clause
    broke. See ``_contract_violation``.

    Raises, in the three cases a verdict cannot honestly be applied:

    - ``GuardrailChainError`` if a guardrail returns a redact carrying no
      content. ``Verdict`` forbids that state, so this is an assertion about the
      library's own consistency rather than a detector failure.
    - ``GuardrailChainError`` if a guardrail returns a redact the chain cannot
      LOCATE: no findings, or a finding without a span. The chain rewrites from
      spans, so an unlocatable redact would leave the decision at ``redact``
      over a string nothing rewrote, which is the same fail-open the no-content
      case is about, arriving one step later. An out-of-range span does NOT
      reach this raise: ``_contract_violation`` bounds every finding's span,
      over every decision, before a redact ever gets here, so a span that fails
      that bound has already become a synthesised ``deny`` two paragraphs up.
    - ``ValueError`` from ``Verdict`` if a guardrail whose ``check`` raised also
      declares a ``kind`` this library does not know. Provenance is synthesised
      from the guardrail's own declarations; no fallback kind exists that would
      not put a claim the detector never made into the audit record.

    **A caller must treat any exception out of ``run`` as a deny.** All three
    cases abandon the run, so there is no ``ChainResult`` and no audit record,
    which is acceptable only because nothing was allowed through. A caller that
    catches and carries on has converted this library from fail-closed to
    fail-open in one line.
    """

    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        self._guardrails = list(guardrails)

    def run(self, content: str, context: Context) -> ChainResult:
        decision: Decision = "allow"
        verdicts: list[Verdict] = []
        found: list[tuple[str, tuple[int, int]]] = []
        digest = saw(content)

        for guardrail in self._guardrails:
            if context.direction not in guardrail.directions:
                continue
            # The try wraps ONLY the check call. Widening it to the loop body would
            # swallow the GuardrailChainError below, which is an assertion about
            # this library's own consistency, not a detector failure.
            try:
                # `content`, never a working value. This argument is the whole of
                # the fix in the docstring: a detector that is handed a rewritten
                # string can match a truncated credential and report success over
                # a tail that is still there.
                verdict = guardrail.check(content, context)
            except Exception as exc:  # noqa: BLE001 - fail closed on any detector bug
                verdict = self._synthesised_deny(guardrail, digest, _bounded(exc))
            else:
                # A `check` that returns rather than raises has said nothing
                # about whether its answer is honest, only that it finished.
                # `_contract_violation` is where the chain marks that answer
                # against what it independently knows.
                violation = self._contract_violation(verdict, guardrail, digest, content)
                if violation is not None:
                    verdict = self._synthesised_deny(guardrail, digest, violation)
            verdicts.append(verdict)
            decision = combine(decision, verdict.decision)
            if verdict.decision == "redact":
                found += self._spans_of(verdict, guardrail, content)

        # `found` is sorted here rather than by each detector, because spans from
        # different guardrails interleave and `_merge` requires text order over
        # the whole set. A tie on the start offset puts the shorter span first;
        # `sorted` is stable, so equal spans keep the order their guardrails ran
        # in and the result is total either way.
        rewritten = _rewrite(content, sorted(found, key=lambda pair: pair[1])) if found else content
        return ChainResult(decision=decision, content=rewritten, verdicts=verdicts)

    @staticmethod
    def _synthesised_deny(guardrail: Guardrail, digest: str, error: str) -> Verdict:
        """A ``deny`` built from the guardrail's OWN declarations, never its return value.

        Both callers in ``run`` reach this because ``guardrail.check`` produced
        something the chain will not pass on: an exception, or a ``Verdict``
        whose account of itself does not match what the chain independently
        knows. Either way, ``provenance`` here is stamped from ``guardrail.kind``,
        ``.name`` and ``.version`` -- the identity the guardrail declared by
        being in the chain at all -- and never from anything the untrustworthy
        return value claimed about itself. A classifier that misbehaves must not
        be recorded as a constraint, whether it misbehaved by raising or by
        lying about what it is.
        """
        return Verdict(
            decision="deny",
            content=None,
            findings=[],
            provenance=Provenance(
                kind=guardrail.kind, detector=guardrail.name, version=guardrail.version
            ),
            saw=digest,
            error=error,
        )

    @staticmethod
    def _contract_violation(
        verdict: object, guardrail: Guardrail, digest: str, content: str
    ) -> str | None:
        """The one clause of the returned-verdict contract this call broke, or ``None``.

        ``check`` returning without raising is not evidence that what it
        returned is true. ``saw``, ``provenance`` and every finding's span are
        the guardrail's OWN account of what it did, and the chain is the only
        party positioned to check that account against reality: it is the one
        that built ``digest`` and handed ``content`` to ``check``, so it is the
        one party that knows what "the text the decision was actually made
        about" actually was. A guardrail attesting to its own provenance is
        marking its own homework, and grading it here is not optional -- a
        ``saw`` over different text, a ``provenance`` naming a different
        detector, or a span into a string the chain never built are wrong
        VALUES sitting in a right-shaped record. Nothing about their shape
        marks them as wrong, which is exactly why this failure is silent
        everywhere else: a false audit record and a true one read identically
        to anything downstream, and only the party holding the original content
        and the original digest can tell them apart.

        Checked in this order because every check after the first assumes it:
        there is no ``.saw`` or ``.provenance`` to read on something that is not
        a ``Verdict`` at all, which is the one shape a detector can return that
        this function must not touch before asking.

        Every branch returns a fixed sentence naming the clause that failed.
        Where a caller-supplied value is included -- a claimed type name, a
        claimed kind, detector or version, a finding's type -- it is bounded by
        `_bounded_str` first, the same way `_bounded` bounds an exception's type
        name: none of those strings is ours, and an unbounded one would let a
        hostile guardrail grow the audit record without limit simply by
        returning a longer lie. ``saw`` is shown in full because it is not
        caller-chosen prose, it is a fixed-length hash, and a hash reveals
        nothing about what it was taken over. The content itself, and any
        span's surrounding text, never appear here: only offsets and lengths,
        which locate a problem without repeating the thing this library exists
        to keep out of a verdict.
        """
        if not isinstance(verdict, Verdict):
            return (
                f"guardrail {guardrail.name!r} returned "
                f"{_bounded_str(type(verdict).__name__)!r} from check(), not a Verdict"
            )
        if verdict.saw != digest:
            return (
                f"guardrail {guardrail.name!r} returned saw={verdict.saw}, but the chain "
                f"computed {digest} over the content it gave check()"
            )
        if verdict.provenance.kind != guardrail.kind:
            return (
                f"guardrail {guardrail.name!r} returned provenance.kind "
                f"{_bounded_str(str(verdict.provenance.kind))!r}, but declares kind "
                f"{guardrail.kind!r}"
            )
        if verdict.provenance.detector != guardrail.name:
            return (
                f"guardrail {guardrail.name!r} returned provenance.detector "
                f"{_bounded_str(verdict.provenance.detector)!r}, but its own name is "
                f"{guardrail.name!r}"
            )
        if verdict.provenance.version != guardrail.version:
            return (
                f"guardrail {guardrail.name!r} returned provenance.version "
                f"{_bounded_str(verdict.provenance.version)!r}, but its own version is "
                f"{guardrail.version!r}"
            )
        for finding in verdict.findings:
            if finding.span is None:
                continue
            start, end = finding.span
            if not 0 <= start < end <= len(content):
                return (
                    f"guardrail {guardrail.name!r} returned a finding of type "
                    f"{_bounded_str(finding.type)!r} whose span {(start, end)!r} does not "
                    f"index into the {len(content)} characters check() was given"
                )
        return None

    @staticmethod
    def _spans_of(
        verdict: Verdict, guardrail: Guardrail, content: str
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

        Repeated here rather than trusted from upstream. `_contract_violation`
        already runs this same bound over every finding of every decision before
        `run` ever calls this method, so a `redact` reaching here today always
        has spans already known to be in range. This method's own check is not
        redundant for that reason: it is what makes `_spans_of` correct on its
        own terms, callable directly against any `Verdict` a test or a future
        caller constructs, rather than correct only in combination with a
        precondition enforced somewhere else in the file.
        """
        if verdict.content is None:
            # Verdict already rejects a redact with no content, so this cannot
            # fire today; mypy still needs the narrowing. It raises rather than
            # falling through because of which way the fallback fails: leaving
            # the content alone would keep the decision at "redact" over content
            # nothing rewrote, telling the caller to forward an un-redacted
            # string. Raising costs nothing and fails closed.
            raise GuardrailChainError(
                f"guardrail {guardrail.name!r} returned a redact with no content"
            )
        if not verdict.findings:
            raise GuardrailChainError(
                f"guardrail {guardrail.name!r} returned a redact with no findings; the "
                "chain rewrites from spans, so it has nothing to redact and would "
                "report a redact over content nothing changed"
            )
        spans: list[tuple[str, tuple[int, int]]] = []
        for finding in verdict.findings:
            if finding.span is None:
                raise GuardrailChainError(
                    f"guardrail {guardrail.name!r} returned a redact whose finding "
                    f"{finding.type!r} carries no span; the chain rewrites from spans, so "
                    "it cannot apply this redaction"
                )
            start, end = finding.span
            if not 0 <= start < end <= len(content):
                raise GuardrailChainError(
                    f"guardrail {guardrail.name!r} returned a redact whose finding "
                    f"{finding.type!r} spans {finding.span!r}, which does not index into "
                    f"the {len(content)} characters it was given"
                )
            spans.append((finding.type, (start, end)))
        return spans
