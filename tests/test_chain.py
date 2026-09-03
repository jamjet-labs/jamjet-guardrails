from typing import cast

import pytest

from jamjet_guardrails import build_chain
from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors import AVAILABLE, build
from jamjet_guardrails.errors import GuardrailChainError, GuardrailUnavailableError
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import (
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Provenance,
    Verdict,
)

OUT = Context(direction="output", origin="model")
IN = Context(direction="input", origin="user")


def _prov(name: str) -> Provenance:
    return Provenance(kind="constraint", detector=name, version="0.1.0")


class Rewriter:
    """Redacts one token wherever it occurs, so the chain's rewrite is observable.

    `replace` is what THIS guardrail puts in its own `Verdict.content`. It is not
    what the chain returns: the chain rewrites from spans and substitutes its own
    placeholder, so a double that reported a rewrite its findings did not locate
    would be testing a shape the chain now refuses. Findings and `content` here
    describe the same edits, which is what the chain requires of a real detector.
    """

    # Annotated at class level so the doubles satisfy the protocol's declared types:
    # a bare `kind = "constraint"` infers `str`, which is not `Kind`.
    name: str
    version: str
    kind: Kind
    directions: frozenset[Direction]

    def __init__(
        self,
        name: str,
        find: str,
        replace: str,
        directions: frozenset[Direction] = frozenset({"input", "output"}),
        type_name: str = "TOK",
    ) -> None:
        self.name, self.version, self.kind = name, "0.1.0", "constraint"
        self.directions = directions
        self._find, self._replace, self._type = find, replace, type_name

    def check(self, content: str, context: Context) -> Verdict:
        spans = []
        start = content.find(self._find)
        while start != -1:
            spans.append((start, start + len(self._find)))
            start = content.find(self._find, start + len(self._find))
        if not spans:
            return Verdict("allow", None, [], _prov(self.name), saw(content))
        return Verdict(
            "redact",
            content.replace(self._find, self._replace),
            [Finding(type=self._type, span=span) for span in spans],
            _prov(self.name),
            saw(content),
        )


class Denier:
    name: str = "denier"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict("deny", None, [Finding(type="NOPE")], _prov("denier"), saw(content))


class Allower:
    """Always allows. The weakest possible verdict, for the never-weakens tests."""

    name: str = "allower"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict("allow", None, [], _prov("allower"), saw(content))


class DenierHandingBackContent:
    """Denies and also hands back content. The chain must ignore that content.

    Only a redact contributes to the chain's rewrite. A deny that supplies a
    string is either a confused guardrail or a hostile one, and either way its
    string must never reach `ChainResult.content`.
    """

    name: str = "denier-with-content"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def check(self, content: str, context: Context) -> Verdict:
        # A SPANNED finding, so that a chain gating on `content is not None`
        # rather than on the decision would have everything it needs to rewrite.
        # With an unspanned finding this double could not tell the two gates
        # apart: the wrong one would raise rather than redact, which is the
        # right outcome reached for the wrong reason. A deny carrying spanned
        # findings is ordinary, and `secrets(on_match="deny")` returns one.
        return Verdict(
            "deny",
            "attacker-supplied",
            [Finding(type="NOPE", span=(0, 8))],
            _prov("denier-with-content"),
            saw(content),
        )


def test_allow_when_nothing_matches() -> None:
    result = GuardrailChain([Rewriter("a", "zzz", "___")]).run("hello", OUT)
    assert result.decision == "allow"
    assert result.content == "hello"
    assert len(result.verdicts) == 1


def test_no_guardrail_sees_another_guardrails_rewrite() -> None:
    """CHANGED from `test_rewrites_apply_sequentially`, which asserted the defect.

    The old test ran `a` (one -> two) then `b` (two -> three) and asserted the
    chain returned "three", i.e. that `b` had been handed `a`'s output. That is
    exactly the mechanism that let a personal-data placeholder split a Slack bot
    token and leave its 24-character tail standing (see
    `test_a_redaction_cannot_split_a_credential_for_the_next_guardrail`). It is
    now specified out of the library, so the test that pinned it is wrong.

    `b` looks for "two", which is not in the input; it never sees `a`'s output,
    so it allows.
    """
    chain = GuardrailChain([Rewriter("a", "one", "two"), Rewriter("b", "two", "three")])
    result = chain.run("one", OUT)
    assert result.content == "[REDACTED:TOK]"
    assert result.decision == "redact"
    assert [v.decision for v in result.verdicts] == ["redact", "allow"]


def test_every_verdict_in_one_run_hashes_the_same_content() -> None:
    """CHANGED with the test above, and the same reason.

    It asserted `verdicts[1].saw == saw("two")`, i.e. that the second guardrail
    had inspected the first one's output. Every guardrail now inspects the input,
    so every `saw` in a run is the input's digest. `saw` still means what it
    always meant: the hash of the exact string that verdict's spans index into.
    """
    chain = GuardrailChain([Rewriter("a", "one", "two"), Rewriter("b", "one", "four")])
    result = chain.run("one", OUT)
    assert [v.saw for v in result.verdicts] == [saw("one"), saw("one")]


def test_deny_beats_redact_regardless_of_order() -> None:
    first = GuardrailChain([Denier(), Rewriter("r", "x", "y")]).run("x", OUT)
    second = GuardrailChain([Rewriter("r", "x", "y"), Denier()]).run("x", OUT)
    assert first.decision == "deny"
    assert second.decision == "deny"
    # ChainResult.content is the merged rewrite, so a deny that follows a redact
    # still carries the redacted string rather than None or the original. It is a
    # real string on every path, which is exactly why the docstring has to say out
    # loud that a deny means DO NOT SEND IT.
    assert second.content == "[REDACTED:TOK]"


def test_a_later_allow_cannot_weaken_an_earlier_deny() -> None:
    """The accumulator must be monotone, not merely order-independent.

    Every verdict in the test above is non-allow, so a fold that reset to allow
    whenever it saw an allow would survive it untouched. Here the LAST verdict is
    the weakest one, and the result must still be the strongest seen. This is the
    "no code path may weaken a decision" property itself.
    """
    result = GuardrailChain([Denier(), Rewriter("r", "x", "y"), Allower()]).run("x", OUT)
    assert result.decision == "deny"
    assert len(result.verdicts) == 3


def test_a_later_allow_cannot_weaken_an_earlier_redact() -> None:
    """The same property one rung down: allow must not undo a redact either."""
    result = GuardrailChain([Rewriter("r", "x", "y"), Allower()]).run("x", OUT)
    assert result.decision == "redact"
    assert result.content == "[REDACTED:TOK]"


def test_a_deny_that_hands_back_content_does_not_rewrite() -> None:
    """Only a redact rewrites; the decision half of that rule is load-bearing.

    Collecting spans from every verdict that carries content, rather than from
    every verdict that decided `redact`, would let a deny (or anything else that
    hands back findings) rewrite, handing the caller content no guardrail ever
    approved and calling it redacted.
    """
    result = GuardrailChain([DenierHandingBackContent()]).run("original", OUT)
    assert result.decision == "deny"
    assert result.content == "original"


def test_direction_filtering_skips_non_matching_guardrails() -> None:
    chain = GuardrailChain([Rewriter("out-only", "x", "y", frozenset({"output"}))])
    result = chain.run("x", IN)
    assert result.decision == "allow"
    assert result.content == "x"
    assert list(result.verdicts) == []  # it did not run, and says so by being absent


def test_the_same_guardrail_does_run_when_the_direction_matches() -> None:
    """Positive control for the test above.

    "no verdict, unchanged content, allow" is also what an inert guardrail
    produces, so on its own the skip test could pass for the wrong reason. This
    runs the SAME double on the SAME content and only changes the direction: it
    rewrites and records a verdict, so the difference is the filter and nothing else.
    """
    chain = GuardrailChain([Rewriter("out-only", "x", "y", frozenset({"output"}))])
    result = chain.run("x", OUT)
    assert result.decision == "redact"
    assert result.content == "[REDACTED:TOK]"
    assert [v.provenance.detector for v in result.verdicts] == ["out-only"]


# The two tests above vary the CONTEXT while holding the double output-only, so
# between them they only prove the filter for direction="output". These two vary it
# the other way. Together the four cover the whole truth table of declared
# directions against context direction, which is what stops the filter being
# reducible to a constant: hardcoding `!= "output"` kills input guardrailing
# outright, and honouring `directions` on input alone lets every guardrail run on
# output no matter what it declared. Both look fine from the output side only.
def test_an_input_only_guardrail_runs_on_input() -> None:
    chain = GuardrailChain([Rewriter("in-only", "x", "y", frozenset({"input"}))])
    result = chain.run("x", IN)
    assert result.decision == "redact"
    assert result.content == "[REDACTED:TOK]"
    assert [v.provenance.detector for v in result.verdicts] == ["in-only"]


def test_an_input_only_guardrail_is_skipped_on_output() -> None:
    chain = GuardrailChain([Rewriter("in-only", "x", "y", frozenset({"input"}))])
    result = chain.run("x", OUT)
    assert result.decision == "allow"
    assert result.content == "x"
    assert list(result.verdicts) == []


def test_verdicts_are_recorded_in_execution_order() -> None:
    chain = GuardrailChain([Rewriter("a", "1", "2"), Rewriter("b", "2", "3")])
    result = chain.run("1", OUT)
    assert [v.provenance.detector for v in result.verdicts] == ["a", "b"]


def test_empty_chain_allows_and_leaves_content_unchanged() -> None:
    result = GuardrailChain([]).run("hello", OUT)
    assert (result.decision, result.content, list(result.verdicts)) == ("allow", "hello", [])


def test_a_redact_without_content_denies_instead_of_forwarding_the_original() -> None:
    """CHANGED from `test_a_redact_without_content_raises_instead_of_forwarding_the
    _original`, which asserted `GuardrailChainError` out of `run`.

    Pin the DIRECTION of an impossible state's fallback. Verdict.__post_init__
    already rejects a redact carrying no content, so the double below has to
    reach around that invariant with object.__setattr__ to build one. What is
    being pinned is what the chain does IF that invariant is ever relaxed.
    Silently skipping the verdict would leave the decision at "redact" while the
    content is the original, i.e. telling the caller "this was redacted, forward
    it" about content that was not: fail-open on the exact axis this library
    sells.

    It is now a synthesised deny rather than a raise, because the chain no
    longer forwards a returned verdict at all: it rebuilds one, and a redact
    with no content is a rebuild it refuses. That refusal is the ordinary
    fail-closed path, so the whole run keeps its audit record instead of losing
    it to an exception, and the decision is stronger than the redact that was
    claimed rather than weaker.
    """

    class BrokenRedactor:
        name: str = "broken"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            verdict = Verdict("redact", "REDACTED", [], _prov("broken"), saw(content))
            object.__setattr__(verdict, "content", None)  # bypasses the invariant
            return verdict

    result = GuardrailChain([BrokenRedactor()]).run("secret", OUT)
    assert result.decision == "deny"
    assert result.content == "secret"  # nothing was rewritten, and nothing claims it was
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "redact carrying no content" in verdict.error
    assert verdict.provenance == _prov("broken")


# ==========================================================================
# The composition leak, and the property that prevents the class of it.
# ==========================================================================

# The shape from the report, with a body that says so. Canonical Slack bot-token
# shape: `xoxb-`, a 10-digit team id, a 13-digit bot-user id and a 24-character
# secret. The middle segment is Luhn-valid and starts with a 2, so the PII
# detector's bare-card branch matches it: a payment card is exactly "13 to 19
# digits, leading 2 to 6, valid check digit", and this segment is all four. The
# secret carries `EXAMPLEONLY` and `notarealtoken` in its own 24 characters, the
# convention `corpora/NOTICE.md` states, so nothing here reads as a live
# credential to a scanner or to a person. The reported token had a random body
# and it is not needed: the leak turns on the middle segment, and the decision,
# the spans and the merged placeholder are identical either way.
SLACK_TOKEN = "SLACK_BOT_TOKEN=xoxb-0000000000-2000000000008-EXAMPLEONLYnotarealtoken"
SLACK_SECRET_TAIL = "EXAMPLEONLYnotarealtoken"


def test_a_redaction_cannot_split_a_credential_for_the_next_guardrail() -> None:
    """The leak this rule was written for, in the order the README teaches.

    Sequentially, `pii` redacted the 13-digit segment first, the placeholder cut
    the token in two, `secrets` then matched only the 16-character prefix
    `xoxb-0000000000-`, and the 24-character secret tail survived into content
    the chain returned as `redact` carrying a SLACK_TOKEN finding:

        SLACK_BOT_TOKEN=[REDACTED:SLACK_TOKEN][REDACTED:CREDIT_CARD]-EXAMPLEONLYnotarealtoken

    A caller branching on `decision` forwards that. Measured over 20,000
    canonical tokens with a random 13-digit second segment: 973 leaked, 4.87%.

    The tail is asserted absent from `content` rather than the whole placeholder
    asserted present, because the failure is the tail SURVIVING and a test that
    checks the replacement text would also pass on output that carried both.
    """
    result = build_chain(["pii", "secrets"]).run(SLACK_TOKEN, OUT)
    assert result.decision == "redact"
    assert SLACK_SECRET_TAIL not in result.content
    assert result.content == "SLACK_BOT_TOKEN=[REDACTED:CREDIT_CARD+SLACK_TOKEN]"


def test_the_two_orders_of_the_same_two_guardrails_agree() -> None:
    """The property, not the instance.

    Order-dependence is what the leak WAS, so this is the assertion that
    prevents the class rather than the case: any rule under which one guardrail
    can consume evidence another needs shows up here as two different strings.
    Reordering the README would have moved the one token above out of the leak
    and left every other shape of overlap in it.

    The three inputs are the three arrangements available: an overlap the leak
    lived in, two disjoint matches (the README's own quickstart), and a nested
    match, a Luhn-valid card sitting inside a JWT-shaped body.
    """
    for text in (
        SLACK_TOKEN,
        "mail alice@example.com and use sk-abcdefghijklmnopqrstuvwxyz012345",
        "id 4111111111111111 mail bob@example.com card 4012 8888 8888 1881",
    ):
        forward = build_chain(["pii", "secrets"]).run(text, OUT)
        backward = build_chain(["secrets", "pii"]).run(text, OUT)
        assert forward.content == backward.content, text
        assert forward.decision == backward.decision, text
        assert {(f.type, f.span) for v in forward.verdicts for f in v.findings} == {
            (f.type, f.span) for v in backward.verdicts for f in v.findings
        }, text


def test_a_redact_the_chain_cannot_locate_raises_rather_than_reporting_a_rewrite() -> None:
    """The no-content assertion, one step over.

    The chain rewrites from spans. A redact carrying no finding, or a finding
    with no span, gives it nothing to place, so keeping the content as it stood
    would report `redact` over a string nothing rewrote: the same fail-open the
    no-content case exists to stop, arriving after the type system rather than
    before it.
    """

    class Unlocatable:
        name: str = "unlocatable"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def __init__(self, findings: list[Finding]) -> None:
            self._findings = findings

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict("redact", "REDACTED", self._findings, _prov("unlocatable"), saw(content))

    for findings in ([], [Finding(type="TOK")], [Finding(type="TOK", span=(0, 4))]):
        chain = GuardrailChain([Unlocatable(findings)])
        if findings and findings[0].span is not None:
            assert chain.run("secret", OUT).content == "[REDACTED:TOK]et"
        else:
            with pytest.raises(GuardrailChainError, match="rewrites from spans"):
                chain.run("secret", OUT)


@pytest.mark.parametrize("span", [(-3, 4), (0, 0), (4, 2), (2, 99), (99, 200)])
def test_a_span_that_does_not_index_into_the_content_becomes_a_synthesised_deny(
    span: tuple[int, int],
) -> None:
    """CHANGED from `test_a_span_that_does_not_index_into_the_content_is_refused`,
    which asserted that `GuardrailChain([BadSpan()]).run(...)` RAISED
    `GuardrailChainError`.

    That was right while the range check lived only inside `_spans_of`, reached
    only for a `redact`. `_contract_violation` now runs the identical bound over
    every finding of every decision before `_spans_of` is ever called, so a
    `redact` with an out-of-range span is caught there first and turned into a
    synthesised `deny` -- the run keeps its audit record rather than losing it
    to an exception. `_spans_of` still raises for a redact with no findings or
    an unspanned finding
    (`test_a_redact_the_chain_cannot_locate_raises_rather_than_reporting_a_rewrite`),
    because those are not out of range, they are unlocatable, and no decision
    the chain could stamp would be honest about what happened to the content.

    A negative start is the one that leaks rather than merely mislabels:
    `content[cursor:-3]` is measured from the END, so the slice emitted in front
    of the placeholder is a long prefix of the ORIGINAL content. An out-of-range
    span un-redacts, which is exactly why it must never reach the rewrite at
    all now, rather than merely being rejected loudly after the fact.
    """

    class BadSpan:
        name: str = "bad-span"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "redact", "x", [Finding(type="TOK", span=span)], _prov("bad-span"), saw(content)
            )

    result = GuardrailChain([BadSpan()]).run("sensitive-value", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "does not index into" in verdict.error
    # The true provenance the guardrail declared, not "x" and not a finding
    # the guardrail's own redact carried.
    assert verdict.provenance == _prov("bad-span")
    assert verdict.content is None
    assert verdict.findings == ()
    # Nothing redacted: the one guardrail in the chain no longer decided redact.
    assert result.content == "sensitive-value"


# ==========================================================================
# The returned-verdict contract: `check` can return without raising and still
# lie. Every test below reproduces one clause of the report -- a hostile
# guardrail named `liar`, version `9.9.9`, kind `constraint`, whose returned
# verdict claimed a different detector, version, kind and saw, all four
# ACCEPTED by the code this file's tests ran against before this change.
# ==========================================================================


class ProvenanceLiar:
    """Declares one identity and, by default, reports it honestly.

    Each keyword lets a test make exactly one returned field false while
    leaving the guardrail's own declared `name`, `version` and `kind` -- and
    every OTHER returned field -- honest. That isolation is the point: a test
    that set every field wrong at once could pass even if the chain checked
    only one of them.
    """

    name: str = "liar"
    version: str = "9.9.9"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(
        self,
        *,
        kind: Kind = "constraint",
        detector: str = "liar",
        version: str = "9.9.9",
        saw_text: str | None = None,
    ) -> None:
        self._kind = kind
        self._detector = detector
        self._version = version
        self._saw_text = saw_text

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind=self._kind, detector=self._detector, version=self._version),
            saw(self._saw_text) if self._saw_text is not None else saw(content),
        )


def test_a_saw_of_text_never_sent_becomes_a_deny_and_the_run_continues() -> None:
    """Clause 2: `verdict.saw == digest`. `saw` is a hash of the text a
    detector CLAIMS it inspected; a chain that never checked it against the
    text it actually sent would accept a hash of anything.
    """
    chain = GuardrailChain([ProvenanceLiar(saw_text="text the chain never sent"), Allower()])
    result = chain.run("the real content", OUT)
    assert result.decision == "deny"
    assert len(result.verdicts) == 2
    liar_verdict, allower_verdict = result.verdicts
    assert liar_verdict.decision == "deny"
    assert liar_verdict.error is not None
    assert "saw" in liar_verdict.error
    assert allower_verdict.decision == "allow"  # the later guardrail still ran


def test_a_wrong_provenance_kind_alone_becomes_a_deny() -> None:
    """Clause 3. `ProvenanceLiar`'s `detector` and `version` stay honest here,
    so a chain that checked only those two would let this one through.

    CHANGED: this asserted `"classifier" in verdict.error`, i.e. that the
    message quoted the CLAIMED kind. That was the wrong thing to want. A
    returned string is chosen by the detector after it has seen the content, so
    a detector whose claimed kind, detector name or finding type IS the content
    puts the content into the audit record through the error message -- the
    exact leak `_bounded` refuses for an exception's message, reached from the
    return side. The claimed value is now asserted ABSENT.
    """
    result = GuardrailChain([ProvenanceLiar(kind="classifier")]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "provenance.kind" in verdict.error  # the clause that failed, named
    assert "classifier" not in verdict.error  # and the claim itself, not repeated
    # The synthesised deny's own provenance is the declared identity, not
    # what the verdict falsely claimed.
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")


def test_a_wrong_provenance_detector_alone_becomes_a_deny() -> None:
    """Clause 4. `ProvenanceLiar`'s `kind` and `version` stay honest here.

    CHANGED with the test above, and for its reason.
    """
    result = GuardrailChain([ProvenanceLiar(detector="pii")]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "provenance.detector" in verdict.error
    assert "pii" not in verdict.error
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")


def test_a_wrong_provenance_version_alone_becomes_a_deny() -> None:
    """Clause 5. `ProvenanceLiar`'s `kind` and `detector` stay honest here.

    CHANGED with the two tests above, and for their reason.
    """
    result = GuardrailChain([ProvenanceLiar(version="0.1.0")]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "provenance.version" in verdict.error
    assert "0.1.0" not in verdict.error
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")


def test_a_detector_returning_something_that_is_not_a_verdict_becomes_a_deny() -> None:
    """Clause 1. Without it, `_contract_violation` reads `.saw` off whatever
    `check` returned and raises `AttributeError` from inside `run`, which is
    exactly the "abandon the run" failure a synthesised deny exists to avoid.
    """

    class ReturnsNonsense:
        name: str = "nonsense"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            return cast(Verdict, "not a verdict at all")

    result = GuardrailChain([ReturnsNonsense()]).run("hello", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "not a Verdict" in verdict.error
    assert verdict.provenance == Provenance(kind="constraint", detector="nonsense", version="0.1.0")


def test_an_out_of_range_span_on_a_deny_is_caught_which_the_old_code_accepted() -> None:
    """The report's second gap, reproduced exactly: a `deny` carrying
    `span=(9999, 10001)` over 22 characters of content. `_spans_of` only ever
    ran on a `redact`, so a `deny`'s spans were never looked at.
    """
    content = "x" * 22
    guardrail_name = "denier-bad-span"

    class DenierWithBadSpan:
        name: str = guardrail_name
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "deny",
                None,
                [Finding(type="NOPE", span=(9999, 10001))],
                _prov(guardrail_name),
                saw(content),
            )

    result = GuardrailChain([DenierWithBadSpan()]).run(content, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "does not index into" in verdict.error
    assert verdict.findings == ()  # the false finding did not reach the audit record either


def test_an_out_of_range_span_on_an_allow_is_caught_too() -> None:
    """The third decision the old code never checked at all. `_spans_of` was
    reachable only via a `redact`, so neither a `deny` nor an `allow` carrying
    an unlocatable finding was ever looked at. Nothing forbids a finding on an
    `allow` -- `Verdict.__post_init__` does not -- so the bound applies here
    too, on the same terms.
    """

    class AllowerWithBadSpan:
        name: str = "allow-bad-span"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "allow",
                None,
                [Finding(type="ODD", span=(0, 999))],
                _prov("allow-bad-span"),
                saw(content),
            )

    result = GuardrailChain([AllowerWithBadSpan()]).run("short", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "does not index into" in verdict.error


def test_a_none_span_is_still_legal_on_any_decision() -> None:
    """The carve-out in clause 6. A classifier finding carries no span at all,
    and that must not be mistaken for one that fails the bound -- on a `deny`
    exactly as much as on the `allow` the invariant tests already cover.
    """

    class ClassifierDenyNoSpan:
        name: str = "classifier-no-span"
        version: str = "0.1.0"
        kind: Kind = "classifier"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "deny",
                None,
                [Finding(type="TOXIC", span=None, confidence=0.97)],
                Provenance(kind="classifier", detector=self.name, version=self.version),
                saw(content),
            )

    result = GuardrailChain([ClassifierDenyNoSpan()]).run("hello", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    # Not a contract violation: this deny is the guardrail's own, not synthesised.
    assert verdict.error is None
    assert verdict.findings[0].span is None
    # The confidence survives the rebuild. A classifier finding without one is
    # a verdict `Verdict` refuses to construct, so losing it here would turn
    # every classifier verdict into a synthesised deny.
    assert verdict.findings[0].confidence == 0.97
    assert verdict.provenance.kind == "classifier"


def test_the_synthesised_deny_carries_the_guardrails_true_provenance_not_the_false_one() -> None:
    """The report's hostile guardrail, all four fields wrong at once: declares
    name="liar", version="9.9.9", kind="constraint"; returns
    provenance.detector="pii", provenance.version="0.1.0",
    provenance.kind="classifier", and a saw of text it was never given. Every
    field the synthesised deny carries must be the DECLARED one.
    """
    liar = ProvenanceLiar(
        kind="classifier", detector="pii", version="0.1.0", saw_text="text the chain never sent"
    )
    result = GuardrailChain([liar]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")
    assert verdict.saw == saw("the real content")
    assert verdict.content is None
    assert verdict.findings == ()


def test_a_later_guardrail_still_runs_after_a_contract_violation_and_cannot_weaken_it() -> None:
    """The failure posture a raised exception already has, now also true of a
    returned lie: the run does not stop, the later guardrail's own verdict is
    on the record, and nothing it decides can talk the deny back down.
    """
    result = GuardrailChain([ProvenanceLiar(kind="classifier"), Allower()]).run(
        "the real content", OUT
    )
    assert result.decision == "deny"  # the allower did not weaken it
    assert len(result.verdicts) == 2
    liar_verdict, allower_verdict = result.verdicts
    assert liar_verdict.decision == "deny"
    assert allower_verdict.decision == "allow"
    assert allower_verdict.error is None  # its own verdict is untouched


# ==========================================================================
# Read once, check strictly, rebuild.
#
# Every test below is a reproduction that walked straight through the first
# version of the checks above: one that validated the object a detector
# returned and then went on using that same object. Validating untrusted data
# and then trusting the mutable, subclassable thing it arrived in is the whole
# defect, and each of these is one way through it -- a `str` subclass whose
# `__ne__` is False, an object whose `__class__` says Verdict, an `int`
# subclass that passes a bound and then indexes as a negative number.
#
# What closes the class is not a better comparison. It is that nothing a
# detector returns is kept: every field is read once, checked strictly, and a
# fresh Verdict is built from the checked reads, stamped with the chain's own
# digest and the chain's own copy of the guardrail's identity.
#
# Each test asserts the same three things, through `_assert_the_record_is_the
# _chains_own`, and then the one thing that is its own: deny, an error, and a
# provenance and `saw` that are the chain's truth rather than the detector's
# claim.
# ==========================================================================

CONTENT = "AKIAIOSFODNN7EXAMPLE and sk-livesecret"


def _assert_the_record_is_the_chains_own(
    verdict: Verdict, *, detector: str, version: str, content: str
) -> None:
    """The three assertions every attack below shares.

    `saw` is compared against the digest of the content the chain was actually
    given, and `provenance` against the identity the guardrail declared, so a
    chain that recorded the detector's claim instead fails here whatever the
    decision was.
    """
    assert verdict.decision == "deny"
    assert verdict.error is not None
    assert verdict.provenance == Provenance(kind="constraint", detector=detector, version=version)
    assert verdict.saw == saw(content)
    # Exact types, not merely equal values: a recorded `str` subclass compares
    # equal to anything it likes, so equality alone cannot tell the chain's own
    # digest from one that only claims to be it.
    assert type(verdict.saw) is str
    assert type(verdict.provenance.detector) is str


class LyingStr(str):
    """A `str` subclass that answers every comparison the way its holder wants.

    `__ne__` returning False is what walked past `verdict.saw != digest`, and
    writing the comparison the other way round buys nothing: a subclass
    instance takes reflected priority on `==` and `!=` whichever side it is on,
    so `digest != verdict.saw` calls this class too. `type(x) is str` is what
    refuses it, because the exact type is the one property it cannot fake.
    """

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return hash(str(self))


class SawLiar:
    """Hashes text the chain never sent, and hides it in a `LyingStr`."""

    name: str = "saw-liar"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            _prov("saw-liar"),
            LyingStr(saw("text the chain never sent")),
        )


def test_a_saw_that_lies_through_a_str_subclass_becomes_a_deny() -> None:
    """Attack 1. The verdict claimed allow over a hash of text nobody sent it.

    The double is proven hostile before it is used: a test whose fixture turns
    out to be inert passes without exercising anything.
    """
    lying = LyingStr(saw("text the chain never sent"))
    assert (lying != saw(CONTENT)) is False  # the `!=` the first version used
    assert (saw(CONTENT) != lying) is False  # and the same the other way round

    result = GuardrailChain([SawLiar()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="saw-liar", version="0.1.0", content=CONTENT
    )
    assert "saw" in (verdict.error or "")


@pytest.mark.parametrize(
    ("field", "lie"),
    [("kind", "classifier"), ("detector", "pii"), ("version", "0.1.0")],
    ids=["kind", "detector", "version"],
)
def test_a_provenance_field_that_lies_through_a_str_subclass_becomes_a_deny(
    field: str, lie: str
) -> None:
    """Attack 1b, the same subclass one field over.

    Provenance is the claim this library exists to make true, so a lie here is
    the one that matters most: a record naming a detector that never ran, at a
    version nobody deployed, over content it never saw.

    One field at a time, and the other two honest, for the reason
    `ProvenanceLiar` above is built the same way: a double that lies about
    everything at once passes even against a chain that checks only one field.
    Written as a single triple-lying double, this test stayed green when the
    `kind` comparison was reverted to the bare `!=` a `str` subclass walks
    through, because the `detector` comparison caught it instead.
    """

    class SubclassLiar:
        name: str = "liar"
        version: str = "9.9.9"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            claimed: dict[str, str] = {"kind": "constraint", "detector": "liar", "version": "9.9.9"}
            claimed[field] = LyingStr(lie)
            return Verdict(
                "allow",
                None,
                [],
                Provenance(
                    kind=cast(Kind, claimed["kind"]),
                    detector=claimed["detector"],
                    version=claimed["version"],
                ),
                saw(content),
            )

    result = GuardrailChain([SubclassLiar()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(verdict, detector="liar", version="9.9.9", content=CONTENT)
    assert f"provenance.{field}" in (verdict.error or "")  # the clause, named
    assert lie not in (verdict.error or "")  # and the claim, not repeated
    # The declared identity is what the record carries, in every field.
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")


class VerdictLookalike:
    """Not a `Verdict`. `isinstance` says otherwise, and that was enough.

    `isinstance` consults `obj.__class__`, which is an ordinary writable
    attribute, so this object passed the type check and then answered its
    `saw` and `provenance` reads as it pleased: the honest value on the read
    that validated it, a different one on every read after. Nothing rebuilt the
    verdict, so what reached the audit record was whatever the later reads
    returned.
    """

    __class__ = Verdict  # type: ignore[assignment]

    def __init__(self, honest_saw: str, honest_provenance: Provenance) -> None:
        self._reads = 0
        self._honest_saw = honest_saw
        self._honest_provenance = honest_provenance
        self.decision = "allow"
        self.content: str | None = None
        self.findings: tuple[Finding, ...] = ()
        self.error: str | None = None

    @property
    def saw(self) -> str:
        self._reads += 1
        return self._honest_saw if self._reads == 1 else saw("text the chain never sent")

    @property
    def provenance(self) -> Provenance:
        if self._reads <= 4:
            return self._honest_provenance
        return Provenance(kind="classifier", detector="pii", version="0.1.0")


class ClassSpoofer:
    name: str = "liar"
    version: str = "9.9.9"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def check(self, content: str, context: Context) -> Verdict:
        return cast(
            Verdict,
            VerdictLookalike(
                saw(content), Provenance(kind="constraint", detector="liar", version="9.9.9")
            ),
        )


def test_an_object_that_only_claims_to_be_a_verdict_becomes_a_deny() -> None:
    """Attack 2. `isinstance` is a claim the object gets to make about itself."""
    spoof = VerdictLookalike(saw(CONTENT), _prov("liar"))
    # The mechanism, asserted rather than described: this is what `isinstance`
    # is worth against an object that is trying, and it is why the chain uses
    # `type(verdict) is Verdict`.
    assert isinstance(spoof, Verdict)
    assert type(spoof) is not Verdict

    result = GuardrailChain([ClassSpoofer()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(verdict, detector="liar", version="9.9.9", content=CONTENT)
    assert "not a Verdict" in (verdict.error or "")


class LyingInt(int):
    """An `int` subclass that passes any bound it is compared against.

    `0 <= start < end <= len(content)` is four comparisons, and a subclass
    takes reflected priority on all of them, so `LyingInt(-3)` clears the
    range check and then slices as -3.
    """

    def __ge__(self, other: object) -> bool:
        return True

    def __gt__(self, other: object) -> bool:
        return True

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True


class UnRedactor:
    """A redact whose span is a pair of lying ints.

    This is the attack that is WORSE than the bug it walked through. With the
    span accepted, `_rewrite` sliced `content[cursor:-3]`, measured from the
    end, and emitted a long prefix of the ORIGINAL content in front of the
    placeholder: `AKIAIOSFODNN7EXAMPLE and sk-livesec[REDACTED:X]t`, the
    credential handed back under a `redact` decision that says it was handled.
    """

    name: str = "unredactor"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "redact",
            "[REDACTED:X]",
            [Finding(type="X", span=(LyingInt(-3), LyingInt(-1)))],
            _prov("unredactor"),
            saw(content),
        )


def test_a_span_of_lying_ints_denies_and_never_reaches_the_rewrite() -> None:
    """Attack 3, and the assertion that matters is about the CONTENT."""
    start, end = LyingInt(-3), LyingInt(-1)
    assert 0 <= start < end <= len(CONTENT)  # the bound, defeated
    assert int(start) == -3  # and the value it actually carries

    result = GuardrailChain([UnRedactor()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="unredactor", version="0.1.0", content=CONTENT
    )
    assert "span" in (verdict.error or "")
    # Nothing was rewritten, and nothing claims it was. The failure this
    # replaces emitted the original content next to a placeholder under a
    # `redact`, which is the one output a caller forwards.
    assert result.content == CONTENT
    assert "[REDACTED" not in result.content


@pytest.mark.parametrize(
    "span",
    [(1, 2, 3), ("a", "b"), 5, (1,), (1, 2, 3, 4), ("1", 2)],
    ids=["three-ints", "two-strings", "not-a-pair", "one-int", "four-ints", "string-and-int"],
)
def test_a_malformed_span_shape_denies_instead_of_abandoning_the_run(span: object) -> None:
    """Attack 4. These raised out of `run` and took the whole run with them.

    The validation sat in an `else:` outside the `try`, so unpacking `(1, 2, 3)`
    or comparing `"a"` against an int raised where nothing caught it: no
    `ChainResult`, no audit record, and every guardrail after this one never
    ran -- from code written to keep a run alive when a detector misbehaves.
    An `Allower` follows the hostile double here for exactly that reason: the
    run has to continue, not merely fail closed.
    """

    class BadSpanShape:
        name: str = "shape"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "allow",
                None,
                [Finding(type="T", span=cast("tuple[int, int]", span))],
                _prov("shape"),
                saw(content),
            )

    result = GuardrailChain([BadSpanShape(), Allower()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    assert len(result.verdicts) == 2  # the run continued and kept its record
    hostile, allower = result.verdicts
    _assert_the_record_is_the_chains_own(
        hostile, detector="shape", version="0.1.0", content=CONTENT
    )
    assert "span" in (hostile.error or "")
    assert allower.decision == "allow"
    assert allower.error is None


def test_a_guardrail_declaring_an_unknown_kind_is_refused_when_the_chain_is_built() -> None:
    """Attack 5. An honest verdict, and a kind this library does not know.

    Checking the kind per verdict put the failure inside the chain's own
    fail-closed path: the synthesised deny is stamped with `guardrail.kind`, so
    `Verdict` refused to build it and raised `ValueError: unknown provenance
    kind` out of `run`. A guardrail that had done nothing wrong except declare
    itself oddly took the whole run down with it, and so did one whose `check`
    merely raised.

    Reading and checking the kind once, when the chain is built, is what makes
    the synthesised deny always constructible. The refusal arrives before any
    content is checked, in the error type `build` and `build_chain` already
    raise for a configuration that cannot check what it says it checks.
    """

    class HonestButOddKind:
        name: str = "odd"
        version: str = "0.1.0"
        kind: Kind = cast(Kind, "heuristic")
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict("allow", None, [], _prov("odd"), saw(content))

    with pytest.raises(GuardrailUnavailableError, match="declares a kind that is not one of"):
        GuardrailChain([HonestButOddKind(), Allower()])


def test_a_two_million_character_name_cannot_grow_the_audit_record() -> None:
    """Attack 6, first half. The name was interpolated up to four times.

    A 2,000,000-character name produced a 4,000,073-character `error`, which is
    a cost every log line, trace and database column downstream of a
    `ChainResult` then pays. The name is the one caller-supplied string a
    message still carries, because it is what tells a reader WHICH check
    misbehaved, so it is carried bounded.
    """
    huge = "N" * 2_000_000

    class HugeName:
        name: str = huge
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "allow",
                None,
                [],
                Provenance(kind="constraint", detector="wrong", version="0.1.0"),
                saw(content),
            )

    result = GuardrailChain([HugeName()]).run(CONTENT, OUT)
    (verdict,) = result.verdicts
    assert verdict.decision == "deny"
    assert verdict.error is not None
    assert len(verdict.error) < 500
    assert "N" * 201 not in verdict.error
    # The record still names the guardrail the caller configured, in full: the
    # bound is on the MESSAGE, which repeats the name, and not on the identity,
    # which is one reference to the caller's own string.
    assert verdict.provenance.detector == huge


def test_no_error_message_repeats_a_string_the_detector_chose() -> None:
    """Attack 6, second half, and the more dangerous one.

    The values a detector returns are chosen AFTER it has seen the content, so
    a detector whose class name or finding type IS the content puts the content
    into the audit record through the error message. That is exactly what
    `_bounded` refuses for an exception's message, reached from the return side
    instead, and bounding it would only have shortened the leak.

    One marker, planted in the three places a message used to quote: the class
    name of a returned non-Verdict, a finding's type on a bad span, and a
    finding's type in the `GuardrailChainError` that an unlocatable redact
    raises.
    """
    marker = "MARKER-2f8a1c-this-string-is-the-content"

    class NotAVerdict:
        name: str = "leaky"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            # A class whose NAME is the marker, built the way a detector's own
            # class is: `type(verdict).__name__` read it straight out.
            return cast(Verdict, type(marker, (), {})())

    class MarkerFindingType:
        name: str = "leaky2"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "deny",
                None,
                [Finding(type=marker, span=(0, 99999))],
                _prov("leaky2"),
                saw(content),
            )

    class MarkerUnlocatableRedact:
        name: str = "leaky3"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict("redact", "x", [Finding(type=marker)], _prov("leaky3"), saw(content))

    for guardrail in (NotAVerdict(), MarkerFindingType()):
        result = GuardrailChain([guardrail]).run(marker, OUT)
        (verdict,) = result.verdicts
        assert verdict.decision == "deny"
        assert verdict.error is not None
        assert marker not in verdict.error

    # The same rule on the one path that still raises rather than denying.
    with pytest.raises(GuardrailChainError) as raised:
        GuardrailChain([MarkerUnlocatableRedact()]).run(marker, OUT)
    assert marker not in str(raised.value)


def test_a_name_that_changes_between_reads_records_the_one_the_chain_validated() -> None:
    """Attack 7. Validate a value, then read it again, and it can be different.

    `guardrail.name` was read to compare against the verdict's claim and read
    AGAIN to stamp the synthesised deny. A property answering "honest-detector"
    to the first and "pii" to the second passed the comparison and signed the
    record as a detector that never ran. Neither read is wrong at its own call
    site, which is why the fix is to have only one: the name is read when the
    chain is built and every verdict is stamped from that copy.
    """

    class MutatingName:
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def __init__(self) -> None:
            self._reads = 0

        @property
        def name(self) -> str:
            self._reads += 1
            return "honest-detector" if self._reads <= 1 else "pii"

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "allow",
                None,
                [],
                Provenance(kind="constraint", detector="pii", version="0.1.0"),
                saw(content),
            )

    guardrail = MutatingName()
    result = GuardrailChain([cast(Guardrail, guardrail)]).run(CONTENT, OUT)
    assert guardrail.name == "pii"  # it is answering differently by now
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="honest-detector", version="0.1.0", content=CONTENT
    )
    assert "pii" not in (verdict.error or "")


@pytest.mark.parametrize(
    ("field", "clause"),
    [("name", "name that is not a str"), ("version", "version that is not a str")],
    ids=["name", "version"],
)
def test_an_identity_field_that_is_not_a_string_is_refused_when_the_chain_is_built(
    field: str, clause: str
) -> None:
    """The other end of the same read. An identity has to be recordable.

    Both fields, because each is what the other's test would let through: a
    check written for `name` alone leaves `version` to be stamped into every
    verdict as whatever object the guardrail handed over.
    """

    class NumericIdentity:
        name: str = "numeric"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
            raise AssertionError("must not run")

    guardrail = NumericIdentity()
    setattr(guardrail, field, 5)
    with pytest.raises(GuardrailUnavailableError, match=clause):
        GuardrailChain([cast(Guardrail, guardrail)])


@pytest.mark.parametrize(
    ("field", "value", "clause"),
    [
        ("content", 5, "content that is neither a str nor None"),
        ("findings", ("not a finding",), "finding that is not a Finding"),
        ("findings", 5, "TypeError raised"),
        ("error", "prose the detector wrote", "error, which only the chain sets"),
        ("decision", "allowed", "decision that is not"),
    ],
    ids=["content", "findings", "findings-not-iterable", "error", "decision"],
)
def test_a_field_of_the_wrong_type_becomes_a_deny(field: str, value: object, clause: str) -> None:
    """The remaining fields, each read once and each checked.

    Every one of these has to reach around `Verdict`'s own invariants with
    `object.__setattr__`, which is precisely the point: a detector that can
    construct a `Verdict` at all can also hand back one whose slots hold
    something else, and the chain is downstream of that.

    `findings=5` is the case that pins WHERE the checking happens rather than
    what it checks. Nothing can enumerate the shapes an attribute might hold, so
    the last line of defence is that reading them is inside the same `try` as
    the `check` call: an integer where a sequence belongs raises on iteration
    and becomes this same deny, where validation sitting outside that `try`
    would take the whole run down.
    """

    class WrongType:
        name: str = "wrong-type"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            verdict = Verdict("allow", None, [], _prov("wrong-type"), saw(content))
            object.__setattr__(verdict, field, value)
            return verdict

    result = GuardrailChain([WrongType()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="wrong-type", version="0.1.0", content=CONTENT
    )
    assert clause in (verdict.error or "")


def test_a_verdict_this_library_cannot_rebuild_becomes_a_deny() -> None:
    """The last exit: fields that pass every check and still make no verdict.

    A constraint's findings carry no confidence -- that pairing is `types.py`'s
    to enforce and is not restated in the chain -- so a constraint handing back
    a confidence-bearing finding is a verdict the chain cannot construct. It
    fails closed by the same door as everything else rather than raising out of
    `run`, and it says the chain could not rebuild rather than blaming the
    detector for an exception it did not throw.
    """

    class TamperedConfidence:
        name: str = "tampered"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            # Tampered AFTER construction, because `Verdict` validates the
            # pairing on the way in: the detector could not have built this one.
            verdict = Verdict(
                "allow", None, [Finding(type="TOK", span=(0, 4))], _prov("tampered"), saw(content)
            )
            object.__setattr__(verdict.findings[0], "confidence", 0.9)
            return verdict

    result = GuardrailChain([TamperedConfidence()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="tampered", version="0.1.0", content=CONTENT
    )
    assert "cannot rebuild" in (verdict.error or "")


def test_a_returned_verdict_is_never_the_object_that_reaches_the_record() -> None:
    """The property the tests above are instances of, stated once.

    An honest guardrail's verdict is not forwarded either. It is rebuilt, and
    what a caller reads is an object this library constructed: the same values,
    a different identity. A chain that appended what it was handed passes every
    equality assertion in this file and fails this one.
    """
    returned: list[Verdict] = []

    class Honest:
        name: str = "honest"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            verdict = Verdict(
                "redact",
                "[REDACTED:TOK] and sk-livesecret",
                [Finding(type="TOK", span=(0, 20))],
                _prov("honest"),
                saw(content),
            )
            returned.append(verdict)
            return verdict

    result = GuardrailChain([Honest()]).run(CONTENT, OUT)
    (recorded,) = result.verdicts
    (original,) = returned
    assert recorded == original  # same values
    assert recorded is not original  # different object
    assert recorded.findings[0] is not original.findings[0]
    assert recorded.provenance is not original.provenance
    assert result.decision == "redact"
    assert result.content == "[REDACTED:TOK] and sk-livesecret"


class ProvenanceLookalike:
    """Not a `Provenance`, one level down from `VerdictLookalike`.

    The verdict is a real `Verdict`; what it carries is not a real
    `Provenance`. `isinstance` says otherwise, and each field is a property
    that answers honestly while it is being compared and differently
    afterwards. A chain that read these three fields to grade them and then
    kept the object would carry it into the audit record for anything
    downstream to read again.
    """

    __class__ = Provenance  # type: ignore[assignment]

    def __init__(self) -> None:
        self._reads = 0

    def _answer(self, honest: str, lie: str) -> str:
        self._reads += 1
        return honest if self._reads <= 3 else lie

    @property
    def kind(self) -> Kind:
        return cast(Kind, self._answer("constraint", "classifier"))

    @property
    def detector(self) -> str:
        return self._answer("lookalike", "pii")

    @property
    def version(self) -> str:
        return self._answer("0.1.0", "9.9.9")

    @property
    def model(self) -> str | None:
        return None

    @property
    def revision(self) -> str | None:
        return None

    @property
    def threshold(self) -> float | None:
        return None


def test_a_provenance_that_only_claims_to_be_a_provenance_becomes_a_deny() -> None:
    """The `__class__` spoof one level down, where the claim actually lives.

    Checking the verdict's type and then reading `verdict.provenance.kind` off
    whatever it carries moves the same hole one attribute deeper: the fields
    grade honestly and the object goes on answering for itself afterwards.
    """

    class SpoofedProvenance:
        name: str = "lookalike"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict("allow", None, [], cast(Provenance, ProvenanceLookalike()), saw(content))

    lookalike = ProvenanceLookalike()
    assert isinstance(lookalike, Provenance)  # what `isinstance` is worth here
    assert type(lookalike) is not Provenance

    result = GuardrailChain([SpoofedProvenance()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="lookalike", version="0.1.0", content=CONTENT
    )
    assert "not a Provenance" in (verdict.error or "")
    assert type(verdict.provenance) is Provenance


def test_a_confidence_that_is_not_a_number_becomes_a_deny() -> None:
    """The one finding field the span checks do not cover.

    A confidence is `float | None` and reaches the rebuild, so it is checked
    like everything else. Without the check `float(confidence)` decides the
    outcome: `"high"` raises `ValueError` and the run still fails closed, but
    the record then blames the detector for an exception it never threw, and a
    `float` subclass with its own `__float__` is not refused at all.
    """

    class OddConfidence:
        name: str = "odd-confidence"
        version: str = "0.1.0"
        kind: Kind = "classifier"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "deny",
                None,
                [Finding(type="TOXIC", span=None, confidence=cast(float, "high"))],
                Provenance(kind="classifier", detector="odd-confidence", version="0.1.0"),
                saw(content),
            )

    result = GuardrailChain([OddConfidence()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "confidence is neither a number nor None" in verdict.error
    assert verdict.provenance == Provenance(
        kind="classifier", detector="odd-confidence", version="0.1.0"
    )
    assert verdict.saw == saw(CONTENT)


def test_a_finding_type_that_lies_through_a_str_subclass_becomes_a_deny() -> None:
    """A finding's type is the one detector-chosen string the record keeps.

    `Finding` validates nothing, so a detector can put anything in `type` and
    the `Verdict` around it constructs cleanly. That string is what a corpus
    row matches, what a placeholder prints and what an operator filters an
    audit log by, so a `str` subclass there is a record that answers "yes" to
    every filter it is ever compared against.
    """

    class LyingFindingType:
        name: str = "typeliar"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "redact",
                "[REDACTED:EMAIL]",
                [Finding(type=cast(str, LyingStr("EMAIL")), span=(0, 4))],
                _prov("typeliar"),
                saw(content),
            )

    result = GuardrailChain([LyingFindingType()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    _assert_the_record_is_the_chains_own(
        verdict, detector="typeliar", version="0.1.0", content=CONTENT
    )
    assert "finding whose type is not a str" in (verdict.error or "")
    assert verdict.findings == ()
    assert result.content == CONTENT  # the redact it claimed was never applied


@pytest.mark.parametrize(
    ("field", "value", "clause"),
    [
        ("model", 5, "provenance.model or provenance.revision"),
        ("revision", 5, "provenance.model or provenance.revision"),
        ("threshold", "high", "provenance.threshold"),
    ],
    ids=["model", "revision", "threshold"],
)
def test_a_provenance_detail_of_the_wrong_type_becomes_a_deny(
    field: str, value: object, clause: str
) -> None:
    """`model`, `revision` and `threshold` are the fields the chain cannot grade.

    It knows what a guardrail is called and what kind of check it is; it has no
    way to know which model weights answered. They are copied through rather
    than dropped, because a classifier's record without them is not a
    provenance record at all -- and copied means rebuilt, so they are checked
    like everything else on the way in.
    """

    class OddDetail:
        name: str = "detail"
        version: str = "0.1.0"
        kind: Kind = "classifier"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            provenance = Provenance(kind="classifier", detector="detail", version="0.1.0")
            object.__setattr__(provenance, field, value)
            return Verdict("allow", None, [], provenance, saw(content))

    result = GuardrailChain([OddDetail()]).run(CONTENT, OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert clause in (verdict.error or "")
    assert verdict.provenance == Provenance(kind="classifier", detector="detail", version="0.1.0")
    assert verdict.saw == saw(CONTENT)


def test_a_classifiers_model_and_threshold_survive_the_rebuild() -> None:
    """The positive control for the test above: checked is not dropped.

    A rebuild that stamped only the three identity fields would pass every
    hostile test in this file and quietly delete the only record of WHICH model
    decided, which is the provenance a classifier exists to carry.
    """

    class Classifier:
        name: str = "classy"
        version: str = "0.1.0"
        kind: Kind = "classifier"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "deny",
                None,
                [Finding(type="TOXIC", span=None, confidence=0.97)],
                Provenance(
                    kind="classifier",
                    detector="classy",
                    version="0.1.0",
                    model="detoxify",
                    revision="abc123",
                    threshold=0.5,
                ),
                saw(content),
            )

    result = GuardrailChain([Classifier()]).run(CONTENT, OUT)
    (verdict,) = result.verdicts
    assert verdict.error is None
    assert verdict.provenance.model == "detoxify"
    assert verdict.provenance.revision == "abc123"
    assert verdict.provenance.threshold == 0.5
    assert verdict.findings[0].confidence == 0.97


def test_a_declared_direction_that_is_not_an_exact_str_cannot_force_a_run() -> None:
    """The chain decides which guardrails run, against a set it built itself.

    `directions` is caller-supplied too, and set membership consults `__hash__`
    first and `__eq__` only on a collision, so a member whose hash is the hash
    of "output" and whose `__eq__` says yes runs a guardrail in a direction it
    never declared. Every verdict it then records describes a check the caller's
    configuration did not ask for, in a direction the guardrail does not claim
    to support. Coercing the members to exact strings when the chain is built
    settles membership before any of that can be asked.
    """

    class CollidingStr(str):
        def __hash__(self) -> int:
            return hash("output")

        def __eq__(self, other: object) -> bool:
            return True

        def __ne__(self, other: object) -> bool:
            return False

    class Sideways:
        name: str = "sideways"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({cast(Direction, CollidingStr("sideways"))})

        def check(self, content: str, context: Context) -> Verdict:  # pragma: no cover
            return Verdict("deny", None, [], _prov("sideways"), saw(content))

    # The double is hostile, asserted rather than assumed.
    assert "output" in Sideways.directions

    result = GuardrailChain([Sideways()]).run(CONTENT, OUT)
    assert result.verdicts == ()
    assert result.decision == "allow"
    assert result.content == CONTENT


# ==========================================================================
# The positive control: the bundled checks, unaffected.
#
# The old control ran one benign string that every check allows, so it
# exercised the allow path and nothing else: a rebuild that dropped every
# finding, mangled a span or lost a redact's content would have passed it. It
# now runs every registered check, in both directions, over inputs that produce
# allow, redact AND deny, and compares what the chain recorded against what the
# same guardrail returned when called directly.
# ==========================================================================

# Which decisions each registered check can produce. `pii` has no deny to
# produce: it redacts what it finds and is not configurable. Asserted exactly
# rather than as a subset, so a check that gains or loses a decision has to be
# noticed here.
_DECISIONS_PRODUCED: dict[str, frozenset[Decision]] = {
    "injection-structural": frozenset({"allow", "redact", "deny"}),
    "pii": frozenset({"allow", "redact"}),
    "rules": frozenset({"allow", "redact", "deny"}),
    "secrets": frozenset({"allow", "redact", "deny"}),
}

# The `on_match` settings each check accepts, on top of its published default.
_ON_MATCH: dict[str, tuple[Decision, ...]] = {
    "injection-structural": ("redact", "deny"),
    "pii": (),
    "rules": ("redact", "deny"),
    "secrets": ("redact", "deny"),
}

# One input per detection path, plus one that every check allows. Between them
# they reach every decision in the table above.
_SAMPLES = (
    "hello, this is ordinary text.",
    "mail alice@example.com about it",
    "key AKIAIOSFODNN7EXAMPLE here",
    "read this \u202e reversed",
    "hello \U000e0041\U000e0042 world",
    "ticket JIRA-1234 on db.corp.example about project bluebird",
    "",
)

_BOTH_DIRECTIONS: tuple[tuple[Direction, Context], ...] = (("input", IN), ("output", OUT))


def test_the_bundled_check_tables_name_every_registered_check() -> None:
    """A check added to the registry cannot skip the control below by omission."""
    assert set(_DECISIONS_PRODUCED) == set(AVAILABLE)
    assert set(_ON_MATCH) == set(AVAILABLE)


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_the_chain_records_exactly_what_a_bundled_check_returned(name: str) -> None:
    """The positive control for every test above it in this file.

    Each check is built the way a caller builds one -- `build`, with the same
    options `docs/conformance.md` prints beside the published `rules` row --
    then called DIRECTLY and run through a chain over the same input, and the
    two verdicts are compared field by field. A detector that honestly declares
    its own kind, name and version, and hashes and spans the content it was
    actually given, must come through the chain unchanged in every field; if
    one did not, every caller of this library would see contract-error denies,
    lost findings or a lost redaction on ordinary traffic.
    """
    expected = _DECISIONS_PRODUCED[name]
    seen: set[Decision] = set()
    for on_match in (None, *_ON_MATCH[name]):
        options = dict(options_for(name))
        if on_match is not None:
            options["on_match"] = on_match
        guardrail = build(name, **options)
        for direction, context in _BOTH_DIRECTIONS:
            if direction not in guardrail.directions:
                continue
            for text in _SAMPLES:
                direct = guardrail.check(text, context)
                result = GuardrailChain([guardrail]).run(text, context)
                (through,) = result.verdicts
                assert through.error is None, (name, direction, text)
                assert through.decision == direct.decision, (name, direction, text)
                assert through.saw == direct.saw == saw(text), (name, direction, text)
                assert through.content == direct.content, (name, direction, text)
                assert tuple(through.findings) == tuple(direct.findings), (name, direction, text)
                assert through.provenance == direct.provenance, (name, direction, text)
                seen.add(through.decision)
    assert seen == expected


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_a_bundled_redaction_still_rewrites_the_content_the_chain_returns(name: str) -> None:
    """The verdicts can agree while the rewrite is lost.

    `ChainResult.content` is built from the spans the chain collected, so a
    rebuild that dropped a finding, or kept it and lost the span, shows up here
    and nowhere above: the decision would still be `redact` and the content
    would come back untouched, which is the fail-open this library exists to
    prevent.
    """
    if "redact" not in _DECISIONS_PRODUCED[name]:  # pragma: no cover - table is exhaustive
        pytest.skip(f"{name} produces no redact")
    options = dict(options_for(name))
    if "redact" in _ON_MATCH[name]:
        # `injection-structural` denies by default, so its redact path is only
        # reachable through this option, and it is the redact path this test is
        # about.
        options["on_match"] = "redact"
    guardrail = build(name, **options)
    redacted = False
    for text in _SAMPLES:
        result = GuardrailChain([guardrail]).run(text, OUT)
        if result.decision != "redact":
            continue
        redacted = True
        assert "[REDACTED:" in result.content
        assert result.content != text
    assert redacted, f"no sample made {name} redact, so this test asserted nothing"
