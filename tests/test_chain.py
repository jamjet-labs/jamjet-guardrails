from typing import cast

import pytest

from jamjet_guardrails import build_chain
from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors import AVAILABLE, build
from jamjet_guardrails.errors import GuardrailChainError
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import Context, Direction, Finding, Kind, Provenance, Verdict

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


def test_a_redact_without_content_raises_instead_of_forwarding_the_original() -> None:
    """Pin the DIRECTION of an impossible state's fallback.

    Verdict.__post_init__ already rejects a redact carrying no content, so this
    cannot happen today and the double below has to reach around that invariant
    with object.__setattr__ to build one. What is being pinned is what the chain
    does IF that invariant is ever relaxed. Silently skipping the verdict would
    leave the decision at "redact" while the content is the original, i.e.
    telling the caller "this was redacted, forward it" about content that was
    not: fail-open on the exact axis this library sells. It raises instead.

    The raise is deliberately not a synthesised deny. Task 5 wraps guardrail.check
    for detectors that fail; this is the chain finding itself in a state its own
    types forbid, and conflating the two would bury a library bug in a verdict.
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

    with pytest.raises(GuardrailChainError):
        GuardrailChain([BrokenRedactor()]).run("secret", OUT)


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
    """
    result = GuardrailChain([ProvenanceLiar(kind="classifier")]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "kind" in verdict.error
    assert "classifier" in verdict.error  # the CLAIMED kind, not the declared one
    # The synthesised deny's own provenance is the declared identity, not
    # what the verdict falsely claimed.
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")


def test_a_wrong_provenance_detector_alone_becomes_a_deny() -> None:
    """Clause 4. `ProvenanceLiar`'s `kind` and `version` stay honest here."""
    result = GuardrailChain([ProvenanceLiar(detector="pii")]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "detector" in verdict.error
    assert "pii" in verdict.error  # the CLAIMED detector, not the declared one
    assert verdict.provenance == Provenance(kind="constraint", detector="liar", version="9.9.9")


def test_a_wrong_provenance_version_alone_becomes_a_deny() -> None:
    """Clause 5. `ProvenanceLiar`'s `kind` and `detector` stay honest here."""
    result = GuardrailChain([ProvenanceLiar(version="0.1.0")]).run("the real content", OUT)
    assert result.decision == "deny"
    (verdict,) = result.verdicts
    assert verdict.error is not None
    assert "version" in verdict.error
    assert "0.1.0" in verdict.error  # the CLAIMED version, not the declared one
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


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_every_bundled_check_still_passes_its_own_returned_verdict_contract(name: str) -> None:
    """The positive control for every test above it in this file.

    Built the way a caller builds one -- `build`, with the same options
    `docs/conformance.md` prints beside the published `rules` row
    (`options_for`) -- and run through the same chain a hostile guardrail is
    graded by. A detector that honestly declares its own kind, name and
    version, and hashes and spans the content it was actually given, must
    never trip `_contract_violation`; if one did, every caller of this
    library would see contract-error denies on ordinary traffic.
    """
    guardrail = build(name, **options_for(name))
    result = GuardrailChain([guardrail]).run("hello, this is ordinary text.", OUT)
    assert len(result.verdicts) == 1
    assert result.verdicts[0].error is None
