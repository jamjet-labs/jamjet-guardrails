import pytest

from jamjet_guardrails import build_chain
from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.errors import GuardrailChainError
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

# The exact token from the report. Canonical Slack bot-token shape: `xoxb-`, a
# 10-digit team id, a 13-digit bot-user id and a 24-character secret. The middle
# segment is Luhn-valid and starts with a 2, so the PII detector's bare-card
# branch matches it: a payment card is exactly "13 to 19 digits, leading 2 to 6,
# valid check digit", and this segment is all four.
SLACK_TOKEN = "SLACK_BOT_TOKEN=xoxb-2411756141-2412093090608-8dyRy9NUsIbEXCKV0LZ7XkGx"
SLACK_SECRET_TAIL = "8dyRy9NUsIbEXCKV0LZ7XkGx"


def test_a_redaction_cannot_split_a_credential_for_the_next_guardrail() -> None:
    """The leak this rule was written for, in the order the README teaches.

    Sequentially, `pii` redacted the 13-digit segment first, the placeholder cut
    the token in two, `secrets` then matched only the 16-character prefix
    `xoxb-2411756141-`, and the 24-character secret tail survived into content
    the chain returned as `redact` carrying a SLACK_TOKEN finding:

        SLACK_BOT_TOKEN=[REDACTED:SLACK_TOKEN][REDACTED:CREDIT_CARD]-8dyRy9NUsIbEXCKV0LZ7XkGx

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
def test_a_span_that_does_not_index_into_the_content_is_refused(span: tuple[int, int]) -> None:
    """Spans now drive the rewrite, so they arrive from caller-supplied code.

    A negative start is the one that leaks rather than merely mislabels:
    `content[cursor:-3]` is measured from the END, so the slice emitted in front
    of the placeholder is a long prefix of the ORIGINAL content. An out-of-range
    span un-redacts.
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

    with pytest.raises(GuardrailChainError, match="does not index into"):
        GuardrailChain([BadSpan()]).run("sensitive-value", OUT)
