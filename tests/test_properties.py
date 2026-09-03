"""Properties, over generated input, of the invariants `docs/conformance.md` states.

Every property here is a sentence already in the contract. That is the bar and
it is deliberate: a property that fails is then a conformance failure and not a
new opinion about how this library should behave, and a port that fails one has
learned something about itself rather than about this implementation's taste.

Why generated input at all, next to the hand-written cases. Every Critical this
package has recorded was a CLASS of input nobody wrote down, not a wrong answer
on a case somebody thought of:

- `finditer` resumed at the END of each match, so a decoy of the same shape
  joined in front of a real credential left the real one's start position never
  tried, and a whole 36-character token stood behind a redaction placeholder.
- Two overlapping spans, one kept and one dropped, left 15 of the 19 characters
  of a credit card in output this library called redacted.
- Sequential rewriting let one detector's placeholder split a credential, so the
  next detector matched only the stump and reported success.
- A fold that changes length, casefolding expanding and stripping
  default-ignorable code points contracting, makes a naive index wrong in
  opposite directions.

Each is a violation of an invariant with a counterexample a few characters long,
and each shipped past a suite that had a test for every case its author thought
of. The tests here do not restate those cases; `tests/test_spans.py`,
`tests/test_fold.py`, `tests/test_chain.py`, `tests/test_authoring.py` and
`tests/test_injection_structural.py` hold them. These state the rule the cases
are instances of.

The settings are in `tests/conftest.py`, which explains why they are fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from jamjet_guardrails import (
    Context,
    Decision,
    Direction,
    Finding,
    GuardrailChain,
    Kind,
    Limits,
    PatternGuardrail,
    Provenance,
    Verdict,
    build,
    combine,
)
from jamjet_guardrails._fold import casefold_view, fold
from jamjet_guardrails._spans import _merge, _rewrite, _scan
from jamjet_guardrails.detectors.injection_structural import (
    INJECTION_TYPES,
    InjectionStructuralGuardrail,
)
from jamjet_guardrails.protocol import saw

IN = Context(direction="input", origin="user")

# The shape `_Region.placeholder` writes. Used to take a rewritten string back
# apart, which is what makes "no claimed character survives" checkable rather
# than merely stated. Every content strategy below draws from an alphabet with
# no `[`, so a bracket in a rewritten string can only have come from a
# placeholder and this pattern cannot eat content that happens to look like one.
_PLACEHOLDER = re.compile(r"\[REDACTED:[A-Z0-9_+]*\]")

Found = list[tuple[str, tuple[int, int]]]

# Content for the span properties: ten letters, no `[`, so the reassembly above
# is exact. WHAT the characters are is irrelevant to span arithmetic, since what
# matters is which offsets are claimed, so the alphabet is small on purpose and
# Hypothesis spends its budget on the span shapes instead.
_PLAIN = st.text(alphabet="abcdefghij", min_size=1, max_size=40)

# Type names in the domain `authoring._TYPE_NAME` enforces. Three of them, and a
# repeated draw is wanted: two spans claiming one region under the SAME name is
# what `_Region.claim` de-duplicates, and two under different names is what its
# sorted placeholder has to name in full.
_TYPE_NAME = st.sampled_from(["AA", "BB", "CC"])


@st.composite
def _content_and_spans(draw: st.DrawFn) -> tuple[str, Found]:
    """Content, and typed spans into it, sorted the way `_merge` requires.

    Sorted here rather than asserted, because sorted input is a documented
    PRECONDITION of `_merge` and not a case: it tests each span against the
    running end of the region it is extending and looks no further back, so
    feeding it an unsorted list measures nothing about the merge.
    """
    content: str = draw(_PLAIN)
    limit = len(content)
    spans: list[tuple[int, int]] = draw(
        st.lists(
            st.tuples(st.integers(0, limit - 1), st.integers(1, limit)).filter(
                lambda pair: pair[0] < pair[1]
            ),
            max_size=8,
        )
    )
    found: Found = [(draw(_TYPE_NAME), span) for span in spans]
    return content, sorted(found, key=lambda pair: pair[1])


def _covered(found: Found) -> set[int]:
    """Every offset any span claims, by union rather than by merging.

    A second and deliberately naive implementation of what is under test.
    `_merge` walks the list once carrying a running end; this builds a set. Two
    implementations agreeing on generated input is evidence. One implementation
    compared against itself is not.
    """
    return {offset for _, (start, end) in found for offset in range(start, end)}


def _kept(content: str, covered: set[int]) -> str:
    """The characters no span claimed, in order."""
    return "".join(char for offset, char in enumerate(content) if offset not in covered)


# ==========================================================================
# Span arithmetic
# ==========================================================================


@given(_content_and_spans())
def test_merged_regions_are_sorted_disjoint_and_cover_exactly_the_union(
    case: tuple[str, Found],
) -> None:
    """The defect: an overlap resolved by KEEPING one span and DROPPING the other.

    That shipped, and it left 15 of the 19 characters of a credit card in a
    string this library reported as redacted. The union is what makes it
    checkable in the direction that matters: a merge that drops a span covers
    LESS than the union, and every offset of the difference is a character a
    caller was told had been removed.

    Sorted and disjoint are the precondition `_rewrite` consumes. It walks the
    regions once with a cursor that only moves forward, so a pair out of order
    emits a slice measured backwards and a pair that overlaps emits the
    overlapping stretch twice.
    """
    _, found = case
    regions = _merge(found)

    assert [(region.start, region.end) for region in regions] == sorted(
        (region.start, region.end) for region in regions
    )
    for region in regions:
        assert region.start < region.end, region
    for earlier, later in pairwise(regions):
        assert earlier.end <= later.start, (earlier, later)

    merged = {offset for region in regions for offset in range(region.start, region.end)}
    assert merged == _covered(found), (found, [(r.start, r.end) for r in regions])


@given(_content_and_spans())
def test_a_region_names_every_type_that_claimed_any_part_of_it(case: tuple[str, Found]) -> None:
    """The defect: a merged region labelled with only the first type that reached it.

    `docs/conformance.md` leaves the placeholder's TEXT unspecified and does not
    leave the audit record unspecified: a region claimed by more than one
    guardrail "becomes one replacement naming every type that claimed any part of
    it". A region naming one of two checks tells a reader the other never fired
    over those bytes, and that reading is what makes
    `[REDACTED:CREDIT_CARD+SLACK_TOKEN]` worth printing at all.

    Both directions, because each fails silently on its own: a type that claimed
    part of the region and is missing from the placeholder, and a type in the
    placeholder that claimed none of it.
    """
    _, found = case
    for region in _merge(found):
        claimants = {
            type_name
            for type_name, (start, end) in found
            if start < region.end and end > region.start
        }
        assert set(region.types) == claimants, (region, claimants, found)
        assert region.placeholder == f"[REDACTED:{'+'.join(sorted(claimants))}]"


@given(_content_and_spans())
def test_no_character_any_span_claimed_survives_the_rewrite(case: tuple[str, Found]) -> None:
    """The redaction guarantee, stated over `_rewrite` itself.

    The defect: a claimed character left standing in the output. Both recorded
    leaks are instances, the dropped overlapping span and the token whose start
    was never tried, and both were invisible to a test that asserted the
    placeholder was present, because the placeholder WAS present and the
    credential was beside it.

    Checked by taking the rewritten string apart rather than by comparing it to
    an expected one: strip every placeholder, and what is left must be exactly
    the characters no span claimed, in order. Comparing against an expected
    rewrite would restate the implementation instead of checking it.
    """
    content, found = case
    rewritten = _rewrite(content, found)
    assert _PLACEHOLDER.sub("", rewritten) == _kept(content, _covered(found)), (
        content,
        found,
        rewritten,
    )


@given(_content_and_spans())
def test_rewriting_from_the_merged_regions_is_the_same_rewrite(case: tuple[str, Found]) -> None:
    """Idempotence with respect to region boundaries.

    The defect: a second rewriting pass. Rewriting one guardrail's spans and then
    the next guardrail's over the result is what let a personal-data placeholder
    cut a Slack token in half and leave its 24-character tail standing in content
    the chain called redacted, measured at 4.87% of canonical bot tokens. The fix
    was to merge everything and rewrite once, and this is the invariant that fix
    leans on: feeding the merged regions back in yields the same regions and the
    same output, so there is nothing a second pass could add.
    """
    content, found = case
    regions = _merge(found)
    again: Found = sorted(
        (
            (type_name, (region.start, region.end))
            for region in regions
            for type_name in region.types
        ),
        key=lambda pair: pair[1],
    )
    assert _rewrite(content, again) == _rewrite(content, found)
    assert [(region.start, region.end, sorted(region.types)) for region in _merge(again)] == [
        (region.start, region.end, sorted(region.types)) for region in regions
    ]


@given(st.sampled_from([r"a+b?", r"[ab]+", r"a.{0,3}", r"\w+@\w+", r"ab|abc"]), _PLAIN)
def test_scan_finds_every_match_at_every_start_position(pattern: str, content: str) -> None:
    """The `finditer` Critical, as a rule rather than as the four cases that found it.

    The defect: `finditer` resumes at the END of each match, so a decoy of the
    same shape joined directly in front of a real credential runs its greedy body
    past the real one's prefix and leaves the scan resuming INSIDE it. That start
    is then never tried and the credential stands behind the placeholder. It cost
    a GitHub token 36 characters, AWS 19 of 20, and an email address its whole
    domain half.

    Stated as coverage, which is what a redaction actually needs: whatever the
    pattern matches when ANCHORED at any offset must lie inside the union of what
    `_scan` kept. That admits the containment filter, which drops a match wholly
    inside one already kept because the container covers every offset it covered,
    and it refuses the overlap suppression that leaked before.
    """
    compiled = re.compile(pattern)
    kept = {offset for match in _scan(compiled, content) for offset in range(*match.span())}
    for start in range(len(content) + 1):
        match = compiled.match(content, start)
        if match is None or match.start() == match.end():
            continue
        assert set(range(match.start(), match.end())) <= kept, (
            pattern,
            content,
            start,
            match.span(),
        )


def test_a_zero_width_match_at_the_end_of_the_content_terminates() -> None:
    r"""The scan used to HANG here, which is why this one is not a property.

    A property test cannot express it: against the defect this does not fail, it
    never returns, and it takes the suite with it. So the inputs are written out.

    `Pattern.search` CLAMPS a `pos` past the end of the string back to
    `len(content)`. A pattern that matches zero-width at the end is therefore
    found again at that same offset on every pass, the containment filter drops
    the repeat as no longer than the one already kept, `pos` is set back to the
    same `len(content) + 1`, and nothing changes. `PatternGuardrail` refuses a
    pattern that matches the empty string OUTRIGHT and says in its own comment
    that it cannot refuse one gated by a lookbehind, so `(?<=a)` over "a" -- and
    `\ba*` over "0", which needs no lookbehind at all -- reached `_scan` through
    the caller-configured `rules` check and spun forever. A guardrail that never
    returns is the one failure a fail-closed library cannot report.

    What happens now is what `authoring.py`'s own comment always claimed: the
    zero-width match is reported once and the chain refuses it. That refusal is
    `test_a_word_boundary_gated_pattern_reports_a_zero_width_span`, which is a
    known miss rather than a fixed one.
    """
    for pattern, content in ((r"(?<=a)", "a"), (r"(?<=a)b*", "xa"), (r"\ba*", "0"), (r"\ba*", "")):
        spans = [match.span() for match in _scan(re.compile(pattern), content)]
        assert all(0 <= start <= end <= len(content) for start, end in spans), (
            pattern,
            content,
            spans,
        )


# ==========================================================================
# Folded views and the offsets that lead back from one
# ==========================================================================

# A character that casefolds LONGER, one the fold below deletes, and ordinary
# text, drawn into one string on purpose. The two folds `_fold` documents move an
# index in OPPOSITE directions, and a string carrying only one of them cannot
# tell an off-by-the-expansion from an off-by-the-deletion, because either error
# is zero on the other half.
_EXPANDS = "ßẞﬁİ"
_DELETED = "\u200b\u200d\u00ad\ufeff\u2060"
_FOLDABLE = st.text(
    alphabet=st.one_of(
        st.sampled_from(_EXPANDS),
        st.sampled_from(_DELETED),
        st.sampled_from("aAzZ09 .Σςİı"),
    ),
    max_size=24,
)


def _casefold_and_strip(char: str) -> str:
    """Casefold, then delete the default-ignorable carriers: expands AND contracts.

    Not a fold this package ships. It is the composition of the two the module
    docstring of `_fold` names as its reason to exist, which is what puts both
    length changes into one view and therefore into one offset map.
    """
    return "" if char in _DELETED else char.casefold()


@given(_FOLDABLE)
def test_the_offset_map_has_exactly_one_entry_per_view_character(source: str) -> None:
    """The defect: an index into the view read against the source.

    Length equality is the smallest statement of it and the one that fails first.
    A character that produced two view characters appears twice in the map and
    one that produced none does not appear at all, so any map built by a rule
    other than this one is wrong for any fold that is not length-preserving.

    Monotonicity is in the same test because it is what `span` relies on and
    nothing else asserts: `span` reads `origin[start]` and `origin[end - 1]` and
    calls the pair a range, which it is only if the map never goes backwards.
    """
    for per_char in (str.casefold, _casefold_and_strip, str.upper, lambda char: char * 2):
        view = fold(source, per_char)
        assert len(view.origin) == len(view.text)
        assert view.source_length == len(source)
        assert all(0 <= index < len(source) for index in view.origin)
        assert list(view.origin) == sorted(view.origin)


@given(_FOLDABLE, st.data())
def test_a_view_span_maps_back_to_the_source_run_that_produced_it(
    source: str, data: st.DataObject
) -> None:
    """The defect: a span reported into the view instead of into the source.

    A detector matching over a folded view still has to report a span into the
    string the chain was given, because that is what `saw` hashes, what every
    other detector's spans index into and what the corpus labels. Off by the
    expansion in one direction and off by the deletion in the other, and a
    redaction then removes the wrong characters while reporting that it removed
    the right ones.

    Three claims, and the third is the one a redaction needs. The source span is
    non-empty and in range. Its ends are the source characters that produced the
    first and last characters of the match, so it is tight. And re-folding that
    source run REPRODUCES the matched view text, so the run is closed over
    everything that produced the match: on a contracting fold that closure is
    what keeps a launderer's zero-width space inside the redacted region instead
    of standing in content reported as rewritten.
    """
    view = fold(source, _casefold_and_strip)
    if not view.text:
        return
    start = data.draw(st.integers(0, len(view.text) - 1))
    end = data.draw(st.integers(start + 1, len(view.text)))

    source_start, source_end = view.span(start, end)
    assert 0 <= source_start < source_end <= len(source)
    assert view.origin[start] == source_start
    assert view.origin[end - 1] == source_end - 1
    refolded = fold(source[source_start:source_end], _casefold_and_strip).text
    assert view.text[start:end] in refolded, (source, start, end, source_start, source_end)


@given(st.text(alphabet=st.characters(codec="utf-8"), max_size=24))
def test_per_character_casefolding_agrees_with_whole_string_casefolding(source: str) -> None:
    """`casefold_view`'s docstring says the two agree on every input measured. Measure it.

    The defect it would catch: a Unicode version in which `str.casefold` maps a
    SEQUENCE of characters, which a per-character fold cannot see. The
    per-character form is the one that carries an offset map, so the two must
    agree or the map describes a different string from the one a whole-string
    casefold produced, and `PatternGuardrail._banned_spans` reports source spans
    out of exactly that map.

    Over the encodable code point space rather than a sample of it, which is the
    only way this claim means anything.
    """
    assert casefold_view(source).text == source.casefold()


# ==========================================================================
# The Unicode state machine in injection-structural
# ==========================================================================

# Every range below is here because the detector reads it. `st.text()` with its
# default alphabet reaches none of them: it draws overwhelmingly from ordinary
# BMP letters, and a tag character or an unbalanced isolate would arrive roughly
# never.

# U+E0000..U+E007F, the tag block. TAG SPACE through TAG TILDE mirror printable
# ASCII one for one and render as nothing, which is the whole smuggling
# primitive. U+E007F CANCEL TAG is in range on purpose: it terminates the three
# RGI subdivision flag sequences, so it is what the one exemption keys on.
_TAG = st.integers(min_value=0xE0000, max_value=0xE007F).map(chr)

# The nine bidi CONTROLS `_bidi_spans` pairs: U+202A..U+202E, the legacy
# embeddings with the override and PDF, and U+2066..U+2069, the isolates with
# PDI. Drawn independently of each other so unbalanced sequences are the common
# case rather than the rare one, which is the half of the space a hand-written
# balanced example cannot reach.
_BIDI_CONTROL = st.sampled_from("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")

# The three directional MARKS, U+200E, U+200F and U+061C. Deliberately not
# controls: the detector EXCLUDES them from the zero-width signal, and an
# exclusion is only exercised by input that reaches it.
_BIDI_MARK = st.sampled_from("\u200e\u200f\u061c")

# Zero-width and other default-ignorable carriers: ZWSP, ZWNJ, ZWJ, WORD JOINER,
# the BOM, the Mongolian vowel separator, COMBINING GRAPHEME JOINER, SOFT HYPHEN,
# a variation selector, and the four invisible mathematical operators
# U+2061..U+2064 that carried a payload at one character per bit before the
# default-ignorable table replaced a hand-picked list.
_ZERO_WIDTH = st.sampled_from(
    "\u200b\u200c\u200d\u2060\ufeff\u180e\u034f\u00ad\ufe0f\u2061\u2062\u2063\u2064"
)

# Combining marks, including two viramas of combining class 9, Devanagari and
# Bengali, plus a nukta, an accent and an Arabic hamza. The joiner exemption
# walks from a joiner across marks to the letter underneath, so a mark is what
# makes that walk run at all.
_MARK = st.sampled_from("\u093c\u094d\u09cd\u0301\u0654")

# Astral characters, every one of which is a surrogate PAIR in UTF-16 and four
# bytes in UTF-8: the flag base U+1F3F4 the tag exemption keys on, an emoji that
# takes a ZWJ sequence, a skin-tone modifier, a musical invisible, a Duployan
# format character, a plane-2 letter and a variation selector supplement member.
_ASTRAL = st.sampled_from("\U0001f3f4\U0001f469\U0001f3fb\U0001d173\U0001bca0\U00020000\U000e0100")

# Lone surrogates, U+D800..U+DFFF. A Python `str` may hold one and UTF-8 cannot
# encode it, so this is the one class of input on which the detector and the hash
# it stamps disagree about whether the content can exist at all.
_LONE_SURROGATE = st.integers(min_value=0xD800, max_value=0xDFFF).map(chr)

# Ordinary text. It carries U+2029 PARAGRAPH SEPARATOR, the bidi class B
# character `_bidi_spans` flushes both its stacks on, and letters of the six
# scripts the joiner exemption is written for: Hebrew, Arabic, Devanagari, Thai,
# Mongolian and Hiragana. Escaped rather than written literally, for the reason
# `injection_structural.py` gives about its own controls: a literal is invisible
# in this file, invisible in the diff that adds it, and invisible in review.
_ORDINARY = st.sampled_from("aZ09 .\n\r\t\u2029\u05d0\u0627\u0915\u0e01\u1820\u3042")

_ENCODABLE_CHAR = st.one_of(_TAG, _BIDI_CONTROL, _BIDI_MARK, _ZERO_WIDTH, _MARK, _ASTRAL, _ORDINARY)
_STRUCTURAL_TEXT = st.text(alphabet=_ENCODABLE_CHAR, max_size=32)
_ANY_STR = st.text(alphabet=st.one_of(_ENCODABLE_CHAR, _LONE_SURROGATE), max_size=32)

_INJECTION = InjectionStructuralGuardrail()
_INJECTION_REDACTS = InjectionStructuralGuardrail(on_match="redact")


@given(_ANY_STR)
def test_the_span_machinery_never_raises_on_any_string_a_str_can_hold(content: str) -> None:
    """The defect: a detector that raises on input rather than deciding about it.

    A raise inside `check` becomes a synthesised deny and the chain survives it,
    which is exactly why this is worth generating for: the failure is invisible
    in the decision, since deny is what a payload gets too, and shows up only as
    a guardrail that has stopped classifying and started refusing everything of
    some shape.

    Over lone surrogates as well, which is where the domain genuinely splits and
    the split is stated rather than smoothed over. A `str` may hold one and UTF-8
    cannot encode it, so `saw` raises `UnicodeEncodeError` and `check` cannot
    produce a verdict about content the contract's own hash cannot describe. The
    span machinery underneath has no such excuse: it indexes code points, and it
    answers here.

    The span bound is the clause the chain enforces on every finding of every
    decision, asserted at the source as well. A detector whose spans are out of
    range becomes a chain-wide synthesised deny with a message naming the clause
    and not the reason.
    """
    found = _INJECTION._matches(content)
    assert found == sorted(found, key=lambda pair: pair[1])
    for type_name, (start, end) in found:
        assert type_name in INJECTION_TYPES
        assert 0 <= start < end <= len(content), (type_name, start, end)


@given(_STRUCTURAL_TEXT)
def test_every_injection_verdict_satisfies_every_verdict_invariant(content: str) -> None:
    """The invariants `docs/conformance.md` lists for `Verdict`, over generated input.

    The defect: a verdict well-formed on the inputs somebody wrote down and
    malformed on a shape nobody did. `Verdict.__post_init__` refuses most of
    these itself, so what is asserted here is what it cannot see: that `saw` is
    the digest of THIS content rather than merely a well-formed digest, that
    every finding type is one of the three this check declares, and that a
    constraint's findings carry no confidence.

    `saw` is the load-bearing one. The hash is what a chain replays from and what
    ties a span to the string it indexes, so a detector hashing a normalised or
    stripped copy of its input would report spans into one string and a digest of
    another, which no single verdict reveals.
    """
    verdict = _INJECTION.check(content, IN)
    assert verdict.decision in ("allow", "deny")
    assert verdict.saw == saw(content)
    assert verdict.provenance.kind == "constraint"
    assert verdict.provenance.detector == _INJECTION.name
    assert verdict.error is None
    assert (verdict.decision == "deny") == bool(verdict.findings)
    for finding in verdict.findings:
        assert finding.confidence is None
        assert finding.type in INJECTION_TYPES
        assert finding.span is not None
        start, end = finding.span
        assert 0 <= start < end <= len(content)


def _tag_spelling(code: str) -> str:
    """A subdivision code written in tag characters, terminated by CANCEL TAG.

    The payload half of an RGI flag sequence, without its U+1F3F4 base. Drawn
    alongside arbitrary tag runs because it is the only input that reaches the
    exemption at all: `_is_valid_flag_sequence` decodes the run and compares it
    against a closed set of three, so a random run of tag characters takes the
    early return and never exercises the rule that the character BEFORE the run
    must be the flag base. Dropping that rule left this property green until
    these three strings were in the strategy.
    """
    return "".join(chr(0xE0000 + ord(char)) for char in code) + "\U000e007f"


_FLAG_BODY = _tag_spelling("gbsct")
_TAG_PAYLOAD = st.one_of(
    st.lists(_TAG, min_size=1, max_size=8).map("".join),
    st.sampled_from([_tag_spelling(code) for code in ("gbeng", "gbsct", "gbwls")]),
)


@example(payload=_FLAG_BODY, prefix="a", suffix="")
@given(
    _TAG_PAYLOAD,
    st.text(alphabet="abcXYZ 0", max_size=10),
    st.text(alphabet="abcXYZ 0", max_size=10),
)
def test_ordinary_text_around_a_tag_payload_moves_its_span_by_the_prefix_length(
    payload: str, prefix: str, suffix: str
) -> None:
    """The defect: a signal that only fires at the edges of the input.

    Two ways that happens, and both have shipped in this file's neighbourhood. A
    walk that reads a neighbour without checking the bounds wraps around the end
    of the string, which is what
    `test_a_joiner_at_the_very_front_does_not_read_its_neighbour_off_the_back`
    holds for the joiner walk. And an exemption keyed on the character BEFORE a
    run behaves differently when there is no character before it: the tag
    exemption reads `content[start - 1]`, so a run at offset zero and the same
    run one character in take different paths through it.

    So the property is translation. A payload detected on its own is still
    detected with ordinary Latin text around it, under the same type, and its
    span moves by exactly the length of what was put in front. Latin
    deliberately, and no flag base: a U+1F3F4 in the prefix would make a
    flag-shaped run exempt, which is the exemption working rather than a
    detection lost. The seeded example is the tag spelling of `gbsct` behind one
    Latin character, which is the input that separates "the run decodes to a
    subdivision code" from "the run stands on a flag base": the second is what
    the exemption requires, and only a run that decodes reaches the test at all.
    """
    alone = [span for name, span in _INJECTION._matches(payload) if name == "INVISIBLE_TAG_CHARS"]
    if not alone:
        return
    surrounded = [
        span
        for name, span in _INJECTION._matches(prefix + payload + suffix)
        if name == "INVISIBLE_TAG_CHARS"
    ]
    assert surrounded == [(start + len(prefix), end + len(prefix)) for start, end in alone], (
        payload,
        prefix,
        suffix,
    )


@given(_STRUCTURAL_TEXT)
def test_redacting_leaves_no_reported_character_standing(content: str) -> None:
    """The defect this check's own docstring calls out: an unsorted span list.

    `_matches` re-sorts what its three signals produce, because each signal scans
    the whole input independently and the concatenation is not in span order.
    `_merge` looks only at the running end of the region it is extending, so an
    out-of-order list makes it emit a region starting AFTER a control the
    placeholder claims to have removed. `deny` is this check's default and a deny
    never reaches the rewrite, so the whole failure stays invisible until a
    caller configures `redact`, which is what this property configures.

    Stated as coverage rather than as an expected string, for the reason
    `test_no_character_any_span_claimed_survives_the_rewrite` gives: what must be
    true is that every reported character is gone, not that the output reads a
    particular way.
    """
    verdict = _INJECTION_REDACTS.check(content, IN)
    if verdict.decision == "allow":
        assert verdict.content is None
        return
    assert verdict.decision == "redact"
    assert verdict.content is not None
    covered = {
        offset
        for finding in verdict.findings
        if finding.span is not None
        for offset in range(*finding.span)
    }
    assert _PLACEHOLDER.sub("", verdict.content) == _kept(content, covered), (
        content,
        verdict.content,
    )


# ==========================================================================
# The chain contract
# ==========================================================================


@dataclass
class _Scripted:
    """A well-behaved guardrail that returns a scripted verdict and records its input.

    Well-behaved is the point. `tests/test_chain.py` is largely about guardrails
    that LIE, and the chain's answer to those is to rebuild or to deny. What is
    generated here is the honest population, where the chain's job is to combine
    and to compose rather than to refuse.
    """

    name: str
    version: str
    kind: Kind
    directions: frozenset[Direction]
    decision: Decision
    spans: tuple[tuple[int, int], ...]
    seen: list[str] = field(default_factory=list)

    def check(self, content: str, context: Context) -> Verdict:
        self.seen.append(content)
        provenance = Provenance(kind=self.kind, detector=self.name, version=self.version)
        findings = [Finding(type="AA", span=span) for span in self.spans]
        digest = saw(content)
        if self.decision == "redact":
            rewritten = _rewrite(content, [("AA", span) for span in sorted(self.spans)])
            return Verdict("redact", rewritten, findings, provenance, digest)
        if self.decision == "deny":
            return Verdict("deny", None, findings, provenance, digest)
        return Verdict("allow", None, findings, provenance, digest)


@st.composite
def _content_and_chain(draw: st.DrawFn) -> tuple[str, list[_Scripted]]:
    """Content, and a handful of honest guardrails scripted over it.

    A `redact` is always given at least one span, because a redact the chain
    cannot LOCATE is the one shape that abandons the run by contract, and
    abandoning the run is a different property from the ones below. Empty content
    therefore scripts no redact at all: no span in an empty string satisfies
    `0 <= start < end <= 0`.
    """
    content: str = draw(st.text(alphabet="abcdefghij", max_size=30))
    limit = len(content)
    span = st.tuples(st.integers(0, max(limit - 1, 0)), st.integers(1, max(limit, 1))).filter(
        lambda pair: pair[0] < pair[1] <= limit
    )
    guardrails: list[_Scripted] = []
    for position in range(draw(st.integers(0, 4))):
        allowed: list[Decision] = ["allow", "deny"] if limit == 0 else ["allow", "deny", "redact"]
        decision: Decision = draw(st.sampled_from(allowed))
        spans: list[tuple[int, int]] = draw(
            st.lists(span, min_size=1 if decision == "redact" else 0, max_size=3)
        )
        directions: frozenset[Direction] = draw(
            st.sampled_from(
                [
                    frozenset({"input"}),
                    frozenset({"output"}),
                    frozenset({"input", "output"}),
                ]
            )
        )
        guardrails.append(
            _Scripted(
                name=f"g{position}",
                version="0.1.0",
                kind="constraint",
                directions=directions,
                decision=decision,
                spans=tuple(spans),
            )
        )
    return content, guardrails


@given(_content_and_chain())
def test_the_decision_is_the_restrictive_combination_of_the_verdicts(
    case: tuple[str, list[_Scripted]],
) -> None:
    """The defect: a code path that WEAKENS a decision.

    "No code path may weaken a decision" is the whole of the combination section,
    and the shape it forbids is a later guardrail talking an earlier deny back
    down. `docs/conformance.md` also records that the corpora cannot see this at
    all: scoring runs one guardrail against one case and compares that single
    verdict, so combination is specified there and measured nowhere.

    The two consequences the document draws are asserted with it, because both
    are the direction a port fails in. A chain that runs no guardrail returns
    `allow` over anything, and a guardrail whose declared directions exclude the
    context's is SKIPPED rather than allowed, which is a different thing: a skip
    records no verdict, and an allow records one saying the content was checked.
    """
    content, guardrails = case
    result = GuardrailChain(guardrails).run(content, IN)

    running: Decision = "allow"
    for verdict in result.verdicts:
        running = combine(running, verdict.decision)
    assert result.decision == running

    ran = [g for g in guardrails if "input" in g.directions]
    assert len(result.verdicts) == len(ran)
    assert [verdict.provenance.detector for verdict in result.verdicts] == [g.name for g in ran]
    assert [g.seen for g in guardrails if "input" not in g.directions] == [
        [] for g in guardrails if "input" not in g.directions
    ]
    if not ran:
        assert result.decision == "allow"
        assert result.content == content


@given(_content_and_chain())
def test_no_guardrail_sees_a_rewritten_string_and_every_verdict_hashes_the_original(
    case: tuple[str, list[_Scripted]],
) -> None:
    """The defect: sequential rewriting, which is the leak the single-pass rule exists for.

    A personal-data check redacts a Luhn-valid 13-digit run inside a Slack bot
    token, its placeholder splits the token, the credential check then matches
    only the 16-character prefix, and the 24-character secret tail survives into
    content the chain returns as `redact` with a `SLACK_TOKEN` finding. Measured
    at 4.87% of canonical bot tokens in one guardrail order and 0% in the other,
    which is the tell: a rule whose safety depends on configuration order is the
    defect, not the ordering.

    Two assertions, and neither implies the other. Every guardrail that ran was
    handed the string the chain was given, so no rewrite reaches a detector. And
    every verdict's `saw` is the digest of that same string, which is what makes
    every span in one run index into one string.
    """
    content, guardrails = case
    result = GuardrailChain(guardrails).run(content, IN)
    digest = saw(content)
    for guardrail in guardrails:
        assert guardrail.seen in ([], [content]), guardrail.seen
    for verdict in result.verdicts:
        assert verdict.saw == digest


@given(_content_and_chain())
def test_no_byte_inside_a_reported_span_survives_a_chain_redaction(
    case: tuple[str, list[_Scripted]],
) -> None:
    """The redaction guarantee end to end, over a chain rather than over a rewrite.

    The defect: a `redact` decision over a string that still contains what the
    findings say was removed. Both recorded leaks land here, and so does a third
    shape only a chain can produce, two guardrails claiming overlapping stretches
    where merging the spans is the difference between removing both and removing
    one.

    Only a `redact` contributes, which is contract rather than detail. Content
    returned beside an `allow` or a `deny` is ignored, so a chain rewriting from
    a deny's findings would remove characters no caller was told about and, worse,
    would make a deny look forwardable.
    """
    content, guardrails = case
    result = GuardrailChain(guardrails).run(content, IN)
    covered = {
        offset
        for verdict in result.verdicts
        if verdict.decision == "redact"
        for finding in verdict.findings
        if finding.span is not None
        for offset in range(*finding.span)
    }
    assert _PLACEHOLDER.sub("", result.content) == _kept(content, covered), (
        content,
        result.content,
    )


# The four bundled checks over one string. Built once at module scope: each
# constructor compiles its patterns, and rebuilding per example would put that
# cost inside the deadline `tests/conftest.py` sets.
_BUNDLED = GuardrailChain(
    [
        build("injection-structural", on_match="redact"),
        build("pii"),
        build("secrets", on_match="redact"),
        PatternGuardrail(
            name="rules",
            version="0.1.0",
            patterns={"TICKET_ID": r"\bJIRA-\d{4,}\b"},
            banned={"PROJECT_CODENAME": ("project bluebird",)},
            on_match="redact",
        ),
    ]
)

# Fragments each bundled check has an opinion about, glued together in generated
# order: an address and a card for `pii`, two credential shapes for `secrets`, a
# ticket id and a codename for `rules`, and the three structural signals. The
# JOINS are the point rather than the fragments. Two detections butted together,
# one inside another and one straddling a third are the shapes that produced
# every composition defect in this repository, and they are what a corpus of one
# labelled case per line cannot carry.
_BUNDLED_PIECE = st.sampled_from(
    [
        "alice@example.com",
        "4111 1111 1111 1111",
        "sk-abcdefghijklmnopqrstuvwxyz012345",
        "JIRA-1234",
        "Project Bluebird",
        "\U000e0069\U000e0067\U000e006e",
        "\u202e",
        "\u200b\u200b\u200b",
        "ordinary words ",
        "\n",
    ]
)
_BUNDLED_CONTENT = st.lists(_BUNDLED_PIECE, max_size=8).map("".join)


@given(_BUNDLED_CONTENT)
def test_no_byte_the_bundled_checks_reported_survives_their_chain(content: str) -> None:
    """The same guarantee over the four shipped checks instead of scripted ones.

    The defect: a composition failure between two REAL detectors, which is the
    only kind this repository has actually shipped. `docs/conformance.md` says
    plainly that nothing measured here covers it, since scoring calls one
    guardrail on one case, so "no score in this repository was computed over a
    chain". That is how the Slack-token leak lived: each guardrail was right
    about the text it was handed.

    The fragments are glued in generated order so detections land adjacent,
    nested and straddling. That is the input shape a corpus of one labelled case
    per line cannot express, and it is where the two spans that have to merge
    come from.
    """
    result = _BUNDLED.run(content, IN)
    for verdict in result.verdicts:
        assert verdict.error is None, verdict.error
        assert verdict.saw == saw(content)
    covered = {
        offset
        for verdict in result.verdicts
        if verdict.decision == "redact"
        for finding in verdict.findings
        if finding.span is not None
        for offset in range(*finding.span)
    }
    assert _PLACEHOLDER.sub("", result.content) == _kept(content, covered), (
        content,
        result.content,
    )


# ==========================================================================
# PatternGuardrail
# ==========================================================================

# Pattern pieces that all match at least one character, so every pattern built
# from them constructs: `PatternGuardrail` refuses one that matches the empty
# string, and a strategy that mostly generated refused patterns would spend its
# budget on the constructor rather than on the scan.
_ATOM = st.sampled_from(
    ["a", "b", "0", "-", "@", ".", r"\d", r"\w", r"\s", "[ab]", "[^ab]", "[a-c0-9]"]
)
_QUANTIFIER = st.sampled_from(["", "", "+", "{1,3}", "{2}", "?", "*"])
_REQUIRED_QUANTIFIER = st.sampled_from(["", "", "+", "{1,3}", "{2}"])


@st.composite
def _pattern(draw: st.DrawFn) -> str:
    r"""A pattern that cannot match the empty string and cannot nest unbounded repeats.

    Both exclusions are the constructor's rather than this strategy's opinion,
    and the second is why no quantifier is ever applied to a group here: `(a+)+`
    is refused at construction, so generating it would exercise a refusal
    `tests/test_authoring.py` already pins by name.

    The first piece takes a quantifier with a minimum of one, which is what makes
    the whole pattern non-empty-matching and therefore constructible. `\b` is
    allowed in front of it for the same reason the published fixture pattern
    `\bJIRA-\d{4,}\b` has one: that is what a real rule looks like. It is safe
    HERE only because the body cannot match empty. See
    `test_a_word_boundary_gated_pattern_reports_a_zero_width_span` for what
    happens when it can.
    """
    body: str = draw(_ATOM) + draw(_REQUIRED_QUANTIFIER)
    for _ in range(draw(st.integers(0, 3))):
        body += draw(_ATOM) + draw(_QUANTIFIER)
    if draw(st.booleans()):
        alternative = draw(_ATOM) + draw(_REQUIRED_QUANTIFIER)
        body = f"(?:{body}|{alternative})"
    if draw(st.booleans()):
        body = r"\b" + body
    return body


# The sharp s and the capital sharp s casefold to "ss", two characters for one,
# so a banned substring found in the folded view spans a DIFFERENT number of
# source characters. That is the width change `_banned_spans` exists to handle,
# and content without one cannot tell a source span from a view span: on pure
# ASCII the two are equal and every offset is right by accident.
#
# A literal `s` and `S` are in the alphabet for a second reason, and without them
# one whole failure mode is unreachable. The banned substring is "ss", so an
# OVERLAPPING pair of occurrences needs three matching view characters in a row;
# drawn from the sharp s alone the view only ever gains "ss" two at a time, every
# occurrence starts on a source character boundary, and a walk that skipped the
# overlapping one would still cover every offset. Measured: without `s` in this
# alphabet, resuming the banned walk past the END of each match instead of one
# past its start leaves
# `test_a_banned_substring_reports_the_span_of_the_source_that_carries_it` green.
_PATTERN_CONTENT = st.text(alphabet="ab c0-9@.sSßẞ\n\t", max_size=30)

_BANNED = "ss"

# The character limit the guardrail below is built with, and the two lengths that
# straddle it. Seeded as explicit examples rather than left to the draw, because
# the boundary is one input wide and a bounded, derandomised run reaches it by
# luck: measured, a mutation making the limit fire AT its bound rather than one
# past it was caught on one version of the content alphabet and missed on the
# next, purely because widening the alphabet moved the derandomised sequence.
# A boundary a property depends on is a boundary the property should name.
_MAX_CHARS = 12


@example(patterns=["a"], content="a" * _MAX_CHARS)
@example(patterns=["a"], content="a" * (_MAX_CHARS + 1))
@given(st.lists(_pattern(), min_size=1, max_size=3), _PATTERN_CONTENT)
def test_a_pattern_guardrail_never_raises_and_reports_spans_inside_its_content(
    patterns: list[str], content: str
) -> None:
    """The defect: a caller's own rule turning a check into an exception.

    `rules` is the one check whose patterns come out of a user's configuration
    file, so its inputs are the least constrained in the package and its failures
    are the ones this repository cannot see coming. A raise out of `check` is a
    synthesised deny inside a chain and an unhandled exception outside one, and
    `PatternGuardrail`'s own docstring contemplates "a caller holding one
    guardrail".

    The span bound is the clause `docs/conformance.md` puts on every finding of
    every decision, asserted at the detector rather than at the chain. The chain
    checks it too and refuses a verdict that breaks it, which means a detector
    that breaks it is reported as a contract violation naming a clause: true, and
    not the same as knowing which pattern did it.

    Content is drawn from an alphabet with no lone surrogates, which is a real
    boundary rather than tidiness. `Limits(max_bytes=...)` counts UTF-8 bytes and
    `saw` hashes UTF-8, so a lone surrogate raises `UnicodeEncodeError` out of
    both, for the reason
    `test_the_span_machinery_never_raises_on_any_string_a_str_can_hold` records.

    All THREE sources are configured together, and that is what the rewrite
    assertion needs. Patterns, banned substrings and a size limit each scan the
    whole input independently and their results concatenate in source order,
    which is not span order; `_matches` re-sorts, and `_merge` looks only at the
    running end of the region it is extending, so an unsorted list makes the
    guardrail's own `Verdict.content` keep a character its findings say it
    removed. A chain re-sorts every span it collects before rewriting, so that
    failure is invisible through a chain and visible only here, in the rewrite
    `PatternGuardrail`'s own docstring offers to "a caller holding one
    guardrail".
    """
    guardrail = PatternGuardrail(
        name="generated",
        version="0.1.0",
        patterns={f"T{index}": pattern for index, pattern in enumerate(patterns)},
        banned={"BANNED": [_BANNED]},
        limits=Limits(max_chars=_MAX_CHARS),
        on_match="redact",
    )
    verdict = guardrail.check(content, IN)
    assert verdict.saw == saw(content)
    covered: set[int] = set()
    for finding in verdict.findings:
        assert finding.confidence is None
        assert finding.span is not None
        start, end = finding.span
        assert 0 <= start < end <= len(content), (patterns, content, finding.span)
        covered |= set(range(start, end))
    if verdict.decision == "allow":
        assert not verdict.findings
        return
    assert verdict.content is not None
    assert _PLACEHOLDER.sub("", verdict.content) == _kept(content, covered), (
        patterns,
        content,
        verdict.content,
    )


@given(_PATTERN_CONTENT)
def test_a_banned_substring_reports_the_span_of_the_source_that_carries_it(content: str) -> None:
    """The defect: a span reported into the folded view rather than into the source.

    `docs/conformance.md` fixes this one: banned substrings "match
    case-insensitively, over a case-folded view, and the span reported is the
    SOURCE span. Where folding changes a character's width the two differ." A
    view span reported as a source span is off by the expansion, so the redaction
    removes the wrong characters while the finding says otherwise, and the corpus
    cannot see it on the ASCII where the two coincide.

    Stated as containment rather than equality, because equality is false and the
    difference is the point: a fold that EXPANDS means one source character
    produced two view characters, so a match on the first half of a product spans
    the whole source character and folding it back gives more than the match.
    What must hold is that the source run the finding names carries the banned
    text, which is exactly what makes redacting that run remove it.

    The second half is the `finditer` Critical arriving on the other engine.
    `_banned_spans` walks the view with `find`, resuming one past the previous
    match's START rather than its end, for the same reason `_scan` resumes at
    `match.start() + 1`: resuming past the end skips an overlapping occurrence,
    and a skipped occurrence is a banned substring standing in content the
    verdict reports as redacted. So EVERY occurrence in the view has to be
    covered, not merely every one the walk happened to reach.
    """
    guardrail = PatternGuardrail(
        name="banned-only", version="0.1.0", banned={"BANNED": [_BANNED]}, on_match="redact"
    )
    verdict = guardrail.check(content, IN)
    covered: set[int] = set()
    for finding in verdict.findings:
        assert finding.type == "BANNED"
        assert finding.span is not None
        start, end = finding.span
        assert 0 <= start < end <= len(content)
        assert _BANNED in casefold_view(content[start:end]).text, (content, finding.span)
        covered |= set(range(start, end))

    view = casefold_view(content)
    for offset in range(len(view.text) - len(_BANNED) + 1):
        if not view.text.startswith(_BANNED, offset):
            continue
        source_start, source_end = view.span(offset, offset + len(_BANNED))
        assert set(range(source_start, source_end)) <= covered, (content, offset)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a word-boundary or lookbehind gated pattern reports a zero-width span; the "
        "chain refuses the verdict, so the composition fails closed rather than leaking"
    ),
)
@example(pattern=r"\ba*", content="0")
@given(st.sampled_from([r"\ba*", r"(?<=a)b*"]), st.text(alphabet="ab0", min_size=1, max_size=4))
def test_a_word_boundary_gated_pattern_reports_a_zero_width_span(
    pattern: str, content: str
) -> None:
    r"""A KNOWN MISS, pinned rather than fixed, and not weakened to make it pass.

    `docs/conformance.md` requires every finding's span to satisfy
    `0 <= start < end <= len(content)`, and `PatternGuardrail` can break it. The
    constructor probes `pattern.search("")` and refuses a pattern that matches
    the empty string OUTRIGHT; a pattern gated by `\b` or by a lookbehind does
    not match at position zero of the empty string, so it constructs and then
    matches zero-width against real content. `\ba*` over "0" reports a finding at
    (0, 0).

    Not fixed, on the merits rather than for the cost. The alternative, dropping
    zero-width matches inside the scan, turns a broken rule into a check that
    runs and never matches, which is the "configured and silent" failure this
    constructor already refuses five other ways, and it would rewrite a design
    decision `authoring.py` records at length. What the source says happens is
    what happens, and the assertion below is that containment: the chain refuses
    the verdict and synthesises a deny, so nothing is forwarded. A known miss
    whose containment nobody checks is just a miss.

    Fixing it makes this XPASS and the suite red, which is the alarm.

    The containment is asserted only where a zero-width finding actually appears,
    and the ordering matters: on the inputs where these patterns behave normally
    the guardrail returns an ordinary `redact`, so asserting a deny unconditionally
    would make this test fail for a second, unrelated reason and hide which of the
    two an `xfail` was recording.
    """
    guardrail = PatternGuardrail(
        name="gated", version="0.1.0", patterns={"T": pattern}, on_match="redact"
    )
    verdict = guardrail.check(content, IN)
    if any(f.span is not None and f.span[0] == f.span[1] for f in verdict.findings):
        result = GuardrailChain([guardrail]).run(content, IN)
        assert result.decision == "deny"
        assert result.content == content
        assert result.verdicts[0].error is not None
    for finding in verdict.findings:
        assert finding.span is not None
        start, end = finding.span
        assert 0 <= start < end <= len(content), (pattern, content, finding.span)
