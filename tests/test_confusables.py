"""The confusables constraint, and the two halves of every rule in it.

Both halves of both rules are load-bearing, and a check that keeps only one half
of either passes almost every input anybody would think to type while denying a
language. So the tests below come in pairs: the spoof that must fire, and the
ordinary sentence that must not, chosen so that dropping the condition under
test flips exactly one of them.

Every test here was watched to FAIL against a mutation of the thing it guards,
with `__pycache__` cleared between runs, and the whole selector was run
UNMUTATED first: a selector naming a test that does not exist also exits
non-zero and reads exactly like a kill. The mutations:

- the identifier-profile condition dropped from `_confusable_into`;
- the majority-script condition dropped from `_confusable_into`;
- the wildcard-only-prototype guard dropped from `_confusable_into`;
- the Highly Restrictive test dropped;
- the Highly Restrictive table widened to "Latin plus any one script";
- the first-code-point preference dropped from the majority tie-break;
- `WHOLE_SCRIPT_CONFUSABLE` scanned over tokens instead of over labels;
- the non-Latin requirement dropped from the whole-script rule;
- the single-script requirement dropped from the whole-script rule;
- default-ignorable code points admitted into tokens;
- the ASCII fast path made to report rather than to skip;
- the sort dropped from `_matches`;
- `allow` accepted as a decision;
- an `on_detect` mapping missing a direction accepted;
- `identifier_allowed` made to return True for everything.

THREE OF THEM SURVIVED THE FIRST TEST WRITTEN FOR THEM, and each survival is
recorded beside the test it rewrote, because each one is the same mistake: a
test that reaches the guard through an input the guard is not what refuses.
`iPhone対応` does not hold the Highly Restrictive table down, `copy.example.org`
does not hold the non-Latin clause down, and a zero-width space does not hold
the default-ignorable exclusion down.
"""

from __future__ import annotations

import pytest

from jamjet_guardrails.detectors import build
from jamjet_guardrails.detectors.confusables import (
    _IGNORABLE,
    CONFUSABLES_TYPES,
    ConfusablesGuardrail,
    _confusable_into,
    _is_token_character,
    _mixed_script_spans,
    _tokens,
    _whole_script_spans,
)
from jamjet_guardrails.detectors.injection_structural import (
    _EMBED_CLOSE,
    _EMBED_OPEN,
    _ISOLATE_CLOSE,
    _ISOLATE_OPEN,
    _TAG_END,
    _TAG_START,
    _ZERO_WIDTH,
)
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.types import Context

IN = Context(direction="input", origin="user")
OUT = Context(direction="output", origin="model")

# Written as escapes, never as literals. A Cyrillic small a inside a Latin word
# is invisible in this file, invisible in the diff that adds it and invisible in
# the review that should have caught it, which is the whole reason this check
# exists. `corpora/NOTICE.md` makes the same argument for the corpus.
CYRILLIC_A = "\u0430"
CYRILLIC_O = "\u043e"
CYRILLIC_ER = "\u0440"
CYRILLIC_EM = "\u043c"
CYRILLIC_DE = "\u0434"
CYRILLIC_BE = "\u0431"
CYRILLIC_EF = "\u0444"
CYRILLIC_U = "\u0443"
CYRILLIC_PALOCHKA = "\u04cf"
GREEK_RHO = "\u03a1"
ZWSP = "\u200b"
# Default-ignorable AND a letter or a mark, which is the combination the token
# scan has to exclude by hand. A zero-width space is `Cf` and is refused by the
# category test on its own, so a test written with one asserts nothing about the
# exclusion: measured, admitting default-ignorable code points into tokens
# changes no verdict on any input built from U+200B.
GRAPHEME_JOINER = "\u034f"  # COMBINING GRAPHEME JOINER, Mn
HANGUL_FILLER = "\u115f"  # HANGUL CHOSEONG FILLER, Lo

# `paypal` written entirely in Cyrillic: the whole-script confusable this check
# reports inside a host label and deliberately does not report in prose.
CYRILLIC_PAYPAL = (
    CYRILLIC_ER + CYRILLIC_A + CYRILLIC_U + CYRILLIC_ER + CYRILLIC_A + CYRILLIC_PALOCHKA
)


def decision(content: str, context: Context = IN) -> str:
    return ConfusablesGuardrail().check(content, context).decision


# ==========================================================================
# MIXED_SCRIPT_CONFUSABLE: both conditions, one test per condition per side.
# ==========================================================================


def test_one_substituted_letter_that_reads_as_the_majority_script_is_reported() -> None:
    """The attack, in its smallest form. `pаypal` is six glyphs and two scripts."""
    content = f"Sign in at p{CYRILLIC_A}ypal to confirm."
    verdict = ConfusablesGuardrail().check(content, IN)
    assert verdict.decision == "deny"
    assert [f.type for f in verdict.findings] == ["MIXED_SCRIPT_CONFUSABLE"]
    (finding,) = verdict.findings
    assert finding.span is not None
    assert content[finding.span[0] : finding.span[1]] == f"p{CYRILLIC_A}ypal"


def test_a_latin_brand_with_a_cyrillic_ending_whose_letters_do_not_read_as_latin() -> None:
    """The negative the whole identifier-profile condition exists for.

    `iPhoneом` is mixed script and its majority is Latin, so a check that asked
    only whether the minority letters had a LATIN prototype would deny it: both
    do. Cyrillic em folds to U+028D LATIN SMALL LETTER TURNED W, which is
    outside the UTS #39 identifier profile and is therefore not a character any
    brand is written in, so the token imitates nothing.

    Mutation-checked: dropping the `identifier_allowed` clause in
    `_confusable_into` makes this deny.
    """
    content = f"Купил iPhone{CYRILLIC_O}{CYRILLIC_EM} вчера"
    assert decision(content) == "allow"


def test_a_latin_letter_folding_to_latin_is_not_a_spoof_of_a_cyrillic_majority() -> None:
    """The majority-script condition, from the side that shows it is not free.

    A Cyrillic word carrying one Latin `o` is mixed script, and the Latin `o`
    folds to Latin `o`, which is not the majority script. A rule that dropped
    the majority condition and asked only "does this fold to something" would
    report every such token.

    Mutation-checked: removing the `script in _scripts_of(point)` clause makes
    this deny.
    """
    content = "Москва и Питeр"
    assert decision(content) == "allow"


def test_a_token_of_one_script_is_never_a_mixed_script_confusable() -> None:
    """Russian prose is Russian prose. The corpus carries 18 more of these."""
    assert decision("Компания Google") == "allow"


# ==========================================================================
# Highly Restrictive: the permitted combinations are Unicode's table.
# ==========================================================================


def test_japanese_text_carrying_a_latin_brand_passes_by_rule() -> None:
    """Latin plus Han plus Hiragana plus Katakana is Highly Restrictive.

    No exemption, no brand list: `iPhone対応` is one token in three scripts and
    UTS #39 section 5.2 permits that combination.

    THIS CASE DOES NOT HOLD THE HIGHLY RESTRICTIVE TEST DOWN, and saying so is
    the point of writing it here. Measured: with `_HIGHLY_RESTRICTIVE` deleted
    this still allows, because no kanji folds to a Latin letter, so the second
    condition refuses it anyway. The case below is the one that holds the table,
    and it was found by mutation rather than by reasoning.
    """
    assert decision("iPhone対応のアプリ", OUT) == "allow"


def test_korean_text_carrying_a_latin_brand_passes_by_rule() -> None:
    """Latin plus Han plus Hangul is the third permitted combination."""
    assert decision("KakaoTalk으로 연락 주세요", OUT) == "allow"


def test_han_beside_katakana_is_permitted_and_it_is_the_table_that_permits_it() -> None:
    """`3カ月間` is ordinary Japanese for a period of three months.

    It is also, character for character, the shape the mixed-script rule is
    built to report: the token is Han and Katakana, its majority is Han, and the
    one Katakana character folds to U+529B, a Han character inside the
    identifier profile. Only the Highly Restrictive table stops it.

    The token has to be `3カ月間` on its own, which is why the sentence around it
    is a specification line rather than prose. Written as `契約は3カ月間有効です`
    the hiragana join the same token, four characters are then outside the Han
    majority, and the three that are hiragana fold to nothing Han, so the second
    condition refuses it and the table is not what saved it. That version was
    written first and the mutation survived it.

    Mutation-checked: deleting the `_HIGHLY_RESTRICTIVE` test makes this deny,
    and it is the ONLY test in this file that it makes deny. Katakana folds into
    Han for ten characters at 16.0.0 and Hangul and Bopomofo do the same, so the
    table is load-bearing on real CJK text rather than on constructed input.
    """
    assert decision("無料期間: 3カ月間", OUT) == "allow"


def test_latin_with_cyrillic_is_not_a_permitted_combination() -> None:
    """The false-reject control on the two tests above.

    A table widened to "Latin plus any one other script" would pass all three,
    and Latin plus Cyrillic is the whole of the attack.

    Mutation-checked: widening `_HIGHLY_RESTRICTIVE` that way makes this allow.
    """
    assert decision(f"Contact {CYRILLIC_A}pple support") == "deny"


def test_a_tie_is_broken_by_the_first_code_point_and_the_direction_matters() -> None:
    """`gооd` with two Cyrillic o denies; `cодe` with two Cyrillic letters does not.

    Both tokens are two Latin letters and two Cyrillic ones, so the counts tie
    and the tie-break decides which half is the minority. The first code point
    of `gооd` is Latin, so the Cyrillic pair is the minority and folds onto
    Latin; the first code point of `cодe` is Latin but its Cyrillic pair is
    `о` and `д`, and `д` folds to nothing outside Cyrillic.

    Mutation-checked: replacing the first-code-point preference with the
    alphabetically first candidate makes `gооd` allow, because `Cyrillic` sorts
    before `Latin` and the Latin half then becomes the minority. A test written
    with a Greek-first token cannot see that: `Greek` sorts before `Latin` too,
    so both spellings agree there and the mutation survives.
    """
    assert decision(f"That looks g{CYRILLIC_O}{CYRILLIC_O}d to me.") == "deny"
    assert decision(f"The c{CYRILLIC_O}{CYRILLIC_DE}e review is blocked.") == "allow"


def test_a_prototype_of_nothing_but_wildcards_is_not_evidence() -> None:
    """Cyrillic be folds to the DIGIT 6, which is Common and so is every script.

    A prototype with no script of its own imitates no word in any script, and
    counting it would make the majority-script test turn on punctuation and
    digits: `all(script in ... for point in [])` is vacuously true, so without
    the guard EVERY character with such a prototype folds into every script at
    once.

    Mutation-checked: removing the `if not substantive` guard from
    `_confusable_into` makes both assertions below flip.
    """
    from jamjet_guardrails._unicode import skeleton

    assert skeleton(CYRILLIC_BE).text == "6"
    assert not _confusable_into(CYRILLIC_BE, "Latin")
    assert decision(f"the ID{CYRILLIC_BE} field") == "allow"


# ==========================================================================
# WHOLE_SCRIPT_CONFUSABLE: only where a spoof has a target.
# ==========================================================================


def test_a_single_script_label_that_reads_as_latin_is_reported_inside_a_url() -> None:
    content = f"Go to https://{CYRILLIC_PAYPAL}.com to reset."
    verdict = ConfusablesGuardrail().check(content, IN)
    assert verdict.decision == "deny"
    assert [f.type for f in verdict.findings] == ["WHOLE_SCRIPT_CONFUSABLE"]
    (finding,) = verdict.findings
    assert finding.span is not None
    assert content[finding.span[0] : finding.span[1]] == CYRILLIC_PAYPAL


@pytest.mark.parametrize(
    "content",
    [
        "Write to support@{}.com about it.",
        "Follow @{} for updates.",
        "Try https://{}.com:8443/login",
        "Open https://user@{}.com/reset",
    ],
)
def test_the_three_contexts_a_whole_script_confusable_is_reported_in(content: str) -> None:
    """A host label, an email domain label and a handle, plus a port and userinfo.

    The last two are there because the authority is not the host: a port left on
    the label makes it mixed with digits and userinfo read as the host reports
    the wrong run entirely.
    """
    assert decision(content.format(CYRILLIC_PAYPAL)) == "deny"


def test_the_same_word_in_prose_is_a_word() -> None:
    """The condition that keeps this rule from denying a language.

    `сор` is Russian for rubbish and `ухо` is an ear, and both are built
    entirely out of letters that fold to Latin. A whole-script rule applied to
    prose reports every one of them.

    Mutation-checked: scanning the content instead of the labels makes this
    deny, and makes the two Russian words below deny as well.
    """
    assert decision(f"The word {CYRILLIC_PAYPAL} appears in the body.") == "allow"
    assert decision("Сор и ухо") == "allow"


def test_a_genuine_cyrillic_domain_is_not_a_whole_script_confusable() -> None:
    """`почта.рф`, which is the measurement this whole check turned on.

    Cyrillic er folds to `p` and Cyrillic ef to U+0278 LATIN SMALL LETTER PHI,
    so BOTH labels of the Russian ccTLD have Latin prototypes and neither is
    written in characters a Latin label is written in. A check that asked only
    about script denies every `.рф` domain there is.

    Mutation-checked: dropping the `identifier_allowed` clause makes this deny
    on the `рф` label.
    """
    assert decision("https://почта.рф/вход") == "allow"
    assert decision(f"https://мвд.{CYRILLIC_ER}{CYRILLIC_EF}/news") == "allow"


def test_a_latin_host_label_is_never_a_whole_script_confusable() -> None:
    """The rule requires a NON-Latin single-script label.

    The label has to be non-ASCII to hold the clause down, and that is the whole
    lesson of this test: an ASCII label never reaches the script test at all,
    because the ASCII fast path skips it first. Measured, `copy.example.org`
    still allows with the clause deleted, so a test written with it asserts
    nothing. `café` is Latin, is not ASCII, and folds character for character to
    Latin characters inside the identifier profile.

    Mutation-checked: dropping the `"Latin" in resolved` clause makes this deny,
    and with it every internationalised Latin hostname there is.
    """
    assert decision("Go to https://copy.example.org/ instead.") == "allow"
    assert decision("Buy at https://caf\u00e9-noir.example.com/ today.") == "allow"


def test_a_mixed_label_is_not_a_whole_script_confusable() -> None:
    """It is a MIXED one, reported by the other signal and over the token.

    The two signals are mutually exclusive on one token by construction: mixed
    needs an empty resolved script set and whole-script needs a non-empty one.

    Mutation-checked: dropping the single-script requirement makes this report
    both types over overlapping spans.
    """
    content = f"Visit https://{CYRILLIC_A}pple.com/verify"
    verdict = ConfusablesGuardrail().check(content, IN)
    assert [f.type for f in verdict.findings] == ["MIXED_SCRIPT_CONFUSABLE"]


def test_an_ipv6_literal_yields_no_labels() -> None:
    """`[::1]` splits into pieces a label scan would read as labels."""
    assert _whole_script_spans("http://[2001:db8::1]:8080/health") == []


# ==========================================================================
# Tokens, and the code points that end one.
# ==========================================================================


def test_an_apostrophe_and_a_hyphen_end_a_token() -> None:
    """Splitting is the conservative direction, and it is what makes the two
    ordinary Russian spellings of a foreign brand pass.

    `iPhone-ом` and `iPhone'ом` are two single-script tokens each. Neither
    punctuation mark is named in the module: they end a token because they are
    not letters, marks or numbers.
    """
    ending = CYRILLIC_O + CYRILLIC_EM
    assert _tokens(f"iPhone-{ending}") == [(0, 6), (7, 9)]
    assert _tokens(f"iPhone'{ending}") == [(0, 6), (7, 9)]


def test_a_default_ignorable_code_point_ends_a_token() -> None:
    """The disjointness requirement, from the token scan's side.

    A token that ran ACROSS a zero-width space would report a span containing
    it, and a contiguous span containing that code point shares it with
    `ZERO_WIDTH_SMUGGLING`.

    Mutation-checked: admitting default-ignorable code points into tokens makes
    the span below one run of eight rather than two of one and seven, and makes
    `test_this_check_claims_no_code_point_any_structural_signal_claims`
    meaningless.
    """
    assert _tokens(f"{CYRILLIC_A}{GRAPHEME_JOINER}pple") == [(0, 1), (2, 6)]
    assert _tokens(f"{CYRILLIC_A}{HANGUL_FILLER}pple") == [(0, 1), (2, 6)]
    assert _tokens(f"{CYRILLIC_A}{ZWSP}pple") == [(0, 1), (2, 6)]
    assert decision(f"Sign in at {CYRILLIC_A}{GRAPHEME_JOINER}pple support") == "allow"


def test_this_check_claims_no_code_point_any_structural_signal_claims() -> None:
    """Pairwise against all three `injection-structural` signals.

    A finding from this check and a finding from that one must never cover the
    same code point, or a merged placeholder would name two checks for one
    character and the two corpora would be scoring overlapping claims. The
    property holds because every code point either signal there reports is
    default-ignorable, and `_IGNORABLE` is IMPORTED from that module rather than
    restated, so the two cannot drift apart.

    Asserted over the sets rather than over an input, for the reason that module
    asserts its own three: a run of inputs that happened to work is not the
    invariant.
    """
    tags = {chr(point) for point in range(_TAG_START, _TAG_END + 1)}
    bidi = set(_EMBED_OPEN) | set(_ISOLATE_OPEN) | {_EMBED_CLOSE, _ISOLATE_CLOSE}
    zero_width = set(_ZERO_WIDTH)

    # Non-vacuous on both sides: an empty set is disjoint from everything.
    assert (len(tags), len(bidi)) == (128, 9)
    assert len(zero_width) > 3000
    assert len(_IGNORABLE) > 4000

    # Derived from `_is_token_character` itself, not from a second copy of its
    # rule: a test that re-implements the thing it guards cannot see the thing
    # it guards change. Measured, the `- _IGNORABLE` spelling passed with the
    # exclusion deleted from the function.
    claimable = {chr(point) for point in range(0x110000) if _is_token_character(chr(point))}
    assert len(claimable) > 100_000, "the claimable set emptied; this test would prove nothing"

    assert not claimable & tags
    assert not claimable & bidi
    assert not claimable & zero_width


def test_every_span_this_check_reports_lies_inside_a_token_or_a_label() -> None:
    """The property the assertion above rests on, checked on real content.

    Sets are the proof; this is the run that says the spans really do come from
    the token scan and not from somewhere that never consulted it.
    """
    content = (
        f"Sign in at p{CYRILLIC_A}ypal or https://{CYRILLIC_PAYPAL}.com "
        f"and re{GRAPHEME_JOINER}ad thi{HANGUL_FILLER}s{ZWSP * 4} line."
    )
    verdict = ConfusablesGuardrail().check(content, IN)
    assert verdict.findings
    for finding in verdict.findings:
        assert finding.span is not None
        run = content[finding.span[0] : finding.span[1]]
        assert not set(run) & _IGNORABLE, f"{run!r} covers a default-ignorable code point"


def test_ordinary_ascii_prose_reaches_neither_signal() -> None:
    """The fast path, asserted as behaviour rather than as an optimisation.

    Mutation-checked: making the ASCII branch fall through rather than skip
    changes no verdict anywhere, which is the point; making it return a finding
    fails here.
    """
    assert _mixed_script_spans("The quick brown fox jumps over the lazy dog.") == []
    assert decision("SELECT * FROM users WHERE id = 42;", OUT) == "allow"


# ==========================================================================
# The verdict, the spans and the decision.
# ==========================================================================


def test_findings_from_both_signals_come_back_in_span_order() -> None:
    """The sort in `_matches` is what `_merge` requires of its input.

    Each signal walks left to right, so each list is ordered on its own and it
    is the concatenation that is not: here the whole-script label comes first in
    the content and second in the concatenation.

    Mutation-checked: returning `found` unsorted fails this.
    """
    content = f"https://{CYRILLIC_PAYPAL}.com and then p{CYRILLIC_A}ypal"
    verdict = ConfusablesGuardrail().check(content, IN)
    assert [f.type for f in verdict.findings] == [
        "WHOLE_SCRIPT_CONFUSABLE",
        "MIXED_SCRIPT_CONFUSABLE",
    ]
    spans = [f.span for f in verdict.findings]
    assert None not in spans
    assert spans == sorted(spans)  # type: ignore[type-var]


def test_a_redaction_removes_the_whole_token_and_not_only_the_substituted_letter() -> None:
    """A placeholder over one letter leaves `p[REDACTED]ypal` on the page, which
    still reads as the brand."""
    content = f"Sign in at p{CYRILLIC_A}ypal now."
    verdict = ConfusablesGuardrail(on_detect="redact").check(content, IN)
    assert verdict.decision == "redact"
    assert verdict.content == "Sign in at [REDACTED:MIXED_SCRIPT_CONFUSABLE] now."


def test_the_decision_can_differ_by_direction() -> None:
    guardrail = ConfusablesGuardrail(on_detect={"input": "redact", "output": "deny"})
    content = f"p{CYRILLIC_A}ypal"
    assert guardrail.check(content, IN).decision == "redact"
    assert guardrail.check(content, OUT).decision == "deny"


def test_a_mapping_that_omits_a_direction_is_refused() -> None:
    """The alternative is a KeyError from inside check, which names nothing.

    Mutation-checked: removing the `missing` guard makes this raise KeyError
    from `check` instead, and only for content that matched.
    """
    with pytest.raises(GuardrailUnavailableError, match="no decision for"):
        ConfusablesGuardrail(on_detect={"input": "deny"})


def test_a_mapping_that_names_a_direction_this_check_does_not_run_in_is_refused() -> None:
    with pytest.raises(GuardrailUnavailableError, match="never be asked about"):
        ConfusablesGuardrail(
            on_detect={"input": "deny", "output": "deny", "sideways": "deny"}  # type: ignore[dict-item]
        )


def test_allow_is_refused_as_a_decision() -> None:
    """A check configured to allow on a detection runs and cannot act.

    Mutation-checked: accepting `allow` makes this construct cleanly and then
    return `allow` over a spoofed hostname.
    """
    with pytest.raises(ValueError, match="must be 'redact' or 'deny'"):
        ConfusablesGuardrail(on_detect="allow")


def test_the_registry_builds_this_check_with_both_directions() -> None:
    guardrail = build("confusables")
    assert guardrail.name == "confusables"
    assert guardrail.kind == "constraint"
    assert guardrail.directions == frozenset({"input", "output"})


def test_the_declared_types_are_the_types_it_can_report() -> None:
    """A type nobody can produce has no recall figure; one nobody declares
    cannot be labelled."""
    produced = set()
    for content in (
        f"p{CYRILLIC_A}ypal",
        f"https://{CYRILLIC_PAYPAL}.com",
    ):
        produced |= {f.type for f in ConfusablesGuardrail().check(content, IN).findings}
    assert produced == CONFUSABLES_TYPES


def test_the_empty_content_allows_rather_than_raising() -> None:
    assert decision("") == "allow"


def test_a_greek_capital_rho_standing_in_for_p_is_reported() -> None:
    """Greek, not only Cyrillic. The corpus carries Armenian too."""
    assert decision(f"{GREEK_RHO}aypal charged your card.", OUT) == "deny"
