"""Structural instruction-smuggling constraint.

Injection that is visible in the TEXT is a classifier's problem. Injection that
is invisible in the RENDERING is not: it is a property of the bytes, it needs no
model, and a model trained on natural language will not see it at all. Those two
facts are why this check exists in the core package while the classifier lives
behind an extra.

Every signal here is chosen for one property: no legitimate document produces it
by accident. That is what makes a published precision number defensible, and it
is why Private Use Area characters are NOT a signal -- see the note by
`_ZERO_WIDTH` for what happens to a check that fires on a developer's shell
prompt.
"""

from __future__ import annotations

import unicodedata

from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import (
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Provenance,
    Verdict,
)

INJECTION_TYPES = frozenset(
    {
        "INVISIBLE_TAG_CHARS",
        "BIDI_OVERRIDE",
        "ZERO_WIDTH_SMUGGLING",
    }
)

_VERSION = "0.1.0"

# U+E0000..U+E007F. TAG SPACE (U+E0020) through TAG TILDE (U+E007E) mirror
# printable ASCII one-for-one, which is the whole smuggling primitive: any
# instruction can be written invisibly and survives copy-paste, most log
# viewers, and every model tokenizer that does not strip them.
_TAG_START = 0xE0000
_TAG_END = 0xE007F
# U+E007F CANCEL TAG, the terminator every RGI flag sequence ends with. Held as
# a character rather than as a codepoint because the test below is `endswith`,
# which is total: an empty run answers False instead of raising on `run[-1]`.
_CANCEL_TAG = "\U000e007f"
# U+1F3F4 WAVING BLACK FLAG. The base of every RGI subdivision flag sequence and
# the ONLY context in which a tag run is ordinary text.
_FLAG_BASE = 0x1F3F4
# The RGI subdivision flag set is CLOSED: Unicode defines exactly these three
# sequences and no others. A closed set gets an ALLOWLIST, not a test of its
# members' shape, and that distinction is the whole of what is recorded here.
#
# The shape test this replaced was BYPASSABLE, written down rather than quietly
# deleted because the failure generalises. It required a U+1F3F4 base, a single
# trailing CANCEL TAG, tag letters throughout, and at most five of them: four
# conditions, each load-bearing, each pinned by a test that was watched to fail.
# It still allowed a complete injection. U+1F3F4 is NOT itself a tag character,
# so it ENDS the maximal tag run before it, and the exemption is applied per
# run; chaining bases therefore chains exempt runs, and the five-letter bound
# capped per-run capacity rather than total capacity. Measured against the
# shipped code: "Summarise this. " followed by seven repeats of U+1F3F4 + up to
# five tag letters + CANCEL TAG carried "ignore all previous instructions" past
# this deny-by-default detector with `_tag_spans` returning []. The payload is
# 32 characters, so six of the seven flags carry five each and the seventh
# carries the remaining two: seven visible black flags for 32 smuggled
# characters, and NOT a flat five per flag. That exact input is kept as
# `test_chained_flag_bases_do_not_smuggle_a_payload`.
#
# An allowlist has no such seam: an exempt run is one of exactly three fixed
# strings, so it carries no attacker-chosen bytes at all and a row of them is a
# row of flags. The general rule, for the next exemption written into this
# module: an exemption that APPROXIMATES a closed set with a shape test is
# chainable, and chaining turns the exemption into precisely the channel it was
# written to deny.
#
# MEMBERSHIP CAN BE SOFTENED IN TWO DIRECTIONS AND ONLY ONE WAS PINNED.
# `any(code.startswith(known) ...)` accepts a code that EXTENDS a real one, and
# `test_six_tag_letters_are_too_many_to_be_a_subdivision_code` catches it.
# `any(known.startswith(code) ...)` accepts every PREFIX instead: twelve strings
# rather than three, the empty one included. Measured against the file BEFORE
# `test_a_tag_run_shorter_than_a_subdivision_code_is_not_exempt` was written,
# that second mutation changed no test and no corpus case, so the allowlist
# could have been widened fourfold in silence. Twelve values chosen per visible
# flag base is not nothing, and bases chain, which is the seam above in a
# smaller form. That test holds the other direction now.
_RGI_SUBDIVISION_CODES = frozenset({"gbeng", "gbsct", "gbwls"})


def _is_valid_flag_sequence(content: str, start: int, end: int) -> bool:
    """Whether a tag run at [start, end) is an RGI subdivision flag's payload.

    `[start, end)` must be a run of TAG characters, which is what `_tag_spans`
    passes and what the decode below relies on: it subtracts `_TAG_START`, and
    `chr` raises on anything under U+E0000.

    Three conditions, and each one closes a laundering route:
      - the character before the run is U+1F3F4, so a flag-shaped run standing
        on its own is not exempt;
      - the run ENDS with CANCEL TAG, since the code is read as everything
        before the last character and `gbsct` plus one more letter would
        otherwise read as `gbsct`;
      - what is left decodes to one of the three codes Unicode defines.
    """
    if start == 0 or ord(content[start - 1]) != _FLAG_BASE:
        return False
    run = content[start:end]
    if not run.endswith(_CANCEL_TAG):
        return False
    code = "".join(chr(ord(ch) - _TAG_START) for ch in run[: -len(_CANCEL_TAG)])
    return code in _RGI_SUBDIVISION_CODES


def _tag_spans(content: str) -> list[tuple[int, int]]:
    """Maximal runs of Unicode tag characters, minus valid flag sequences."""
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(content)
    while index < length:
        if not _TAG_START <= ord(content[index]) <= _TAG_END:
            index += 1
            continue
        start = index
        while index < length and _TAG_START <= ord(content[index]) <= _TAG_END:
            index += 1
        if not _is_valid_flag_sequence(content, start, index):
            spans.append((start, index))
    return spans


# Two families, and they do NOT pop each other. U+202A..U+202E are the legacy
# embeddings and overrides, closed by PDF; U+2066..U+2068 are the isolates,
# closed by PDI. Counting both in one stack lets `LRE ... PDI` balance, which is
# the exact shape a Trojan Source payload uses to look tidy while the embedding
# it opened is still open.
_EMBED_OPEN = frozenset("\u202a\u202b\u202d\u202e")
_EMBED_CLOSE = "\u202c"
_ISOLATE_OPEN = frozenset("\u2066\u2067\u2068")
_ISOLATE_CLOSE = "\u2069"


def _bidi_spans(content: str) -> list[tuple[int, int]]:
    """Every bidi control that no partner of its own family terminates.

    Balanced controls are ordinary formatting and are NOT reported: real Arabic
    and Hebrew text uses them, and a check that denies a language is not a
    security control, it is a check that gets switched off. What is reported is
    imbalance -- and imbalance comes in two shapes, which are reported on
    DIFFERENT grounds. An earlier version of this paragraph gave one ground for
    both, and it was false of half of them.

      - An unbalanced INITIATOR leaves a scope open with no end, so whatever
        divergence it causes runs to the end of the paragraph rather than to a
        point the author chose. The claim is that the divergence is UNBOUNDED,
        not that it is always present: measured with GNU FriBidi 1.0.16, an
        unclosed LRE in front of left-to-right text renders that text unchanged,
        because an embedding onto the direction the text already has reorders
        nothing. What cannot happen is for the author to bound it.
      - An unbalanced TERMINATOR reorders NOTHING AT ALL. Measured the same way,
        `harmless<PDF> text` and `harmless<PDI> text` render byte-identically to
        `harmless text`. They are reported as a malformed control sequence
        rather than as a reordering: a terminator that closes nothing is a
        document that was cut, or a probe. That is a weaker ground than the one
        above and it is stated separately rather than borrowed from it.
        `inj-0019` and `inj-0020` are the pair, and `corpora/NOTICE.md`
        discloses them, because the realistic population is not an attacker: it
        is a pipeline that splits a document across a balanced `LRE ... PDF`,
        which puts an unclosed initiator in one chunk and a stray terminator in
        the next. `inj-0139` and `inj-0140` are those two chunks.

    Two counters, one per family, are not enough on their own, because a pairing
    this function accepts has to be a pairing a RENDERER accepts. Both rules
    below were measured against GNU FriBidi 1.0.16; the first closes a bypass,
    the second removes a false positive:

      - UAX #9 X7 ignores a PDF while the innermost open initiator is an
        ISOLATE, so `RLO ... LRI ... PDF ... PDI` leaves the override in force
        past the PDI. Per-family counters call that balanced. Measured: the text
        AFTER the PDI still renders reversed, which it can only do while the
        override is open. Pinned by
        `test_a_pdf_inside_an_isolate_does_not_close_an_override_outside_it`.

      - UAX #9 X6a pops down to the isolate initiator, so an embedding opened
        inside an isolate is terminated by the PDI and is NOT reported. Measured:
        the text after the PDI renders unchanged, so the effect cannot escape the
        isolate, which is the containment a balanced pair also has. Pinned by
        `test_an_isolate_terminates_an_embedding_opened_inside_it`.

    One PDI can pop many embeddings, but every control is pushed once and popped
    at most once, so the whole scan stays linear in the length of the input.
    """
    embeds: list[int] = []
    isolates: list[int] = []
    unbalanced: list[int] = []
    for index, char in enumerate(content):
        if char in _EMBED_OPEN:
            embeds.append(index)
        elif char in _ISOLATE_OPEN:
            isolates.append(index)
        elif char == _EMBED_CLOSE:
            if embeds and (not isolates or isolates[-1] < embeds[-1]):
                embeds.pop()
            else:
                unbalanced.append(index)
        elif char == _ISOLATE_CLOSE:
            if isolates:
                opened = isolates.pop()
                while embeds and embeds[-1] > opened:
                    embeds.pop()
            else:
                unbalanced.append(index)
        elif unicodedata.bidirectional(char) == "B":
            # A control's scope ends at its paragraph (UAX #9 P1), so nothing
            # open here is ever closed and a PDF after this point closes none of
            # it. Without the split, an attacker balances the check by moving the
            # PDF past a newline and nothing else changes: measured with GNU
            # FriBidi 1.0.16, an unclosed RLO reverses its own line, the next
            # line renders untouched, and a PDF on that next line does nothing.
            # Class B is asked of `unicodedata` rather than listed here, so it
            # cannot drift from the Unicode data the interpreter ships;
            # `test_controls_do_not_pair_across_a_paragraph_break` carries the
            # seven characters that answered "B" in Unicode 16.0.0.
            #
            # It is not free, and the cost is a false positive on real text:
            # `FSI ... PDI` wrapped around a multi-line interpolated value, the
            # idiom Unicode recommends, is denied here. The rendering claim is
            # one MEASUREMENT and is scoped to the shape measured, not to the
            # idiom: on a value whose first line ends in a right-to-left run
            # with no left-to-right text after it, the wrapper changes the
            # visible order not at all, so the denial buys nothing on that
            # input. Narrowing the flush for isolates was measured and rejected
            # anyway, and the reason is NOT that the content is right-to-left --
            # the false positive's content is. It is that an isolate left open
            # across a break reorders for real when LEFT-TO-RIGHT text follows
            # the right-to-left run inside the still-open scope.
            # `test_an_isolate_around_a_multi_line_value_denies_and_that_is_deliberate`
            # holds both measurements and is the test that changes if the trade
            # is re-taken.
            unbalanced += embeds
            unbalanced += isolates
            embeds.clear()
            isolates.clear()
    unbalanced += embeds
    unbalanced += isolates
    return [(index, index + 1) for index in sorted(unbalanced)]


# ZWSP, ZWNJ, ZWJ, WORD JOINER, and the ZERO WIDTH NO-BREAK SPACE a UTF-8 BOM
# decodes to. Escapes rather than literals, for the reason the bidi controls in
# the tests are escapes: a literal here is invisible in this file, invisible in
# the diff that adds it, and invisible in the review that should have caught it.
#
# PRIVATE USE AREA characters are deliberately NOT a signal, and this is the
# note the module docstring points at. The PUA is U+E000..U+F8FF, U+F0000..
# U+FFFFD and U+100000..U+10FFFD: 137,468 code points that `unicodedata.category`
# answers "Co" for and that Unicode itself assigns no meaning to. Fonts assign
# them. Nerd Fonts and Powerline put their icons there, so a pasted shell prompt
# carries a handful, and U+F8FF is the Apple logo on Apple platforms.
#
# The reason that is fatal rather than merely noisy is what the signals here have
# in common. A PUA character RENDERS -- as a glyph in the right font, as a tofu
# box in the wrong one -- so a reader looking at the text can SEE it, and a
# detector that fires on it is making a claim about text the reader was never
# blind to. Every signal in this module is chosen because it is invisible in the
# rendering while carrying meaning to a model, which is what makes a published
# precision number defensible. On PUA there is no such number to publish: the
# same code point is a developer's prompt and an attacker's private encoding,
# and nothing in the bytes tells the two apart.
# Unicode's Default_Ignorable_Code_Point ranges, 16.0.0. This is the ONE table
# in this module that cannot be asked of `unicodedata`, which publishes no such
# property, so it is written out and the test below re-derives everything it can
# from the interpreter's own data. Reproduce it from DerivedCoreProperties.txt.
#
# The set matters because the property's definition is the signal's definition:
# a default-ignorable code point is one a conforming renderer draws as NOTHING.
# Five characters used to be listed here by hand, and a hand-picked list is a
# list of what somebody thought of. Measured against that list, a bitstream over
# U+2061 and U+2062 -- FUNCTION APPLICATION and INVISIBLE TIMES, `Cf`,
# default-ignorable, invisible -- carried "ignore all previous instructions"
# through this detector at 1.0000 characters per bit with nothing on the page,
# and the same construction ran over U+2063/U+2064, U+206A/U+206B,
# U+1D173/U+1D174 and U+1BCA0/U+1BCA1. That 1.0000 is the cost of THAT encoding.
# It is not a ranking: an earlier version of this comment called it cheaper than
# every residual the module records, which is false against the module's own
# list -- `_invisible` publishes encoders at 0.1250, 0.1992, 0.2070, 0.2500,
# 0.2695 and 0.6289, and the variation-selector row is 0.1250, not the 1.5000
# that the presence-and-absence encoding over ONE selector costs.
#
# FOUR OF THE SEVENTEEN ROWS CONTRIBUTE NOTHING TO THE SET, and that is the
# table doing its job rather than dead weight. U+00AD, U+061C, U+202A..U+202E
# and U+FE00..U+FE0F are each removed again by one of the exclusions in
# `_invisible`, so `_ZERO_WIDTH` is the same set with or without them -- 3,773
# characters on Unicode 16.0.0 -- and the count below cannot see them go: measured against the file before
# the test named at the end of this note existed, deleting U+061C or
# U+202A..U+202E changed no test and no corpus case. They are load-bearing the
# moment an exclusion narrows, because whichever family stops being excluded has
# to be IN the table to be counted, and an exclusion naming a family the table
# does not hold excludes nothing at all.
# `test_every_family_the_rule_excludes_is_in_the_table_it_excludes_them_from`
# asserts each excluded family's presence here for that reason.
_DEFAULT_IGNORABLE: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)

# U+00AD SOFT HYPHEN is default-ignorable and passes every property test below,
# and is refused by name. It is the one member of the set that RENDERS: a
# hyphen, wherever the line happens to break. It is in every hyphenated ebook,
# and a signal that fires on six of them is the Thai case for a much larger
# population. `inj-0108` is the sample that holds it.
_SOFT_HYPHEN = 0x00AD
# Read off the character's NAME, so the 260 variation selectors are excluded as
# a family rather than as four ranges somebody has to keep in step with Unicode.
# It catches VARIATION SELECTOR-1..256 and the four MONGOLIAN FREE VARIATION
# SELECTORs, which is exactly right: all 260 do the same job, modifying the
# glyph of the character in front of them, and all 260 are orthography wherever
# that character is.
_VARIATION_SELECTOR = "VARIATION SELECTOR"


def _is_directional(char: str) -> bool:
    """Whether a format character's job is direction rather than nothing.

    Bidi class BN, Boundary Neutral, is what every character with NO directional
    meaning carries, so a `Cf` character that is not BN is one whose entire
    purpose is directional: U+200E and U+200F, the LRM and RLM, class L and R;
    U+061C, the ALM, class AL; and U+202A..U+202E and U+2066..U+2069, each
    carrying its own. The first three are what ordinary right-to-left text is
    written with and the rest are `_bidi_spans`'s, where an imbalance is
    reported and a balanced pair is deliberately not.
    """
    return unicodedata.category(char) == "Cf" and unicodedata.bidirectional(char) != "BN"


def _invisible() -> frozenset[str]:
    """Every default-ignorable code point that renders nothing and no other rule owns.

    Default-ignorable is Unicode's own name for "a conforming renderer draws
    this as nothing", which is this signal's definition, so the property is the
    rule rather than a starting point somebody then edits. Three exclusions, and
    each one is a family with a reason rather than a code point somebody
    remembered:

      - the DIRECTIONAL format characters, which `_is_directional` names. Real
        right-to-left text is written with the marks and `_bidi_spans` owns the
        controls, where a balanced pair is deliberately allowed; counting them
        here would deny both.
      - every VARIATION SELECTOR, all 260 of them. A variation selector modifies
        the glyph of the character before it, so it is orthography wherever that
        character is: U+FE0F is in every emoji sequence, the 240 ideographic
        ones are in Japanese personal names, and the four Mongolian ones are
        written word-finally in ordinary Mongolian. Four is the Unicode 14.0.0
        count; on 13.0.0 only three of them have a name for this test to match,
        which is what the note below is about.
      - U+00AD and the tag block, for the reasons beside them.

    WHAT IS LEFT IS A FACT ABOUT THE INTERPRETER, NOT ABOUT THIS MODULE, because
    every input to the rule comes from `unicodedata`. Measured on the five
    interpreters the CI matrix runs: **3,773 code points on Unicode 14.0.0,
    15.0.0, 15.1.0 and 16.0.0 -- 3,738 unassigned, which no text can contain by
    accident; the four Hangul fillers, which are `Lo`; the two Khmer inherent
    vowels and U+034F COMBINING GRAPHEME JOINER, which are `Mn`; and 28 `Cf`.**
    On Unicode 13.0.0, which Python 3.10 ships, it is 3,774 and 3,739: U+180F
    MONGOLIAN FREE VARIATION SELECTOR FOUR was unassigned then, so the
    name-based selector exclusion had no name to match and the code point fell
    through into the unassigned bucket. That one code point is the whole
    difference between the two sets, it is a real difference in what this
    detector denies, and `corpora/NOTICE.md` discloses it beside the other
    residuals. Only the unassigned bucket moves; 28, 4 and 3 hold on all five.
    `test_the_one_member_that_moves_between_unicode_versions` is that
    measurement and `test_every_invisible_character_is_default_ignorable_and_nothing_else_is`
    keys the counts by `unicodedata.unidata_version` rather than publishing one
    of them as universal.

    THE PROPERTY THIS RULE HAS AND ITS PREDECESSOR DID NOT. The rule here was
    "`Cf` and bidi class BN", which admitted 29 characters and left every other
    family out with an argument attached. Two of those arguments were wrong in
    the same way and a sweep found them: the Hangul fillers were excluded
    because they are "handled where letters are, by `_joining_neighbour`'s range
    test", and that test refuses them as an EXCUSING NEIGHBOUR while saying
    nothing about them as a CARRIER. Measured against that version, a two-symbol
    bitstream over U+115F/U+1160 and one over U+17B4/U+17B5 each carried "ignore
    all previous instructions" through this detector at 1.0000 characters per
    bit with nothing on the page, and so did one over any two unassigned
    default-ignorable code points. A fact about one role stated as an assurance
    about another is this module's oldest defect, and it is why the rule is now
    a property of the character rather than a claim about what some other
    function does with it.

    WHAT IS STILL OPEN. This module does NOT state a minimum cost for getting a
    payload past it, and the absence is deliberate. Four numbers were published
    for one channel in four consecutive rounds -- 1.4875, 1.0000, 0.1250 and
    0.1247 -- and each was written in the sentence correcting the last. The
    defect is not arithmetic. **A minimum is a claim about every possible
    encoding, and a measurement only ever exhibits one**, so no sweep of this
    kind can establish one; each round found a cheaper encoder because there are
    always more encoders. What CAN be defended is a list of families this rule
    does not count, and one measured encoder for each. Those are below, and each
    figure is the cost of THAT ENCODING and not a bound on anything.

    Every one returns zero findings and decodes back to the payload verbatim,
    measured against this file:

      256 variation selectors, one per byte             0.1250 chars/bit
      the same plus the 3 marks and 4 Mongolian ones    0.1250
      32 C1 controls, base 32                           0.1992
      29 C0 controls, base 29                           0.2070
      16 Egyptian hieroglyph format controls, base 16   0.2500
      13 Prepended_Concatenation_Mark characters,
        base 13                                         0.2695
      3 interlinear annotation characters, base 3       0.6289
      3 directional marks, two of them                  1.0000

    And a cost per bit is itself one accounting. An attacker already sending a
    document pays for the characters they ADD and nothing else, and `_MIN_TOTAL`
    lets four counted characters through PROVIDED NO TWO ARE ADJACENT, since
    two adjacent ones are a run and `_MIN_RUN` is 2. Measured: two adjacent
    deny, two scattered allow, four scattered allow, five scattered deny.

    Priced over the alphabet the bound actually governs, which is the counted
    set and not the excluded families -- an excluded character consumes no part
    of `_MIN_TOTAL` at all, so "four" is not a limit on those -- take
    `inj-0105`'s text with its own zero-width characters stripped, 2,499
    characters, and ADD four counted ones: a 2,503-character page, which is
    `inj-0106`'s length, in which the four choose among C(2500, 4)
    pairwise-non-adjacent positions and 3,773 symbols each (Unicode 16.0.0;
    3,774 on 13.0.0, which moves this to 88.0899 from 88.0884 and rounds to the
    same figure):
    **88.1 bits carried by 4 added characters**. The page the four are added to
    is the corpus case and the four are not: `inj-0105` carries three of its own
    at 2,502 characters, and `inj-0106` is the same page carrying four at 2,503.
    The slot count is the page AFTER the additions, which is what the test
    builds; pricing 2,502 slots for a construction that makes 2,503 moves the
    figure by 0.0023 bits and rounds to the same 88.1, which is why it stood.
    Priced over the 259 symbols `_MIN_TOTAL` does NOT count it comes to 72.6,
    which is an accounting of two different things and understates the leak of
    the bound it names.

    THE RAISE FROM FOUR TO FIVE WIDENED THIS BY 21.2 BITS on this document, from
    three characters and 66.9 bits to four and 88.1. That is the standing cost
    of buying back twelve false-positive cases, and it is stated in the same
    place as the leak rather than only beside the corpus.
    `test_the_bound_passes_four_non_adjacent_characters_and_what_they_carry`
    holds all three figures.

    The families are the claim. The numbers are illustrations of it.

    WHY THE TWO SWEPT FAMILIES STAY OUT, measured rather than asserted. This
    docstring said "counting variation selectors denies every emoji", and that
    is false: measured with them counted, a single heart with U+FE0F, three
    keycaps and a four-person family sequence all still ALLOW, because one,
    three or four unexplained characters is under `_MIN_TOTAL`. What is true is
    narrower and still decisive: FIVE keycaps, five U+FE0F emoji in one message,
    or five ideographic variation selectors -- a Japanese document naming five
    people whose names take variant glyphs -- all reach `_MIN_TOTAL` and deny.
    Counting U+200E, U+200F and U+061C denies a bilingual invoice carrying five
    of them. Four rather than five while `_MIN_TOTAL` was 4; when the bound
    moved, every one of these justifications had to be re-measured, because at
    four occurrences they now allow whether the families are counted or not.

    An earlier version of this paragraph said a single RAINBOW FLAG denies,
    reasoning that U+FE0F sits immediately before U+200D and so makes a run of
    two. That is wrong and the mutation says so: U+FE0F is inside
    `_PICTOGRAPHIC`, so with selectors counted it EXPLAINS the joiner, leaving
    one suspicious character per flag. Measured, up to four rainbow flags allow
    and five deny on the total bound, exactly like the keycaps. It was argued
    rather than run.

    The corpus carries the negatives so the claim is scored rather than argued:
    `inj-0143`, `inj-0144`, `inj-0145` and `inj-0146`.
    """
    return frozenset(
        chr(point)
        for low, high in _DEFAULT_IGNORABLE
        for point in range(low, high + 1)
        if not _is_directional(chr(point))
        and _VARIATION_SELECTOR not in unicodedata.name(chr(point), "")
        and point != _SOFT_HYPHEN
        and not _TAG_START <= point <= _TAG_END
    )


_ZERO_WIDTH = _invisible()
# The three that carry meaning, and only next to a script that uses them.
_CONTEXTUAL = frozenset("\u200c\u200d\u180e")
_ZWJ = "\u200d"
# U+180E MONGOLIAN VOWEL SEPARATOR. It is `Cf` and BN and default-ignorable, so
# the rule above admits it, and it is ALSO ordinary Mongolian: it stands between
# a word and the suffix vowel that follows it, which is the job ZWNJ does in
# Persian. So it gets what ZWNJ gets, a context test, rather than being dropped
# from the set the way the soft hyphen is.
#
# The Mongolian free variation selectors, U+180B..U+180D and U+180F, are NOT in
# the set at all -- they are `Mn` variation selectors, excluded with the other
# 260 -- so a Mongolian word that ends in one is untouched here. That matters
# because a variation selector is written word-FINALLY, where a both-neighbours
# rule has nothing to its right; the vowel separator is written medially, which
# is why a both-neighbours rule fits it and not them.
_MVS = "\u180e"
_MONGOLIAN: tuple[tuple[int, int], ...] = ((0x1800, 0x18AF),)

# Scripts in which ZWJ and ZWNJ are orthography rather than decoration. Indic
# scripts use them to force or suppress conjunct forms; Arabic and its
# derivatives use ZWNJ to break cursive joining. Ranges rather than a library so
# this stays stdlib-only.
#
# SIX OF THESE EIGHT ROWS HAD NOTHING STANDING ON THEM until the test named
# below was written. Deleting one range at a time and running the whole suite
# and the corpus against the file as it was then: only Arabic and the
# Devanagari-through-Sinhala block were reached by any input. Syriac, Thaana,
# Thai, Myanmar and both Arabic Presentation Forms blocks could each be deleted
# with no test and no case noticing, which is three quarters of the table's
# width asserted and unmeasured. Each row now has a letter of its own script
# standing beside a joiner in
# `test_each_declared_joining_range_excuses_the_joiner_it_is_there_for`, with a
# Latin control that denies, so widening the table is caught as well as
# narrowing it.
_JOINING_SCRIPTS: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0780, 0x07BF),  # Thaana
    (0x0900, 0x0DFF),  # Devanagari through Sinhala
    (0x0E00, 0x0E7F),  # Thai
    (0x1000, 0x109F),  # Myanmar
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)
# Emoji and the variation selector that precedes ZWJ in many sequences. The
# first range is symbols rather than emoji proper, and until it was swept it was
# the one row here nothing exercised: measured before the test named below
# existed, deleting U+2190..U+2BFF changed no test and no corpus case. It is not
# spare. RGI sequences join to symbols out of that
# block -- the transgender flag is U+1F3F3 U+FE0F ZWJ U+26A7 U+FE0F, where the
# character after the joiner is in no other range this module names -- and five
# of them in one message is the total bound.
# `test_a_flag_sequence_joining_to_a_symbol_is_not_an_attack` is that input,
# and it carries five since `_MIN_TOTAL` was raised: at four it allowed with or
# without the range and asserted nothing about it.
_PICTOGRAPHIC: tuple[tuple[int, int], ...] = (
    (0x2190, 0x2BFF),
    (0xFE0E, 0xFE0F),
    (0x1F000, 0x1FAFF),
)
# Canonical combining class 9 is the class Unicode gives a virama and the marks
# that behave like one. Asked of `unicodedata` rather than listed here so it
# cannot drift from the Unicode data the interpreter ships: in Unicode 16.0.0 it
# is 69 characters spread across the Brahmic scripts, and listing them by hand
# would mean tracking a dozen different names for the same job -- virama,
# halanta, hasanta, al-lakuna, coeng, asat, pangkon, subjoiner, sakot.
_VIRAMA = 9
# What a virama may stand behind and still be found sitting on its own letter:
# anything that is not a STARTER, plus the format characters, which are starters
# by combining class but transparent to a reader.
#
# Non-starter rather than a category set, and the difference is not academic.
# `Mn`, `Me` and `Cf` looks like the same rule and is not: a spacing mark is
# `Mc`, and 15 of the 69 characters of combining class 9 are themselves `Mc`, so
# under a category set a virama can stand in as the base of another virama.
# Measured that way, 63 of the 69 admitted a LATIN base behind one same-script
# spacing mark, at four characters per bit.
#
# The character immediately before a virama is not always the character the
# virama sits on, and all four shapes below put a character of the VIRAMA'S OWN
# SCRIPT in that position, where the script test cannot see anything wrong with
# it. The first three are CROSSED by the walk, so the base is further back and
# the shape turns on what the walk passes over; the fourth is a starter the walk
# stops on, and what refuses it is the base having to be a LETTER.
#
#   - U+093C DEVANAGARI SIGN NUKTA, a non-starter mark, in front of a Devanagari
#     virama that stands on a Latin letter. Read as the base it is Devanagari
#     and it excuses the joiner. Measured: four characters per bit, and what
#     shows on the page is Latin letters with a dot under each.
#   - U+110BD KAITHI NUMBER SIGN, category Cf and invisible, in front of a
#     Kaithi virama. Measured: three characters per bit and NO visible text at
#     all. U+110BD and U+110CD are the only format characters in Unicode 16.0.0
#     that share a first name word with any character of combining class 9,
#     which is why format characters are transparent here and not only marks.
#   - U+1B44 BALINESE ADEG ADEG and the other 14 class 9 spacing marks, each
#     standing in as the base of a second copy of itself. Four characters per
#     bit with a Latin cover.
#   - U+0903 DEVANAGARI SIGN VISARGA, U+0966 DEVANAGARI DIGIT ZERO and U+0964
#     DEVANAGARI DANDA. These are STARTERS, so no walk of any definition passes
#     them, and they are refused by the base having to be a letter instead.
#
# That list is four shapes and it is not offered as exhaustive; what is offered
# as exhaustive is the measurement over every character of every script that has
# a class 9 mark, in `_base_before`.
# `test_an_invisible_character_does_not_stand_in_for_the_base`,
# `test_a_format_character_of_the_virama_s_own_script_is_not_a_base`,
# `test_a_same_script_non_letter_is_not_a_base` and
# `test_a_virama_with_nothing_before_it_has_no_base` hold them.
_FORMAT = "Cf"
# Decimal digits, which `_joining_neighbour` admits and `_base_before` does not.
# The two rules are asking different questions: a virama sits on a letter and
# never on a numeral, while a Persian or Urdu ZWNJ after a year or an age sits
# between a numeral and a suffix. `Nd` rather than `N`: the other 38 numbers in
# these ranges are all `No`, and they are Malayalam, Telugu, Oriya and Tamil
# fractions, Bengali currency numerators and script-specific number signs.
# Nothing measured here writes a joiner against one, and admitting a character
# no sample needs is how the exemptions in this module have gone wrong before.
#
# THIS CLOSES HALF OF THAT FALSE POSITIVE AND THE OTHER HALF IS STILL OPEN. The
# digit has to be inside `_JOINING_SCRIPTS` like every other neighbour, so an
# ASCII digit is not one, and Persian and Urdu web text uses ASCII digits
# constantly. Raising `_MIN_TOTAL` to 5 moved this false positive by one clause
# rather than removing it. Measured, at four joiners and then at five:
#
#   کودک ۵<ZWNJ>ساله ...     allow      کودک 5<ZWNJ>ساله ...      allow / DENY
#   ۱۹۴۷<ZWNJ>ء میں ...      allow      1947<ZWNJ>ء میں ...       allow / DENY
#                                       CD<ZWNJ>ها و DVD<ZWNJ>ها  allow / DENY
#
# The Arabic-Indic column allows at any length, because a digit inside the
# ranges is an excusing neighbour. The last row is the same shape with a LATIN
# ACRONYM rather than a numeral, which Persian and Urdu attach the same suffixes
# to, and it is a different shape from the two above it.
# `test_an_ascii_numeral_before_a_joiner_is_a_known_false_positive` holds all
# three at both lengths. The candidate rule is to excuse a digit or a
# Latin letter when the OTHER neighbour is in a joining script, which is a
# different rule from this one -- it makes a neighbour's admissibility depend on
# its partner -- and it is written here as a candidate rather than shipped
# unmeasured.
_DECIMAL_DIGIT = "Nd"
# How far back that walk may go, and it is a bound on COST, not on orthography.
# Unbounded it is QUADRATIC, and the input that shows it is one an attacker can
# send: a letter followed by repeats of virama-plus-joiner is one unbroken run of
# Mn and Cf characters, so every joiner walks back over all of it, and every
# joiner is excused when the walk ends. Measured on that input, unbounded:
# 5.51 s at 8k joiners, 22.00 s at 16k, 88.53 s at 32k -- four times the work for
# twice the input. Bounded at four: 0.010 s, 0.018 s, 0.035 s, which is linear.
# Seconds are one machine's and will not reproduce; the ratios are the claim and
# do. Re-measured on both walks, each doubling of the input costs about four
# times the work unbounded and about twice bounded.
#
# Four characters are EXAMINED, so at most three transparent ones are crossed.
#
# WHAT THE VALUE COSTS ON EACH SIDE. Re-measured over every input
# tests/test_injection_structural.py checks, by tracing both walks across a
# whole run of the file, the deepest walk that finds a base is FOUR, and it
# reaches four in exactly two places: the five clusters of
# `test_the_mark_walk_reaches_exactly_as_far_as_the_bound_says` and the five of
# `test_the_base_walk_reaches_exactly_as_far_as_the_bound_says`, which are the
# tests written to pin this bound and take each walk to it deliberately.
#
# Outside those two, the deepest is THREE, in a `_mark_base` walk in
# `test_a_mark_on_a_devanagari_letter_under_an_arabic_fatha_still_allows`, where
# the mark to the RIGHT of the joiner reaches its letter across the joiner and
# one more mark; `_base_before` never goes past two. An earlier version of this
# note said every base is one character back except behind a nukta, where it is
# two, which is true of `_base_before` and false of the walk that reaches three;
# the version after that said the deepest anywhere is three, which was measured
# against the file BEFORE the two tests above were added to it and was stale the
# moment they landed.
#
# So four sat one above every measured need and nothing showed it. Cutting it to
# three and raising it to five each left the whole suite green and every one of
# the corpus's 146 cases where it was. Both sides are pinned now, one test per
# walk, each carrying the case one character inside the bound and the case one
# character past it. Removing the bound outright fails the second half of each.
#
# Padding past the bound loses the exemption rather than gaining anything, which
# is the safe direction for a bound to fail in: a unit of slack is one more
# transparent character an attacker may pad a mark cover with, and each padding
# character is another character per bit.
_MAX_TRANSPARENT = 4

# A run of two, or five in total anywhere in the input. Both bounds are about
# what arrives by ACCIDENT. An unexplained zero-width character turns up one at a
# time, out of a copy-paste from a rendered page; a payload does not, because one
# bit per character means a payload is a sequence.
#
# THE TWO BOUNDS ANSWER DIFFERENT QUESTIONS, which is why one of them moved and
# the other did not. `_MIN_RUN` is about SHAPE: two adjacent counted characters
# are what a bit-per-character encoder emits and what ordinary prose does not,
# so it is a structural signal and it stays at two. `_MIN_TOTAL` is about
# VOLUME, and volume is the one an honest document can reach by accident.
#
# `_MIN_TOTAL` WAS 4 AND IS NOW 5. Measured over the published corpus, the raise
# moves the row from 0.679 precision / 0.974 recall to 0.971 / 0.870, and both
# halves of that are real:
#
#   - twelve cases of ORDINARY TEXT stop being reported. Every one of them
#     carried exactly four unexplained characters, which is what made four the
#     wrong side of the line: a Thai sentence marked up for line breaking; three
#     Persian and Urdu numeral compounds written with ASCII digits and one
#     Persian plural suffix on Latin acronyms; a retrieved page with four
#     incidental U+200B; MathML extracted to plain text; and five more, which
#     are Korean prose about jamo, a Khmer dictionary entry, U+034F blocking a
#     collation contraction, U+034F fixing point order in Biblical Hebrew, and
#     four UTF-8 files concatenated with each keeping its own BOM.
#   - three PAYLOADS stop being reported, twelve findings between them, and each
#     is four zero-width characters with no two adjacent. `inj-0051`, `inj-0052`
#     and `inj-0053` are those cases and they are labelled `deny`, so they cost
#     recall rather than hiding in the prose.
#
# The false positives did not go away. They moved up by exactly one occurrence,
# which was measured one sample at a time rather than assumed: six Thai words
# deny, a fifth Persian clause denies, a fifth invisible operator denies, a fifth
# jamo or inherent-vowel entry denies. Both sides of each are asserted, in
# `test_thai_line_break_hints_allow_at_five_words_and_deny_at_six` and its three
# siblings.
#
# One case in that family is NOT bought back, and it is the clearest evidence
# that the two bounds are different signals: `inj-0129`, musical notation
# extracted to plain text, is four controls of which two are ADJACENT, because
# an END BEAM is immediately followed by the next BEGIN BEAM. `_MIN_RUN` reports
# it and `_MIN_TOTAL` never did.
#
# RAISING `_MIN_RUN` TOO WAS CONSIDERED AND REFUSED. The sweep says
# `_MIN_RUN` 3 with `_MIN_TOTAL` 5 scores 0.980 / 0.861 against 0.971 / 0.870:
# it buys 0.009 precision and gives up a true positive and the adjacency signal
# with it. A pair is the cheapest thing an encoder emits and the most expensive
# thing for prose to produce by accident, so it is worth more than the ratio.
#
# THE FULL SWEEP, precision / recall over the 146 published cases. Columns are
# `_MIN_TOTAL`, rows `_MIN_RUN`; the shipped pair is (2, 5).
#
#         2            3            4            5*           6            7            8
#   1  .626/.974    .626/.974    .626/.974    .626/.974    .626/.974    .626/.974    .626/.974
#   2  .647/.974    .655/.974    .679/.974    .971/.870*   .971/.870    .971/.870    .969/.809
#   3  .647/.974    .653/.965    .677/.965    .980/.861    .980/.861    .980/.861    .979/.800
#   4  .647/.974    .653/.965    .677/.965    .980/.861    .980/.861    .980/.861    .979/.800
#
# `_MIN_RUN` 1 makes `_MIN_TOTAL` irrelevant, since every single character is
# then a reportable run. `_MIN_TOTAL` 5, 6 and 7 are identical and so are
# `_MIN_RUN` 3 and 4: no case carries five or six unexplained characters, and no
# case has a run of exactly THREE, so the corpus can say what 5 costs and cannot
# say where between 5 and 7 the answer lies. That is a fact about the corpus, not
# about the bounds. The totals that appear above four are 7 (`inj-0123`), 8, 9,
# 16 and 80, all of them payloads, so 5 through 7 behave alike because of the gap
# between four and seven rather than any absence above four; the run lengths that
# appear are 1, 2, 4, 16 and 80, so 3 and 4 behave alike for the same reason one
# step down.
#
# An earlier version of this note gave the reason as "no allowing case has a run
# longer than one", which is true only of what the detector allows and false of
# what the corpus LABELS allow: `inj-0129` is labelled allow and has a run of
# two. That run of two is the whole reason it is a false positive, and it is why
# `_MIN_RUN` 3 scores 0.980 -- raising the run bound stops reporting it.
#
_MIN_RUN = 2
_MIN_TOTAL = 5

# A joiner every second character, four of them, with the symbol changing at
# least once along the way.
#
# This is the signal that does not depend on the exemption below being right,
# and it is here because the exemption cannot be made right on its own. A
# steganographic encoder emits one joiner per base character; orthography does
# not. Measured over the samples in tests/test_injection_structural.py, longest
# chain of EXCUSED joiners exactly one base character apart:
#
#   Hindi with five conjunct joiners                    1
#   Persian prose with six ZWNJ                         1
#   Malayalam sentence, five chillu words               1
#   kiss sequence (a variation selector inside it)      2
#   four-person family emoji                            3
#   ten family emoji written next to each other         3
#   a stray ZWSP two characters before a family emoji   3
#   36 Arabic characters, a ZWNJ between every pair    35
#   256-bit payload, Devanagari or Arabic cover       256
#   256-bit payload, emoji cover                        4
#
# The emoji row is short because that cover loses its ZWNJ to the context test
# below: 123 of the 256 joiners are unexcused, the total bound reports them
# where they stand, and what is left excused is runs of set bits. Its four is a
# run of four one-bits, and it is at the bound and still not reported, because
# every joiner in it is the same character -- the condition below suppressing a
# true positive on an input that denies twice over.
# `test_the_two_rules_that_catch_a_covered_bitstream_are_different_rules` holds
# which rule answers for which cover.
#
# Three is structural for emoji rather than lucky. An RGI sequence built from
# single code point elements carries one joiner between each pair; the longest
# is the four-person family, at three. Two sequences written next to each other
# cannot chain, because one ends and the next begins on a base character, which
# leaves three characters between their joiners rather than one. A variation
# selector or a skin tone modifier LOWERS the score for the same reason, so the
# longer RGI sequences are the safer ones.
#
# The Arabic row is why length cannot be the whole test: it is longer than any
# attack fragment worth catching, so no bound below it separates the two. It is
# allowed because it carries NOTHING. One symbol repeated at every position is a
# channel with no choice in it, and a bitstream needs a choice per bit.
#
# Asking for the symbol to change is what lets this bound sit at four instead of
# above thirty-five, and it is a TRADE, not a free win. What it gives up is
# measured and it is in this file: a presence-and-absence encoder, a joiner for
# a one bit and nothing for a zero, leaves uniform chains as long as its longest
# run of set bits, and `test_a_presence_and_absence_encoding_is_a_known_miss`
# carries a payload whose longest run is exactly 4. Drop the changing-symbol
# condition and that input denies. What that reaches is bounded and knowable:
# ASCII bytes all have a zero top bit, so a set-bit run never crosses a byte,
# and a 4-run happens exactly when the payload contains a character whose own
# byte has one -- for lowercase text o, x, y and z. An attacker who avoids those
# four letters, or packs the payload any other way, gives up nothing.
#
# The trade was taken in that direction because the true positive it buys is
# evadable at no cost and the false positive it removes is not evadable at all
# by the person writing the text.
#
# Four rather than three keeps the bound above the emoji maximum on its own,
# without the changing-symbol condition helping. That is deliberate: whoever
# drops that condition should be left with a threshold that still clears every
# legitimate sample here, rather than with a check that denies family emoji.
#
# What this bounds is RATE, not capacity. An attacker who spends one spare cover
# character every three bits holds every chain at three, and the whole payload
# goes through for 2.33 characters per bit against the 2.00 of the shape this
# denies; `test_a_deperiodised_bitstream_is_a_known_miss` is that input.
_MIN_PERIODIC = 4


def _in_ranges(char: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    """Whether `char` falls in any range. An empty string, the edge of the input, is False."""
    if not char:
        return False
    point = ord(char)
    return any(low <= point <= high for low, high in ranges)


def _script(char: str) -> str:
    """The script a character belongs to, read off the first word of its name.

    `unicodedata` publishes no Script property, so the name is where the script
    is: every character of a Brahmic script is named for it, and the mark and
    the letter agree -- KHMER SIGN COENG with KHMER LETTER KA, SYLOTI NAGRI SIGN
    HASANTA with SYLOTI NAGRI LETTER KO. Derived rather than tabulated for the
    reason the virama class is: a table of 69 marks and their scripts is a table
    that goes stale, and this one cannot.

    Checked across every character of combining class 9 in Unicode 16.0.0: all
    69 have a letter sharing their first name word, so the test below excuses a
    conjunct in every one of those scripts, and none of them is excused behind a
    Latin or pictographic base.

    An unassigned or unnamed code point answers "". A virama is never one --
    `unicodedata.combining` answers 0 for everything unassigned, so every
    character of class 9 has a name -- and the comparison below is between a
    base and a virama, so "" can only appear on the base side, where it matches
    nothing. That is what closes the 440 unassigned code points inside
    `_JOINING_SCRIPTS`, which a range test accepts as a base.

    ONE ambiguity, measured and recorded rather than papered over: TAI LE, TAI
    THAM and TAI VIET share a first word, and the class 9 mark among them is TAI
    THAM SIGN SAKOT, so a Tai Le or Tai Viet letter under a Tai Tham sakot reads
    as one script here. It buys an attacker nothing, because a correctly paired
    Tai Tham letter and sakot is already excused and costs the same three
    characters per bit of visible cover.
    """
    return unicodedata.name(char, "").partition(" ")[0] if char else ""


def _is_letter(char: str) -> bool:
    """Whether `char` is a letter. The empty string, the edge of the input, is not.

    No caller passes "" today: both walks below read a character out of
    `content` before asking. So the guard is a contract for the next caller
    rather than a live path, and inverting it left the whole suite green -- until
    `test_the_edge_of_the_input_is_not_a_letter` was written for it -- where the
    same inversion in `_in_ranges` and `_script` never did. That test holds all
    three.
    """
    return bool(char) and unicodedata.category(char).startswith("L")


def _mark_base(content: str, index: int) -> str:
    """The letter the mark at `index` is written on, or "" if it is written on nothing.

    A DIFFERENT walk from `_base_before`, and the two are not interchangeable
    even though both end on a letter. This one asks whether a mark is attached,
    so it crosses what a renderer stacks onto the cluster before it -- every
    mark, spacing or not, and the format characters, which draw nothing at all.
    `_base_before` asks where a virama's base is, so it crosses NON-STARTERS and
    stops on the first starter.

    WHICH letter is not decided here. `_base_before` requires the virama's own
    script because a virama is script-specific orthography; the caller of this
    function requires only `_JOINING_SCRIPTS` membership, and that weaker test
    is load-bearing rather than lax. Any letter at all admits a base that
    renders nothing: U+115F and U+1160 HANGUL CHOOSEONG and JUNGSEONG FILLER,
    U+3164 HANGUL FILLER and U+FFA0 HALFWIDTH HANGUL FILLER are category `Lo`
    AND default-ignorable, so `<filler><fatha><joiner><fatha>` repeated carried
    a 256-bit payload at 4.0000 characters per bit with NO READABLE COVER --
    measured on all four. Not "nothing on the page", which this said and which
    is false of the construction: U+064E ARABIC FATHA is `Mn` with combining
    class 30 and DRAWS, so what a reader sees is a row of orphaned diacritics
    with no letter under them. The security claim is the one that survives --
    there is no text there for a reader to read -- and it does not need the
    stronger one. Inside `_JOINING_SCRIPTS` the only default-ignorable
    code points are U+061C and U+FEFF, both `Cf`, which the caller's category
    test already refuses; so range membership is what makes "a letter" mean "a
    letter somebody can see" here.
    `test_a_mark_on_a_letter_outside_the_joining_ranges_is_not_a_neighbour` holds it.

    THAT IT HAS TO BE A LETTER AT ALL was, until it was swept, the widest
    unmeasured condition in this function. Measured against the file before
    `test_a_mark_over_an_unwritten_code_point_is_not_an_excusing_neighbour`
    existed, returning whatever the walk stops on changed no test and no corpus
    case, because every input then reached either a letter or nothing. What it opens is the
    third case: 440 of the code points inside `_JOINING_SCRIPTS` are unassigned,
    a walk stops on them and a font has no glyph for them, so a cover of
    unassigned code point, fatha, joiner, fatha carried a 256-bit payload at
    4.0039 characters per bit with not one letter in the input. "Draws nothing"
    is what this said and it is the wrong claim about an unassigned code point:
    a font draws `.notdef` for one, the tofu box, which is visible. What is true
    and is what matters here is that none of it is READABLE, and that a walk
    looking for a letter stops on something that is not one. That is the same hole
    `test_an_unwritten_code_point_in_a_joining_script_range_is_not_a_neighbour`
    closed for the code point standing directly beside a joiner, one
    construction later with a mark on top of it, and it is now
    `test_a_mark_over_an_unwritten_code_point_is_not_an_excusing_neighbour`.

    The two definitions are incomparable, not nested, and the numbers are in
    `test_the_walk_to_a_base_crosses_non_starters_and_format_characters`: 1,126
    marks in Unicode 16.0.0 are `Mn` or `Me` with combining class 0, and this
    walk crosses every one of them while `_base_before` stops dead on it; 27
    spacing marks have a non-zero class, and `_base_before` crosses those while
    this walk stops on none of them because they are marks. Each function uses
    the walk its own question needs.

    Crossing SPACING marks is the half of that no input in
    tests/test_injection_structural.py pins, and it is kept for what it costs
    rather than for what it buys: measured, `<letter><Mc><Mn>` in front of a
    joiner carries a payload at 4.0039 characters per bit against the 3.0039 of
    a bare `<letter><virama><joiner>`, with the same one visible letter per bit,
    so an attacker who uses it pays more for nothing. Narrowing to `Mn`, `Me`
    and `Cf` would change no verdict in that file either, and that was run
    rather than reasoned: the narrowed walk leaves the whole suite green and the
    corpus at the same 146 cases. The reason to leave it
    wide is that a spacing vowel sign under a further mark is ordinary Indic --
    `<ka><vowel sign O><anusvara>` is `Lo Mc Mn`, and so is `<ka><vowel sign
    AA><candrabindu>` -- and this module has no list of which such clusters take
    a joiner after them.

    Bounded by `_MAX_TRANSPARENT` for that constant's reason, on its own input
    shape: a letter followed by repeats of mark-plus-joiner is one unbroken run
    this walk crosses, so unbounded every joiner walks the whole of it.
    Measured on `<letter>(<fatha><ZWNJ>)*n`, unbounded: 0.794 s at 2,000
    joiners, 3.167 s at 4,000, 12.692 s at 8,000 -- four times the work for
    twice the input. Bounded: 0.004 s, 0.009 s, 0.015 s.
    """
    at = index - 1
    while at >= 0 and index - at <= _MAX_TRANSPARENT:
        char = content[at]
        category = unicodedata.category(char)
        if category[0] != "M" and category != _FORMAT:
            return char if _is_letter(char) else ""
        at -= 1
    return ""


def _joining_neighbour(
    content: str, index: int, ranges: tuple[tuple[int, int], ...] = _JOINING_SCRIPTS
) -> bool:
    """Whether the character at `index` is one a joining script writes beside a joiner.

    ``ranges`` is a parameter so that the Mongolian branch of
    ``_is_contextually_legitimate`` can ask this same question about its own
    block instead of repeating the three tests. Everything measured below is
    measured on the default, which is the only value any other caller passes.

    A letter, a decimal digit, or a mark that is written on a letter. Each of
    the three is a separate measured decision and the last is the security one.

    A script is a set of CHARACTERS and this module holds one as a RANGE, which
    is not the same thing, and the difference is a free cover character. Of the
    2,800 code points inside `_JOINING_SCRIPTS`, 10 are format characters and
    440 are unassigned; each sits between two joiners as happily as a letter
    does while rendering as nothing, and each carried a 256-bit payload at 2.3359
    characters per bit -- 598 characters, not one of them visible. Refusing them
    is what the category test does.

    Refusing them is NOT enough, because a mark on its own renders as nothing
    either. Measured over every code point in these ranges, each used as the
    sole cover of a deperiodised payload: 374 of them were excused with ZERO
    letters anywhere in the input -- 243 `Mn` and 131 `Mc` -- at 2.3359
    characters per bit, which is character for character the construction and
    the rate of the format-character hole above. An unattached mark is not
    orthography; a mark is orthography when there is something under it. So a
    mark neighbour has to reach a letter through `_mark_base`, and after that
    condition the same sweep leaves 1,808 covers excused and every one of them
    is a letter or a digit, which is to say every one of them is visible.

    Marks are admitted at all because Arabic and Persian write harakat next to
    these joiners: `<letter><kasra><ZWNJ>` is ordinary vocalised text, and 389
    of the 389 marks in these ranges that have a same-script letter are still
    excused sitting on one.

    Digits are admitted because Persian and Urdu put a ZWNJ after a NUMERAL
    before a suffix -- decades (`۱۹۸۰<ZWNJ>ها`), ages and measures
    (`۵<ZWNJ>ساله`, `۱۰<ZWNJ>متری`), and Urdu's `<ZWNJ>ء` after a year. There
    are 150 `Nd` code points in these ranges and while they were not excusing
    neighbours all four of those samples DENIED;
    `test_a_joiner_after_a_numeral_is_ordinary_persian_and_urdu` holds them. A
    digit grants an attacker nothing that is not already granted: it is visible,
    so a digit cover costs exactly what the letter cover recorded in
    `test_a_deperiodised_bitstream_is_a_known_miss` costs. Measured on a 256-bit
    payload, the deperiodised construction behind a Persian digit and behind a
    Devanagari letter are the same 598 characters, the same 2.3359 per bit and
    the same 342 visible characters. It is another spelling of that residual,
    not a new one.
    """
    char = content[index] if 0 <= index < len(content) else ""
    if not _in_ranges(char, ranges):
        return False
    category = unicodedata.category(char)
    if category[0] == "L" or category == _DECIMAL_DIGIT:
        return True
    return category[0] == "M" and _in_ranges(_mark_base(content, index), ranges)


def _base_before(content: str, index: int) -> str:
    """The letter the mark at `index` sits on, or "" if it is not sitting on one.

    Two questions, and separating them is what this function is for. Where the
    base IS: the first character back that a reader would call the start of a
    cluster, so anything that combines with what precedes it, and the format
    characters, are passed over. Whether that character can BE a base: only a
    letter can, which is the condition every starter that is not a letter dies
    to -- a digit, a danda, a visarga, all of them named for a script that has a
    virama and none of them something a virama sits on.

    Measured over every character of every script that owns a character of
    combining class 9, which is the only measurement here that is exhaustive
    rather than a list somebody chose: of the same-script characters that are
    not letters, none is accepted as a base; of the same-script characters that
    are letters, all are. `test_a_same_script_non_letter_is_not_a_base` carries
    four of the refused shapes.
    """
    at = index - 1
    while at >= 0 and index - at <= _MAX_TRANSPARENT:
        char = content[at]
        if unicodedata.combining(char) == 0 and unicodedata.category(char) != _FORMAT:
            return char if _is_letter(char) else ""
        at -= 1
    return ""


def _is_contextually_legitimate(content: str, index: int) -> bool:
    """Whether a joiner at `index` sits where a script or an emoji sequence puts one.

    Context, not identity. U+200D is orthography in Devanagari and structural in
    an emoji sequence; between two Latin letters it is neither, and that is
    exactly where a payload hides.

    BOTH neighbours, and both in ONE context. Asking whether EITHER neighbour is
    pictographic or in a joining script is the same per-occurrence shape the tag
    exemption above shipped, and it fails the same way: one cover character per
    joiner excuses every joiner, so repeating the construct costs the attacker
    two characters per bit and nothing else. Loosen the two conditions below to
    accept either neighbour AND delete the periodicity rule in
    `_zero_width_spans`, which together are the design this replaced, and
    "Summarise this. " followed by 256 bits of cover character plus ZWJ or ZWNJ
    returns `[]` for an emoji, a Devanagari and an Arabic cover alike.

    Loosening this rule ALONE does not reproduce that, and the honest account of
    why is that the periodicity rule catches all three covers on its own. What
    this rule adds is the case periodicity cannot see: an attacker who breaks up
    the spacing to stay under that bound is still caught behind an emoji cover
    and is NOT caught behind a joining-script one, which is the asymmetry
    `test_a_deperiodised_bitstream_is_a_known_miss` holds. So the tests that
    answer for this rule are that one and
    `test_a_joiner_needs_both_neighbours_in_one_context`, not the covered
    bitstream, which denies either way.

    ZWNJ is never legitimate between two pictographics, and that is the half of
    this rule that pays for itself: emoji sequences join with ZWJ, no RGI
    sequence uses ZWNJ anywhere, so behind an emoji cover the attacker loses one
    of the two symbols a bitstream needs. Measured on the 256-bit payload above,
    it leaves 123 of the joiners unexcused and the total bound reports them.

    A virama BEFORE the joiner excuses it whatever follows, because that is the
    one place a both-neighbours rule cannot look: a Malayalam chillu is
    consonant, virama, ZWJ, and it ENDS a word, so the character after the
    joiner is a space or a full stop. Without this, five ordinary Malayalam words
    in a sentence are five unexplained joiners, which is exactly the total bound.
    `test_a_joiner_at_the_end_of_a_word_is_still_orthography` is that sentence,
    and it carries five since `_MIN_TOTAL` was raised: at four it allowed whether
    this branch existed or not.

    The VIRAMA'S OWN BASE has to be in a joining script too, and that condition
    is the repair of a bypass this branch shipped without it. Written as "any
    character of combining class 9" it excused a joiner behind a virama sitting
    on anything at all, which is a cover character wearing a mark: base, virama,
    joiner carries a bit for three characters, both symbols stay available
    because neither is excused by its neighbours, and the joiners are three apart
    so the periodicity rule never sees them. Measured that way, an emoji base and
    a LATIN base both carried a 256-bit payload past this detector at three
    characters per bit. The emoji cover is the one the condition above had just
    closed; the Latin cover was not available at any price before this branch
    existed. `test_a_virama_does_not_excuse_a_joiner_standing_on_a_foreign_base`
    holds both.

    THE TRADE, since a rule this tight makes one. What it refuses that Unicode
    would call a cluster is: a virama whose base is in a different script from
    it, a virama with more than THREE characters of marks and format characters
    between it and its base, a virama whose base is not a letter, and a virama
    with no base at all.

    None of those four is orthography in any script, and the evidence is a
    measurement built from character properties rather than from a list of
    inputs somebody chose, which is the only kind that can find what the author
    did not think of. Over every character of every script that owns a character
    of combining class 9 in Unicode 16.0.0: all 3,519 same-script LETTERS are
    excused as bases, and all 10,668 covers built from the same-script
    NON-letters -- behind a Latin base, behind a pictographic one, standing
    alone, and doubled -- are refused. An earlier version of this paragraph said
    none was excused above a Latin base while 63 of the 69 were, and it said so
    on the strength of four covers picked by hand.

    The one soft edge is recorded in `_script`: TAI LE, TAI THAM and TAI VIET
    share a first name word, so this cannot tell them apart.

    The script test is a VETO rather than one more way to pass, which is why this
    branch returns instead of falling through, and WHAT THE VETO IS DOING HAS
    NARROWED since that was written. A virama is in the Devanagari through
    Sinhala range itself, so a joiner behind one already satisfies half of the
    both-neighbours rule below, and the input this note has always cited is
    `<latin><virama><joiner><devanagari>` -- alternating bases, the same channel
    at the SAME price, not a wider one: alternating, Latin-only and
    Devanagari-only all measure 769 characters for 256 bits.

    That input no longer reaches the question, and saying it did was a claim
    about a rule that has since tightened. `_joining_neighbour` asks a mark to
    reach a LETTER INSIDE the joining ranges, and a Latin letter is not one, so
    with this branch turned into a fall-through the alternating cover still
    denies. Measured against the file before the test named below existed, that
    mutation left the whole suite green and every corpus case where it was.

    What the veto is still the only thing refusing is a base whose script
    differs from its virama's while BOTH sit inside the ranges. A Bengali letter
    under a Devanagari virama satisfies every condition the rule below asks --
    the virama is a mark in range that reaches a letter in range, and the
    character after the joiner is a letter in range -- so a fall-through excuses
    that stream whole, at the 3.0039 characters per bit the table above records
    for a matched base. `test_a_virama_over_a_letter_of_another_script_is_not_excused`
    is that input, and it exists because nothing here held the veto.

    What is left is any base whose script matches its virama's, and separately
    any pair of neighbours the rule below excuses, which is letters, decimal
    digits, and marks written on a letter of a joining script. Neither of those
    two branches now excuses a joiner with nothing visible around it:
    `_is_contextually_legitimate` answers False for `<ALM><ZWNJ><ALM>`, for a
    run of unattached marks, and for a mark on a Hangul filler, and
    `_joining_neighbour` carries the sweeps that closed all three. That is a
    statement about THESE branches and not about the module -- the pictographic
    branch below still excuses a wholly invisible payload, and the last
    paragraph here says at what rate.

    THE CONVENTION for every rate in this module, because three of them have
    been wrong for want of one: a message carries the WHOLE payload, and the
    character count includes the one trailing cover character the last joiner
    needs as a right neighbour. Measured that way on a 256-bit payload:

        <letter><virama><joiner>              769 chars  3.0039/bit  257 visible
        <letter>(<virama><joiner>)*2          641        2.5039      129
        <letter>(<joiner><virama>)*3          599        2.3398       87
        <letter>(<joiner><virama>)*4          577        2.2539       65   DENY
        deperiodised, letter or digit cover   598        2.3359      342

    The rows are five measurements and the table is the whole of what they say.
    An earlier version of this paragraph ranked them -- "the cheapest thing
    either branch still allows", then "the least costly thing", which is the
    same claim with a new adjective -- and a superlative over everything a
    branch allows is unbounded quantification inferred from five rows. It also
    contradicted the paragraph four lines below it. What the rows do support:
    the deperiodised cover and the virama cover differ by a hair per bit while
    the virama cover costs a QUARTER of the visible text, 87 characters against
    342, so visible cost rather than rate is what separates those two. An
    earlier version also quoted 2.3373 for the virama row, which is 596
    characters over the 255 bits a 3-bit block size actually carried rather than
    over the payload.

    Neither number bounds this function and NOTHING here bounds the module. The
    pictographic branch below excuses a presence-and-absence payload at 1.5000
    characters per bit over any of 503 code points, and WITH NOTHING VISIBLE
    over exactly two of them. This said "with nothing visible" of all 503, which
    is false of the 501: not one of them is default-ignorable, so a font draws
    `.notdef` for each and the cover is a row of tofu boxes. Only U+FE0E and
    U+FE0F carry it invisibly.
    `test_a_variation_selector_bitstream_is_a_known_miss` is that measurement.
    It is the cost of that encoding and not a minimum: `_invisible` records the
    families this module does not count at all, with a measured encoder for
    each, and records why no minimum is stated.
    """
    char = content[index]
    if char not in _CONTEXTUAL:
        return False
    before = content[index - 1] if index > 0 else ""
    after = content[index + 1] if index + 1 < len(content) else ""
    if char == _MVS:
        # Its own branch, and it returns rather than falling through, for the
        # reason the virama branch does: the Mongolian block is not in
        # `_JOINING_SCRIPTS`, so falling through would reach a rule that cannot
        # answer for it and would answer False for every real occurrence.
        #
        # `_joining_neighbour` over the Mongolian block, and NOT a bare range
        # test, which is what this branch shipped as for one measurement. U+180E
        # is itself inside U+1800..U+18AF, so a bare range test makes a run of
        # separators excuse ITSELF: measured, `note` followed by four U+180E and
        # `end` allowed, with the one character the branch was added to catch
        # defeating the check. Asking for a letter, a digit, or a mark written
        # on one closes it, and it still admits the real sequence, where a free
        # variation selector sits between the letter and the separator.
        #
        # WHICH BOUND REPORTS IT has changed, and the note used to name the
        # wrong one. Consecutive separators are ADJACENT, so `_MIN_RUN` reports
        # them from two onwards, which is BEFORE the total bound reaches them at
        # `_MIN_TOTAL`: measured, `note` plus two, three or four U+180E denies
        # with the total at two, three and four and the run bound the only thing
        # firing; at five the total reaches it as well, having been beaten to it.
        # An earlier version of this sentence said the total bound never reaches
        # them and then listed five as one of the values, which is false at the
        # last value it named. That makes
        # `test_a_run_of_mongolian_vowel_separators_does_not_excuse_itself` a
        # run-bound test, which is why the raise to `_MIN_TOTAL = 5` left it
        # discriminating while the four-occurrence Mongolian negatives beside it
        # stopped.
        #
        # The residual this leaves is the one this module already records rather
        # than a new one: a Mongolian LETTER is visible, so presence-and-absence
        # of a separator behind a Mongolian cover costs an attacker one visible
        # character per bit, which is what
        # `test_a_deperiodised_bitstream_is_a_known_miss` measures for a
        # Devanagari cover.
        return _joining_neighbour(content, index - 1, _MONGOLIAN) and _joining_neighbour(
            content, index + 1, _MONGOLIAN
        )
    if before and unicodedata.combining(before) == _VIRAMA:
        return _script(_base_before(content, index - 1)) == _script(before)
    if _joining_neighbour(content, index - 1) and _joining_neighbour(content, index + 1):
        return True
    return char == _ZWJ and _in_ranges(before, _PICTOGRAPHIC) and _in_ranges(after, _PICTOGRAPHIC)


def _chains(indices: list[int], step: int) -> list[list[int]]:
    """Maximal groups in which each index is exactly `step` after the one before it.

    Both rules below are a chain at a different step, which is why this is one
    function and not two loops. Step 1 gives the maximal RUNS of adjacent
    characters; step 2 gives the joiners that are one base character apart.

    EXACTLY, and no test can distinguish that from "at most", which is recorded
    here rather than repaired because it is an equivalence and not a gap.
    Indices arrive ascending and distinct, so at step 1 a difference of one is
    the only difference that can satisfy either spelling. At step 2 the two
    spellings differ only on ADJACENT excused joiners, and a joiner is excused
    only when the character before it is a virama, a character
    `_joining_neighbour` accepts over either range, or a pictographic; ZWJ, ZWNJ
    and the Mongolian vowel separator are none of those, in any context, so the
    second of two adjacent joiners is never excused and the difference never
    arises. Loosening the comparison to `<=` leaves the
    whole suite green and the corpus unmoved, and that is the argument above
    rather than a missing test.
    """
    groups: list[list[int]] = []
    for index in indices:
        if groups and index - groups[-1][-1] == step:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _zero_width_spans(content: str) -> list[tuple[int, int]]:
    """Zero-width characters that no script, no emoji sequence and no accident explains.

    Two rules over one split of the input, and neither one subsumes the other.
    The first reports what the exemption did not excuse, by run and by total.
    The second reports what it DID excuse, when the excused joiners arrive one
    per base character with the symbol changing: that is a bitstream wearing a
    cover, and every joiner in it has a neighbour that excuses it.

    The periodicity chain is built from the EXCUSED joiners alone. One that was
    not excused is already reported where it stands, and letting it join a chain
    would make a stray ZWSP two characters ahead of a family emoji into a
    four-long chain built out of three identical ZWJ;
    `test_a_stray_zero_width_space_before_a_family_emoji_is_not_a_payload` is
    that input. `len(set(...)) > 1` is the changing symbol: only ZWJ and ZWNJ can
    be excused, so more than one distinct character in a chain means both of
    them, which is the choice a bit needs.

    Sorted, because the two rules produce spans that interleave. `_matches`
    re-sorts everything it collects, so nothing downstream depends on this one,
    but a caller reading spans out of one signal should not have to know that.
    That made it invisible: measured before
    `test_each_signal_sorts_the_spans_it_returns_on_its_own` existed, removing
    this sort, and the matching one in `_bidi_spans`, changed no test and no
    corpus case, because every caller goes through `_matches`. Both are pinned at
    the level where the claim is made now, by that test, which calls the
    two helpers directly with the inputs that put their two sources out of
    order: a stray pair AFTER a periodic chain here, and an isolate opened
    before an embedding at a paragraph break there.
    """
    suspicious: list[int] = []
    exempt: list[int] = []
    for index, char in enumerate(content):
        if char in _ZERO_WIDTH:
            bucket = exempt if _is_contextually_legitimate(content, index) else suspicious
            bucket.append(index)

    spans = [
        (run[0], run[-1] + 1)
        for run in _chains(suspicious, 1)
        if len(suspicious) >= _MIN_TOTAL or len(run) >= _MIN_RUN
    ]
    spans += [
        (chain[0], chain[-1] + 1)
        for chain in _chains(exempt, 2)
        if len(chain) >= _MIN_PERIODIC and len({content[index] for index in chain}) > 1
    ]
    return sorted(spans)


class InjectionStructuralGuardrail:
    """Detects instruction smuggling in the encoding rather than in the words."""

    # Annotated with the Literal types, not bare assignments: a bare
    # `kind = "constraint"` infers `str`, and protocol attribute matching is
    # invariant, so it would not satisfy `kind: Kind`.
    name: str = "injection-structural"
    version: str = _VERSION
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input"})

    def __init__(self, on_match: Decision = "deny") -> None:
        # Defaults to deny, unlike `secrets`, and the asymmetry is deliberate. A
        # credential in output is a leak the caller usually still wants the rest
        # of; smuggled instructions in input are the whole reason the input is
        # suspect. A caller who has thought about it can choose `redact`, which
        # is meaningful here only because this attack IS a removable substring:
        # stripping a run of tag characters leaves the visible text intact and
        # the payload gone. The classifier has no such option.
        if on_match not in ("redact", "deny"):
            raise ValueError(f"on_match must be 'redact' or 'deny', got {on_match!r}")
        self._on_match: Decision = on_match

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every span every signal claims, sorted by span. Nothing is dropped.

        Sorted BY SPAN is a precondition of what consumes this list, not a
        formatting preference. `_merge` tests each span against the running end
        of the region it is extending and looks no further back, so a list in
        any other order makes it emit regions that are wrong rather than merely
        untidy, and `_rewrite` writes those out. A tie on the start offset puts
        the shorter span first, and `sorted` is stable, so equal spans keep the
        order the signals produced them in.

        Holding that falls to whoever adds a signal here, because each signal
        scans the whole input independently: their results concatenate in signal
        order, which is not span order, and the combined list has to be re-sorted
        before it is returned. Nothing local catches a miss. `deny` is the
        default and a deny never reaches `_rewrite`, so an unsorted return stays
        invisible until a caller configures `redact`.

        Measured, not assumed, and no longer a no-op. Each signal walks left to
        right, so each list is ordered on its own and it is the concatenation
        that is not. Re-measured against this file as it stands, replacing the
        sort with `return found` fails THREE tests --
        `test_findings_from_both_signals_come_back_in_span_order`,
        `test_findings_from_all_three_signals_come_back_in_span_order` and
        `test_redacting_both_signals_leaves_neither_control_standing`. This note
        said two and named the first and the last, having been written when the
        third did not exist and never re-run afterwards.

        The redaction one is what shows the cost: `_merge` folds the earlier
        bidi span into the later tag region and emits one region that starts
        after the control, so the redacted output keeps the override while the
        placeholder claims to have removed it.

        WHY ONE SORT IS ENOUGH, which is an invariant of the three signals and
        was nowhere written down. THEIR CHARACTER SETS ARE PAIRWISE DISJOINT.
        `_tag_spans` reports only U+E0000..U+E007F; `_bidi_spans` reports only
        the nine controls in `_EMBED_OPEN`, `_ISOLATE_OPEN`, `_EMBED_CLOSE` and
        `_ISOLATE_CLOSE`; `_zero_width_spans` reports only members of
        `_ZERO_WIDTH`. The two overlaps that could exist are closed on purpose
        in `_invisible`: the tag range is excluded by
        `not _TAG_START <= point <= _TAG_END`, and every one of the nine bidi
        controls is removed by `_is_directional`. So no two signals can ever
        claim the same code point, no two spans from different signals can be
        equal, and `sorted` therefore puts this list in an order `_merge` can
        consume in one pass -- provably sufficient rather than merely adequate
        on the inputs anybody has tried. Widening one of the three ranges into
        another's territory is what would break it, silently, so
        `test_the_three_signals_claim_pairwise_disjoint_character_sets` asserts
        the disjointness rather than leaving it to this paragraph.
        """
        found = [("INVISIBLE_TAG_CHARS", span) for span in _tag_spans(content)]
        found += [("BIDI_OVERRIDE", span) for span in _bidi_spans(content)]
        found += [("ZERO_WIDTH_SMUGGLING", span) for span in _zero_width_spans(content)]
        return sorted(found, key=lambda pair: pair[1])

    def check(self, content: str, context: Context) -> Verdict:
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        findings = [Finding(type=name, span=span) for name, span in found]
        if self._on_match == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
