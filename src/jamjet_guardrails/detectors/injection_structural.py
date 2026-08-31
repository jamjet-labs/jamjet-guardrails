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
# shipped code: "Summarise this. " followed by seven repeats of U+1F3F4 + five
# tag letters + CANCEL TAG carried "ignore all previous instructions" past this
# deny-by-default detector with `_tag_spans` returning [], for a cost of one
# visible black flag per five smuggled characters. That exact input is kept as
# `test_chained_flag_bases_do_not_smuggle_a_payload`.
#
# An allowlist has no such seam: an exempt run is one of exactly three fixed
# strings, so it carries no attacker-chosen bytes at all and a row of them is a
# row of flags. The general rule, for the next exemption written into this
# module: an exemption that APPROXIMATES a closed set with a shape test is
# chainable, and chaining turns the exemption into precisely the channel it was
# written to deny.
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
    imbalance, because that is what makes the rendered order diverge from the
    logical order a parser -- or a model -- actually reads for the whole rest of
    the paragraph, rather than for a stretch the author closed.

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
            unbalanced += embeds
            unbalanced += isolates
            embeds.clear()
            isolates.clear()
    unbalanced += embeds
    unbalanced += isolates
    return [(index, index + 1) for index in sorted(unbalanced)]


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
        that is not. Replacing the sort with `return found` now fails
        `test_findings_from_both_signals_come_back_in_span_order` and
        `test_redacting_both_signals_leaves_neither_control_standing`, and the
        second of those is the one that shows the cost: `_merge` folds the
        earlier bidi span into the later tag region and emits one region that
        starts after the control, so the redacted output keeps the override
        while the placeholder claims to have removed it.
        """
        found = [("INVISIBLE_TAG_CHARS", span) for span in _tag_spans(content)]
        found += [("BIDI_OVERRIDE", span) for span in _bidi_spans(content)]
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
