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

        Measured, not assumed: replacing the sort with `return found` leaves the
        whole suite green today. `_tag_spans` walks left to right and is the only
        signal, so its output is already in span order and the sort is currently
        a no-op. It is the SECOND signal that makes this load-bearing, which is
        why the sort goes in now rather than then.
        """
        found = [("INVISIBLE_TAG_CHARS", span) for span in _tag_spans(content)]
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
