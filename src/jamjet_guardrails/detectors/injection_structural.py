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
_TAG_LETTER_START = 0xE0020
_TAG_LETTER_END = 0xE007E
_CANCEL_TAG = 0xE007F
# U+1F3F4 WAVING BLACK FLAG. The base of every RGI subdivision flag sequence and
# the ONLY context in which a tag run is ordinary text.
_FLAG_BASE = 0x1F3F4
# The longest RGI subdivision code is five tag letters (gbeng, gbsct, gbwls).
# Bounded so the exemption cannot be stretched into a channel: at six or more,
# whatever it is, it is not one of the three sequences Unicode defines.
_MAX_FLAG_TAG_LETTERS = 5


def _is_valid_flag_sequence(content: str, start: int, end: int) -> bool:
    """Whether a tag run at [start, end) is an RGI subdivision flag's payload.

    Four conditions, and each one closes a laundering route:
      - the character before the run is U+1F3F4;
      - the run ends with exactly one CANCEL TAG, and it is the last character;
      - every other character is a tag LETTER, so a run carrying anything
        outside U+E0020..U+E007E is not a flag whatever surrounds it;
      - at most five letters, which is the longest code Unicode assigns.
    """
    if start == 0 or ord(content[start - 1]) != _FLAG_BASE:
        return False
    run = content[start:end]
    if not run or ord(run[-1]) != _CANCEL_TAG:
        return False
    body = run[:-1]
    if not body or len(body) > _MAX_FLAG_TAG_LETTERS:
        return False
    return all(_TAG_LETTER_START <= ord(ch) <= _TAG_LETTER_END for ch in body)


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
