"""Personal-data constraint over four conservative, high-signal types.

Constraint, not classification: each type is a regex, so the same input always
produces the same finding and the same decision. Constraint does not mean "one
regex per type". A bare digit run is matched by pattern and then VALIDATED by its
check digit, which is still a pure function of the input and still produces the
same spans every time; it is conditional, not probabilistic, and it carries no
confidence.

Overlapping matches are MERGED, never dropped. Two patterns that claim
overlapping stretches of the input are each right about their own bytes, so
keeping one and discarding the other leaves the discarded one's bytes in the
output. In a redactor an ambiguous span has to resolve toward more redaction,
never less.
"""

from __future__ import annotations

import re

from jamjet_guardrails._spans import _rewrite, _scan
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import Context, Direction, Finding, Kind, Provenance, Verdict

PII_TYPES = frozenset({"EMAIL", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER"})


def _luhn(digits: str) -> bool:
    r"""The check digit every payment card carries.

    This is what makes a bare digit run safe to match. Without it,
    `[2-6]\d{12,18}` matches every order number and timestamp id in the corpus;
    with it, nine in ten random runs are rejected. It is NOT sufficient on its
    own: the one in ten it passes is still one machine-written log line in ten,
    which is why the leading digit is constrained beside it.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# A separator only fences a value off when it CONTINUES a digit run. The earlier
# `(?<![\d-])` / `(?![\d-])` turned away any adjacent hyphen at all, so a label
# hid the value it labelled: "ssn-123-45-6789" came back allow with the SSN
# whole and liftable, and so did "card-4111 1111 1111 1111". A hyphen with a
# digit on its far side is a longer dashed run, which is the case these were
# written for and the only case they should turn away.
_BEFORE = r"(?<!\d)(?<!\d-)"
_AFTER = r"(?!\d)(?!-\d)"

# U+0300 to U+036F, Combining Diacritical Marks: the tails NFD leaves behind when
# it decomposes an accented Latin, Greek or Cyrillic letter. `\w` does NOT match
# them, because a nonspacing mark is not alphanumeric, so without this range the
# decomposed spelling of a real address is a COMPLETE MISS while the composed
# spelling redacts: "jose" plus U+0301 then "@example.com" has a mark before the
# `@`, no start position reaches the `@`, and nothing matches. Text that has been
# through NFD wholesale (an Apple filesystem, some JSON pipelines) is spelled
# that way throughout, so this is a real shape and not a curiosity. Applied to
# the local part AND the domain, since a decomposition normalises both halves.
_MARKS = "\u0300-\u036f"

# Order is not load-bearing for the output. Overlapping matches merge and every
# contributing type is named, so no pattern can win a contested span at another's
# expense. Order settles only the listing of two findings sharing one span, which
# these four patterns cannot produce anyway: no two of them match the same text.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Repetition is BOUNDED, not open. Unbounded `+` on the local part is
    # quadratic: `.` is in the class, so a word boundary holds after every dot and
    # there are O(n) start positions each scanning O(n) characters to find an `@`.
    # Measured on "a." * n + "@": 4.0x per doubling, 3.3 seconds at 64 KB, on a
    # component that reads attacker-influenced text in both directions. The bounds
    # are RFC 5321's: local part at most 64 octets, domain at most 255. Linear
    # after the change, 2.0x per doubling and 8.9 ms at 64 KB. An atomic group
    # would also fix it but needs Python 3.11, and this package supports 3.10.
    #
    # There is deliberately NO word-boundary anchor at EITHER end.
    #
    # Leading: a bound plus a start anchor is a redaction bypass. The bound applies
    # to the WHOLE local part, so 60 characters of padding before a real address
    # put the `@` out of reach of every legal start, and "contact " + "u"*60 +
    # "alice@example.com" came back allow with the address intact.
    #
    # Trailing: one word character after the TLD defeated it, so "a@b.com9" and
    # "a@b.com" plus any letter outside ASCII went unredacted. Over 150,000
    # randomised trials, 45,516 delimited addresses survived whole with the anchor
    # and 0 without it.
    #
    # What the anchor was doing, stated accurately: buying precision. It suppressed
    # "myimage@sha256.abc123" and "build@node.js18", tokens whose dotted tail looks
    # like a TLD and is followed by a digit, and those now redact. The shape is
    # essentially absent from real text: across 400 MB of source, logs, lockfiles
    # and docs it produced ZERO new matches, so the price is close to nothing in
    # practice while the evasion it enabled was a whole liftable address.
    #
    # A TLD allowlist WOULD separate the two, and is rejected on its own terms
    # rather than as impossible. `.abc`, `.zip`, `.mov`, `.dev`, `.sh` and `.rs`
    # are all delegated gTLDs, so a natural class of these tokens survives any
    # list; and an allowlist buys precision by paying recall, which is the
    # direction this component must not trade in, while putting a list that goes
    # stale on its own inside a security boundary.
    #
    # `\w` rather than `[A-Za-z0-9_]`, and NO `re.UNICODE`. For a str pattern
    # `\w` is Unicode ALREADY, so passing the flag would be a no-op wearing a
    # guard's clothes: delete it and nothing changes, which is worse than not
    # having it, because the next reader would believe it was load-bearing. What
    # actually widens the class is the class itself. The ASCII version made a
    # whole deliverable address a complete miss, "alice@example." plus the
    # Cyrillic ccTLD among them, which is the most consequential shape this
    # detector had.
    #
    # The TLD is LETTERS, not `\w`, and this is the one place the wider class was
    # rejected rather than taken. `\.\w{2,}` reads a trailing numeric label as a
    # TLD, so "root@192.168.1.100" becomes an EMAIL: an SSH target, in every
    # server log there is, redacted as a person. Letters keep it an allow and
    # cost nothing real, because no delegated TLD contains a digit except the
    # `xn--` A-labels, which stop at their own hyphen under either class. Checked
    # against 70 real ASCII TLDs: `[^\W\d_]{2,}` and the old `[A-Za-z]{2,}` agree
    # on every one, so the widening costs no precision on the ASCII side at all.
    #
    # A NEW AXIS OF VERSION DEPENDENCE arrives with this line, and nothing else
    # in this file has one. `\w` and `[^\W\d_]` are defined by the INTERPRETER's
    # Unicode database, so which code points they admit moves with the Python
    # version; the ASCII classes they replaced could not. Every fixture in the
    # suite uses long-assigned code points, so 3.10 and 3.14 agree today and
    # there is no CI risk now. It is recorded because a future recall change
    # arriving from a Python upgrade rather than from a diff is the kind of thing
    # nobody thinks to look for.
    (
        "EMAIL",
        re.compile(rf"[\w{_MARKS}.%+-]{{1,64}}@[\w{_MARKS}.-]{{1,255}}\.[^\W\d_]{{2,}}"),
    ),
    #
    # TWO GAPS ARE LEFT OPEN HERE ON PURPOSE, and they are recall, not oversight.
    # An unseparated `123456789` and a space- or dot-separated `123 45 6789` are
    # both real ways to write an SSN and neither is matched. Nine bare digits is
    # also a zip+4, an order number, a part number and half the identifiers in
    # any log, and unlike a card number this type carries NO check digit to buy
    # the precision back: the contiguous-card pattern below is only safe because
    # Luhn rejects roughly nine in ten runs, and there is no equivalent here. A
    # pattern that matched them would fire on ordinary log lines all day, so the
    # miss is the cheaper error. Task 14's corpus carries both shapes as expected
    # findings anyway, so the published recall number pays for them rather than
    # hiding them.
    ("US_SSN", re.compile(rf"{_BEFORE}\d{{3}}-\d{{2}}-\d{{4}}{_AFTER}")),
    #
    # The 4-6-5 alternative is how Amex actually prints a card, and it was a
    # complete miss: "3782 822463 10005" has no four-group layout to match.
    # No Luhn behind this one, deliberately. Grouping IS the evidence, it is
    # already pinned by tests and corpora, and putting a check digit behind it
    # would turn existing expected-redact cases into silent allows.
    (
        "CREDIT_CARD",
        re.compile(
            rf"{_BEFORE}(?:\d{{4}}[ -]\d{{6}}[ -]\d{{5}}|(?:\d{{4}}[ -]){{3}}\d{{4}}){_AFTER}"
        ),
    ),
    #
    # The grammar, stated rather than counted: an optional `+1`, then either a
    # parenthesised area code or three digits and a separator, then three digits,
    # a SEPARATOR, and four digits; or, on its own branch, `+1` and ten bare
    # digits. An earlier version of this comment said "four layouts", naming the
    # four fixtures instead of describing what the pattern accepts, and the count
    # was wrong the moment two separator characters were allowed. Count nothing
    # here; say what the shape is.
    #
    # BOTH separators are required, and the second one was optional for exactly
    # one revision. Optional, `NNN[-.]NNNNNNN` is a phone number, and that reads
    # a UUID's middle as one:
    #
    #     0ad830af-c60c-657d-c147-7008575ea35c
    #         ->  0ad830af-c60c-657d-c[REDACTED:PHONE_NUMBER]ea35c
    #
    # 371 of 200,000 random UUIDv4s were hit with it optional and ZERO with it
    # required, and no fixture of any of the four target layouts needs it: the
    # whole suite passes with the character removed. A correlation id losing its
    # middle to a finding that asserts a phone number nobody wrote is a worse
    # outcome than any recall the character bought, and it bought none.
    #
    # The bare-digit branch requires the `+1`, and that is the whole of its
    # precision. Ten unseparated digits with the country code dropped is a Unix
    # timestamp in seconds, which appears in nearly every log line there is; the
    # `+1` is the evidence that turns the run into a phone number, exactly as
    # Luhn is for a bare card run. It does fire on a signed integer of the same
    # width, "+12345678901", which is recorded as a known cost rather than
    # guarded: `+1` followed by ten digits is E.164 for a US number, and the two
    # are not distinguishable from the characters alone.
    #
    # KNOWN PRECISION COST of the dot separator: a dotted digit run of the shape
    # 3.3.4 reads as a phone number wherever it occurs, so "192.168.100.4001"
    # redacts. Every VALID IPv4 stays an allow, and structurally so rather than
    # by luck: an octet is at most 255, so it is at most three digits, and this
    # pattern's final group is four consecutive digits. The shape that collides
    # is a dotted quad whose last group is four digits, which is not an address.
    (
        "PHONE_NUMBER",
        re.compile(
            rf"{_BEFORE}(?:"
            r"\+1[ -]?\d{10}"
            r"|(?:\+1[ -]?)?(?:\(\d{3}\)[ ]?|\d{3}[-.])\d{3}[-.]\d{4}"
            rf"){_AFTER}"
        ),
    ),
)

# The contiguous card cannot live in the table above, because it is the one match
# this detector makes CONDITIONALLY. `\d{13,19}` on its own matches every order
# number, timestamp id and hash fragment in the corpus; behind Luhn it rejects
# nine in ten random runs, which is the difference between a usable pattern and
# one that must not ship.
#
# TWO conditions, not one, and the leading digit is the one that was missing.
# Luhn alone leaves 10.0% of digit runs standing, measured as exactly 10.0000%
# over 20,000 consecutive epoch-millisecond values and 10.0045% over 200,000
# uniform runs. Epoch-ms is THIRTEEN digits and sits in nearly every machine
# written log line, so one log line in ten carried a CREDIT_CARD finding:
#
#     {"event":"click","ts":1767205800009,"user":"u42"}
#         ->  {"event":"click","ts":[REDACTED:CREDIT_CARD],"user":"u42"}
#
# The first digit of a card is its Major Industry Identifier, and payment cards
# are issued only under MII 2 to 6 (ISO/IEC 7812). 0, 1, 7, 8 and 9 are ISO/TC
# 68, airlines, petroleum, healthcare and telecom, and national assignment. So
# `[2-6]` is a constraint from a published standard rather than a threshold
# tuned to a fixture, and it halves what Luhn leaves: 10.0045% to 4.9595% over
# uniform runs, and 0.4305% to 0.2155% over 200,000 sha256 digests.
#
# WHAT IT COSTS, all three of them, because none is zero:
#
#   1. A card issued under an MII outside 2 to 6 is a complete miss. None is
#      issued today; this is a deliberate trade against a future assignment.
#   2. THE EPOCH PROTECTION EXPIRES. It works because epoch-ms currently begins
#      with a 1, and epoch-ms first carries a leading 2 on 2033-05-18T03:33:20Z,
#      after which timestamps sit INSIDE the MII range again: measured 10.0000%
#      before the guard and 10.0000% after it on values past that boundary.
#      Epoch-microseconds, sixteen digits, crosses on the same date. This is a
#      dated guard and it needs revisiting, not a permanent fix.
#   3. Other Luhn-carrying identifiers are still matched, and typed CREDIT_CARD.
#      IMEIs and ICCIDs both carry a Luhn check digit, and an IMEI beginning 49
#      sits squarely inside Visa's range: "imei 490154203237518" redacts under a
#      CREDIT_CARD finding. Redacting it is the right direction, since an IMEI is
#      personal data too; the TYPE in the audit record is wrong, and no leading
#      digit separates the two.
_CONTIGUOUS_CARD = re.compile(r"(?<!\d)[2-6]\d{12,18}(?!\d)")


def _contiguous_cards(text: str) -> list[tuple[str, tuple[int, int]]]:
    """Digit runs that carry a valid check digit.

    A run that fails Luhn is not reported AT ALL, neither in the output nor in
    the findings. Reporting it would spend the precision the check exists to buy.

    Scanned with `_scan` rather than `finditer` for the same reason every other
    pattern here is, even though the two agree on this one: the lookarounds pin
    the match to a whole digit run, so there is no second start to skip. That is
    an argument about today's pattern, not about the scan, and the scan is where
    a Critical lived once already.
    """
    return [
        ("CREDIT_CARD", (match.start(), match.end()))
        for match in _scan(_CONTIGUOUS_CARD, text)
        if _luhn(match.group())
    ]


_VERSION = "0.1.0"


class PiiGuardrail:
    """Redacts personal data to typed placeholders."""

    # Annotated with the Literal types, not bare assignments: a bare
    # `kind = "constraint"` infers `str`, and protocol attribute matching is
    # invariant, so it would not satisfy `kind: Kind` and mypy --strict fails
    # at the registry in Task 8.
    name: str = "pii"
    version: str = _VERSION
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(self, types: frozenset[str] | None = None) -> None:
        """Select which types to look for. Refuses a selection that looks for none.

        The two refusals raise DIFFERENT errors, and the split is the one this
        package already draws elsewhere:

        - **A type outside the domain** is a bad argument value and stays a
          ``ValueError``, the detector's own. `build` deliberately does not wrap
          a detector's constructor, so this reaches the caller unrelabelled;
          `tests/test_registry.py` pins that boundary.
        - **A legal but INERT selection** is an availability problem, and raises
          ``GuardrailUnavailableError`` like the other four shapes of "a check
          that is configured and would not check". `types=frozenset()` built a
          guardrail that returned allow, no findings and untouched content over
          an email, an SSN, a card and an AWS key at once. That is exactly the
          output `build_chain` refuses to build from an empty list of names, and
          reachable from the same kind of configuration:
          `guardrails: {pii: {types: []}}`. `build_chain([])` and a guardrail
          declaring no runnable direction both raise
          ``GuardrailUnavailableError`` for being inert; so does this.

        `types=None` still means all four, and it is not the same as an empty
        set: `selected = PII_TYPES if not types else types` collapses the two
        and re-enables every pattern for the one caller who asked for none.
        """
        selected = PII_TYPES if types is None else types
        unknown = selected - PII_TYPES
        if unknown:
            raise ValueError(f"unknown PII types: {sorted(unknown)}")
        if not selected:
            raise GuardrailUnavailableError(
                "pii was configured with an empty set of types, so it would check "
                "nothing and allow any content at all. Expected some of "
                f"{sorted(PII_TYPES)}, or no types= argument to select them all"
            )
        self._patterns = tuple((t, p) for t, p in _PATTERNS if t in selected)
        # The contiguous card is a second SITE where the type selection applies,
        # and it is not in the table the line above filters. A caller who asked
        # for EMAIL alone must not have card numbers redacted out from under
        # them.
        self._scan_contiguous_cards = "CREDIT_CARD" in selected

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every match of every selected pattern, in text order. Nothing is dropped.

        Sorted by span, so a tie on the start offset puts the shorter match first.
        `sorted` is stable, so two matches with an identical span would keep
        pattern-table order and the result is total either way.

        Spans, not match objects. The contiguous card arrives as a span from a
        validated scan rather than from the table, and it has to merge with the
        grouped pattern exactly as two patterns already do: a card written
        "4111111111111111" inside text that also matches something else must
        collapse into one region, not stand beside it.
        """
        found = [
            (pii_type, match.span())
            for pii_type, pattern in self._patterns
            for match in _scan(pattern, content)
        ]
        if self._scan_contiguous_cards:
            found += _contiguous_cards(content)
        return sorted(found, key=lambda pair: pair[1])

    def check(self, content: str, context: Context) -> Verdict:
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        # One finding per match, each carrying its own original span, even where
        # several collapsed into one placeholder. The output loses that detail by
        # design; the audit record must not.
        findings = [Finding(type=pii_type, span=span) for pii_type, span in found]

        # Spans, not match objects: the shared merge takes spans so that a
        # detector which WALKS a region rather than matching it, or VALIDATES one
        # as the contiguous card does, can feed the same pipeline. The spans are
        # the ones already sorted above, in that order.
        #
        # This rewrite is THIS detector's own view of the input, offered for a
        # caller holding one guardrail. A chain does not thread it: it merges
        # these spans with every other guardrail's and rewrites once, through
        # this same `_rewrite`.
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
