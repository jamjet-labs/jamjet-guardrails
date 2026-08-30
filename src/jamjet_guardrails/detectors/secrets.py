"""Credential-leakage constraint.

Prefix-anchored patterns only. Entropy scoring is where these detectors earn
their reputation for noise, and precision is the number this project publishes.

Overlapping matches are MERGED, never dropped and never rebuilt one match at a
time. Two patterns that claim overlapping stretches of the input are each right
about their own bytes, so anything that keeps one and walks past the other
leaves the other's bytes in the output. In a redactor an ambiguous span has to
resolve toward more redaction, never less.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from jamjet_guardrails._spans import _rewrite, _scan
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

SECRET_TYPES = frozenset(
    {
        "AWS_ACCESS_KEY",
        "GITHUB_TOKEN",
        "OPENAI_KEY",
        "ANTHROPIC_KEY",
        "SLACK_TOKEN",
        "PRIVATE_KEY",
        "JWT",
    }
)

# ANTHROPIC_KEY precedes OPENAI_KEY defensively. It is not load-bearing, and the
# original claim about WHY has been corrected: overlapping matches merge here, so
# no pattern can win a contested span at another's expense and order cannot
# decide what gets redacted. What it decides is the order of two findings sharing
# one span in the audit record; the placeholder names every contributing type,
# sorted, so even that is invisible in the output.
#
# The other half of the claim was measured rather than trusted. OPENAI does not
# match an Anthropic key: "ant-" is not one of its named prefixes and the hyphen
# stops its pure-alphanumeric branch after three characters. Checked against
# "sk-ant-api03-" bodies of 20, 95 and 92 characters with `-` and `_` mixed in,
# and against a bare "sk-ant-" plus 16: ANTHROPIC_KEY was the only match every
# time. The one shape that does put both patterns on one span is a key whose body
# CONTAINS "sk-" ("sk-ant-api03-sk-" plus 32 alphanumerics), and merging is what
# handles that, not order.
#
# OPENAI_KEY has two branches on purpose. `sk-` plus 32+ pure alphanumerics is
# the legacy shape. The prefixed branch covers `sk-proj-`, `sk-svcacct-` and
# `sk-admin-`, whose bodies contain `-` and `_`; a single permissive
# `sk-[A-Za-z0-9_-]{32,}` would cover those too but also fires on any long
# kebab-case identifier beginning `sk-`, which is why the prefixes are named
# rather than the character class widened.
# NO WORD-BOUNDARY ANCHORS. Task 6 shipped both bypasses in the EMAIL pattern
# and spent two review rounds removing them; the same holes were measured in
# THIS table before it was written:
#   'x' + 'AKIAIOSFODNN7EXAMPLE'  -> complete miss, whole key liftable
#   'AKIAIOSFODNN7EXAMPLE' + '9'  -> complete miss
#   'x' + 'ghp_...', 'x' + 'sk-ant-...', 'x' + 'xoxb-...', 'x' + 'eyJ...'
#                                 -> complete miss, every one
# A leading \b means one character of padding hides a live credential; a
# trailing \b means one character of suffix does. Both were verified to close
# with the anchors removed at zero cost on the negatives below. In a detector,
# ambiguity resolves toward MORE detection.
#
# JWT's three segments are BOUNDED, and it is the only pattern here that needs
# to be. Unbounded, every "eyJ" in the input is a candidate start whose first
# segment scans the rest of the input looking for a `.` that is not there, which
# is quadratic: on "eyJa" repeated it cost 4.0x per doubling, 870 ms at 64 KB and
# 14.2 SECONDS at 256 KB, on a component that reads attacker-influenced text in
# both directions. Bounded, the same input is LINEAR, 4.0x per 4x of input:
# 112 ms at 64 KB, 459 ms at 256 KB, 1.92 s at 1 MB. The other six patterns were
# measured on the same shape and are linear, for the per-pattern reasons given
# with the bounds below. An earlier version of this comment explained that with
# "a long run that reaches the end of the input MATCHES for them, so the scan is
# consumed rather than repeated". That model is FALSE and the Critical below
# disproved it: matches are computed per start position, so a match being found
# does not stop the next start from being tried. Deleted rather than left
# standing, because a reader who follows it repeats the bug.
#
# The header bound was 1024 and that was WRONG, on measured header sizes rather
# than on a guess: plain RS256 encodes to 36 characters, `jwk` with an RSA-2048 key is
# 584, one x5c certificate is 1670 and a two-certificate chain is 3292. x5c
# headers are ordinary in Open Banking, eIDAS and mTLS-bound JWS, so 1024 sat
# UNDER a whole class of real tokens and this pattern would have missed every one
# of them. 4096 clears a two-certificate chain. It is the bound that costs
# something to raise, since it is what the failing scan pays, and the cost is
# 456 ms on 256 KB of the adversarial shape against 109 ms at 1024, still linear
# and still inside the test's 2.0 second budget. Payload 65536 (a fat enterprise
# access token is a few thousand) and signature 8192 (an RSA-4096 signature is
# 683) are far from anything real. KNOWN RECALL LIMIT: a token that exceeds any
# of the three bounds is a complete miss, not a shorter match.
# Only the header bound is plausibly reachable at all now, and only by a
# certificate chain longer than two. Task 14's corpus must label that a miss
# rather than counting it as clean text.
#
# Padding cannot force that shape from outside: a failed match at one start does
# not end the scan, so a real token's own "eyJ" is still a later start with a
# short header, and lengthening the header means editing inside it, which
# destroys the signature. There is a test for that.
#
# KNOWN MISSES, recorded because a miss nobody wrote down is a recall number
# nobody can explain. `github_pat_` fine-grained personal access tokens, which
# are GitHub's modern default, and `xapp-` Slack app-level tokens are BOTH
# complete misses: neither shape is in this table, and no pattern here comes
# close enough to catch one by accident. That is the brief's list of types, not
# an oversight in the patterns, and it is left alone deliberately rather than
# widened here. Task 15 must not publish a recall number that leaves them
# invisible. There is a test naming both, so they cannot become silent.
#
# THREE BODIES ARE BOUNDED FOR COST, and the condition that forces it is worth
# stating exactly, because it is not "unbounded is bad". Trying every start
# position is what closed the decoy bypass; it turns an unbounded greedy body
# quadratic only when the body's character class can swallow the pattern's OWN
# prefix, because then every prefix occurrence starts a match that runs to the
# end of the input and there are O(n) of them. Measured end to end on 256 KB of
# each pattern's own prefix repeated:
#
#   xoxb-    repeated   8.97 SECONDS   4.0x per doubling   -> bounded
#   sk-ant-  repeated   6.22 SECONDS   4.0x                -> bounded
#   sk-proj- repeated   5.54 SECONDS   4.0x                -> bounded
#   AKIA, ghp_, sk- (legacy branch), -----BEGIN, eyJa      -> already linear
#
# The linear five are linear for a reason each: AWS is fixed length, GITHUB's
# body excludes the `_` its prefix ends with, OPENAI's legacy branch excludes the
# `-`, and JWT is bounded. PRIVATE_KEY is no longer a pattern at all. So OPENAI's second
# branch stays unbounded on purpose: `-` is not in its class, so it cannot reach
# across `sk-`.
#
# 1024, against JWT's 4096, and the difference is the data. The longest token
# OBSERVED in these three families is 171 characters (sk-svcacct; xoxb 56, xoxp
# 77, sk-ant-api03 108, sk-proj 164), but observed length is the weaker number:
# Slack's own changelog tells integrators to plan for tokens up to 255
# characters. So the honest headroom is 4x against published vendor guidance,
# not 6x against what happens to exist today, and anyone tightening this to 256
# would land exactly on Slack's stated ceiling. 4096 would cost 3.5x more on the
# dense shape for no recall at all. JWT's 4096 exists because a real
# two-certificate x5c header measured 3292.
#
# A body over the bound is NOT a miss here, unlike JWT: these patterns have no
# required suffix, so the match stops at the bound and the tail stands. The
# prefix and the first 1024 characters go, so what is left is not liftable, and
# nothing real is within a factor of five of the bound. Pinned from both sides.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS_ACCESS_KEY", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    ("GITHUB_TOKEN", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-[A-Za-z0-9\-_]{16,1024}")),
    (
        "OPENAI_KEY",
        re.compile(r"sk-(?:(?:proj|svcacct|admin)-[A-Za-z0-9_-]{32,1024}|[A-Za-z0-9]{32,})"),
    ),
    ("SLACK_TOKEN", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,1024}")),
    (
        "JWT",
        re.compile(r"eyJ[A-Za-z0-9_-]{4,4096}\.[A-Za-z0-9_-]{4,65536}\.[A-Za-z0-9_-]{4,8192}"),
    ),
)

# PRIVATE_KEY is NOT a regex. Three rounds of review turned "how far may I look"
# into a literal three times over, and every literal was a cliff: a terminated
# block bounded at 4096 missed a key logged line by line, raising it to 8192 just
# moved the miss to a longer key, and a line branch that required an UNBROKEN run
# of good lines from the first one lost the whole body to a single Proc-Type line
# or one interleaved log entry. Both of those were measured, not argued: an
# encrypted PEM logged line by line leaked 50 of 50 body lines, and one
# interleaved line leaked 25 of 50, each under a PRIVATE_KEY finding that would
# have scored a recall hit.
#
# The shape of the mistake is worth naming once: a PEM block is a LINE-ORIENTED
# FORMAT with an optional terminator, and a regex has to encode "keep looking" as
# a number. So the header is found by pattern and the body is walked in Python,
# which needs no bound at all.
#
# It is still a CONSTRAINT. The walk is deterministic and a pure function of its
# input: the same bytes always produce the same spans, which is what the
# constraint/classifier split means. Its findings carry no confidence, and
# `Verdict` enforces that. Constraint does not mean "one regex".
#
# The walk, from each header:
#   - segments are the remainder of the header's own line, then each following
#     line, split on a real newline OR a literal backslash-n, which is how a key
#     embedded in JSON is spelled;
#   - a segment holding an END marker closes the block and the walk stops there;
#   - a segment holding another BEGIN marker stops the walk before it, which is
#     also what keeps the walks non-overlapping and the whole scan linear;
#   - a segment counts as key material when at least 16 of its characters are
#     base64 and they make up at least half of it, which is what separates a
#     logged key line (64 base64 of 97 characters) from a sentence carrying a
#     git SHA (40 of 108);
#   - up to 8 consecutive non-key segments are tolerated, so RFC 1421 headers, a
#     blank line and interleaved log noise do not end the block. Only an accepted
#     segment advances the span, so tolerated noise is never redacted on its own.
#
# The body is kept only if the block was terminated, or two segments were
# accepted, or a single segment was overwhelmingly base64. That last case is a
# one-line PEM; the two-segment rule is what stops a single sentence mentioning a
# header and a container digest from being swallowed.
#
# SEVEN NUMBERS, not two. An earlier version of this comment said the rewrite
# took six numeric bounds down to two, and that was simply not counted: there are
# seven, and three of them have demonstrated cliffs. Listed so nobody has to
# count again, with what each one costs when it bites:
#
#   _PEM_WINDOW 1024        work, not reach: a segment is classified and searched
#                           for markers on its first 1024 characters, and a line
#                           of a megabyte is still covered in full once
#                           classified. Without it a dense run of headers with no
#                           newline is quadratic. CLIFF: a line opening with a
#                           kilobyte of prose and carrying the key after it.
#   _PEM_MISS_TOLERANCE 8   CLIFF: more than 8 consecutive non-key lines end a
#                           block. This is the tunable k.
#   _PEM_DENSITY 0.45       a fallback, behind the 40-character run. CLIFF in its
#                           own right while it was the PRIMARY rule, and the
#                           worst kind: it partitioned deployments by logging
#                           format rather than keys by size. It was 0.5, which
#                           sat at the median of the real distribution; it is now
#                           0.45, inside the interval that separates the two
#                           measured edges. Those edges, re-derived from the
#                           fixtures rather than quoted: prose carrying a
#                           24-character token is 0.4211, a 16-column body line
#                           in a both-sides envelope is 0.4848, and a 32-column
#                           one is 0.6531. So the separating interval is
#                           (0.4211, 0.6531] and any value in it splits them. An
#                           earlier version of this comment claimed no value
#                           could, which was arithmetic run backwards.
#   _PEM_LONG_RUN 40        the primary rule. No cliff found on wrap width: PEM
#                           wraps at 64, 70 or 76, so a body line clears it in
#                           every envelope measured. It is NOT true that nothing
#                           can shorten a run, which an earlier version of this
#                           comment asserted: two standard transforms insert
#                           characters INSIDE one. PHP's json_encode escapes a
#                           forward slash by default, which drops 25.7% of
#                           64-column lines below a 40-run, and base64url output
#                           replaces `+` and `/` with `-` and `_`, which drops
#                           50.9%. Both measured here over 200,000 random
#                           64-column lines rather than carried from a summary.
#                           Both matter
#                           little for whole keys today, since a block needs only
#                           two accepted segments, but the false version of this
#                           sentence would justify raising this bound toward 64,
#                           where it would bite hard.
#   _PEM_SOLO_DENSITY 0.9   an UNTERMINATED block whose body is a SINGLE accepted
#                           segment, with no 40-character run and under 90%
#                           base64, loses that segment. The run rule now backs it
#                           up, which closed a leak this comment had recorded
#                           INVERTED: it said a one-line key behind a short
#                           prefix survives, and the truth was the other way
#                           round. Envelopes of 0, 3 and 6 characters were
#                           covered; 11 and 33 lost the whole key, while the line
#                           carried a 63-character run throughout.
#   _BASE64_RUN {16,}       a run under 16 characters is not key material at all,
#                           so a body wrapped below 16 columns is missed in every
#                           envelope. Separately, and this is the bigger one, an
#                           unterminated key's FINAL short line survives when the
#                           envelope is long: up to 39 characters, measured on
#                           standard openssl 64-column output inside standard
#                           logback JSON, not on the exotic wrapping an earlier
#                           version of this comment blamed.
#   _MARKER_OVERLAP 48      how far a marker search reaches past a segment, so a
#                           marker across a window seam is still read whole.
#
# FOUR MORE LIMITS that are not thresholds of their own. Two are inherent to
# anchoring on a header: an attacker who emits `-----END-----` truncates the
# block early, and a body with no header before it is invisible. One is run
# fragmentation: anything that breaks a body line into pieces shorter than 40
# drops it back onto the density fallback, which is where the two transforms
# above land it. And `_PEM_WINDOW` is listed as a cliff above but is UNPINNED:
# cutting it from 1024 to 128 leaves the whole suite green, so nothing would
# notice it moving.
#
# EVERY ONE of these leaks with `decision == "redact"` and a PRIVATE_KEY finding
# while part of the body survives. Task 14 must score COVERAGE, never finding
# presence; a corpus that counts findings scores all of them as hits.
# Two details here survive mutation, recorded rather than left to be rediscovered.
# Dropping the `\r?` from the line break leaves the whole suite green, because a
# stray carriage return sits at the end of a segment and every classifier here
# works on stripped text; it stays for tidier spans, not for behaviour. And
# finding headers with `finditer` instead of `_scan` is likewise invisible: the
# decoy bypass needs a match that ENDS inside a later match's start, and a BEGIN
# marker cannot contain another BEGIN marker's opening. `_scan` stays because the
# reasoning that makes it safe is about today's literal, not about the rule.
_PEM_BEGIN = re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----")
_LINE_BREAK = re.compile(r"\r?\n|\\n")
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{16,}")
_HEX_ONLY = re.compile(r"[0-9a-fA-F]+")

_PEM_WINDOW = 1024
_PEM_MISS_TOLERANCE = 8
_PEM_DENSITY = 0.45
_PEM_LONG_RUN = 40
_MARKER_OVERLAP = 48
_PEM_SOLO_DENSITY = 0.9


def _segments(content: str, pos: int) -> Iterator[tuple[int, int]]:
    """The remainder of the line at `pos`, then every following line.

    A segment never exceeds `_PEM_WINDOW`, and that bound is on WORK rather than
    on reach: a line longer than the window is handed back in window-sized
    pieces, each classified on its own, so a single line of a megabyte is still
    covered in full. Searching for the next break across the whole remainder
    instead was quadratic on input with no newlines in it: 213 SECONDS on 1 MB of
    repeated headers, against 51 ms once the search is bounded.
    """
    while pos <= len(content):
        limit = min(len(content), pos + _PEM_WINDOW)
        brk = _LINE_BREAK.search(content, pos, limit)
        if brk is not None:
            yield pos, brk.start()
            pos = brk.end()
            continue
        yield pos, limit
        if limit >= len(content):
            return
        pos = limit


def _looks_like_key_material(window: str) -> bool:
    """Whether one segment reads as part of a key body rather than as prose.

    Two ways to qualify, and the second was added because the first alone lost a
    real shape. Half the segment being base64 separates a logged key line (64 of
    97 characters) from a sentence carrying a git SHA (40 of 108). But a key
    WRAPPED AT 16 COLUMNS and logged with a 33-character timestamp is 16 of 49,
    a third, and reads as prose by that rule: it leaked every body line until the
    axes were crossed and the case appeared.

    So a segment also qualifies when its base64 run ENDS it. That is true of any
    line that is a prefix plus key material, and false of a sentence, which
    carries on after its hash. It is safe to be this permissive per segment only
    because the block still has to clear the two-segment rule below: a single
    prose line that happens to end in a digest is not a key body.
    """
    # A run of nothing but hex digits is a hash, not key material: git SHAs,
    # sha256 image digests and MD5 sums are all hex, and prose is full of them.
    # Base64 uses 64 symbols of which 22 are hex, so a 64-character line of real
    # key material being accidentally all-hex has probability (22/64)**64, which
    # is zero for this purpose. Dropping hex-only runs is what stops a changelog
    # like "added in commit 3f2a... / reverted in commit 9c8b..." after a header
    # mention from reading as two lines of key body.
    runs = [
        match for match in _BASE64_RUN.finditer(window) if not _HEX_ONLY.fullmatch(match.group())
    ]
    covered = sum(len(match.group()) for match in runs)
    if covered < 16:
        return False
    # A single unbroken run of 40 or more is key material whatever surrounds it,
    # and this is the rule that carries the weight. PEM wraps at 64, 70 or 76
    # columns, so a body line's run clears 40 in every real format, while prose
    # does not carry 40-character unbroken base64 tokens.
    #
    # It replaced a density threshold as the primary test because density asks
    # the wrong question. A length bound asks "how far may I look", a property of
    # the KEY; density asks "how much of the line is key", a property of the
    # victim's LOGGING CONFIGURATION. That partitions deployments rather than
    # keys: measured across the eleven envelopes the fixtures carry, plain is
    # 1.00, syslog RFC5424 0.60, pino JSON 0.47, logback JSON 0.34 and GCP
    # logging 0.3459, so a
    # threshold of 0.5 sat at the MEDIAN of the real distribution and every shop
    # on logback leaked every key it ever logged while every shop on logfmt
    # leaked none. A length bound degrades gradually as keys grow; a density
    # bound does not degrade at all, it is binary per deployment.
    if any(len(match.group()) >= _PEM_LONG_RUN for match in runs):
        return True
    stripped = window.strip()
    if stripped and covered / len(stripped) >= _PEM_DENSITY:
        return True
    return bool(runs) and runs[-1].end() == len(window.rstrip())


def _private_key_spans(content: str) -> list[tuple[int, int]]:
    """Every PEM block, found by walking forward from each header."""
    spans: list[tuple[int, int]] = []
    for header in _scan(_PEM_BEGIN, content):
        start, body = header.span()
        last_good, accepted, misses = body, 0, 0
        strongest = 0.0
        longest = 0
        terminated = False

        for index, (seg_start, seg_end) in enumerate(_segments(content, body)):
            # Marker searches reach a little past the segment so that a marker
            # lying across a window boundary is still seen whole.
            reach = content[seg_start : min(len(content), seg_end + _MARKER_OVERLAP)]
            window = content[seg_start:seg_end]

            closing = _PEM_END.search(reach)
            if closing is not None:
                last_good = seg_start + closing.end()
                terminated = True
                break

            # Another header ends this block, which is also what keeps the walks
            # non-overlapping. Whatever sits BEFORE it is still classified: a
            # one-line key followed immediately by the next header would
            # otherwise lose its body.
            opening = _PEM_BEGIN.search(reach)
            if opening is not None:
                window = window[: opening.start()]

            if _looks_like_key_material(window):
                last_good = seg_start + len(window)
                accepted, misses = accepted + 1, 0
                stripped = window.strip()
                runs = [
                    m.group()
                    for m in _BASE64_RUN.finditer(window)
                    if not _HEX_ONLY.fullmatch(m.group())
                ]
                covered = sum(len(run) for run in runs)
                strongest = max(strongest, covered / len(stripped) if stripped else 0.0)
                longest = max(longest, max((len(run) for run in runs), default=0))
            elif opening is None and window.strip() and index > 0:
                # Two things do not count as evidence of leaving the block, and
                # both were off-by-ones that cost a tolerated line.
                #
                # An EMPTY segment: the newline straight after a header makes
                # one, and RFC 1421 puts a blank line before its body.
                #
                # The HEADER'S OWN LINE REMAINDER, which is segment zero. In JSON
                # or with any trailing text it is non-empty and not key material,
                # so it spent a miss and the recorded tolerance of eight was
                # really seven for exactly the envelopes that need it most.
                misses += 1
                if misses > _PEM_MISS_TOLERANCE:
                    break
            if opening is not None:
                break

        # A single accepted segment keeps the body when it is overwhelmingly
        # base64 OR when it carries a full-length run. The density half alone was
        # a leak of exactly the kind round five condemned: whether a one-line key
        # survived depended on how long its LOG WRAPPER was, so an envelope of 0,
        # 3 or 6 characters was covered and one of 11 or 33 characters lost the
        # whole key, while the line itself carried a 63-character run the whole
        # time. The run rule already knew it was key material; only this rule did
        # not ask it.
        solo = accepted == 1 and (strongest >= _PEM_SOLO_DENSITY or longest >= _PEM_LONG_RUN)
        spans.append((start, last_good if (terminated or accepted >= 2 or solo) else body))
    return spans


_VERSION = "0.1.0"


class SecretsGuardrail:
    """Detects credentials by their issuer prefix."""

    # Annotated with the Literal types, not bare assignments: a bare
    # `kind = "constraint"` infers `str`, and protocol attribute matching is
    # invariant, so it would not satisfy `kind: Kind`.
    name: str = "secrets"
    version: str = _VERSION
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(self, on_match: Decision = "redact") -> None:
        if on_match not in ("redact", "deny"):
            raise ValueError(f"on_match must be 'redact' or 'deny', got {on_match!r}")
        self._on_match: Decision = on_match

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every span every detector claims, in text order. Nothing is dropped.

        Spans rather than match objects, because PRIVATE_KEY is walked rather
        than matched and the two have to arrive in the same pipeline.

        Sorted by span, which is a total order over the pairs: a tie on the start
        offset puts the shorter match first, and `sorted` is stable, so two
        matches with an identical span keep pattern-table order.

        NO TEST CAN CURRENTLY SEE THAT, and it is recorded rather than assumed:
        sorting by start alone leaves the whole suite green. Every pattern here
        begins with a distinct literal, and the one pair that shares a prefix,
        ANTHROPIC_KEY and OPENAI_KEY, cannot match at the same offset, so no
        input produces a tie for the two keys to disagree about. Sorting by span
        is what keeps that true of a table someone adds to later, and `_merge`
        depends on it: it compares each span against the running end of the
        region it is extending.
        """
        found = [
            (secret_type, match.span())
            for secret_type, pattern in _PATTERNS
            for match in _scan(pattern, content)
        ]
        found += [("PRIVATE_KEY", span) for span in _private_key_spans(content)]
        return sorted(found, key=lambda pair: pair[1])

    def check(self, content: str, context: Context) -> Verdict:
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        # One finding per match, each carrying its own original span, even where
        # several collapsed into one placeholder. The output loses that detail by
        # design; the audit record must not.
        findings = [Finding(type=secret_type, span=span) for secret_type, span in found]

        # A deny carries no rewritten content: only a redact contributes spans to
        # the chain's rewrite, so nothing here can put the credential back into
        # the content the chain returns. The findings name types and spans, never
        # the matched text, so the credential is not in the audit record either.
        if self._on_match == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))

        # This detector's own view of the input, for a caller holding one
        # guardrail. A chain merges these spans with every other guardrail's and
        # rewrites once, through this same `_rewrite`.
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
