"""Span arithmetic shared by the detectors and by the chain.

Private by name and private on purpose. This is how a redaction finds and
combines spans, not part of the surface the package offers callers, and nothing
here is re-exported from the package root.

Two copies of this logic were defensible while there were two detectors. It
stopped being defensible when one Critical, `finditer` leaving whole credential
bodies standing, had to be repaired by typing the same fix into both files and
the same oracle fix into both test files. The next detector would have made
three copies.

It lives at the package root rather than under `detectors/` because `chain.py`
now merges spans too: a chain applies every guardrail's redactions in a single
pass, and that pass has to be the SAME merge the detectors use, not a fourth
copy of it. `detectors/__init__.py` imports `chain`, so leaving this module
under `detectors/` would have made the chain import through a package that
imports the chain.

The four units below are the only place a redaction decides WHICH bytes it
covers, so each one carries the reasoning for why it behaves the way it does.
None of it is style. Every paragraph is a bug that shipped or nearly shipped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class _Region:
    """One stretch of the input that a single placeholder replaces."""

    start: int
    end: int
    # A list in first-seen order, deliberately not a set. Nothing here may depend
    # on set iteration order, which varies with PYTHONHASHSEED: with a set, a build
    # that lost the sort below still produced sorted output most of the time, so no
    # test could pin the sort. It was pinned in about 92% of runs, which is not
    # pinned. First-seen order differs from sorted order for real input, which
    # makes the sort testable.
    types: list[str]

    def claim(self, type_name: str) -> None:
        """Record a type as contributing to this region, at most once."""
        if type_name not in self.types:
            self.types.append(type_name)

    @property
    def placeholder(self) -> str:
        """Names every type that claimed any part of the region, sorted to be stable."""
        names = "+".join(sorted(self.types))
        return f"[REDACTED:{names}]"


def _scan(pattern: re.Pattern[str], content: str) -> list[re.Match[str]]:
    """Every match of one pattern, trying EVERY start position.

    NOT `finditer`, and this is a Critical, not a preference. `finditer` resumes
    at the END of each match, so a decoy of the same shape joined directly in
    front of a real credential runs its greedy body past the real one's prefix,
    stops on the first character outside the body class, and leaves the scan
    resuming INSIDE the real credential. Its start is then never tried:

        'ghp_' + 40 chars + 'ghp_0000EXAMPLEONLY0000notarealtoken0000'
            ->  '[REDACTED:GITHUB_TOKEN]_0000EXAMPLEONLY0000notarealtoken0000'

    A whole 36-character body left standing behind a publicly known 4-character
    prefix. Measured the same way: AWS lost 19 of its 20 characters, both OPENAI
    branches their whole bodies, and a JWT its payload and signature.

    The same mechanism was live on the personal-data patterns, where a greedy
    domain runs into the next address's local part, stops at its `@`, and hides
    the real address entirely:

        'a@b.comalice@example.com'  ->  '[REDACTED:EMAIL]@example.com'

    The domain half of a working address, in the output of a redactor.

    Resuming at `match.start() + 1` tries every start, at no measured cost: the
    literal-prefix optimisation still applies, because `search` keeps it.

    The containment filter is NOT the overlap suppression that leaked before.
    Dropping a match CONTAINED in one already kept cannot uncover a character,
    since the container covers every offset it covered; partially overlapping
    matches, which is what was dropped before, are all kept. Starts increase and
    only a longer end is kept, so ends increase too, and testing the last kept
    match is enough.

    How much it does depends on the patterns, measured over the generated corpus
    each detector sweeps: 34,816 of 40,756 matches dropped on the personal-data
    patterns, where every suffix of a local part is itself an address, and 158 of
    5,449 on the credential patterns. On the personal-data side it is what keeps
    the audit record honest, since without it "alice@example.com" reports five
    findings for one address. On the credential side the shape that reaches it is
    two credentials of one type joined directly, so that the run is one match and
    the second key is another ending at the same offset; those two are then
    reported as ONE finding over the whole run. The redaction is unaffected; a
    corpus that scored recall by counting findings rather than checking coverage
    would miscount it, which is a note for Task 14.
    """
    matches: list[re.Match[str]] = []
    pos = 0
    while (match := pattern.search(content, pos)) is not None:
        if not matches or match.end() > matches[-1].end():
            matches.append(match)
        pos = match.start() + 1
    return matches


def _merge(found: Sequence[tuple[str, tuple[int, int]]]) -> list[_Region]:
    """Collapse overlapping spans into maximal regions, in text order.

    Overlapping spans are MERGED, never dropped. Two patterns that claim
    overlapping stretches of the input are each right about their own bytes, so
    keeping one and discarding the other leaves the discarded one's bytes in the
    output. That shipped once and leaked 15 of the 19 characters of a credit
    card. In a redactor an ambiguous span has to resolve toward more redaction,
    never less.

    Spans rather than match objects, because a detector may WALK a span rather
    than match it, as the PEM private-key body is walked, and the two have to
    arrive in the same pipeline.

    `found` must already be sorted by span. Spans that merely touch, where one
    ends exactly where the next begins, do not overlap and stay separate: the
    strict `<` is what keeps "a@b.com.a@b.com" two placeholders rather than one,
    and an AWS key butted against a PRIVATE KEY header two rather than one.

    A later span is tested against the region's RUNNING end, not against the end
    of the span that opened it, so a chain of pairwise overlaps collapses into a
    single region and no member of that chain can escape it.
    """
    regions: list[_Region] = []
    for type_name, (start, end) in found:
        if regions and start < regions[-1].end:
            regions[-1].end = max(regions[-1].end, end)
            regions[-1].claim(type_name)
        else:
            regions.append(_Region(start, end, [type_name]))
    return regions


def _rewrite(content: str, found: Sequence[tuple[str, tuple[int, int]]]) -> str:
    """Replace every merged region with its placeholder, in ONE pass over `content`.

    `found` must already be sorted by span, which is what `_merge` requires.

    One pass, and that is the whole point rather than an efficiency note. Both
    detectors wrote this loop out, and the chain needed a third copy the day it
    stopped threading each guardrail's rewrite into the next one. Rewriting
    twice is what let a personal-data placeholder cut a credential in half and
    leave its tail standing, so the operation that must never happen more than
    once is the one that should exist in exactly one place.

    Every offset it reads indexes into `content`. A caller that hands it spans
    computed against some OTHER string gets silent nonsense, so the chain
    validates spans against the content it inspected before calling this.
    """
    pieces: list[str] = []
    cursor = 0
    for region in _merge(found):
        pieces.append(content[cursor : region.start])
        pieces.append(region.placeholder)
        cursor = region.end
    pieces.append(content[cursor:])
    return "".join(pieces)
