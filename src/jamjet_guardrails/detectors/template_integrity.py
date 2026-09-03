"""Content that claims to be a turn, a role or a template of the conversation.

A chat model does not see a conversation. It sees one string, in which turn
boundaries are ordinary characters that the serving stack wrote: `<|im_start|>`
for a ChatML model, `[INST]` for Llama 2 and Mistral, `<start_of_turn>` for
Gemma. Retrieved content that carries those same characters is asking the model
to read the rest of it as a turn the operator never sent. Nothing about the
words has to be adversarial; the structure is the attack.

Three shapes of that claim, and each is structural rather than semantic:

- a delimiter a real model repository declares, read from its own tokenizer
  configuration at a pinned revision (`_template_markers.py`);
- a line that opens the way a transcript opens, with a privileged role and a
  colon;
- a tag that claims a privileged name the content has no authority to claim.

**The matching view is the design of this check.** Every signal reads a FOLDED
view of the content rather than the content: Default_Ignorable code points
removed, NFKD applied, and the UTS #39 confusable skeleton taken. So one zero
width space inside `<|im_start|>` does not save it, a fullwidth vertical line
does not save it, and neither does one Cyrillic letter. Spans map back through
`jamjet_guardrails._fold`, so a redaction removes the launderer's zero-width
space along with the marker rather than leaving it standing inside content the
verdict reports as rewritten.

NFKD rather than the NFKC the phase 3 design names, and the two are the same
question asked twice. NFKC is canonical COMPOSITION applied to NFKD, so two
strings have the same NFKC form exactly when they have the same NFKD form, and
this check only ever asks whether two strings are equal. Composition merges
characters, and a merge has no offset map in the direction this module needs:
one view character would carry several source indices, and `_Folded.origin`
holds one. Decomposing is the form of the same equivalence that keeps the map.

**No exemption is on by default.** `exempt_code_fences` is off, and the
documentation says it is a bypass rather than a refinement: an attacker can
wrap an injection in a fence and the model may obey it anyway. Documentation
quoting a marker therefore fires under the default, the corpus labels those
cases `allow`, and they are what the published precision pays for. That is the
trade this check makes on purpose and `corpora/NOTICE.md` names the cases.

Cost, and the shape of it. Linear in the content length on both of its paths.
29.7 ms median for one megabyte of the seeded input recorded in
`docs/performance.md`, on an Apple M3 Max under CPython 3.14.5, with the median
rising 3.8x to 4.0x per 4x of input from 4 KB upward and 3.7x on the first step,
where a 33 microsecond call is still mostly fixed cost. That input carries no
marker, so the figure is the folding pass and the three scans over it and none
of the offset map. Content that FIRES pays for the map as well, which is four
Python-level passes building one integer per view character, and it costs 31x:
923 ms for the same megabyte with one marker in it. `docs/performance.md`
records that second figure beside the first rather than hiding it in an average,
because a large document seeded with one marker is the input an attacker chooses
and the one that pays for it. `scripts/measure_throughput.py` reruns the first,
the page carries the snippet that reruns the second, and it states the machine,
the input and the method for both.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from jamjet_guardrails._fold import _Folded, compose, fold
from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.detectors._template_markers import MARKERS
from jamjet_guardrails.detectors.injection_structural import _DEFAULT_IGNORABLE
from jamjet_guardrails.errors import GuardrailUnavailableError
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

TEMPLATE_INTEGRITY_TYPES = frozenset(
    {
        "CHAT_TEMPLATE_MARKER",
        "ROLE_PREFIX_LINE",
        "FAKE_SYSTEM_TAG",
    }
)

_VERSION = "0.1.0"

# Deny in both directions. A marker in INPUT is retrieved content claiming a
# turn; a marker in OUTPUT is a model handing the next agent in the chain a
# forged boundary, which is the same claim one hop later. Neither is a value the
# caller wanted to keep with a hole cut in it, which is why this differs from
# `url-exfiltration`, where redaction on input keeps a page worth reading.
_DEFAULT_ON_DETECT: Mapping[Direction, Decision] = {"input": "deny", "output": "deny"}


# ==========================================================================
# The folded view
# ==========================================================================

# Every Default_Ignorable code point, imported from the check that CLAIMS them
# rather than tabulated again here. Two tables of "what renders as nothing"
# would disagree the first time one of them was tightened, and the disagreement
# is not cosmetic: this module's whole pairwise-disjointness argument with
# `injection-structural` is that every code point that check reports is one this
# fold DELETES, so no character can ever trigger a signal in both. One table is
# what makes that a property rather than a coincidence, and
# `test_template_integrity.py::test_no_character_injection_structural_claims_can_trigger_a_signal`
# asserts it.
_IGNORABLE: frozenset[int] = frozenset(
    point for low, high in _DEFAULT_IGNORABLE for point in range(low, high + 1)
)

# `str.translate` tables, both FINITE and both built once. A table that computed
# entries on demand through `__missing__` would be smaller to write and would
# grow one entry per distinct code point the caller sends, which for content an
# attacker chooses is a cache with a ceiling of the whole code space. These two
# hold exactly the code points the fold changes; every other character misses
# the table and `str.translate` leaves it alone, which is both the correct
# answer and the fast one.
_STRIP: dict[int, str] = {point: "" for point in _IGNORABLE}

_PROTOTYPE_TABLE: dict[int, str] | None = None


def _prototypes() -> dict[int, str]:
    """The confusables table as a `str.translate` table, built on first use.

    `jamjet_guardrails._unicode.confusables` is 176 KiB of source and 21 ms to
    compile, and `import jamjet_guardrails` must not pay for it: the registry
    imports every detector module, so an import at the top of this file would
    put that cost on every caller of every other check.
    `tests/test_unicode.py::test_importing_the_package_loads_no_unicode_table`
    holds that, and this indirection is what keeps it true.
    """
    global _PROTOTYPE_TABLE
    if _PROTOTYPE_TABLE is None:
        from jamjet_guardrails._unicode.confusables import PROTOTYPES

        _PROTOTYPE_TABLE = {ord(source): prototype for source, prototype in PROTOTYPES.items()}
    return _PROTOTYPE_TABLE


def _view_text(content: str) -> str:
    """The folded view, with no offset map and no per-character Python loop.

    Character for character the same string `_view` below produces, computed
    with the interpreter's own normaliser and two `str.translate` calls instead
    of four passes of `jamjet_guardrails._fold.fold`. It is 30x faster on a
    megabyte, and the reason to have both is that the map is what costs: content
    that matches nothing never needs to know which source character produced
    which view character, and that is nearly all content.

    WHY THE TWO AGREE, which is not an empirical claim about inputs anybody has
    tried. `unicodedata.normalize("NFKD", s)` is canonical ordering applied to
    the per-character compatibility decomposition of `s`, which is exactly what
    `fold(s, per-character NFKD)` produces and `skeleton` then reorders in its
    first NFD pass; mapping each character through the confusables table is a
    per-character rule either way; and the final NFD is `skeleton`'s own last
    step. So this is the same four operations in the same order, with the
    bookkeeping left out.
    `tests/test_template_integrity.py::test_the_fast_view_is_the_folded_view_character_for_character`
    sweeps both over the corpus and over the shapes that make normalisation
    interesting.
    """
    stripped = content.translate(_STRIP)
    decomposed = unicodedata.normalize("NFKD", stripped)
    return unicodedata.normalize("NFD", decomposed.translate(_prototypes()))


def _view(content: str) -> _Folded:
    """The folded view AND the way back to the content it was built from.

    Built only when a signal has already fired on `_view_text`, because the
    offset map is one integer per view character and nothing needs it until a
    span has to be reported.
    """
    from jamjet_guardrails._unicode import skeleton

    stripped = fold(content, lambda character: "" if ord(character) in _IGNORABLE else character)
    decomposed = fold(stripped.text, _compatibility_decomposition)
    return compose(compose(stripped, decomposed), skeleton(decomposed.text))


def _compatibility_decomposition(character: str) -> str:
    """NFKD of ONE character, which is that character's full decomposition."""
    return unicodedata.normalize("NFKD", character)


# ==========================================================================
# CHAT_TEMPLATE_MARKER
# ==========================================================================

_MARKER_PATTERN: re.Pattern[str] | None = None


def _marker_pattern() -> re.Pattern[str]:
    """One alternation of every marker in the table, FOLDED at load.

    Folded here rather than in `_template_markers.py`, which stores every marker
    exactly as the repository declared it. A generated table that had already
    been through NFKD and a confusable skeleton would carry one interpreter's
    Unicode version in its bytes and would no longer show the string a source
    actually declares, and the byte-identity guard on that module would then be
    holding a normalisation rather than a fetch.

    Longest first, so `finditer` cannot report a shorter marker where a longer
    one starts at the same offset. No folded marker is a substring of another
    today, which
    `tests/test_template_integrity.py::test_no_folded_marker_hides_inside_another`
    asserts, so the ordering changes nothing now; it is here because the day the
    table gains one that is, the alternative is a finding whose span covers half
    a marker.
    """
    global _MARKER_PATTERN
    if _MARKER_PATTERN is None:
        folded = sorted(
            {_view_text(marker) for marker in MARKERS}, key=lambda text: (-len(text), text)
        )
        _MARKER_PATTERN = re.compile("|".join(re.escape(text) for text in folded))
    return _MARKER_PATTERN


def _marker_spans(view: str) -> list[tuple[int, int]]:
    """Every marker occurrence in the view, in view coordinates."""
    return [(match.start(), match.end()) for match in _marker_pattern().finditer(view)]


# ==========================================================================
# ROLE_PREFIX_LINE and FAKE_SYSTEM_TAG
# ==========================================================================

# The roles an attacker gains something by claiming. `user` and `human` are
# DELIBERATELY absent: content that impersonates the user is asking the model to
# treat it as what it already is, so the shape carries no privilege, and the
# populations it would hit are enormous -- `user:` is a line in every database
# configuration, every connection string dump and every transcript quoted in a
# bug report. The corpus carries those as negatives.
_ROLE_LABELS: tuple[str, ...] = ("system", "developer", "assistant")

# The same three, plus the four nouns a fabricated container is named after. A
# tag is a weaker claim than a turn boundary, so the vocabulary is wider: a
# model asked to obey `<system_prompt>` inside retrieved content has been handed
# a structure the content had no authority to declare.
_TAG_LABELS: tuple[str, ...] = _ROLE_LABELS + ("instruction", "prompt", "rules", "policy")


def _folded_class(character: str) -> str:
    """A regex class matching this character's folded form in ANY case.

    The lexicon above is written in lowercase and the view is not case folded,
    so a label has to be matched in whatever case the content wrote it. That
    cannot be `re.IGNORECASE` over the plain word, because the fold is not
    case-preserving: the confusables table maps lowercase `m` to `rn` and leaves
    uppercase `M` alone, so `system` arrives in the view as `systern` and
    `SYSTEM` arrives as `SYSTEM`, and no case-insensitive spelling of one is a
    spelling of the other. It maps `I` to `l` and leaves `i` alone, which is the
    same problem one letter over and also the reason `1` matches here: the table
    maps the digit to `l` as well, so leetspeak is folded out for free rather
    than by a rule about digits.

    Per CHARACTER, so this is linear in the label rather than exponential in how
    many of its letters the fold treats asymmetrically.
    """
    forms = {_view_text(form) for form in (character, character.lower(), character.upper())}
    return "(?:" + "|".join(re.escape(form) for form in sorted(forms)) + ")"


def _folded_word(word: str) -> str:
    """A regex source matching `word` as it appears in the view, in any case."""
    return "".join(_folded_class(character) for character in word)


_ROLE_PATTERN: re.Pattern[str] | None = None
_TAG_LABEL_PATTERN: re.Pattern[str] | None = None


def _role_pattern() -> re.Pattern[str]:
    """A line that opens with a privileged role and a colon.

    `[ \\t]` and not `\\s` for the leading run: `\\s` matches a newline, so a
    blank line would be consumed as this line's own indentation and the position
    test below would then be asked about the wrong line.
    """
    global _ROLE_PATTERN
    if _ROLE_PATTERN is None:
        labels = "|".join(_folded_word(label) for label in _ROLE_LABELS)
        _ROLE_PATTERN = re.compile(rf"^[ \t]*({labels})[ \t]*:", re.MULTILINE)
    return _ROLE_PATTERN


def _tag_label_pattern() -> re.Pattern[str]:
    global _TAG_LABEL_PATTERN
    if _TAG_LABEL_PATTERN is None:
        _TAG_LABEL_PATTERN = re.compile("|".join(_folded_word(label) for label in _TAG_LABELS))
    return _TAG_LABEL_PATTERN


# An XML-shaped tag, and nothing looser. The name must follow `<` or `</` with
# no space between, because `< system >` is prose about a system and `<system>`
# is a claim to be one, and every renderer that matters agrees: HTML and XML
# both refuse whitespace after the opening angle bracket. Loosening it is what
# would put every `a < b and c > d` comparison in the corpus.
#
# `[^\W\d_]` is "a letter" derived from the character classes rather than spelled
# as a range, so a tag named in a non-Latin script is a tag here too.
_TAG = re.compile(r"</?([^\W\d_][\w.:-]*)(?:[ \t\r\n][^<>]*)?/?>")


def _claims_a_role(name: str) -> bool:
    """Whether a tag name claims one of the privileged names.

    Underscores and hyphens are removed before the test, so `<system_prompt>`,
    `<system-prompt>` and `<systemprompt>` are one claim written three ways and
    an attacker gains nothing by picking the spelling the check has not seen.
    Containment rather than equality, for the same reason: `<systemPrompt>` and
    `<assistant_instructions>` are the shapes this is written for, and a rule
    that demanded the whole name would catch neither.

    What containment costs is real and is not hidden: `<policyholder>` and
    `<systemd_unit>` are tags in ordinary documents and both fire.
    `corpora/NOTICE.md` names them with the measurement.
    """
    return _tag_label_pattern().search(name.replace("_", "").replace("-", "")) is not None


def _tag_spans(view: str) -> list[tuple[int, int]]:
    """Every privileged-looking tag in the view, in view coordinates."""
    return [
        (match.start(), match.end())
        for match in _TAG.finditer(view)
        if _claims_a_role(match.group(1))
    ]


def _opens_a_turn(view: str, line_start: int) -> bool:
    """Whether a line at `line_start` sits where a transcript turn would.

    At the start of the content, or after a BLANK line, and nowhere else. That
    restriction is the whole precision of this signal and it is not free in
    either direction.

    What it buys: `System:` and `Assistant:` are field labels in a
    specification, a key in a configuration block, a run-in heading in a
    paragraph, and a caption under a diagram, and in every one of those shapes
    the line before them carries text. A rule that fired on any line-initial
    role label would report each of those, and the population of documents
    holding one is every document.

    What it costs, stated rather than discovered later: content that writes
    `system:` on the line directly after a sentence is not reported, and an
    attacker who reads this paragraph will delete a blank line.
    `corpora/NOTICE.md` discloses that as a residual with the case that shows
    it, because a hole named with its cost is a boundary and one left unnamed is
    a channel.
    """
    if line_start == 0:
        return True
    # `view[:line_start]` ends with the newline that opened this line, because
    # the caller found `line_start` with a MULTILINE `^`.
    preceding = view[: line_start - 1]
    return preceding[preceding.rfind("\n") + 1 :].strip() == ""


def _role_spans(view: str) -> list[tuple[int, int]]:
    """Every role-prefix line in the view, in view coordinates.

    The span runs from the LABEL to the colon and does not include the leading
    indentation. A redaction then replaces the claim and leaves the line's shape
    alone, which is the same rule `secrets` follows in covering the credential
    rather than the run it sits in.
    """
    return [
        (match.start(1), match.end())
        for match in _role_pattern().finditer(view)
        if _opens_a_turn(view, match.start())
    ]


# ==========================================================================
# The code-fence exemption
# ==========================================================================

# An opening or closing fence: up to three spaces of indentation, then three or
# more backticks or three or more tildes. CommonMark's own rule, and the
# indentation allowance is part of it rather than a kindness: a fence inside a
# list item is indented.
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*$", re.MULTILINE)

# A run of backticks. A code span opens on a run of N and closes on the next run
# of EXACTLY N, which is what lets ``a ` b`` hold a literal backtick.
_TICKS = re.compile(r"`+")


def _fenced_regions(content: str) -> list[tuple[int, int]]:
    """Fenced code blocks, as spans of the ORIGINAL content.

    Over the content and not the view, deliberately. A fence is a rendering
    feature, and a fence written with a laundered backtick is not a fence to any
    renderer either, so recognising one would exempt a region no reader would
    see as code. The exemption has to agree with the renderer or it is exempting
    something else.
    """
    regions: list[tuple[int, int]] = []
    opening: re.Match[str] | None = None
    for match in _FENCE.finditer(content):
        if opening is None:
            opening = match
            continue
        marker = opening.group(1)
        closing = match.group(1)
        if closing[0] == marker[0] and len(closing) >= len(marker):
            regions.append((opening.start(), match.end()))
            opening = None
    if opening is not None:
        # An unclosed fence runs to the end of the content. That is CommonMark's
        # rule and it is also the safe reading here: the alternative is deciding
        # the fence never opened, which would make an attacker's trailing
        # backticks turn the exemption off for the text underneath it.
        regions.append((opening.start(), len(content)))
    return regions


def _inline_regions(content: str, fenced: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Inline code spans, outside the fenced blocks already found."""
    regions: list[tuple[int, int]] = []
    runs = [
        match
        for match in _TICKS.finditer(content)
        if not any(start <= match.start() < end for start, end in fenced)
    ]
    index = 0
    while index < len(runs):
        opening = runs[index]
        width = len(opening.group())
        for offset in range(index + 1, len(runs)):
            if len(runs[offset].group()) == width:
                regions.append((opening.start(), runs[offset].end()))
                index = offset + 1
                break
        else:
            index += 1
    return regions


def _code_regions(content: str) -> list[tuple[int, int]]:
    fenced = _fenced_regions(content)
    return fenced + _inline_regions(content, fenced)


def _inside(span: tuple[int, int], regions: Sequence[tuple[int, int]]) -> bool:
    """Whether a span lies WHOLLY inside one region.

    Wholly, not overlapping. A marker half in and half out of a code span is not
    quoted code, it is a marker that happens to start next to a backtick, and
    exempting it would make the exemption reachable by writing one.
    """
    return any(start <= span[0] and span[1] <= end for start, end in regions)


class TemplateIntegrityGuardrail:
    """Detects content claiming to be a turn, a role or a template of the chat."""

    name: str = "template-integrity"
    version: str = _VERSION
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(
        self,
        on_detect: Decision | Mapping[Direction, Decision] | None = None,
        exempt_code_fences: bool = False,
    ) -> None:
        """Refuses, at construction, every configuration that would check less.

        - `exempt_code_fences` given anything but a real `bool`. A configuration
          file that hands this option the STRING `"false"` would otherwise turn
          the bypass on, because every non-empty string is truthy, and the only
          symptom would be a check that quietly stopped reporting quoted
          markers. `types` and `directions` are not options here at all, so this
          is the one place a caller can narrow what runs.
        - an `on_detect` mapping that omits a direction, or names one this
          guardrail does not declare: the first is a `KeyError` from inside
          `check`, which fails closed and names nothing; the second is a policy
          silently dropped.
        - `allow` as a decision. A check configured to allow on a detection is a
          check that runs and cannot act.
        """
        if not isinstance(exempt_code_fences, bool):
            raise ValueError(  # noqa: TRY004
                f"exempt_code_fences must be a bool, not the "
                f"{type(exempt_code_fences).__name__} {exempt_code_fences!r}; every "
                "non-empty string is truthy, so a configuration file that spells this "
                "option 'false' would switch the bypass on"
            )

        if on_detect is None:
            resolved: dict[Direction, Decision] = dict(_DEFAULT_ON_DETECT)
        elif isinstance(on_detect, str):
            resolved = {direction: on_detect for direction in self.directions}
        else:
            missing = sorted(self.directions - set(on_detect))
            if missing:
                raise GuardrailUnavailableError(
                    f"template-integrity declares directions {sorted(self.directions)} but "
                    f"on_detect names no decision for {missing}; the alternative is a "
                    "KeyError from inside check, which fails closed and names nothing"
                )
            extra = sorted(set(on_detect) - self.directions)
            if extra:
                raise GuardrailUnavailableError(
                    f"template-integrity declares directions {sorted(self.directions)} but "
                    f"on_detect also names {extra}, which this guardrail would never be "
                    "asked about; a policy for a direction it does not declare would be "
                    "silently dropped"
                )
            resolved = {direction: on_detect[direction] for direction in self.directions}
        for direction, decision in sorted(resolved.items()):
            if decision not in ("redact", "deny"):
                raise ValueError(
                    f"on_detect for {direction!r} must be 'redact' or 'deny', got "
                    f"{decision!r}. A check configured to allow on a detection is a "
                    "check that runs and cannot act"
                )

        self._on_detect: Mapping[Direction, Decision] = resolved
        self._exempt_code_fences = exempt_code_fences

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every span every signal claims, sorted by span, deduplicated by pair.

        SORTED BY SPAN is a precondition of `_spans._merge`, which tests each
        span against the running end of the region it is extending and looks no
        further back. The three signals each scan the whole view independently,
        so their results concatenate in signal order, which is not span order.

        The early return is where nearly every call ends, and it is why this
        check costs what it does. `_view_text` produces the same view as `_view`
        and carries no offset map; content in which no signal fires needs no map
        and never builds one. Content in which one does pays for the map once,
        and the three scans run once each over the mapless view rather than
        again over the mapped one.
        """
        view_text = _view_text(content)
        markers = _marker_spans(view_text)
        roles = _role_spans(view_text)
        # A tag lying WHOLLY inside a marker is that marker, reported once. The
        # confusable fold maps the vertical line to `l`, so `<|system|>` reaches
        # the tag scan as `<lsysternl>`, which is a tag name containing a role
        # label, and three table entries hit it: `<|assistant|>`, `<|system|>`
        # and `<｜Assistant｜>`. Without this filter each of those is two findings
        # over one span, which is the same defect `_spans._scan` drops a
        # contained match for: one construct in the content, several lines in
        # the audit record. Dropping it cannot uncover a character, because the
        # marker span covers every offset the tag span did, and it cannot lower
        # a verdict either, because the marker is reported over that same run.
        #
        # The one place it changes an outcome is with `exempt_code_fences` on
        # and a pipe-delimited role marker inside a fence: the marker is
        # exempted and no tag stands in for it. That is the fence bypass doing
        # exactly what it is documented to do, and it is why this filters
        # against every marker rather than only the kept ones -- filtering
        # against the kept ones would make the option turn one type into
        # another, which is worse to explain and no safer.
        tags = [span for span in _tag_spans(view_text) if not _inside(span, markers)]
        if not (markers or roles or tags):
            return []

        view = _view(content)
        exempt = _code_regions(content) if self._exempt_code_fences else ()
        found: dict[tuple[tuple[int, int], str], None] = {}
        for start, end in markers:
            span = view.span(start, end)
            # The exemption covers markers ALONE, which is what the phase 3
            # design says and is narrower than the reason it gives for existing.
            # An exemption is a candidate bypass, so it is written to the word:
            # a role-prefix line, and a fake system tag that is not itself part
            # of a marker, still fire inside a fence with the option on, so this
            # option cannot be used to get either of them past the check.
            if not _inside(span, exempt):
                found[(span, "CHAT_TEMPLATE_MARKER")] = None
        for start, end in roles:
            found[(view.span(start, end), "ROLE_PREFIX_LINE")] = None
        for start, end in tags:
            found[(view.span(start, end), "FAKE_SYSTEM_TAG")] = None
        return sorted(((name, span) for span, name in found), key=lambda pair: pair[1])

    def check(self, content: str, context: Context) -> Verdict:
        """Refuses a direction this guardrail does not declare, before matching.

        The chain filters on `directions` before it calls `check`, and a caller
        holding one guardrail does not, so this method holds the line the chain
        holds for it. Answering `allow` for a direction this guardrail never
        declared would report that the content was checked.
        """
        if context.direction not in self.directions:
            raise GuardrailUnavailableError(
                f"{self.name!r} was asked to check direction {context.direction!r} but "
                f"declares only {sorted(self.directions)}; answering would report that "
                "content was checked in a direction this guardrail never declared"
            )
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        findings = [Finding(type=type_name, span=span) for type_name, span in found]
        if self._on_detect[context.direction] == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
