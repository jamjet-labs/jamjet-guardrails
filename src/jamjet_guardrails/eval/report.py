"""Generate the published artifacts: benchmarks.json and BENCHMARKS.md.

Nothing here computes anything. Task 10 measured the guardrail, Task 12 refuses
to let the measurement regress, and this module only writes the two files a
reader sees. That is not low stakes. A formatter that drops a column, prints
one field under another's heading, or renders two artifacts that disagree
publishes a document that misstates a measured result: the number is right and
the claim is still false, and unlike a leak nobody can discover it by trying an
input.

Four rules do the work here, and each is a choice about which way an
uninformative case falls.

**A ratio never travels without its denominator and its support.** Precision
1.000 over one case and over five hundred render identically, so every table
publishes the counts behind the ratio, and the headline table publishes the
number of cases scored as well. The counts say what was FOUND; only the case
count says what was MEASURED. A corpus of clean negatives handled perfectly
produces the same three zeros as a corpus nobody scored.

**A rounded score never crosses a boundary it did not reach.** A measurement
strictly between 0 and 1 must not publish as 1.000 or 0.000 at any width.
Precision 9999/10000 beside a false positive, rendered 1.000, is a claim the
arithmetic contradicts on the same row. The two files guard this at their own
widths and therefore on different scores, and what they publish instead is a
bound rather than a measurement, so both facts are stated in the artifact
rather than left for a reader to infer from a figure that looks exact.

**Every escaper belongs to a context, and there are three.** An ordinary table
cell, a code span inside a table row, and a heading each need different
treatment, and the escaping that keeps one intact corrupts another. A value is
never handed to the escaper for the context next door.

**Nothing measured is stated, never implied by an absence.** An empty report,
a corpus with no per-type breakdown, a corpus with no misses, and a miss list
capped at zero each say so in words. A section that is simply skipped produces
the same document as a section that had nothing to report.

**Both artifacts derive from one measurement and one ordering.** Neither file
is rendered from the other's output: rounding an already rounded score moves it
in whichever direction the first rounding went, and the direction that flatters
is up. Two files that disagree about which miss came first are worse than one
file, because the reader has no way to tell which is the artifact and which is
the typo.

**Every cell is escaped.** Type names and case ids come out of a corpus, and a
third-party corpus is not ours to write. One unescaped pipe shifts every value
after it one column to the left, so precision publishes under the Recall
heading, and the row still looks like a table.
"""

from __future__ import annotations

from collections.abc import Sequence

from jamjet_guardrails.eval.metrics import Evaluation, Failure

# Decimals kept in benchmarks.json, and decimals shown in BENCHMARKS.md. BOTH
# round the measured value itself, and neither rounds the other's output.
#
# Rendering the display from the stored value instead is the tempting shortcut,
# because it makes the markdown equal to the JSON figure at display width by
# construction. It also rounds twice, and a double rounding moves a number in
# whichever direction the first rounding went. Recall 41/43 is 0.95348..., which
# is 0.953 at three decimals and 0.9535 at four, and 0.9535 rendered at three
# decimals is 0.954. That is a published score raised above what the measurement
# supports, on ordinary small counts, and up is the direction that flatters.
#
# The cost is that a reader who rounds the stored 0.9535 by hand gets 0.954 and
# the table says 0.953. Two roundings of one measurement at two widths, which is
# a narrower gap than publishing a number the measurement does not support.
_STORED = 4
_SHOWN = 3

# The interior values either width can publish. A measurement strictly inside
# (0, 1) that rounds onto a boundary is clamped to these rather than published
# on it: 999999/1000000 rounds to exactly 1.0 at four decimals, and a stored 1.0
# beside a false positive is a claim the counts on the same row contradict.
#
# A clamped figure is a BOUND, and clamping is not the modest direction at both
# ends. At the top it lowers the score, which is safe. At the bottom it RAISES
# it: a true precision of 5e-06 publishes as 0.0001, which is up, and up is the
# direction that flatters. The clamp stays anyway, because the alternative at
# that end is publishing 0.0 beside TP = 1, which is the same defect with a
# larger error and no notation warning the reader. The artifact says the figure
# is a bound and says the counts recover the exact ratio, which is what makes
# the overstatement recoverable rather than silent.
_STORED_CEILING = round(1.0 - 10.0**-_STORED, _STORED)
_STORED_FLOOR = round(10.0**-_STORED, _STORED)

# The markdown has the same defect at three decimals and cannot clamp, because
# 0.999 read as a score is as wrong as 1.000. It says which side of the boundary
# the value fell instead.
_SHOWN_CEILING = f">{1.0 - 10.0**-_SHOWN:.{_SHOWN}f}"
_SHOWN_FLOOR = f"<{10.0**-_SHOWN:.{_SHOWN}f}"

# Ranked by what a guardrail user loses. A wrong decision means it did not block
# the bad thing, which is the whole job; a missed finding is a leak; a false
# alarm is noise. A kind this module does not recognise sorts AHEAD of all
# three, because ranking it last would make it the first thing the miss cap
# hides, and an unrecognised failure is the last one worth hiding.
_UNRANKED = -1
_KIND_RANK = {"decision_mismatch": 0, "false_negative": 1, "false_positive": 2}

# Twelve columns, and a rule with twelve cells. A rule that does not match its
# header stops the whole table rendering as a table, which is why
# test_a_pipe_in_a_value_cannot_shift_a_column counts the rules too.
_OVERALL_HEADER = (
    "| Check | Corpus | Source | Version | Cases | Precision | Recall | F1 "
    "| TP | FP | FN | Wrong decisions |"
)
_OVERALL_RULE = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"

_NOTHING_MEASURED = (
    "# Benchmarks",
    "",
    "**No evaluations were run.** No check was measured against any corpus, so this",
    "file records no result. It is not a score of zero and it is not a pass; nothing",
    "was measured at all.",
)


def _stored(value: float) -> float:
    """The machine-readable figure: the measurement, at four decimals.

    Clamped off both boundaries, never onto one. An exact 1.0 stays 1.0, and an
    exact 0.0 stays 0.0; it is only a value that got there by rounding that is
    pulled back inside.
    """
    rounded = round(value, _STORED)
    if 0.0 < value < 1.0:
        if rounded >= 1.0:
            return _STORED_CEILING
        if rounded <= 0.0:
            return _STORED_FLOOR
    return rounded


def _shown(value: float) -> str:
    """The displayed figure: the measurement, at three decimals.

    The measurement, not ``_stored`` of it. See the note on ``_STORED``.

    The boundary test is on the RENDERED text rather than on a second rounding
    of the value, because the rendered text is the thing that would be wrong.
    """
    text = f"{value:.{_SHOWN}f}"
    if 0.0 < value < 1.0:
        if float(text) >= 1.0:
            return _SHOWN_CEILING
        if float(text) <= 0.0:
            return _SHOWN_FLOOR
    return text


def _inline(value: str) -> str:
    """Flatten anything that would end the line early.

    A newline inside a heading splits it in two, and inside a table row it ends
    the table. Corpus text is not printed, but corpus-authored ids and type
    names are.
    """
    for control in ("\r\n", "\r", "\n", "\t"):
        value = value.replace(control, " ")
    return value


# The ASCII punctuation characters that open a GFM inline construct, listed
# member by member with the construct each one opens. Not a claim about "every
# character GFM can read", which was a universal standing on four examples:
# tests/test_report.py walks this list literally, one value per member, so each
# one below is pinned by its own assertion and a member removed from the set
# fails on that member rather than on whichever example happened to cover it.
#
# `!` is NOT here. It is markup only immediately before `[`, and `[` is escaped
# unconditionally, so `!` can never open an image. It was added for completeness
# and removed again when its mutation survived: a member no value can pin is a
# member doing no work, and keeping it would restore the universal this comment
# exists to avoid.
#
#   \\  the escape itself   `  code span      *  emphasis
#   [ ] link               < > autolink and raw HTML
#   &  entity              ~  strikethrough   |  cell boundary
#
# Block constructs (#, -, +, digits) are absent because a table cell is parsed
# as inline content, so nothing here can start a heading or a list.
#
# `_` is deliberately NOT in the set. CommonMark does not open emphasis on an
# underscore between two alphanumerics, so `US_SSN` and `false_negative` are
# already literal, and escaping them would publish `US\\_SSN` into a file people
# read as raw text in diffs. The residual is a value that is entirely
# `_wrapped_`, which would render emphasised; it cannot shift a column, and it
# is the one case where this set trades fidelity for legibility.
_INLINE_SYNTAX = frozenset("\\`*[]<>&~|")

# The same set minus the pipe, for a heading. A heading is never split into
# cells, so `\|` there is not an escape a renderer strips, it is a backslash
# published next to a pipe. Derived from the cell set rather than written out
# again, so a member added to one is added to both.
_HEADING_SYNTAX = _INLINE_SYNTAX - {"|"}


def _cell(value: str) -> str:
    """Escape a value so it can neither become a column boundary nor markup.

    One pass, so the escape character and the characters being escaped cannot
    get out of order: escaping the pipe before the backslash would double the
    backslash of the escape itself and leave the pipe live again.

    The boundary half is what keeps the table a table. The markup half is what
    keeps the value a value: a case id of ``a`b`c`` published raw renders as
    ``abc``, and ``<img onerror=...>`` published raw is markup in a document
    this project asks people to trust. Corpus ids and type names reach here,
    and ``corpus.py`` validates both as non-empty strings and nothing more.
    """
    return "".join("\\" + char if char in _INLINE_SYNTAX else char for char in _inline(value))


def _code_cell(value: str) -> str:
    """A table cell for a value published as code. Returns the WHOLE cell.

    Two escaping rules apply to one string here and they disagree, so the
    backticks are conditional rather than the escaping.

    GFM splits a row on pipes first, consuming a backslash and the punctuation
    after it as a pair, and only then parses the cell. Inside a code span it
    strips ``\\|`` and processes no other escape. So a code span needs its
    backslashes left alone to render, and the splitter needs them doubled to
    stay out of the way. A value carrying ``\\|`` cannot satisfy both: doubling
    hands the splitter ``\\\\|``, which is an escaped BACKSLASH followed by a
    live cell boundary, and not doubling publishes the wrong text. There is no
    third encoding, because the pipe a code span renders must arrive as ``\\|``
    and the backslash before it must arrive as ``\\\\``.

    So values with no backslash in them get the code span, and the rest fall
    back to an ordinary cell, where ``_cell``'s doubling is exactly right: the
    inline parser turns ``\\\\`` back into one backslash, and the reader sees
    the same id ``benchmarks.json`` stores. Monospace is worth less than a
    column boundary and less than the right characters.

    A backtick falls back for the same reason: it would end the code span early.
    The rule is any backslash rather than an odd-numbered run of them before a
    pipe, because a rule a reader can check in one glance is worth more here
    than the last drop of monospace on a pathological id.
    """
    if "\\" in value or "`" in value:
        return _cell(value)
    escaped = _inline(value).replace("|", "\\|")
    return f"`{escaped}`"


def _heading(value: str) -> str:
    """Escape a value published as text inside a heading.

    A heading is inline content, so everything that is markup in a cell is
    markup here too, and the pipe is the single difference: it delimits nothing
    outside a table, so escaping it would publish a backslash a reader can see.

    Round 2's sweep recorded these sites as already correct, and that was true
    under the criterion of the day, which was "can this break a column". It
    cannot: a heading has no columns. When round 3 widened the criterion to
    "can this reach the page as markup" the old verdict expired, and the table
    was not re-judged, so `<img src=x onerror=alert(1)>` as a corpus name went
    on rendering raw in every section heading while the same value one line
    down was escaped correctly.
    """
    return "".join("\\" + char if char in _HEADING_SYNTAX else char for char in _inline(value))


def _code_heading(value: str) -> str:
    """A value published as code inside a HEADING rather than a table row.

    No pipe escaping. A heading is never split into cells, so the ``\\|`` that
    keeps a table row intact would publish a literal backslash here.
    """
    if "`" in value:
        # `_heading`, NOT `_cell`. The cell escaper would put `\|` into a
        # heading, which is the exact string this module's own test says must
        # never appear there; that test could not see it, because its fixture
        # has no backtick and so never reaches this branch.
        return _heading(value)
    return f"`{_inline(value)}`"


def _worst_first(failures: Sequence[Failure]) -> list[Failure]:
    """Order the misses so the cap truncates the least important ones.

    ``sorted`` is stable, so failures of one kind keep the order the corpus put
    them in. Both artifacts call this, which makes the markdown list a true
    prefix of the JSON one: "the rest are in benchmarks.json" then tells a
    reader where to carry on rather than where to start searching.
    """
    return sorted(failures, key=lambda f: _KIND_RANK.get(f.kind, _UNRANKED))


def to_json(evaluations: Sequence[Evaluation]) -> dict[str, object]:
    """The machine-readable artifact. Task 12's gate and any consumer read this.

    Built field by field rather than with ``dataclasses.asdict``, which cannot
    walk a frozen ``Evaluation`` at all: ``per_type`` is a ``MappingProxyType``,
    which is neither picklable nor JSON-serialisable. The comprehension over
    ``.items()`` yields a plain dict, which is.
    """
    return {
        "results": [
            {
                "detector": ev.detector,
                "detector_version": ev.detector_version,
                "corpus_name": ev.corpus_name,
                "corpus_source": ev.corpus_source,
                "corpus_version": ev.corpus_version,
                # The denominator, beside the identity of what it counts and
                # ahead of every ratio it qualifies. The three counts below say
                # what was found; this says what was scored.
                "cases": ev.cases,
                "precision": _stored(ev.overall.precision),
                "recall": _stored(ev.overall.recall),
                "f1": _stored(ev.overall.f1),
                "true_positives": ev.overall.true_positives,
                "false_positives": ev.overall.false_positives,
                "false_negatives": ev.overall.false_negatives,
                # Precision and recall count FINDINGS. A guardrail that locates
                # every span and then returns the wrong decision scores 1.000 on
                # both, and a corpus of negatives only scores 1.000 for a
                # detector that denies everything. This count is the independent
                # signal, and it is what a guardrail user actually cares about:
                # did it block the bad thing.
                "decision_mismatches": ev.decision_mismatches,
                # Support travels with every per-type score. Precision alone
                # cannot distinguish a type measured over one case from one
                # measured over five hundred, and a type with nothing to find
                # scores 1.0 by the empty-set convention.
                #
                # Sorted by type, like the markdown table, because both files
                # are committed and CI diffs them against a fresh run: an order
                # that follows whatever dict the caller built is a spurious diff
                # at best and two artifacts disagreeing at worst.
                "per_type": {
                    t: {
                        "precision": _stored(m.precision),
                        "recall": _stored(m.recall),
                        "true_positives": m.true_positives,
                        "false_positives": m.false_positives,
                        "false_negatives": m.false_negatives,
                    }
                    for t, m in sorted(ev.per_type.items())
                },
                # Every miss, not the capped list. This is the file the markdown
                # sends the reader to for the ones it did not print.
                "failures": [
                    {
                        "case_id": f.case_id,
                        "kind": f.kind,
                        "expected": f.expected,
                        "predicted": f.predicted,
                    }
                    for f in _worst_first(ev.failures)
                ],
            }
            for ev in evaluations
        ]
    }


def _guidance(max_misses: int) -> list[str]:
    """Everything a reader needs to not be misled by the table, AFTER the table.

    Each PARAGRAPH here is pinned by a test asserting one substring of it, and
    each was added because a figure above it could otherwise be read as
    something it is not. Paragraph granularity, not sentence: a sentence can be
    edited or dropped inside a pinned paragraph without failing anything, and
    eight of the sentences below are unpinned on their own. Saying "every
    sentence is pinned" would be the same universal-over-a-handful the
    ``_INLINE_SYNTAX`` comment was rewritten to remove, three functions up.

    The attribution paragraph is the one exception to "pinned in
    tests/test_report.py": its test lives in tests/test_corpora.py, beside the
    corpora whose licence asks for it, and it renders the real ones rather than
    a fixture. An earlier version of this paragraph counted the paragraphs
    below, and the count was wrong before this one was added to them. A number
    that has to be maintained by hand to stay true is not worth what it says.

    This is a lot of prose to put between a reader and the numbers they opened
    the file for, so it goes under them. Nothing is cut; the order is what
    changed.
    """
    return [
        "## How to read this",
        "",
        "Generated by CI. In-repo and third-party corpora are reported separately and",
        "never merged: numbers measured only on a corpus we wrote are self-graded.",
        "",
        # The pointer is here rather than in a hand-edited BENCHMARKS.md because
        # CI regenerates that file: a line added to the artifact by hand is
        # overwritten on the next run, and the licence condition would be
        # discharged only until then. It names no dataset, which this formatter
        # cannot know: the Source column above carries that, and the notice
        # carries the rest. CC BY 4.0 says in as many words that a link to a
        # resource holding the required information is a reasonable way to
        # attribute, which is what this is.
        "Provenance travels with the numbers. Every corpus's dataset, authors, URL and",
        "SPDX licence are recorded in `corpora/NOTICE.md`, and the Source column above",
        "names which one each row was measured on. Where a corpus licence requires",
        "attribution wherever the material is used, that file is where it is given,",
        "and a published figure is a use.",
        "",
        "Overall precision, recall and the wrong-decision count are **gated**: CI",
        "refuses a change that lowers a score beyond a small tolerance, or that gets",
        "one more decision wrong than last time. The per-type numbers and F1 are",
        "published rather than thresholded, and CI still requires the committed",
        "artifacts to match a fresh run, so any of them moving fails the build until a",
        "human commits the new value in the same pull request.",
        "",
        "Cases is how many labelled cases the check was scored on. TP, FP and FN are",
        "the findings matched, the findings reported that no label covers, and the",
        "labelled findings missed. The counts say what was found; the case count says",
        "what was measured, and a corpus of clean negatives handled perfectly scores",
        "0 / 0 / 0 on all three. The counts are published beside every ratio in both",
        "tables, and the case count beside the overall ratios, because 1.000 over one",
        "case and 1.000 over five hundred read alike.",
        "",
        "Each file rounds the measurement itself, once, at its own width: three",
        "decimals here, four in `benchmarks.json`. A figure here is NOT the stored one",
        "rounded again, so the two can differ by one in the third decimal, and when",
        "they do this file is the one that gave up the digit.",
        "",
        "Two further guards keep a score off a boundary it never reached, one per",
        "file, and they do NOT act on the same scores. This table shows",
        (
            f"`{_SHOWN_CEILING}` or `{_SHOWN_FLOOR}` for a score that would otherwise "
            "print as 1.000 or"
        ),
        (
            f"0.000 at three decimals. `benchmarks.json` clamps to {_STORED_CEILING} or "
            f"{_STORED_FLOOR} only"
        ),
        "when its own four-decimal figure would land on 1 or 0, which is a narrower",
        "set of scores: 3333/3334 is hedged here and stored there as 0.9997.",
        "",
        "A hedged or clamped figure is a bound, not a measurement, and the clamp is",
        "not always the modest direction: a precision of 0.000005 is stored as",
        f"{_STORED_FLOOR}, which is more than it earned. TP, FP and FN are published",
        "beside every ratio so that a bound can be checked. Where the denominator is",
        "not zero, precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)` and F1 is",
        "`2*TP/(2*TP+FP+FN)`, so any of the three can be recomputed exactly from the",
        "row it sits on.",
        "",
        "Where the denominator IS zero there is nothing to recompute, and the ratio",
        "is a convention rather than a measurement: it is 1.000 when the other side",
        "of it is empty too, and 0.000 when it is not. So a check that predicted",
        "nothing scores 1.000 for precision on a corpus with nothing to find, and",
        "0.000 on a corpus it missed everything in. Those two are told apart by",
        "their FN column, which is 0 on the first and not on the second. The first",
        "of them is the 0 / 0 / 0 row the Cases paragraph is about, and THERE the",
        "case count is the only published number that separates a clean sweep over",
        "40 cases from nothing measured at all over 0.",
        "",
        "Misses are listed worst first, wrong decisions ahead of missed findings ahead",
        f"of false alarms, capped at {max_misses} per corpus. Every miss is in",
        "`benchmarks.json`, and every score to four decimals there unless the sentence",
        "above clamped it.",
    ]


def _title(ev: Evaluation) -> str:
    """Which detector, on which corpus, from where, at which version.

    All four, because any three of them can repeat. Detector and source alone
    do not identify a corpus, and detector, name and source do not identify a
    VERSION of one: re-running a check against an edited corpus would otherwise
    head two sections identically, and the misses under them differ. The
    headline table separates those rows by its Version column, so the sections
    have to as well.
    """
    return (
        f"{_heading(ev.detector)} on {_heading(ev.corpus_name)} "
        f"({_heading(ev.corpus_source)}, {_code_heading(ev.corpus_version)})"
    )


def to_markdown(evaluations: Sequence[Evaluation], max_misses: int = 5) -> str:
    """The human-readable artifact, and the table the README carries.

    Raises:
        ValueError: for a negative ``max_misses``. Slicing accepts it and
            silently means "all but the last few", so a caller asking for a
            shorter list would get a differently truncated one instead.
    """
    if max_misses < 0:
        raise ValueError(f"max_misses must not be negative, got {max_misses}")

    # Read ONCE, before anything else looks at it. The four passes below are
    # four separate `for` loops, so an iterator would render the headline rows
    # and then find nothing left for the per-type and worst-misses sections:
    # every score published, every miss silently gone, and a document that looks
    # complete. The annotation says Sequence, and this module exists to be
    # called by people whose code the annotation does not check.
    # `detectors.build_chain` carries the same fix for the same reason.
    evaluations = list(evaluations)

    if not evaluations:
        # An empty table under a heading promising benchmarks reads as a
        # rendering fault at best, and as measured and fine at worst. The
        # guidance is dropped with it: it explains how to read numbers that are
        # not there.
        return "\n".join(_NOTHING_MEASURED) + "\n"

    # The table first, then how to read it. A reader who opened this file wants
    # a number, and a reader who wants the caveats will scroll.
    lines = [
        "# Benchmarks",
        "",
        "One row per check measured against one corpus, with the counts behind",
        "every score. How to read them is under the table.",
        "",
        _OVERALL_HEADER,
        _OVERALL_RULE,
    ]
    for ev in evaluations:
        lines.append(
            f"| {_cell(ev.detector)} | {_cell(ev.corpus_name)} | {_cell(ev.corpus_source)} | "
            f"{_code_cell(ev.corpus_version)} | {ev.cases} | "
            f"{_shown(ev.overall.precision)} | "
            f"{_shown(ev.overall.recall)} | {_shown(ev.overall.f1)} | "
            f"{ev.overall.true_positives} | {ev.overall.false_positives} | "
            f"{ev.overall.false_negatives} | {ev.decision_mismatches} |"
        )

    lines += [""] + _guidance(max_misses)

    for ev in evaluations:
        lines += ["", f"## Per type: {_title(ev)}", ""]
        if not ev.per_type:
            # Said rather than skipped. A missing section is indistinguishable
            # from a section that had nothing to report.
            lines.append("No findings of any type were expected or predicted on this corpus.")
            continue
        lines += [
            "| Type | Precision | Recall | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {_cell(t)} | {_shown(m.precision)} | {_shown(m.recall)} | "
            f"{m.true_positives} | {m.false_positives} | {m.false_negatives} |"
            for t, m in sorted(ev.per_type.items())
        ]

    for ev in evaluations:
        lines += ["", f"## Worst misses: {_title(ev)}", ""]
        if not ev.failures:
            lines.append("No misses on this corpus.")
            continue
        shown = _worst_first(ev.failures)[:max_misses]
        if shown:
            # A header with no rows under it is a table claiming to list
            # something, so it is only emitted when there is something in it.
            lines += ["| Case | Kind | Expected | Predicted |", "|---|---|---|---|"]
            lines += [
                f"| {_code_cell(f.case_id)} | {_cell(f.kind)} | {_cell(f.expected)} | "
                f"{_cell(f.predicted)} |"
                for f in shown
            ]
        remaining = len(ev.failures) - len(shown)
        if remaining > 0:
            lines += ["", f"...and {remaining} more, in `benchmarks.json`."]

    return "\n".join(lines) + "\n"
