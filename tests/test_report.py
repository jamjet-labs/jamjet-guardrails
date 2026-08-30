"""The published artifacts: benchmarks.json and BENCHMARKS.md.

The brief's fifteen tests come first, verbatim apart from two adaptations mypy
strict forces on them:

  * every test and helper is annotated, and
  * ``to_json`` returns ``dict[str, object]``, so ``payload["results"][0]`` is
    an index into ``object`` and does not typecheck. The three ``_entry`` /
    ``_mapping`` / ``_number`` helpers below narrow it. The assertions they feed
    are the brief's, unchanged.
"""

import json
from collections.abc import Callable, Mapping
from typing import cast

import pytest

from jamjet_guardrails.eval.metrics import Evaluation, Failure, FailureKind, Metrics
from jamjet_guardrails.eval.report import to_json, to_markdown


def _ev(
    source: str = "in-repo",
    failures: list[Failure] | None = None,
    name: str = "mailbox-set",
) -> Evaluation:
    # corpus_name deliberately shares no substring with corpus_source or
    # corpus_version. A fixture whose name contains its source cannot tell you
    # which of the two a report column actually rendered.
    return Evaluation(
        corpus_name=name,
        corpus_version="abc123def456",
        corpus_source=source,
        detector="pii",
        detector_version="0.1.0",
        cases=20,
        overall=Metrics(true_positives=9, false_positives=1, false_negatives=1),
        per_type={"EMAIL": Metrics(9, 1, 1)},
        failures=[Failure("pii-0007", "false_negative", "EMAIL@(3, 9)", "nothing")]
        if failures is None
        else failures,
    )


def _entries(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    results = payload["results"]
    assert isinstance(results, list)
    for entry in results:
        assert isinstance(entry, dict)
    return results


def _entry(payload: Mapping[str, object]) -> Mapping[str, object]:
    return _entries(payload)[0]


def _mapping(within: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = within[key]
    assert isinstance(value, dict)
    return value


def _number(within: Mapping[str, object], key: str) -> float:
    value = within[key]
    assert isinstance(value, float)
    return value


def _headline(md: str, column: str) -> str:
    """The value under a named column of the headline table's first data row.

    By NAME, looked up in the rendered header, so a test asking for Precision
    cannot silently start reading Recall when a column is inserted. That is the
    failure this whole module is about.
    """
    header = [cell.strip() for cell in _cells(_rows(md)[0])]
    row = [cell.strip() for cell in _cells(_rows(md)[2])]
    assert len(header) == len(row), f"header and row disagree: {header} vs {row}"
    return row[header.index(column)]


def test_json_records_detector_source_and_corpus_version() -> None:
    entry = _entry(to_json([_ev()]))
    assert entry["detector"] == "pii"
    assert entry["corpus_source"] == "in-repo"
    assert entry["corpus_version"] == "abc123def456"
    assert entry["precision"] == 0.9
    assert entry["recall"] == 0.9


def test_json_keeps_sources_separate_never_merged() -> None:
    payload = to_json([_ev("in-repo"), _ev("third-party")])
    assert [r["corpus_source"] for r in _entries(payload)] == ["in-repo", "third-party"]


def test_markdown_reports_each_source_on_its_own_row() -> None:
    md = to_markdown([_ev("in-repo"), _ev("third-party")])
    assert "in-repo" in md and "third-party" in md
    assert md.count("| pii |") == 2


def test_markdown_publishes_the_worst_misses() -> None:
    md = to_markdown([_ev()])
    assert "Worst misses" in md
    assert "pii-0007" in md
    assert "false_negative" in md


def test_markdown_caps_the_miss_list_and_says_how_many_were_omitted() -> None:
    failures = [Failure(f"c{i}", "false_negative", "EMAIL", "nothing") for i in range(9)]
    md = to_markdown([_ev(failures=failures)], max_misses=3)
    assert "c0" in md
    assert "c8" not in md
    assert "6 more" in md


def test_markdown_says_so_when_there_are_no_misses() -> None:
    assert "No misses" in to_markdown([_ev(failures=[])])


def test_json_is_serialisable() -> None:
    json.dumps(to_json([_ev()]))


def test_markdown_row_carries_name_source_and_version_as_distinct_columns() -> None:
    """Three separate facts, sharing no substring in the fixture, so dropping
    any one of them fails here. A published row that cannot say which corpus it
    measured is not evidence."""
    md = to_markdown([_ev(source="third-party", name="mailbox-set")])
    assert "mailbox-set" in md
    assert "third-party" in md
    assert "abc123def456" in md


def test_json_publishes_support_counts_per_type() -> None:
    """A precision of 1.000 over one case and over five hundred must not render
    alike. Under the empty-set convention a type with nothing to find scores
    perfectly, so a per-type score without its support is unfalsifiable."""
    per_type = _mapping(_mapping(_entry(to_json([_ev()])), "per_type"), "EMAIL")
    assert per_type["true_positives"] == 9
    assert per_type["false_positives"] == 1
    assert per_type["false_negatives"] == 1


def test_markdown_shows_per_type_support() -> None:
    # The whole row, not a bare "9": precision renders as 0.900, so a substring
    # check for "9" passes with every support column dropped.
    md = to_markdown([_ev()])
    assert "| EMAIL | 0.900 | 0.900 | 9 | 1 | 1 |" in md


def test_the_two_artifacts_agree_to_within_half_a_display_unit() -> None:
    """Two published artifacts, one set of numbers.

    RESTATED from the brief, which asserted that the markdown contains the JSON
    figure rendered at display width. That holds only for a fixture like this
    one, scoring 0.900, where every rounding of every rounding agrees. It is
    false by design for 41/43, where the stored figure is 0.9535 and the table
    says 0.953, and asserting it would contradict
    test_neither_artifact_is_rendered_from_the_other_and_rounded_twice one
    screen below.

    What actually holds is the invariant that matters to a reader comparing the
    two: each file rounds the same measurement at its own width, so they cannot
    differ by more than those two widths can express.
    """
    for ev in (_ev(), _built(Metrics(41, 2, 2)), _built(Metrics(7, 2, 3))):
        entry = _entry(to_json([ev]))
        md = to_markdown([ev])
        for key, column in (("precision", "Precision"), ("recall", "Recall")):
            # Half of the four-decimal width plus half of the three-decimal
            # width. Anything larger means one of them is not the measurement.
            assert abs(_number(entry, key) - float(_headline(md, column))) <= 0.00055


def test_the_artifact_states_its_own_rounding_convention() -> None:
    """A reader who checks the table against `benchmarks.json` finds 0.953
    beside a stored 0.9535 and cannot tell which one is the typo. The reasoning
    for that gap lives in this module's source, where no reader of BENCHMARKS.md
    ever goes, so the artifact has to carry it as well.

    The boundary notations are here for the same reason: a cell reading >0.999
    is unreadable unless the file says what it means.
    """
    md = to_markdown([_ev()])
    assert "rounds the measurement itself" in md
    assert ">0.999" in md
    assert "<0.001" in md
    assert "0.9999" in md
    assert "do NOT act on the same scores" in md
    assert "is a bound, not a measurement" in md
    # The escape hatch the bound depends on: a reader is told they can
    # recompute the exact value, so all three formulas have to be on the page.
    # F1 is hedgeable too, and giving two of the three formulas while claiming
    # every bound is checkable is how the first version of this paragraph came
    # to be false.
    assert "TP/(TP+FP)" in md
    assert "TP/(TP+FN)" in md
    assert "2*TP/(2*TP+FP+FN)" in md
    assert "Where the denominator IS zero" in md
    # And the four-decimal promise has to carry its exception, because a clamped
    # figure is not the four-decimal rounding of anything.
    assert "four decimals there unless" in md


def test_the_two_boundary_guards_have_independent_thresholds() -> None:
    """The claim this replaced said `benchmarks.json` clamps "the same cases"
    the table hedges. It does not. The table hedges from 0.9995 up, where the
    figure would print as 1.000 at three decimals; the JSON clamps only from
    0.99995 up, where its own four-decimal figure would land on 1.

    3333/3334 falls between the two, so it is hedged here and stored there as
    0.9997. A reader following the old sentence to cross-check a `>0.999` would
    expect 0.9999, find 0.9997, and conclude the files disagree, which is the
    exact failure the paragraph exists to prevent. The preamble now names this
    ratio, and this test is what makes that a checked claim.
    """
    high = _built(Metrics(3333, 1, 0))
    assert _headline(to_markdown([high]), "Precision") == ">0.999"
    assert _number(_entry(to_json([high])), "precision") == 0.9997

    low = _built(Metrics(1, 2000, 0))
    assert _headline(to_markdown([low]), "Precision") == "<0.001"
    assert _number(_entry(to_json([low])), "precision") == 0.0005


def test_a_hedged_or_clamped_figure_is_a_true_bound() -> None:
    """`>0.999` has to mean the score really is above 0.999, at both widths and
    on every score that reaches the notation, or the notation is just a
    different way of being wrong.
    """
    for counts in (Metrics(3333, 1, 0), Metrics(9999, 1, 1), Metrics(999999, 1, 1)):
        ev = _built(counts)
        assert _headline(to_markdown([ev]), "Precision") == ">0.999"
        assert _number(_entry(to_json([ev])), "precision") > 0.999
        assert counts.precision > 0.999
    for counts in (Metrics(1, 2000, 0), Metrics(1, 9999, 0), Metrics(1, 999999, 0)):
        ev = _built(counts)
        assert _headline(to_markdown([ev]), "Precision") == "<0.001"
        assert _number(_entry(to_json([ev])), "precision") < 0.001
        assert counts.precision < 0.001


def test_the_counts_beside_a_clamped_figure_recover_the_exact_ratio() -> None:
    """The clamp is NOT the modest direction at the bottom. A true precision of
    5e-06 publishes as 0.0001, which is up, and up flatters. The clamp stays,
    because the alternative is 0.000 beside TP = 1, a larger error with no
    notation warning anyone. What makes the overstatement recoverable is the
    artifact's promise that the counts beside it recover the exact ratio, so
    that promise is checked here rather than merely printed.
    """
    ev = _built(Metrics(1, 199999, 0))
    entry = _entry(to_json([ev]))
    md = to_markdown([ev])
    assert _number(entry, "precision") == 0.0001

    tp, fp = entry["true_positives"], entry["false_positives"]
    assert isinstance(tp, int) and isinstance(fp, int)
    exact = tp / (tp + fp)
    assert exact == 5e-06
    assert exact < _number(entry, "precision"), "the clamp is supposed to overstate here"
    # Published beside the figure, in the row a reader is looking at.
    assert _headline(md, "TP") == "1"
    assert _headline(md, "FP") == "199999"


def test_the_published_formulas_reproduce_the_ratios_they_claim_to() -> None:
    """The artifact gives three formulas and scopes them to rows whose
    denominator is not zero. Checked against the module, including on a row
    where the published figure is a clamped bound rather than the value: that
    is the row where a reader most needs the formula, and the one the claim
    exists for.
    """
    for counts in (Metrics(41, 3, 2), Metrics(7, 2, 3), Metrics(999999, 1, 1)):
        entry = _entry(to_json([_built(counts, per_type={})]))
        tp, fp, fn = (
            entry["true_positives"],
            entry["false_positives"],
            entry["false_negatives"],
        )
        assert isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int)
        assert tp / (tp + fp) == counts.precision
        assert tp / (tp + fn) == counts.recall
        assert 2 * tp / (2 * tp + fp + fn) == pytest.approx(counts.f1)


def test_the_convention_the_artifact_states_is_the_one_the_module_uses() -> None:
    """The rows the formulas cannot reach, and the reason the claim had to be
    scoped: TP+FP = 0 makes precision 0/0, so "the exact value can always be
    recomputed" was false on exactly the rows the same preamble promises four
    paragraphs earlier. The empty-set convention appeared nowhere in the
    artifact, which is the only place a reader would look for it.

    Each assertion below is one clause of the sentence now in the preamble,
    checked against what the module actually publishes.
    """
    md = to_markdown([_ev()])
    assert "1.000 when the other side" in md

    # The formulas must not appear without their scope. That pairing IS the
    # correction, so this reads the paragraph the formulas sit in and checks
    # the scope sits in it too, rather than checking that both strings exist
    # somewhere in the file.
    paragraphs = md.split("\n\n")
    formulas = next(par for par in paragraphs if "TP/(TP+FP)" in par)
    assert "denominator" in formulas, formulas

    # A cross-reference is a claim about ANOTHER paragraph, so it is checked
    # against that paragraph. The first version of this sentence said "the
    # paragraph above", which is the one that had just excluded these rows, so
    # it pointed a reader back at the promise they were exempt from. Presence
    # could not see that; only reading the referent can.
    convention = next(par for par in paragraphs if "Where the denominator IS zero" in par)
    assert "the Cases" in convention, convention
    referent = next(par for par in paragraphs if par.startswith("Cases is how many"))
    assert "0 / 0 / 0" in referent, referent
    assert "denominator" not in referent, "the reference points at the wrong paragraph"
    # The reference is scoped to ONE of the two zero-denominator rows. The other
    # is Metrics(0, 0, N>0), which is not 0 / 0 / 0 and which the Cases paragraph
    # is not about, so the sentence has to say which one it means and what
    # actually separates the pair it does not mean.
    assert "The first" in convention, convention
    assert "FN column" in convention, convention

    # Nothing predicted and nothing expected: the honest perfect row.
    empty = _built(Metrics(0, 0, 0), per_type={})
    assert _headline(to_markdown([empty]), "Precision") == "1.000"
    assert _number(_entry(to_json([empty])), "precision") == 1.0

    # Nothing predicted, three findings missed: 0/0 again, and NOT 1.000.
    missed = _built(Metrics(0, 0, 3), per_type={})
    assert _headline(to_markdown([missed]), "Precision") == "0.000"
    assert _headline(to_markdown([missed]), "TP") == "0"
    assert _headline(to_markdown([missed]), "FP") == "0"

    assert _headline(to_markdown([missed]), "FN") == "3"
    # ...which is what the guidance says tells the two zero-denominator rows
    # apart, and it does: the swept row's FN is 0.
    assert _headline(to_markdown([empty]), "FN") == "0"

    # Nothing expected, three reported: recall's half of the same rule.
    noisy = _built(Metrics(0, 3, 0), per_type={})
    assert _headline(to_markdown([noisy]), "Recall") == "0.000"


def test_the_artifact_defines_the_columns_it_publishes() -> None:
    """Unpinned prose, one paragraph from the one just pinned for being unpinned.
    A `Cases` column with no definition is a number a reader has to guess at.
    """
    md = to_markdown([_ev()])
    assert "Cases is how many labelled cases" in md
    assert "TP, FP and FN are" in md
    # The case count is NOT beside the per-type ratios, and the paragraph used
    # to say both were published beside every ratio.
    assert "the case count beside the overall ratios" in md


def test_the_artifact_states_how_the_misses_were_chosen() -> None:
    """The ordering and the cap are decisions a reader cannot see from the list
    itself: a short list looks the same whether it was truncated or complete.
    """
    md = to_markdown([_ev()], max_misses=3)
    assert "worst first" in md
    assert "capped at 3 per corpus" in md


def test_the_artifact_states_that_the_sources_are_never_merged() -> None:
    """The claim that separates this report from a scores-only one, and it was
    unpinned too.
    """
    assert "never merged" in to_markdown([_ev()])


def test_markdown_states_which_numbers_ci_gates() -> None:
    """Overall precision and recall are gated. Per-type numbers and F1 are
    published but not gated, and the artifact has to say so."""
    md = to_markdown([_ev()])
    assert "gated" in md


def test_the_wrong_decision_count_is_published_in_both_artifacts() -> None:
    """Precision and recall count findings only. A guardrail that locates every
    span and returns the wrong decision scores 1.000 on both, so the decision
    count is the only place that failure shows up."""
    ev = _ev(
        failures=[
            Failure("pii-0007", "decision_mismatch", "redact", "allow"),
            Failure("pii-0008", "false_negative", "EMAIL@(3, 9)", "nothing"),
        ]
    )
    assert _entry(to_json([ev]))["decision_mismatches"] == 1
    assert "Wrong decisions" in to_markdown([ev])


def test_the_wrong_decision_count_is_zero_not_absent_when_there_are_none() -> None:
    """Absent reads as fine. Zero reads as measured."""
    assert _entry(to_json([_ev(failures=[])]))["decision_mismatches"] == 0


def test_a_negative_miss_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="max_misses"):
        to_markdown([_ev()], max_misses=-1)


# Everything below was written after rendering the brief's report for two
# evaluations and reading it as a stranger. Each test was watched failing
# against that implementation; the RED output is in task-11-report.md.


def _built(
    overall: Metrics,
    per_type: dict[str, Metrics] | None = None,
    failures: list[Failure] | None = None,
    name: str = "mailbox-set",
    detector: str = "pii",
    cases: int = 44,
    version: str = "abc123def456",
    source: str = "in-repo",
) -> Evaluation:
    """A fixture whose per-type counts differ from its overall ones.

    Real per-type counts sum to the overall ones, which is exactly why they are
    useless here: a row that read ``per_type`` where it should read ``overall``
    would render identically. These two disagree on every one of the six
    numbers, so each column has to come from the field its heading names.
    """
    return Evaluation(
        corpus_name=name,
        corpus_version=version,
        corpus_source=source,
        detector=detector,
        detector_version="0.1.0",
        cases=cases,
        overall=overall,
        per_type={"EMAIL": Metrics(4, 6, 5)} if per_type is None else per_type,
        failures=[] if failures is None else failures,
    )


def _rows(md: str) -> list[str]:
    return [line for line in md.splitlines() if line.startswith("|")]


def _cells(row: str) -> list[str]:
    """Split on the pipes markdown treats as column separators.

    A backslash escapes the character after it, whatever that character is, so
    this walks the row rather than using a ``(?<!\\\\)`` lookbehind. The
    difference is the whole point: in ``a\\\\|b`` the backslash escapes the
    BACKSLASH and the pipe still separates, which a lookbehind reads as escaped
    and a reader's markdown renderer does not. That is exactly the row an
    escaper applying its two replacements in the wrong order produces.
    """
    cells, current, index = [], "", 0
    while index < len(row):
        char = row[index]
        if char == "\\" and index + 1 < len(row):
            current += row[index : index + 2]
            index += 2
            continue
        if char == "|":
            cells.append(current)
            current = ""
        else:
            current += char
        index += 1
    cells.append(current)
    return cells


# What GFM reads as inline markup. `|` is absent because the table stage has
# already consumed it by the time this runs, and `_` because CommonMark does not
# open emphasis between two alphanumerics and the escaper deliberately leaves it
# alone.
_INLINE_MARKUP = frozenset("`*[]<>&~")


def _render_inline(text: str) -> str:
    """Model GFM's inline stage. Written from the GFM rules, not from ``_cell``.

    Three behaviours, and the third is the one that matters:

      1. a code span renders its content literally;
      2. a backslash before punctuation escapes it, and the backslash goes;
      3. an UNESCAPED markup character is a delimiter, so it contributes
         nothing to the text a reader sees.

    Rule 3 is an approximation of emphasis, links and raw HTML, and it is
    deliberate: its only job is to make this model DISAGREE with the escaper
    whenever a markup character was published unescaped. An earlier version
    recognised only a code span wrapping the whole cell, so ``a`b`c`` looked
    like ``a`b`c`` to it and like ``abc`` to a reader, and the fidelity test
    below could not see the difference.
    """
    out, index = "", 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and not text[index + 1].isalnum():
            out += text[index + 1]
            index += 2
            continue
        if char == "`":
            close = text.find("`", index + 1)
            if close != -1:
                out += text[index + 1 : close]
                index = close + 1
                continue
        if char in _INLINE_MARKUP:
            index += 1
            continue
        out += char
        index += 1
    return out


_SECTIONS = ("## Per type:", "## Worst misses:")


def _rendered(cell: str) -> str:
    """One table cell as a reader sees it.

    The table's own escape first, ``\\|`` to ``|``, which GFM applies
    everywhere including inside code spans, and then the inline stage.
    """
    return _render_inline(cell.strip().replace("\\|", "|"))


def test_the_overall_row_carries_its_support() -> None:
    """The headline row is the one the README quotes, and a ratio without its
    support is unfalsifiable there for exactly the reason it is per type:
    1.000 over one case and over five hundred render alike.
    """
    md = to_markdown([_built(Metrics(7, 2, 3))])
    row = (
        "| pii | mailbox-set | in-repo | `abc123def456` | 44 | 0.778 | 0.700 | 0.737 | 7 | 2 | 3 |"
    )
    assert row in md
    # The per-type row disagrees with the overall row on every number, so this
    # second assertion is what stops one being rendered under the other's
    # heading.
    assert "| EMAIL | 0.400 | 0.444 | 4 | 6 | 5 |" in md


def test_a_row_that_found_nothing_still_publishes_its_zeros() -> None:
    """Metrics(0, 0, 0) scores 1.000 on all three ratios by the empty-set
    convention, and produces no misses either, so every ratio on the row agrees
    that this is the best result published. The three counts are what say no
    findings were involved at all.

    They do NOT say whether anything was measured: see the next test.
    """
    md = to_markdown([_built(Metrics(0, 0, 0), per_type={})])
    assert "| 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |" in md


def test_a_scored_corpus_and_an_unscored_one_are_not_the_same_row() -> None:
    """A corpus of clean negatives handled perfectly produces tp = fp = fn = 0,
    an empty per_type and no failures. Finding nothing there is the CORRECT
    answer and a result worth publishing.

    Without a denominator that row is byte-identical to a corpus nobody scored,
    so the artifact cannot tell a reader which one it is looking at, and any
    prose claiming 0 / 0 / 0 means unmeasured is telling them to throw away a
    real result.
    """
    negatives = _built(Metrics(0, 0, 0), per_type={}, cases=40)
    unscored = _built(Metrics(0, 0, 0), per_type={}, cases=0)
    assert to_markdown([negatives]) != to_markdown([unscored])
    assert _headline(to_markdown([negatives]), "Cases") == "40"
    assert _headline(to_markdown([unscored]), "Cases") == "0"
    assert _entry(to_json([negatives]))["cases"] == 40
    assert _entry(to_json([unscored]))["cases"] == 0


def test_both_artifacts_publish_the_case_count() -> None:
    """The denominator. 44 is not the value of any other number in the fixture,
    so a count taken from the findings or the failures fails here.
    """
    ev = _built(Metrics(7, 2, 3), cases=44)
    assert _entry(to_json([ev]))["cases"] == 44
    assert _headline(to_markdown([ev]), "Cases") == "44"


def test_a_score_inside_the_interval_never_publishes_as_a_perfect_one() -> None:
    """9999/10000 is not 1. Rendered 1.000 beside FP = 1 on the same row, it is
    a claim the arithmetic on that row contradicts, and it is the row the README
    quotes. Up is the flattering direction, which is the one that gets believed.
    """
    ev = _built(Metrics(9999, 1, 1), per_type={"EMAIL": Metrics(9999, 1, 1)})
    md = to_markdown([ev])
    entry = _entry(to_json([ev]))
    assert _headline(md, "Precision") == ">0.999"
    assert _number(entry, "precision") == 0.9999
    assert "| 9999 | 1 | 1 |" in md
    # The per-type table calls the same two helpers, so it is correct today for
    # a reason no assertion held: inlining round(m.precision, 4) into the
    # per-type comprehension left the suite green.
    assert "| EMAIL | >0.999 | >0.999 | 9999 | 1 | 1 |" in md
    assert _number(_mapping(_mapping(entry, "per_type"), "EMAIL"), "precision") == 0.9999


def test_a_score_inside_the_interval_never_publishes_as_a_flat_zero() -> None:
    """The mirror. 1/10000 rendered 0.000 says a detector found nothing right
    when it found one thing right, which understates rather than flatters, and
    is equally false.
    """
    ev = _built(Metrics(1, 9999, 0), per_type={"EMAIL": Metrics(1, 9999, 0)})
    md = to_markdown([ev])
    entry = _entry(to_json([ev]))
    assert _headline(md, "Precision") == "<0.001"
    assert _number(entry, "precision") == 0.0001
    assert "| EMAIL | <0.001 | 1.000 | 1 | 9999 | 0 |" in md
    assert _number(_mapping(_mapping(entry, "per_type"), "EMAIL"), "precision") == 0.0001


def test_the_json_clamps_a_score_that_rounds_onto_a_boundary() -> None:
    """Four decimals has the identical defect three does, and JSON cannot carry
    a `>`. 999999/1000000 rounds to exactly 1.0 and 1/1000000 to exactly 0.0, so
    both are clamped one step inside the interval. Inward is never the
    flattering direction, which is what makes it safe for the gate to compare.
    """
    high = _built(Metrics(999999, 1, 1), per_type={"EMAIL": Metrics(999999, 1, 1)})
    low = _built(Metrics(1, 999999, 0), per_type={"EMAIL": Metrics(1, 999999, 0)})
    for ev, clamped in ((high, 0.9999), (low, 0.0001)):
        entry = _entry(to_json([ev]))
        assert _number(entry, "precision") == clamped
        assert _number(_mapping(_mapping(entry, "per_type"), "EMAIL"), "precision") == clamped
    # F1 goes through the same helper as precision and recall, so the whole row
    # is asserted rather than one cell of it.
    assert "| >0.999 | >0.999 | >0.999 | 999999 | 1 | 1 |" in to_markdown([high])


def test_an_exact_one_still_publishes_as_one() -> None:
    """The guard is about the open interval, not about the values. A detector
    that really did get everything right must not be demoted to >0.999, which
    would be just as false in the other direction.
    """
    ev = _built(Metrics(10, 0, 0), per_type={"EMAIL": Metrics(10, 0, 0)})
    md = to_markdown([ev])
    assert _headline(md, "Precision") == "1.000"
    assert _number(_entry(to_json([ev])), "precision") == 1.0
    assert "| EMAIL | 1.000 | 1.000 | 10 | 0 | 0 |" in md


def test_an_exact_zero_still_publishes_as_zero() -> None:
    ev = _built(Metrics(0, 5, 0), per_type={"EMAIL": Metrics(0, 5, 0)})
    md = to_markdown([ev])
    assert _headline(md, "Precision") == "0.000"
    assert _number(_entry(to_json([ev])), "precision") == 0.0


def test_the_overall_row_publishes_the_wrong_decision_count() -> None:
    """``Wrong decisions`` is a column HEADING, so the brief's assertion that it
    appears holds with the count hardcoded or dropped. This pins the value.
    """
    failures = [
        Failure("pii-0001", "decision_mismatch", "redact", "allow"),
        Failure("pii-0002", "false_negative", "EMAIL@(3, 9)", "nothing"),
        Failure("pii-0003", "decision_mismatch", "deny", "allow"),
    ]
    # Three failures, two of them wrong decisions: a cell rendering len(failures)
    # would say 3, and one rendering a constant would not move between these two.
    assert _rows(to_markdown([_built(Metrics(7, 2, 3), failures=failures)]))[2].endswith("| 2 |")
    assert _rows(to_markdown([_built(Metrics(7, 2, 3))]))[2].endswith("| 0 |")


def test_a_zero_support_type_publishes_its_zeros_in_both_artifacts() -> None:
    """The type with nothing to find is the one the empty-set convention
    flatters, so it is the one that must carry its support.
    """
    ev = _built(Metrics(0, 0, 0), per_type={"PHONE": Metrics(0, 0, 0)})
    assert "| PHONE | 1.000 | 1.000 | 0 | 0 | 0 |" in to_markdown([ev])
    phone = _mapping(_mapping(_entry(to_json([ev])), "per_type"), "PHONE")
    assert phone["true_positives"] == 0
    assert phone["false_positives"] == 0
    assert phone["false_negatives"] == 0


def test_a_corpus_with_no_per_type_breakdown_says_so_rather_than_vanishing() -> None:
    """A section that is simply skipped reads as a section that had nothing to
    report, which is the same document a detector that scored nothing produces.
    """
    md = to_markdown([_built(Metrics(0, 0, 0), per_type={})])
    assert "## Per type: pii on mailbox-set (in-repo, `abc123def456`)" in md
    assert "No findings of any type" in md


def test_every_section_names_the_corpus_it_measured() -> None:
    """Two corpora, one detector, one source. Headed by detector and source
    alone these two sections are indistinguishable, and a reader cannot tell
    which corpus the misses under them came from.
    """
    md = to_markdown([_built(Metrics(7, 2, 3)), _built(Metrics(4, 0, 0), name="ledger-dump")])
    assert "## Worst misses: pii on mailbox-set (in-repo, `abc123def456`)" in md
    assert "## Worst misses: pii on ledger-dump (in-repo, `abc123def456`)" in md


def test_the_worst_misses_are_listed_worst_first() -> None:
    """The heading says worst. Corpus order is not worst order, and under a cap
    it is the wrong decision, the one thing a guardrail user cares about, that
    gets truncated away when it happens to be labelled late in the file.
    """
    failures = [
        Failure("c-noise", "false_positive", "nothing", "EMAIL@(0, 12)"),
        Failure("c-leak", "false_negative", "EMAIL@(3, 9)", "nothing"),
        Failure("c-allowed", "decision_mismatch", "redact", "allow"),
    ]
    md = to_markdown([_built(Metrics(7, 2, 3), failures=failures)], max_misses=1)
    assert "c-allowed" in md
    assert "c-leak" not in md
    assert "c-noise" not in md


def test_an_unrecognised_failure_kind_is_never_the_one_truncated_away() -> None:
    """A kind this module does not know how to rank is not thereby unimportant.
    Ranked last it would be the first thing a cap hides, which is the absent
    reads as fine default in its cheapest form.
    """
    failures = [
        Failure("c-allowed", "decision_mismatch", "redact", "allow"),
        Failure("c-strange", cast(FailureKind, "sabotage"), "redact", "exploded"),
    ]
    md = to_markdown([_built(Metrics(7, 2, 3), failures=failures)], max_misses=1)
    assert "c-strange" in md
    assert "c-allowed" not in md


def test_a_zero_miss_cap_hides_the_table_but_not_the_count() -> None:
    """max_misses=0 with misses to report must not render as the no-misses
    corpus, and must not leave an empty table standing in for a real one.
    """
    failures = [Failure(f"c{i}", "false_negative", "EMAIL", "nothing") for i in range(9)]
    md = to_markdown([_built(Metrics(7, 2, 3), failures=failures)], max_misses=0)
    assert "No misses" not in md
    assert "9 more" in md
    assert "| Case | Kind | Expected | Predicted |" not in md


def test_a_pipe_in_a_value_cannot_shift_a_column() -> None:
    """Corpus content reaches these cells: the type name and the case id are
    validated as non-empty strings and nothing more, and a third-party corpus is
    not ours to fix. One unescaped pipe moves every value after it one column
    left, so precision publishes under the Recall heading. The number is right
    and the claim is false, which is the whole failure mode of a formatter.
    """

    def widths(ev: Evaluation) -> set[int]:
        return {len(_cells(row)) for row in _rows(to_markdown([ev]))}

    benign = _built(
        Metrics(7, 2, 3),
        per_type={"EMAIL": Metrics(4, 6, 5)},
        failures=[Failure("c1", "false_negative", "EMAIL@(3, 9)", "nothing")],
        name="mailbox",
    )
    hostile = _built(
        Metrics(7, 2, 3),
        # A backslash as well as a pipe: an escaper that escapes the pipe before
        # the backslash turns "a|b" into "a\\|b", which renders as a literal
        # backslash followed by a live column separator.
        #
        # The `\|` values are on the CASE ID and the CORPUS VERSION deliberately.
        # Those two are published inside code spans and route through a
        # different escaper, and an earlier version of this fixture put `\|`
        # only in `expected`, which does not: the hostile input never reached
        # the escaper that could not handle it, and a regression there passed
        # this test.
        per_type={"EM|AIL": Metrics(4, 6, 5)},
        failures=[Failure("c\\|1", "false_negative", "EM\\|AIL@(3, 9)", "noth|ing")],
        name="mail|box",
        version="ab\\|123",
    )
    # Three tables, and every row of each one, headers and alignment rows
    # included: a separator row that does not match its header stops the table
    # rendering as a table at all.
    assert widths(benign) == {14, 8, 6}
    assert widths(hostile) == widths(benign)


def test_a_newline_in_a_value_cannot_end_a_row_or_a_heading() -> None:
    """The other half of escaping, and the more destructive half. A pipe shifts
    a row; a newline ENDS it, and everything after the break renders as prose
    sitting under a table that lost its last column. A heading loses its second
    half the same way, taking the corpus name with it.
    """
    ev = _built(
        Metrics(7, 2, 3),
        per_type={"EM\tAIL": Metrics(4, 6, 5)},
        failures=[Failure("c\r\n1", "false_negative", "EMAIL@(3, 9)", "not\nhing")],
        name="mail\nbox",
    )
    md = to_markdown([ev])
    assert "## Worst misses: pii on mail box (in-repo, `abc123def456`)" in md
    assert "| EM AIL | 0.400 | 0.444 | 4 | 6 | 5 |" in md
    assert "| `c 1` | false_negative | EMAIL@(3, 9) | not hing |" in md
    # Three tables of a header, a rule and one row. A value that broke out of a
    # cell leaves a fragment that is no longer a row at all.
    assert len(_rows(md)) == 9


def test_a_reader_sees_the_id_and_version_the_json_stores() -> None:
    """For each value below, in all three contexts, what a reader can read off
    the page equals what the machine artifact stores, and no row loses a column.

    Not a claim about every possible string: a claim about these ten, chosen to
    cover each branch of the escaper and each way GFM can eat a character.

      * ``c\\|1`` is the value that broke this. Escaping only the pipe, for a
        code span, hands GFM's row scanner ``c\\\\|1``: an escaped BACKSLASH,
        and then a live cell boundary. Doubling instead keeps the boundary and
        publishes the wrong characters inside a code span. No encoding does
        both, so a value carrying a backslash gives up its backticks.
      * ``a`b`c`` and ``c\\a*b*d`` take the fallback, where the value is no
        longer inside a code span and every markup character has to be escaped
        by hand. Published raw they render as ``abc`` and ``c\\abd``.
      * ``x*y*z`` and ``<img x>`` take the code-span branch, where the same
        characters are literal and must NOT be escaped.

    All three contexts, because the module's own docstring names three and the
    heading was the one this round was dispatched about. Both published values
    go through it: ``corpus.py`` validates ``id`` and ``version`` as non-empty
    strings and nothing more.
    """
    for value in (
        "c1",
        "c|1",
        "c\\a|b",
        "c\\|1",
        "a`b",
        "ab\\|123",
        "a`b`c",
        "c\\a*b*d",
        "x*y*z",
        "<img x>",
    ):
        ev = _built(
            Metrics(7, 2, 3),
            failures=[Failure(value, "false_negative", "EMAIL", "nothing")],
            version=value,
        )
        md = to_markdown([ev])
        entry = _entry(to_json([ev]))
        failures = entry["failures"]
        assert isinstance(failures, list)

        headline, misses = _cells(_rows(md)[2]), _cells(_rows(md)[-1])
        assert len(headline) == 14, f"{value!r} shifted the headline row: {headline}"
        assert len(misses) == 6, f"{value!r} shifted the miss row: {misses}"
        assert _rendered(headline[4]) == entry["corpus_version"], value
        assert _rendered(misses[1]) == failures[0]["case_id"], value

        # The third context. Every SECTION heading carries the version, and a
        # heading is not a table row, so it reaches a different escaper. The
        # guidance heading is not one of these: it names no corpus.
        headings = [line for line in md.splitlines() if line.startswith(_SECTIONS)]
        assert len(headings) == 2, headings
        for heading in headings:
            assert value in _render_inline(heading), (value, heading)


def test_a_heading_does_not_carry_a_table_escape() -> None:
    """One value, three contexts, three escapers. A heading is never split into
    cells, so the ``\\|`` that keeps a table row intact publishes a literal
    backslash in a heading. The table row two lines above it must still carry
    the escape, so the assertion is scoped to the headings.
    """
    md = to_markdown([_built(Metrics(7, 2, 3), version="ab|123")])
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings
    assert any("`ab|123`" in heading for heading in headings)
    assert not any("\\|" in heading for heading in headings), headings
    # The row still escapes it: the two contexts genuinely differ.
    assert "`ab\\|123`" in _rows(md)[2]


def test_a_generator_of_evaluations_is_not_silently_truncated() -> None:
    """``to_markdown`` walks its argument four times, once for the headline rows
    and once per section kind. Handed an iterator it rendered the scores and
    then found nothing left, so every per-type table and every miss vanished:
    the published artifact kept the numbers and dropped the evidence, with no
    error and a document that looks complete.

    The annotation says Sequence and a generator is already a type error, but
    the gate checks this repo and not a caller's, and the misses are the thing
    this project sells.
    """
    pair = [_ev("in-repo"), _ev("third-party")]
    # Through a Callable alias, not a type: ignore. The signature is correct and
    # mypy enforcing it here is a second guard; this test is about the caller
    # mypy never runs on.
    render = cast(Callable[..., str], to_markdown)
    from_list = to_markdown(pair)
    from_iterator = render(evaluation for evaluation in pair)
    assert from_iterator == from_list
    assert from_iterator.count("## ") == from_list.count("## ") == 5
    assert "pii-0007" in from_iterator


def test_the_opening_line_claims_only_what_the_table_holds() -> None:
    """The first sentence of the artifact, and it was unpinned: deleting both
    lines left the suite green.

    It also over-claimed. "Precision and recall for every check" is a coverage
    claim this table cannot make: it carries one row per evaluation the caller
    handed in, so a detector nobody ran a corpus against is simply absent while
    the opening line says the file covers everything. What IS true is what a row
    is, so that is what it says, and this checks it against the rendered rows
    for one, two and three evaluations.
    """
    assert "One row per check measured against one corpus" in to_markdown([_ev()])
    for count in (1, 2, 3):
        evaluations = [_built(Metrics(7, 2, 3), name=f"corpus-{i}") for i in range(count)]
        rows = _rows(to_markdown(evaluations))
        # header, rule, then exactly one row per evaluation and no more.
        assert len(rows[2 : 2 + count]) == count
        assert not rows[2 + count].startswith("| pii |")


def test_the_case_count_is_the_only_published_number_separating_two_swept_rows() -> None:
    """The guidance says so, so it is checked rather than asserted.

    Two corpora with nothing to find and nothing found differ in no other cell:
    every ratio is 1.000 by the empty-set convention, every count is 0, and the
    wrong-decision column is 0. If any other column moved, the claim would be
    false and the Cases column would not be the thing carrying that distinction.
    """
    swept = to_markdown([_built(Metrics(0, 0, 0), per_type={}, cases=40)])
    unscored = to_markdown([_built(Metrics(0, 0, 0), per_type={}, cases=0)])
    header = [cell.strip() for cell in _cells(_rows(swept)[0])]
    left = [cell.strip() for cell in _cells(_rows(swept)[2])]
    right = [cell.strip() for cell in _cells(_rows(unscored)[2])]
    differing = [header[i] for i, (a, b) in enumerate(zip(left, right)) if a != b]
    assert differing == ["Cases"], differing


def test_the_table_comes_before_the_guidance() -> None:
    """A reader opening this file wants a number. Every sentence of the guidance
    is load-bearing and none is cut, but 55 lines of caveat before the first
    figure is five-sixths of the page, so it sits under the thing it qualifies.
    """
    md = to_markdown([_ev()])
    lines = md.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| Check |"))
    guidance = next(i for i, line in enumerate(lines) if line == "## How to read this")
    assert header < guidance
    # And near the top, not merely before it: a reader should not have to
    # scroll to reach the first number.
    assert header <= 6, f"{header} lines before the table"


def test_a_corpus_authored_value_is_never_raw_markup_in_a_heading() -> None:
    """Round 2's sweep called these three sites correct, under the criterion of
    the day: can this break a column. A heading has no columns, so the answer
    was no and the verdict was true. Round 3 widened the criterion to "can this
    reach the page as markup" and the verdict expired unnoticed, so a corpus
    named ``<img src=x onerror=alert(1)>`` rendered raw in every section heading
    while the same value one line below it was escaped correctly.

    ``corpus.py`` validates name and source as non-empty strings and nothing
    more, and both are third-party authored. ``corpus_version`` is a
    content-derived digest and cannot be hostile, and it was the one heading
    value already escaped.
    """
    builders: tuple[tuple[str, Callable[[str], Evaluation]], ...] = (
        ("detector", lambda v: _built(Metrics(7, 2, 3), detector=v)),
        ("corpus name", lambda v: _built(Metrics(7, 2, 3), name=v)),
        ("corpus source", lambda v: _built(Metrics(7, 2, 3), source=v)),
    )
    for field, build in builders:
        for value in ("<img src=x onerror=alert(1)>", "a*b*c", "a`b`c", "a[x](y)"):
            md = to_markdown([build(value)])
            headings = [line for line in md.splitlines() if line.startswith(_SECTIONS)]
            assert len(headings) == 2, headings
            for heading in headings:
                assert value in _render_inline(heading), (field, value, heading)


def test_every_member_of_the_escape_set_is_pinned_by_its_own_value() -> None:
    """One value per character, and the list is written out HERE rather than
    imported from the module.

    Importing it would make this test enumerate whatever the module currently
    escapes, so a member deleted from the set would delete its own test with it
    and the suite would stay green. That is what happened: two members had
    mutations and the other six did not, and dropping ``< > & [ ] ~`` left 665
    tests passing. ``<`` and ``>`` are the characters the injection argument
    rests on.

    Two values per member, because a character is only dangerous in the position
    where it is markup. ``a\\b`` renders as ``a\\b`` whether or not the backslash
    is escaped, so the lone form pins nothing for that member; it is ``a\\|b``, a
    backslash immediately before a pipe, that GFM's row scanner reads as an
    escaped backslash followed by a LIVE boundary. The lone form was all this
    test had, and the backslash mutation was killed by a different test.

    ``!`` is absent: it is markup only before ``[``, which is always escaped, so
    no value can pin it and it was removed from the set rather than left in as
    an unpinnable member.
    """
    for char, value in [(c, v) for c in "\\`*[]<>&~|" for v in (f"a{c}b", f"a{c}|b")]:
        ev = _built(
            Metrics(7, 2, 3),
            per_type={value: Metrics(4, 6, 5)},
            failures=[Failure(value, "false_negative", value, value)],
            name=value,
            version=value,
        )
        md = to_markdown([ev])
        entry = _entry(to_json([ev]))
        failures = entry["failures"]
        assert isinstance(failures, list)

        rows = _rows(md)
        assert len(_cells(rows[2])) == 14, (char, rows[2])
        assert len(_cells(rows[-1])) == 6, (char, rows[-1])
        # The cell a reader sees, in each of the three contexts.
        assert _rendered(_cells(rows[2])[2]) == entry["corpus_name"], char
        assert _rendered(_cells(rows[-1])[3]) == failures[0]["expected"], char
        assert _rendered(_cells(rows[-1])[1]) == failures[0]["case_id"], char
        for heading in [line for line in md.splitlines() if line.startswith(_SECTIONS)]:
            assert value in _render_inline(heading), (char, heading)


def test_a_heading_never_carries_a_table_escape_on_either_branch() -> None:
    """The earlier version of this test used ``ab|123``, which has no backtick,
    so it took the code-span branch and could not reach the fallback. The
    fallback used the CELL escaper, which puts ``\\|`` into a heading: the guard
    was pinned only on the branch that cannot violate it.
    """
    for version in ("ab|123", "a`b|c"):
        md = to_markdown([_built(Metrics(7, 2, 3), version=version)])
        headings = [line for line in md.splitlines() if line.startswith(_SECTIONS)]
        assert len(headings) == 2
        for heading in headings:
            assert "\\|" not in heading, (version, heading)
            assert version in _render_inline(heading), (version, heading)


def test_two_versions_of_one_corpus_do_not_share_a_section_heading() -> None:
    """Same detector, same corpus name, same source, different content. The
    headline table separates those two rows by its Version column; without the
    version the sections below it do not, and the misses under two identical
    headings cannot be told apart.
    """
    md = to_markdown(
        [
            _built(Metrics(7, 2, 3), failures=[Failure("c1", "false_negative", "E", "nothing")]),
            _built(
                Metrics(4, 0, 0),
                failures=[Failure("c2", "false_negative", "E", "nothing")],
                version="ffffffffffff",
            ),
        ]
    )
    headings = [line for line in md.splitlines() if line.startswith("## Worst misses")]
    assert len(headings) == 2
    assert len(set(headings)) == 2


def test_the_json_carries_every_miss_the_markdown_points_at() -> None:
    """The markdown says the rest are in benchmarks.json. Nothing in the brief
    reads that file's failures at all, so the claim is unchecked: the key could
    be dropped entirely and every one of its tests would still pass.
    """
    # Three kinds, interleaved, so corpus order and worst order disagree. Nine
    # failures of one kind would make the two orders identical and this test
    # would hold for a JSON file that published them in either.
    kinds: list[FailureKind] = ["false_positive", "false_negative", "decision_mismatch"]
    failures = [Failure(f"c{i}", kinds[i % 3], "EMAIL", "nothing") for i in range(9)]
    ev = _built(Metrics(7, 2, 3), failures=failures)
    published = _entry(to_json([ev]))["failures"]
    assert isinstance(published, list)
    assert len(published) == 9
    md = to_markdown([ev], max_misses=3)
    ids = [f["case_id"] for f in published]
    # A prefix, in the same order, so "the rest are in benchmarks.json" tells a
    # reader where to carry on rather than where to start searching.
    for shown in ids[:3]:
        assert f"`{shown}`" in md
    for hidden in ids[3:]:
        assert f"`{hidden}`" not in md


def test_the_json_orders_per_type_the_way_the_markdown_does() -> None:
    """Both artifacts are committed and CI diffs them against a fresh run, so an
    order that follows a caller's dict rather than the type name is a spurious
    diff at best and two artifacts disagreeing at worst.
    """
    ev = _built(Metrics(7, 2, 3), per_type={"ZIP": Metrics(1, 0, 0), "EMAIL": Metrics(4, 6, 5)})
    assert list(_mapping(_entry(to_json([ev])), "per_type")) == ["EMAIL", "ZIP"]
    md = to_markdown([ev])
    assert md.index("| EMAIL |") < md.index("| ZIP |")


def test_neither_artifact_is_rendered_from_the_other_and_rounded_twice() -> None:
    """Each file rounds the measurement once, at its own width.

    41 / 43 is the case that found this: 0.95348..., which is 0.953 at three
    decimals and 0.9535 at four, and 0.9535 rendered at three decimals is 0.954.
    Rendering the table from the stored figure would publish 0.954 for a
    measurement that supports 0.953, on counts a real corpus produces, and up is
    the direction that flatters.

    The brief's same-scores test cannot see this: its fixture scores 0.9, where
    every rounding of every rounding is 0.900.
    """
    ev = _built(Metrics(41, 2, 2))
    entry = _entry(to_json([ev]))
    md = to_markdown([ev])
    assert _number(entry, "recall") == 0.9535
    assert "| 0.953 |" in md
    assert "0.954" not in md


def test_the_json_keeps_more_precision_than_the_markdown_shows() -> None:
    """The markdown is for reading and the JSON is the record. Four decimals is
    what the preamble promises a reader they will find there.
    """
    ev = _built(Metrics(7, 2, 3))
    assert _number(_entry(to_json([ev])), "precision") == 0.7778
    assert "0.778" in to_markdown([ev])


def test_a_report_of_no_evaluations_says_nothing_was_measured() -> None:
    """An empty table under a heading that promises benchmarks reads as a
    rendering fault at best and as measured and fine at worst.
    """
    md = to_markdown([])
    assert "No evaluations were run" in md
    assert "| Check |" not in md
    assert to_json([])["results"] == []
