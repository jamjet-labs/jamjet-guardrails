"""The corpus loader: every published number is measured on what this returns.

The brief's nine tests come first, verbatim apart from the annotations mypy
needs. Everything after them exists because a corpus that loads wrong is a
published number that is wrong, and each of those shapes would otherwise load
quietly and move a score.
"""

import json
from pathlib import Path
from typing import cast, get_args

import pytest

from jamjet_guardrails.eval.corpus import (
    _DECISIONS,
    _DIRECTIONS,
    Case,
    Corpus,
    CorpusError,
    ExpectedFinding,
    load_corpus,
)
from jamjet_guardrails.types import Decision, Direction

CASE: dict[str, object] = {
    "id": "pii-0001",
    "text": "write to alice@example.com",
    "direction": "output",
    "expect": {"decision": "redact", "findings": [{"type": "EMAIL", "span": [9, 26]}]},
    "source": "in-repo",
    "license": "Apache-2.0",
}


def _write(tmp_path: Path, rows: list[dict[str, object]], filename: str = "c.jsonl") -> Path:
    path = tmp_path / filename
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_loads_a_well_formed_case(tmp_path: Path) -> None:
    corpus = load_corpus(_write(tmp_path, [CASE]), name="pii/in-repo")
    case = corpus.cases[0]
    assert len(corpus.cases) == 1
    assert case.id == "pii-0001"
    assert case.expect_decision == "redact"
    assert case.expect_findings[0].type == "EMAIL"
    assert case.expect_findings[0].span == (9, 26)
    assert corpus.source == "in-repo"


# DEVIATION from the brief, recorded rather than silent. The brief's two
# version fixtures are `dict(CASE, id="pii-0002", text="different")`, which
# keeps CASE's `"span": [9, 26]` over a nine-character text: the span slices to
# "". That is exactly "a span that does not fit its text", the shape the loader
# is required to make impossible to load. The fixture is the thing that is
# wrong, so it is replaced by a coherent one -- allow, no findings -- and every
# assertion in both tests is unchanged.
OTHER = dict(CASE, id="pii-0002", text="different", expect={"decision": "allow", "findings": []})


def test_version_is_derived_from_content(tmp_path: Path) -> None:
    """No human bumps this. Adding a case must change it."""
    one = load_corpus(_write(tmp_path, [CASE], "a.jsonl"), name="x")
    other = OTHER
    two = load_corpus(_write(tmp_path, [CASE, other], "b.jsonl"), name="x")
    assert one.version != two.version
    assert len(one.version) == 12


def test_version_is_stable_under_reordering(tmp_path: Path) -> None:
    other = OTHER
    a = load_corpus(_write(tmp_path, [CASE, other], "a.jsonl"), name="x")
    b = load_corpus(_write(tmp_path, [other, CASE], "b.jsonl"), name="x")
    assert a.version == b.version


def test_a_missing_required_field_names_the_line(tmp_path: Path) -> None:
    bad = {k: v for k, v in CASE.items() if k != "license"}
    with pytest.raises(CorpusError, match="line 1"):
        load_corpus(_write(tmp_path, [bad]), name="x")


def test_mixed_sources_in_one_file_are_rejected(tmp_path: Path) -> None:
    """In-repo and third-party numbers stay separable: one file, one source."""
    other = dict(CASE, id="pii-0002", source="third-party")
    with pytest.raises(CorpusError, match="source"):
        load_corpus(_write(tmp_path, [CASE, other]), name="x")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="duplicate"):
        load_corpus(_write(tmp_path, [CASE, dict(CASE)]), name="x")


def test_an_unparseable_line_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(CASE) + "\n{not json\n")
    with pytest.raises(CorpusError, match="line 2"):
        load_corpus(path, name="x")


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(CASE) + "\n\n")
    assert len(load_corpus(path, name="x").cases) == 1


def test_an_empty_corpus_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("\n")
    with pytest.raises(CorpusError, match="no cases"):
        load_corpus(path, name="x")


# --------------------------------------------------------------------------
# Everything below is additional to the brief.
# --------------------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    """A well-formed line with fields replaced. Never mutates CASE."""
    return {**CASE, **overrides}


def _clean(**overrides: object) -> dict[str, object]:
    """A well-formed line whose expectation is empty, so text is free to vary."""
    return _row(expect={"decision": "allow", "findings": []}, **overrides)


def _write_raw(tmp_path: Path, text: str, filename: str = "c.jsonl") -> Path:
    path = tmp_path / filename
    path.write_text(text)
    return path


def _built_case(
    case_id: str = "a",
    text: str = "clean text",
    direction: Direction = "output",
    decision: Decision = "allow",
    findings: tuple[ExpectedFinding, ...] = (),
    source: str = "unit-fixture",
    licence: str = "CC0-1.0",
) -> Case:
    return Case(
        id=case_id,
        text=text,
        direction=direction,
        expect_decision=decision,
        expect_findings=findings,
        source=source,
        license=licence,
    )


# Every value disagrees with anything the loader could plausibly hardcode. The
# brief's fixture is "output"/"redact"/"in-repo"/"Apache-2.0", which are also
# make_case.py's defaults and the plan's example corpus lines, so a loader that
# ignored the file and returned those constants passes all nine brief tests.
ODD: dict[str, object] = {
    "id": "zz-9",
    "text": "quiet line",
    "direction": "input",
    "expect": {"decision": "deny", "findings": [{"type": "NOT_A_REAL_TYPE", "span": None}]},
    "source": "unit-fixture",
    "license": "CC0-1.0",
}


def test_every_field_is_read_from_the_line_not_assumed(tmp_path: Path) -> None:
    """No assertion here can be satisfied by a constant the loader could hold.

    `name` in particular is asserted nowhere in the brief, so a loader ignoring
    its own argument passes every one of those tests -- the exact defect Task 8
    found in its ordering test.
    """
    corpus = load_corpus(_write(tmp_path, [ODD]), name="odd/name-9")
    case = corpus.cases[0]
    assert corpus.name == "odd/name-9"
    assert corpus.source == "unit-fixture"
    assert corpus.license == "CC0-1.0"
    assert case.id == "zz-9"
    assert case.text == "quiet line"
    assert case.direction == "input"
    assert case.expect_decision == "deny"
    assert case.source == "unit-fixture"
    assert case.license == "CC0-1.0"
    assert case.expect_findings[0].type == "NOT_A_REAL_TYPE"
    assert case.expect_findings[0].span is None


def test_the_missing_field_is_named_not_only_the_line(tmp_path: Path) -> None:
    bad = {k: v for k, v in CASE.items() if k != "direction"}
    with pytest.raises(CorpusError, match="direction"):
        load_corpus(_write(tmp_path, [bad]), name="x")


@pytest.mark.parametrize(
    "payload",
    [
        "[1, 2]",
        '"a string"',
        "42",
        "null",
        "true",
        # The one that discriminates. `"id" not in [...]` is a membership test
        # over VALUES, so an array spelling every required field name passes a
        # naive required-key check outright; the next line indexes a list by a
        # string and the loader dies with a bare TypeError instead of naming the
        # line. The five payloads above it are all caught by the key check even
        # with the object test removed, so on their own they pin nothing here.
        '["id", "text", "direction", "expect", "source", "license"]',
    ],
)
def test_a_line_that_is_not_a_json_object_is_rejected(tmp_path: Path, payload: str) -> None:
    with pytest.raises(CorpusError, match="line 1"):
        load_corpus(_write_raw(tmp_path, payload + "\n"), name="x")


def test_an_unrecognised_case_field_is_rejected(tmp_path: Path) -> None:
    """ "licence" is the realistic typo, and `license` is still present beside it.

    Only the unknown-key half of the check can see this one: nothing is missing,
    so a schema that merely required its own fields would load the line and the
    author's edit would go nowhere with no complaint.
    """
    with pytest.raises(CorpusError, match="line 1"):
        load_corpus(_write(tmp_path, [_row(licence="Apache-2.0")]), name="x")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", 5),
        ("id", None),
        ("id", ""),
        ("text", None),
        ("text", ""),
        ("text", ["a"]),
        ("source", []),
        ("source", ""),
        ("license", 3),
        ("license", ""),
    ],
)
def test_a_field_of_the_wrong_type_or_empty_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    """An empty licence is an unknown licence, and this library sells provenance.

    The match is the WHOLE message, not the field name. "text" also occurs in
    the span-fit message ("runs past the end of a N-character text"), so a bare
    `match=field` let a loosened `_require_text` keep three of these green.
    """
    with pytest.raises(CorpusError, match=f"{field} must be a non-empty string"):
        load_corpus(_write(tmp_path, [_row(**{field: value})]), name="x")


@pytest.mark.parametrize("direction", ["sideways", "OUTPUT", "", None, 3, ["output"]])
def test_a_direction_that_is_not_a_Direction_is_rejected(tmp_path: Path, direction: object) -> None:
    with pytest.raises(CorpusError, match="direction"):
        load_corpus(_write(tmp_path, [_row(direction=direction)]), name="x")


@pytest.mark.parametrize("decision", ["block", "REDACT", "", None, 0, ["allow"]])
def test_a_decision_that_is_not_a_Decision_is_rejected(tmp_path: Path, decision: object) -> None:
    with pytest.raises(CorpusError, match="decision"):
        load_corpus(_write(tmp_path, [_row(expect={"decision": decision, "findings": []})]), "x")


def test_the_accepted_directions_track_the_type_module() -> None:
    """Listed literally, then pinned. Adding to the Literal breaks HERE.

    Deriving the accepted set from get_args would admit a new direction into a
    corpus before evaluate() or the report knew what to do with it. Pinning it
    turns that growth into a red test in this module instead.
    """
    assert set(_DIRECTIONS) == set(get_args(Direction))


def test_the_accepted_decisions_track_the_type_module() -> None:
    assert set(_DECISIONS) == set(get_args(Decision))


@pytest.mark.parametrize(
    "expect",
    [
        "redact",
        ["redact"],
        # Spells both keys, so the key check alone would let it through and the
        # next line would index a list with a string.
        ["decision", "findings"],
        None,
        42,
        {"decision": "redact"},
        {"findings": []},
        {},
        {"decision": "redact", "findings": [], "note": "hi"},
    ],
)
def test_expect_must_declare_exactly_a_decision_and_findings(
    tmp_path: Path, expect: object
) -> None:
    """A missing `findings` key must not default to "expects nothing".

    Absent-means-empty is the failure direction this phase keeps producing:
    every real detection on that case becomes a false positive, so a typo in one
    corpus line silently moves a published precision figure.
    """
    with pytest.raises(CorpusError, match="line 1"):
        load_corpus(_write(tmp_path, [_row(expect=expect)]), name="x")


@pytest.mark.parametrize("findings", ["EMAIL", {"type": "EMAIL", "span": None}, 3, None])
def test_findings_must_be_a_list(tmp_path: Path, findings: object) -> None:
    """`tuple("EMAIL")` is five one-character findings, not one finding.

    Matching the whole message matters here too: with the list check removed a
    string or a bare object is still refused, by `expect.findings[0] is not an
    object`, which also contains "findings". Only `3` and `None` killed the
    loose form.
    """
    with pytest.raises(CorpusError, match="expect.findings must be a list"):
        load_corpus(
            _write(tmp_path, [_row(expect={"decision": "redact", "findings": findings})]), "x"
        )


@pytest.mark.parametrize(
    "finding",
    [
        "EMAIL",
        ["EMAIL", [9, 26]],
        # Spells both keys, so the key check passes over it and the next line
        # indexes a list with a string. The same shape as the case-level array
        # above, one level down, and the only fixture here that reaches the
        # object test at all.
        ["type", "span"],
        {"type": "EMAIL"},
        {"span": [9, 26]},
        {"type": "EMAIL", "span": [9, 26], "confidence": 0.9},
        {"type": "", "span": [9, 26]},
        {"type": 7, "span": [9, 26]},
    ],
)
def test_a_finding_must_declare_exactly_a_type_and_a_span(tmp_path: Path, finding: object) -> None:
    """A finding with no `span` key must not silently become span-free.

    A span-free expectation matches on type alone (Task 10), which is a WEAKER
    bar than the exact-span rule the published number claims. Omitting the key
    would quietly buy precision and recall; writing `null` asks for it out loud.
    """
    row = _row(expect={"decision": "redact", "findings": [finding]})
    with pytest.raises(CorpusError, match="line 1"):
        load_corpus(_write(tmp_path, [row]), name="x")


def test_an_explicit_null_span_means_type_alone(tmp_path: Path) -> None:
    row = _row(expect={"decision": "redact", "findings": [{"type": "EMAIL", "span": None}]})
    corpus = load_corpus(_write(tmp_path, [row]), name="x")
    assert corpus.cases[0].expect_findings[0].span is None


@pytest.mark.parametrize(
    "span",
    [
        [9],
        [9, 26, 30],
        ["9", "26"],
        [9.0, 26.0],
        [True, 26],
        [9, False],
        [0, True],
        [-1, 26],
        [26, 9],
        [9, 9],
        [9, 27],
        [0, 27],
        "9:26",
        {"start": 9, "end": 26},
    ],
)
def test_a_span_that_does_not_fit_its_text_is_rejected(tmp_path: Path, span: object) -> None:
    """The text is 26 characters, so [9, 27] points past its end.

    `[True, 26]` is the quiet one: bool is a subclass of int, so it slices as
    [1, 26] and produces a false negative that reads exactly like a detector
    bug.
    """
    row = _row(expect={"decision": "redact", "findings": [{"type": "EMAIL", "span": span}]})
    with pytest.raises(CorpusError, match="line 1"):
        load_corpus(_write(tmp_path, [row]), name="x")


def test_a_span_may_cover_the_whole_text(tmp_path: Path) -> None:
    """The upper bound is inclusive of the end index: [0, len(text)] is legal.

    The brief's own fixture ends at 26 of 26 characters, so an off-by-one here
    would reject the plan's example corpora wholesale.
    """
    row = _row(
        text="alice@example.com",
        expect={"decision": "redact", "findings": [{"type": "EMAIL", "span": [0, 17]}]},
    )
    assert load_corpus(_write(tmp_path, [row]), name="x").cases[0].expect_findings[0].span == (
        0,
        17,
    )


def test_duplicate_ids_name_both_lines(tmp_path: Path) -> None:
    """Which two lines collide is the whole content of the message."""
    with pytest.raises(CorpusError, match="line 3.*line 1"):
        load_corpus(_write(tmp_path, [CASE, _row(id="pii-0002"), dict(CASE)]), name="x")


def test_mixed_sources_name_the_diverging_line(tmp_path: Path) -> None:
    other = dict(CASE, id="pii-0002", source="third-party")
    with pytest.raises(CorpusError, match="line 2"):
        load_corpus(_write(tmp_path, [CASE, other]), name="x")


def test_a_file_mixing_licences_reports_every_one_of_them(tmp_path: Path) -> None:
    """Taking the first case's licence would mislabel the rest of the file.

    The second licence sorts BEFORE the first deliberately. With "CC-BY-4.0" on
    line 2 the expected string is the same under sorted order and under file
    order, and an assertion satisfied by two independent orderings at once pins
    neither -- the defect Task 8 found in its registry-order test.
    """
    other = dict(CASE, id="pii-0002", license="0BSD")
    corpus = load_corpus(_write(tmp_path, [CASE, other]), name="x")
    assert corpus.license == "0BSD, Apache-2.0"


def test_whitespace_only_lines_are_skipped(tmp_path: Path) -> None:
    path = _write_raw(tmp_path, json.dumps(CASE) + "\n   \n\t\n\n")
    assert len(load_corpus(path, name="x").cases) == 1


def test_a_file_of_only_whitespace_is_rejected(tmp_path: Path) -> None:
    """ "Only blank lines" is not "an empty corpus is fine"."""
    with pytest.raises(CorpusError, match="no cases"):
        load_corpus(_write_raw(tmp_path, "\n   \n\t\n"), name="x")


def test_a_completely_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="no cases"):
        load_corpus(_write_raw(tmp_path, ""), name="x")


def test_a_line_without_a_trailing_newline_still_loads(tmp_path: Path) -> None:
    path = _write_raw(tmp_path, json.dumps(CASE))
    assert len(load_corpus(path, name="x").cases) == 1


def test_carriage_returns_do_not_break_a_windows_authored_corpus(tmp_path: Path) -> None:
    path = _write_raw(tmp_path, json.dumps(CASE) + "\r\n\r\n")
    assert len(load_corpus(path, name="x").cases) == 1


def test_a_text_holding_an_exotic_line_separator_still_loads(tmp_path: Path) -> None:
    """JSONL is delimited by "\\n". str.splitlines() is not.

    splitlines() also breaks on U+2028, U+2029 and U+0085, which are ordinary
    characters inside a JSON string and appear in scraped text. Splitting on one
    tears a single valid case into two unparseable halves, so a corpus holding
    one character of real-world punctuation fails to load at all.
    """
    row = _clean(text="one\u2028two")
    path = _write_raw(tmp_path, json.dumps(row, ensure_ascii=False) + "\n")
    assert load_corpus(path, name="x").cases[0].text == "one\u2028two"


def test_a_path_given_as_a_string_still_loads(tmp_path: Path) -> None:
    """`Path(path)` in the loader is tolerance for an untyped caller, so pin it."""
    path = _write(tmp_path, [CASE])
    assert len(load_corpus(cast("Path", str(path)), name="x").cases) == 1


def test_a_file_that_is_not_utf8_is_refused_as_a_corpus_problem(tmp_path: Path) -> None:
    """A stray byte is a malformed corpus, not a traceback for the caller.

    `read_text` raises UnicodeDecodeError, which escaped `load_corpus`
    undocumented while the docstring promised CorpusError for anything wrong
    with the file's content. Task 13's CLI catches CorpusError, so the one byte
    turned "this corpus is malformed" into a stack trace.
    """
    path = tmp_path / "c.jsonl"
    path.write_bytes(b'{"id": "a", "text": "\xff"}\n')
    with pytest.raises(CorpusError, match="UTF-8"):
        load_corpus(path, name="x")


def test_a_repeated_json_key_is_rejected(tmp_path: Path) -> None:
    """Last-wins parsing is invisible to review-by-diff.

    It cannot fudge a number on its own -- whichever value wins is the one
    hashed, so the version still moves -- but two lines that read differently
    and load identically defeat the control this module exists to provide,
    which is that a corpus edit is legible in a diff.
    """
    line = '{"license": "MIT", ' + json.dumps(CASE)[1:]
    with pytest.raises(CorpusError, match="repeats"):
        load_corpus(_write_raw(tmp_path, line + "\n"), name="x")


def test_a_repeated_json_key_inside_a_nested_object_is_rejected(tmp_path: Path) -> None:
    """The hook runs on every object in the line, not just the outermost."""
    line = json.dumps(CASE).replace('"expect": {', '"expect": {"decision": "allow", ', 1)
    with pytest.raises(CorpusError, match="repeats"):
        load_corpus(_write_raw(tmp_path, line + "\n"), name="x")


def test_a_rejected_value_is_never_echoed_into_the_error(tmp_path: Path) -> None:
    """`text` carries PII and credentials by design; a CI log is not the place.

    detectors/secrets.py works to keep matched text out of the audit record. A
    loader that prints the offending value into a build log undoes that on the
    first malformed line, and the line number already identifies the row.
    """
    secret = "sk-live-51H8xQ2vR9mNpL3kT7wY"
    with pytest.raises(CorpusError) as exc:
        load_corpus(_write(tmp_path, [_row(text=[secret])]), name="x")
    assert secret not in str(exc.value)
    assert "list" in str(exc.value)


def test_a_byte_order_mark_is_refused_and_the_message_names_the_fix(tmp_path: Path) -> None:
    """Recorded, not fixed: a BOM is refused where CRLF is tolerated.

    Nothing loads silently and json's own message names the remedy
    ("decode using utf-8-sig"), so the asymmetry costs a reader one error
    message rather than a wrong number.
    """
    path = tmp_path / "c.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(CASE).encode("utf-8") + b"\n")
    with pytest.raises(CorpusError, match="utf-8-sig"):
        load_corpus(path, name="x")


def test_every_error_names_the_file(tmp_path: Path) -> None:
    path = _write_raw(tmp_path, "{not json\n", "named.jsonl")
    with pytest.raises(CorpusError, match="named.jsonl"):
        load_corpus(path, name="x")


def test_a_missing_file_is_not_disguised_as_a_malformed_one(tmp_path: Path) -> None:
    """Deliberate: `FileNotFoundError` says the corpus is absent, not malformed.

    Task 13's CLI catches CorpusError to report a bad corpus and carry on with
    an exit code; a file that is not there is a different fault and must not be
    reported as a schema problem.
    """
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "absent.jsonl", name="x")


def test_changing_a_case_direction_moves_the_version(tmp_path: Path) -> None:
    """`direction` is measured input, so it belongs in the digest with the rest.

    `evaluate` passes `case.direction` straight into `guardrail.check`, which is
    exactly the sense in which the digest claims to cover what is measured.
    Latent today because both bundled detectors take a Context and never read
    it, and not latent for a classifier, which is the obvious Phase 2 case. It
    is also the last cheap moment: once Task 15 writes baselines.json every
    version string is load-bearing and the input set stops being editable.
    """
    out = load_corpus(_write(tmp_path, [_clean(direction="output")], "a.jsonl"), name="x")
    inp = load_corpus(_write(tmp_path, [_clean(direction="input")], "b.jsonl"), name="x")
    assert out.cases[0].text == inp.cases[0].text
    assert out.cases[0].direction != inp.cases[0].direction
    assert out.version != inp.version


def test_a_finding_type_that_renders_like_a_span_does_not_collide(tmp_path: Path) -> None:
    """Why `None` is RENDERED rather than skipped, stated correctly.

    Skipping it would NOT hide a dropped span: the finding count keeps each
    record self-delimiting, so a shorter field list moves the version anyway.
    What skipping loses is injectivity. These two corpora hold different
    expectations -- Task 10 scores them differently, since a span-free finding
    matches on type alone -- and with `None` contributing no field both flatten
    to the same field list and the same version.
    """
    pair = [{"type": "EMAIL", "span": None}, {"type": "0:1", "span": [0, 1]}]
    swapped = [{"type": "EMAIL", "span": [0, 1]}, {"type": "0:1", "span": None}]
    a = _row(text="ab", expect={"decision": "redact", "findings": pair})
    b = _row(text="ab", expect={"decision": "redact", "findings": swapped})
    one = load_corpus(_write(tmp_path, [a], "a.jsonl"), name="x")
    two = load_corpus(_write(tmp_path, [b], "b.jsonl"), name="x")
    assert one.version != two.version


def test_relabelling_a_case_moves_the_version(tmp_path: Path) -> None:
    """The cheapest fudge there is, and the one the derived version must catch.

    Changing an expectation to match whatever the detector currently does turns
    a failing check green. Under a digest of id and text alone that edit is
    invisible: same version, same Task 12 baseline key, better numbers, nothing
    to review. The text is asserted identical here so the only difference
    between the two corpora is the label.
    """
    honest = load_corpus(_write(tmp_path, [CASE], "a.jsonl"), name="x")
    fudged = load_corpus(
        _write(tmp_path, [_row(expect={"decision": "allow", "findings": []})], "b.jsonl"),
        name="x",
    )
    assert honest.cases[0].text == fudged.cases[0].text
    assert honest.cases[0].expect_decision != fudged.cases[0].expect_decision
    assert honest.version != fudged.version


@pytest.mark.parametrize(
    ("edit", "expect"),
    [
        ("decision", {"decision": "deny", "findings": [{"type": "EMAIL", "span": [9, 26]}]}),
        ("type", {"decision": "redact", "findings": [{"type": "US_SSN", "span": [9, 26]}]}),
        ("span moved", {"decision": "redact", "findings": [{"type": "EMAIL", "span": [10, 26]}]}),
        ("span widened", {"decision": "redact", "findings": [{"type": "EMAIL", "span": [9, 25]}]}),
        ("span dropped", {"decision": "redact", "findings": [{"type": "EMAIL", "span": None}]}),
        ("finding removed", {"decision": "redact", "findings": []}),
        (
            "finding added",
            {
                "decision": "redact",
                "findings": [
                    {"type": "EMAIL", "span": [9, 26]},
                    {"type": "PHONE_NUMBER", "span": [0, 5]},
                ],
            },
        ),
    ],
)
def test_every_expectation_edit_moves_the_version(
    tmp_path: Path, edit: str, expect: object
) -> None:
    """Each dimension of the label separately, not just the whole label at once.

    "span dropped" is the sharpest: `null` is a WEAKER expectation than a span,
    so it buys precision and recall on its own, and a digest that rendered None
    as nothing at all would not notice it.
    """
    base = load_corpus(_write(tmp_path, [CASE], "base.jsonl"), name="x")
    edited = load_corpus(_write(tmp_path, [_row(expect=expect)], "edited.jsonl"), name="x")
    assert base.version != edited.version, edit


def test_reordering_findings_within_a_case_moves_the_version(tmp_path: Path) -> None:
    """`expect_findings` is ordered, so its order is content like any other."""
    findings = [{"type": "EMAIL", "span": [9, 26]}, {"type": "US_SSN", "span": [0, 5]}]
    forwards = _row(expect={"decision": "redact", "findings": findings})
    backwards = _row(expect={"decision": "redact", "findings": list(reversed(findings))})
    a = load_corpus(_write(tmp_path, [forwards], "a.jsonl"), name="x")
    b = load_corpus(_write(tmp_path, [backwards], "b.jsonl"), name="x")
    assert a.version != b.version


def test_reordering_lines_that_differ_only_in_expectations_is_stable(tmp_path: Path) -> None:
    """A fixture-regression guard, not a stronger order test.

    It catches nothing the brief's reordering test misses. Replacing that test's
    fixture -- it carried a span that did not fit its text -- left it holding two
    cases that differ in expectations as well as in text, so both tests go red on
    every order mutation. What this one pins is that the property still holds for
    cases differing ONLY in their labels, so a later edit to OTHER cannot quietly
    narrow the order guarantee back to texts alone.
    """
    first = _row(id="pii-0001")
    second = _row(id="pii-0002", expect={"decision": "allow", "findings": []})
    a = load_corpus(_write(tmp_path, [first, second], "a.jsonl"), name="x")
    b = load_corpus(_write(tmp_path, [second, first], "b.jsonl"), name="x")
    assert a.version == b.version


def test_version_changes_when_only_the_text_changes(tmp_path: Path) -> None:
    """Ids alone would let an edit to the content keep the old baseline."""
    a = load_corpus(_write(tmp_path, [_clean(text="one")], "a.jsonl"), name="x")
    b = load_corpus(_write(tmp_path, [_clean(text="two")], "b.jsonl"), name="x")
    assert a.version != b.version


def test_version_changes_when_only_an_id_changes(tmp_path: Path) -> None:
    a = load_corpus(_write(tmp_path, [_clean(id="one")], "a.jsonl"), name="x")
    b = load_corpus(_write(tmp_path, [_clean(id="two")], "b.jsonl"), name="x")
    assert a.version != b.version


def test_version_does_not_depend_on_the_corpus_name_or_file(tmp_path: Path) -> None:
    """The version identifies the CONTENT, so a rename must not move a baseline."""
    a = load_corpus(_write(tmp_path, [CASE], "a.jsonl"), name="pii/in-repo")
    b = load_corpus(_write(tmp_path, [CASE], "b.jsonl"), name="something/else")
    assert a.version == b.version


def test_version_is_twelve_lowercase_hex_characters(tmp_path: Path) -> None:
    version = load_corpus(_write(tmp_path, [CASE]), name="x").version
    assert len(version) == 12
    assert all(c in "0123456789abcdef" for c in version)


def test_the_boundary_between_id_and_text_is_unambiguous() -> None:
    """A separator that can occur inside a field does not separate.

    Joining id and text with a delimiter byte makes ("a", "b<NUL>c") and
    ("a<NUL>b", "c") hash identically, so two different corpora share one
    version and one baseline. Length-prefixing each field removes the question.
    """
    one = Corpus(
        name="x",
        source="unit-fixture",
        license="CC0-1.0",
        cases=(_built_case(case_id="a", text="b\x00c"),),
    )
    two = Corpus(
        name="x",
        source="unit-fixture",
        license="CC0-1.0",
        cases=(_built_case(case_id="a\x00b", text="c"),),
    )
    assert one.version != two.version


def test_a_corpus_with_no_cases_is_refused() -> None:
    """Zero cases score precision 1.0 and recall 1.0 out of nothing."""
    with pytest.raises(CorpusError, match="at least one case"):
        Corpus(name="x", source="in-repo", license="Apache-2.0", cases=())


def test_a_corpus_with_duplicate_ids_is_refused() -> None:
    """The version sorts by id, so a repeated id is also an unstable sort key."""
    with pytest.raises(CorpusError, match="duplicate"):
        Corpus(
            name="x",
            source="unit-fixture",
            license="CC0-1.0",
            cases=(_built_case(text="one"), _built_case(text="two")),
        )


def test_a_corpus_whose_source_disagrees_with_its_cases_is_refused() -> None:
    """`source` is reported next to the number; it may not be decorative."""
    with pytest.raises(CorpusError, match="source"):
        Corpus(name="x", source="third-party", license="CC0-1.0", cases=(_built_case(),))


def test_a_corpus_whose_licence_disagrees_with_its_cases_is_refused() -> None:
    with pytest.raises(CorpusError, match="license"):
        Corpus(name="x", source="unit-fixture", license="Apache-2.0", cases=(_built_case(),))


def test_a_corpus_holding_something_that_is_not_a_case_is_refused() -> None:
    cases = cast("tuple[Case, ...]", ("pii-0001",))
    with pytest.raises(CorpusError, match="Case"):
        Corpus(name="x", source="unit-fixture", license="CC0-1.0", cases=cases)


def test_a_direction_outside_the_type_is_refused_at_construction() -> None:
    """The dataclass guards this too, not only the loader's parse path."""
    with pytest.raises(CorpusError, match="direction"):
        _built_case(direction=cast(Direction, "sideways"))


def test_a_decision_outside_the_type_is_refused_at_construction() -> None:
    with pytest.raises(CorpusError, match="decision"):
        _built_case(decision=cast(Decision, "block"))


def test_findings_given_as_a_bare_string_are_refused_at_construction() -> None:
    """tuple("EMAIL") is five one-character findings, not a TypeError."""
    with pytest.raises(CorpusError, match="must be a list"):
        _built_case(findings=cast("tuple[ExpectedFinding, ...]", "EMAIL"))


def test_a_finding_that_is_not_an_ExpectedFinding_is_refused_at_construction() -> None:
    """Task 10 reads `.span` and `.type` off each of these without asking."""
    with pytest.raises(CorpusError, match="ExpectedFinding"):
        _built_case(findings=cast("tuple[ExpectedFinding, ...]", ("EMAIL",)))


def test_a_span_past_the_end_of_its_text_is_refused_at_construction() -> None:
    with pytest.raises(CorpusError, match="span"):
        _built_case(text="short", findings=(ExpectedFinding("EMAIL", (0, 99)),))


def test_an_empty_span_is_refused_at_construction() -> None:
    """A zero-width span is a non-detection wearing a detection's clothes."""
    with pytest.raises(CorpusError, match="span"):
        ExpectedFinding("EMAIL", (5, 5))


def test_a_negative_span_start_is_refused_at_construction() -> None:
    """text[-1:3] is legal Python and silently the wrong stretch of text."""
    with pytest.raises(CorpusError, match="span"):
        ExpectedFinding("EMAIL", (-1, 3))


def test_a_finding_list_is_frozen_into_the_case() -> None:
    """A caller keeping the list it passed must not be able to edit it after."""
    findings = [ExpectedFinding("EMAIL", (0, 5))]
    case = _built_case(text="alice", findings=cast("tuple[ExpectedFinding, ...]", findings))
    findings.clear()
    assert len(case.expect_findings) == 1
    assert isinstance(case.expect_findings, tuple)


def test_a_case_list_is_frozen_into_the_corpus() -> None:
    cases = [_built_case()]
    corpus = Corpus(
        name="x",
        source="unit-fixture",
        license="CC0-1.0",
        cases=cast("tuple[Case, ...]", cases),
    )
    cases.clear()
    assert len(corpus.cases) == 1
    assert isinstance(corpus.cases, tuple)
