"""Every registered check carries the same six artifacts, or the build says so.

Adding a check to this repository is six files and one registry line, and the
expensive failure is not getting one of them wrong: it is leaving one out. A
check with no conformance section is unportable; one with no NOTICE section
breaks a licence obligation the NOTICE states about itself; one whose types
collide with another's makes a merged placeholder ambiguous and a per-type row
the sum of two measurements.

None of those has any other alarm. Each of them looks exactly like a working
check from every other angle, including a green suite and a published number.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from jamjet_guardrails.detectors import AVAILABLE, TYPES, build
from jamjet_guardrails.eval.corpus import load_corpus
from jamjet_guardrails.eval.fixtures import FIXTURES, options_for

ROOT = Path(__file__).resolve().parent.parent
CORPORA = ROOT / "corpora"
BASELINES = CORPORA / "baselines.json"
NOTICE = CORPORA / "NOTICE.md"
CONFORMANCE = ROOT / "docs" / "conformance.md"
README = ROOT / "README.md"
BENCHMARKS = ROOT / "BENCHMARKS.md"
SCAFFOLD = ROOT / "scripts" / "new_check.py"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

# The claim CONTRIBUTING makes about how much code a check costs. Held to the
# scaffold's own output rather than to a sentence, because "about twenty-five
# lines" is a number in prose and a number in prose is a claim.
_DETECTOR_LINE_BUDGET = 25

CHECKS = sorted(AVAILABLE)


def test_there_is_something_to_check() -> None:
    """The guard on every parametrised test below. An empty registry would
    make all of them vacuously green."""
    assert len(CHECKS) >= 4


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_has_a_corpus_directory_with_at_least_one_corpus(check: str) -> None:
    directory = CORPORA / check
    assert directory.is_dir(), f"{check} is registered with no {directory} directory"
    corpora = sorted(directory.glob("*.jsonl"))
    assert corpora, f"{directory} holds no .jsonl corpus, so {check} publishes no number"


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_has_a_recorded_baseline(check: str) -> None:
    import json

    baselines = json.loads(BASELINES.read_text(encoding="utf-8"))
    keys = [key for key in baselines if key.startswith(f"{check}/")]
    assert keys, f"{check} has no entry in {BASELINES}, so nothing gates its score"


def _headline_table(text: str, path: Path) -> str:
    """The rows of the published headline table, and nothing else in the file.

    README.md carries a second table naming every check ("The checks", with
    backtick-quoted names), and BENCHMARKS.md is generated from the same
    template the headline table in README uses. A file-wide substring match
    cannot tell that second table from this one, so a mutation that deletes a
    row from the headline table alone passed the test this replaces: the row
    was still found, in the OTHER table. The headline table is the one CI
    diffs against the generated benchmarks, so it is the row that carries the
    published number, and this is scoped to it: the header row starting
    "| Check | Corpus" through the next blank line.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("| Check | Corpus"):
            start = index
            break
    else:
        raise AssertionError(f"{path.name} has no headline table (no '| Check | Corpus' row)")
    end = start
    while end < len(lines) and lines[end].strip():
        end += 1
    return "\n".join(lines[start:end])


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_has_a_row_in_both_published_tables(check: str) -> None:
    for path in (BENCHMARKS, README):
        table = _headline_table(path.read_text(encoding="utf-8"), path)
        assert f"| {check} |" in table, (
            f"{check} has no row in the headline table (starting '| Check | Corpus |') "
            f"of {path.name}"
        )


# The document covers two checks through its GENERIC sections rather than
# through one of their own, and says so in a sentence naming both. That
# sentence is the exemption, and it lives in the porting contract rather than
# in a list here on purpose: a hand list in a test is an exemption a new check
# can join by editing the test, and every exemption in this repository that
# approximated a set with a shape turned into the channel it was written to
# deny. Written into the contract instead, claiming it is a reviewable edit to
# a published document.
_COVERED_BY_THE_GENERAL_SECTIONS = "enough to port them"


def _sentences(text: str) -> list[str]:
    """The document as sentences, whitespace collapsed first.

    Collapsing every run of whitespace to one space means a paragraph rewrapped
    across a different set of lines produces the same sentences it did before,
    so a document's line breaks stay free to change without breaking a test
    that has nothing to do with them: this used to require the waiver phrase
    and a check's name to share one LINE, and rewrapping the paragraph that
    covers `pii` and `secrets` broke it for a reason no message explained.
    Splitting on `.`, `!` or `?` followed by whitespace treats the colon in
    "enough to port them: their type names are..." as inside one sentence,
    which it is.
    """
    collapsed = " ".join(text.split())
    return re.split(r"(?<=[.!?])\s+", collapsed)


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_is_covered_by_the_porting_contract(check: str) -> None:
    """A check nobody can port is a check whose corpus cannot grade a port.

    Either the document gives it a section of its own, or the document states,
    in the SAME SENTENCE as the waiver phrase, that its general sections are
    enough for it. Nothing else counts, and in particular the check's name
    appearing anywhere else in the document does not: matching the whole
    document rather than the one sentence would pass for a name that sits in
    an unrelated paragraph and never claims to cover this check at all.
    """
    text = CONFORMANCE.read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    if any(check in heading for heading in headings):
        return
    waivers = [
        sentence
        for sentence in _sentences(text)
        if _COVERED_BY_THE_GENERAL_SECTIONS in sentence and f"`{check}`" in sentence
    ]
    assert waivers, (
        f"docs/conformance.md neither gives {check} a section nor says, in the same "
        f"sentence as {_COVERED_BY_THE_GENERAL_SECTIONS!r}, that its general sections "
        f"are enough for it; headings are {headings}"
    )


@pytest.mark.parametrize("check", CHECKS)
def test_every_in_repo_corpus_is_declared_in_the_notice(check: str) -> None:
    text = NOTICE.read_text(encoding="utf-8")
    assert f"corpora/{check}/in-repo.jsonl" in text, (
        f"corpora/{check}/in-repo.jsonl has no provenance entry in corpora/NOTICE.md"
    )


@pytest.mark.parametrize("check", CHECKS)
def test_every_type_a_check_declares_is_labelled_somewhere_in_its_corpora(check: str) -> None:
    """A type nobody can label has no recall figure, and a per-type row with no
    recall is a capability nobody measured."""
    labelled: set[str] = set()
    for path in sorted((CORPORA / check).glob("*.jsonl")):
        corpus = load_corpus(path, name=f"{check}/{path.stem}")
        for case in corpus.cases:
            labelled |= {finding.type for finding in case.expect_findings}
    unlabelled = sorted(TYPES[check] - labelled)
    assert unlabelled == [], f"{check} declares {unlabelled} and no corpus case expects any of them"


@pytest.mark.parametrize("check", CHECKS)
def test_no_corpus_labels_a_type_its_check_does_not_declare(check: str) -> None:
    """The other direction, and the one that fails quietly: a label nothing can
    produce is a permanent false negative, which lowers a published recall for
    a reason no reader can see."""
    for path in sorted((CORPORA / check).glob("*.jsonl")):
        corpus = load_corpus(path, name=f"{check}/{path.stem}")
        for case in corpus.cases:
            undeclared = sorted({f.type for f in case.expect_findings} - TYPES[check])
            assert undeclared == [], f"{path.name} case {case.id} labels {undeclared}"


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_builds_from_its_fixture_or_from_nothing(check: str) -> None:
    """The harness builds by name, so a check that needs options and has no
    fixture cannot be scored at all, and one whose fixture is wrong fails here
    rather than inside a measurement."""
    guardrail = build(check, **options_for(check))
    assert guardrail.name == check


def test_every_fixture_names_a_registered_check() -> None:
    unknown = sorted(set(FIXTURES) - set(AVAILABLE))
    assert unknown == [], f"{unknown} have fixtures and are not registered"


@pytest.mark.parametrize("check", CHECKS)
def test_a_fixture_selects_only_types_the_check_declares(check: str) -> None:
    fixture = options_for(check)
    named: set[str] = set()
    for key in ("patterns", "banned"):
        value = fixture.get(key)
        if isinstance(value, dict):
            named |= set(value)
    undeclared = sorted(named - TYPES[check])
    assert undeclared == [], f"the {check} fixture names {undeclared}, which it does not declare"


def test_the_scaffold_writes_a_check_whose_detector_fits_the_budget(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "example-check", "--into", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    detector = tmp_path / "src" / "jamjet_guardrails" / "detectors" / "example_check.py"
    assert detector.is_file()
    lines = [
        line
        for line in detector.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) <= _DETECTOR_LINE_BUDGET, (
        f"the scaffolded detector is {len(lines)} lines; CONTRIBUTING claims a check "
        f"costs about {_DETECTOR_LINE_BUDGET}"
    )


def test_the_scaffold_writes_every_artifact_the_completeness_tests_demand(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCAFFOLD), "example-check", "--into", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    for relative in (
        "src/jamjet_guardrails/detectors/example_check.py",
        "corpora/example-check/in-repo.jsonl",
        "tests/test_example_check.py",
    ):
        assert (tmp_path / relative).is_file(), f"the scaffold did not write {relative}"


def test_the_scaffold_refuses_a_name_that_is_already_registered(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "pii", "--into", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "registered" in result.stderr


def test_the_scaffold_refuses_a_name_outside_the_registry_naming_rule(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCAFFOLD), "Bad_Name", "--into", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_contributing_documents_the_scaffold_it_tells_people_to_run() -> None:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    assert "scripts/new_check.py" in text
    assert str(_DETECTOR_LINE_BUDGET) in text
