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


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_has_a_row_in_both_published_tables(check: str) -> None:
    for path in (BENCHMARKS, README):
        text = path.read_text(encoding="utf-8")
        assert f"| {check} |" in text or f"| `{check}` |" in text, (
            f"{check} has no row in {path.name}"
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


@pytest.mark.parametrize("check", CHECKS)
def test_every_check_is_covered_by_the_porting_contract(check: str) -> None:
    """A check nobody can port is a check whose corpus cannot grade a port.

    Either the document gives it a section of its own, or the document states
    that its general sections are enough for it. Nothing else counts, and in
    particular the check's name appearing in passing does not.
    """
    text = CONFORMANCE.read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    if any(check in heading for heading in headings):
        return
    waivers = [
        line
        for line in text.splitlines()
        if _COVERED_BY_THE_GENERAL_SECTIONS in line and f"`{check}`" in line
    ]
    assert waivers, (
        f"docs/conformance.md neither gives {check} a section nor says its general "
        f"sections are enough for it; headings are {headings}"
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
