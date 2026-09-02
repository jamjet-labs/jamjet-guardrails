"""The conformance document is nothing but claims, so it is checked like one.

Almost nothing here asserts that a sentence is present and calls that a test.
Every number, field list, ordering and worked example in `docs/conformance.md`
is RECOMPUTED from the code it describes and then compared against the
document, so the failure this file exists to catch is the document going stale
while the implementation moves underneath it.

That failure has no other alarm. A porter reads this document as the contract
and cannot see the Python; a claim that drifts sends them to build something
this implementation does not do, and the corpora they are graded on will not
tell them which of the two is wrong.

Some claims genuinely cannot be recomputed: that detector internals are
unspecified, that a port is free to lay its modules out as it likes. This file
does not pretend otherwise. What CAN be recomputed is recomputed, including the
single-pass rewriting rule, which `tests/test_chain.py` pins against the
composition it was written for.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

import pytest

from jamjet_guardrails import (
    ChainResult,
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Origin,
    Provenance,
    Verdict,
    build,
    combine,
    saw,
)
from jamjet_guardrails.detectors.injection_structural import INJECTION_TYPES
from jamjet_guardrails.eval.corpus import Case, Corpus, ExpectedFinding, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "conformance.md"
NOTICE = ROOT / "corpora" / "NOTICE.md"
SCREEN = ROOT / "tests" / "test_corpora.py"

REQUIRED_SECTIONS = [
    "## Verdict fields",
    "## Combination order",
    "## Single-pass rewriting",
    "## The saw hash",
    "## Corpus schema",
    "## The injection-structural constraint",
    "## Third-party corpora",
    "## What is deliberately unspecified",
]


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """The body of one H2, up to the next one.

    Asserts the heading is present rather than letting `split(...)[1]` raise an
    IndexError, because a missing section and a section whose content is wrong
    are different failures and only one of them is this helper's fault.
    """
    text = _text()
    assert heading in text, f"the conformance doc has no {heading!r} section"
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


# ==========================================================================
# Shape: the sections the document must carry at all.
# ==========================================================================


def test_conformance_doc_exists() -> None:
    assert DOC.is_file()


def test_conformance_doc_covers_every_required_section() -> None:
    text = _text()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert missing == [], f"conformance doc is missing {missing}"


def test_the_required_sections_appear_in_the_order_this_file_lists_them() -> None:
    """The intro's contract boundary is positional, so position has to be checked.

    It reads "everything above Third-party corpora is contract". That replaced
    an ordinal -- "the first five sections" -- which was wrong the moment a
    sixth contract section was added and which nothing had been checking. The
    replacement is better because it cannot go stale by counting, and it can
    still go stale by MOVING: a contract section dropped below the boundary
    would be silently declared free. Presence alone cannot see that.
    """
    text = _text()
    positions = [text.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions), (
        "the conformance doc's sections are out of order; the intro claims "
        "everything above Third-party corpora is contract, which depends on it. "
        f"Order found: {[s for _, s in sorted(zip(positions, REQUIRED_SECTIONS))]}"
    )


def test_no_design_spec_travels_with_the_code() -> None:
    """INVERTED from `test_the_spec_travelled_with_the_code`, which pinned it in place.

    The design spec was tracked, shipped in the sdist, and held there by the
    assertion this replaces. It carried an unreleased finding about a separate
    product, internal ticket ids and module paths, a roadmap, a status line that
    was false on the day it would have published, and a CLI invocation that does
    not exist. The guards against exactly that copy reach README.md and
    conformance.md and not the file this test pinned.

    Deliberately vague about the finding, and that is the point: an earlier
    draft of this docstring QUOTED it, so the test written to keep the
    disclosure out of the repository put the disclosure into the repository, and
    into every sdist. A guard that must name what it excludes has to name it
    without reproducing it.

    `docs/conformance.md` is the design rationale that belongs in public, and it
    is checked claim by claim against the code. A spec is a working document; it
    is now gitignored, so it can exist locally and cannot ship.

    Tracked files, not files on disk: the working tree is where a spec is
    supposed to live. What must not happen is one being committed, since
    hatchling builds the sdist from what git tracks.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "docs/specs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == [], f"design specs are tracked and would ship in the sdist: {tracked}"


def test_no_document_that_ships_describes_work_that_does_not_exist() -> None:
    """The forward-looking-copy guard, over every Markdown file that ships.

    Two such guards existed and each covered exactly one file: README.md had
    one, this document had one, and the design spec that shipped between them
    carried a roadmap, a section of open questions and a `Status: not yet
    implemented` line. The guards were written for the case in hand and not for
    the case one step over, so the file with the most of what they ban was the
    one neither reached.

    Tracked Markdown is the domain because that is what hatchling puts in the
    sdist, so a document is in scope exactly when a reader can receive it.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked, "no tracked Markdown found; this check would prove nothing"
    promised = {
        name: found
        for name in tracked
        for found in [
            [
                word
                for word in ("coming soon", "todo", "fixme", "roadmap", "not yet implemented")
                if word in (ROOT / name).read_text(encoding="utf-8").lower()
            ]
        ]
        if found
    }
    assert promised == {}, f"published documents promise future work: {promised}"


def test_the_document_describes_only_what_exists() -> None:
    """No roadmap, no pending work, no "coming soon".

    A conformance document that mixes what is specified with what is intended
    gives a porter no way to tell which half they are obliged to match.
    """
    text = _text().lower()
    promised = [word for word in ("coming soon", "todo", "fixme", "roadmap") if word in text]
    assert promised == [], f"the conformance doc promises future work: {promised}"


# ==========================================================================
# Verdict fields: derived from the dataclasses, not from memory.
# ==========================================================================

# Verdict, Provenance and Finding are what the brief names. Context and
# ChainResult are here because `check` takes one and the chain returns the
# other, so a porter who cannot see their fields cannot call the contract.
DOCUMENTED_TYPES: tuple[type[Any], ...] = (Verdict, Provenance, Finding, Context, ChainResult)


def _documented_fields(cls_name: str) -> set[str]:
    """The field names in one type's own table, read as table rows.

    Rows, not "appears somewhere in the section", and the difference is not
    academic. Deleting the `error` row from the `Verdict` table left this test
    green, because `error` is also named in the invariants prose two
    subsections below. A membership test over the whole section grades a
    document on whether it happens to mention a word.
    """
    section = _section("## Verdict fields")
    heading = f"### `{cls_name}`"
    assert heading in section, f"the doc has no {heading} subsection"
    body = section.split(heading, 1)[1].split("\n### ", 1)[0]
    return set(re.findall(r"^\|\s*`([^`]+)`\s*\|", body, re.MULTILINE))


@pytest.mark.parametrize("cls", DOCUMENTED_TYPES, ids=lambda c: str(c.__name__))
def test_the_verdict_fields_section_tabulates_every_field_of_every_documented_type(
    cls: type[Any],
) -> None:
    """Derived from `dataclasses.fields`, so adding a field is a red test here.

    A field added to `Verdict` and left out of the document is a piece of the
    contract a second implementation never learns about, and every other check
    in this repository would stay green.

    Both directions, because a rename produces both faults at once: the new
    name is undocumented and the old row still stands, describing a field this
    library no longer has.
    """
    documented = _documented_fields(cls.__name__)
    names = {f.name for f in dataclasses.fields(cls)}
    assert names, f"{cls.__name__} has no fields; this check would prove nothing"
    assert sorted(names - documented) == [], (
        f"{cls.__name__} fields with no row in the doc: {sorted(names - documented)}"
    )
    assert sorted(documented - names) == [], (
        f"the doc gives {cls.__name__} rows for fields it does not have: "
        f"{sorted(documented - names)}"
    )


@pytest.mark.parametrize(
    ("name", "alias"),
    [("Decision", Decision), ("Direction", Direction), ("Kind", Kind), ("Origin", Origin)],
)
def test_the_document_names_every_value_of_every_literal_in_the_contract(
    name: str, alias: Any
) -> None:
    """A value domain a porter cannot see is a value domain they will not accept."""
    values = get_args(alias)
    assert values, f"{name} has no values; this check would prove nothing"
    text = _text()
    missing = [value for value in values if f"`{value}`" not in text]
    assert missing == [], f"{name} values absent from the conformance doc: {missing}"


# ==========================================================================
# Combination order: the ranking is recovered from `combine` itself.
# ==========================================================================


def test_the_combination_order_section_states_the_order_combine_implements() -> None:
    """The order is recovered from behaviour, never copied from the document.

    Rank each decision by how many decisions it survives combination with.
    `deny` wins against all three, `redact` against two, `allow` against
    itself, which recovers the severity order without reading `_SEVERITY`.
    Reorder severity in types.py and this fails, pointing at the sentence that
    is now wrong rather than at the code that is now right.
    """
    values: tuple[Decision, ...] = get_args(Decision)
    rank = {v: sum(1 for other in values if combine(v, other) == v) for v in values}
    assert len(set(rank.values())) == len(values), f"combination order is not total: {rank}"
    order = sorted(values, key=lambda v: -rank[v])
    expected = " > ".join(f"`{v}`" for v in order)
    assert expected in _section("## Combination order"), (
        f"the doc does not state the order combine implements, which is {expected}"
    )


# ==========================================================================
# The saw hash: length and worked vectors, both recomputed.
# ==========================================================================

SAW_VECTORS = ("", "alice@example.com")


def test_the_saw_section_publishes_the_digest_length_saw_produces() -> None:
    section = _section("## The saw hash")
    stated = re.search(r"(\d+) lowercase hex characters", section)
    assert stated is not None, "the saw section does not state the digest length"
    assert int(stated.group(1)) == len(saw("anything at all"))


@pytest.mark.parametrize("content", SAW_VECTORS)
def test_the_saw_section_publishes_vectors_this_implementation_reproduces(content: str) -> None:
    """A worked digest is what a second implementation can actually check itself against."""
    assert saw(content) in _section("## The saw hash"), (
        f"the doc does not publish the digest of {content!r}, which is {saw(content)}"
    )


# ==========================================================================
# Corpus schema: the version rule, recomputed field by field.
# ==========================================================================

BASE_CASE: dict[str, Any] = {
    "id": "a1",
    "text": "mail alice@example.com",
    "direction": "output",
    "expect_decision": "redact",
    "expect_findings": (ExpectedFinding(type="EMAIL", span=(5, 22)),),
    "source": "in-repo",
    "license": "Apache-2.0",
}


def _corpus(**overrides: Any) -> Corpus:
    case = Case(**{**BASE_CASE, **overrides})
    return Corpus(name="example", source=case.source, license=case.license, cases=(case,))


# Each entry is a field the version rule claims to cover, and one edit to it.
# The edit is chosen to be otherwise valid, so a version that does not move
# means the digest genuinely ignores that field rather than that the case was
# rejected.
HASHED_FIELDS: list[tuple[str, dict[str, Any]]] = [
    ("id", {"id": "a2"}),
    ("text", {"text": "mail alice@example.org"}),
    ("direction", {"direction": "input"}),
    ("expect.decision", {"expect_decision": "deny"}),
    ("expect.findings", {"expect_findings": (ExpectedFinding(type="US_SSN", span=(5, 22)),)}),
    ("expect.findings", {"expect_findings": (ExpectedFinding(type="EMAIL", span=None),)}),
    (
        "expect.findings",
        {
            "expect_findings": (
                ExpectedFinding(type="EMAIL", span=(5, 22)),
                ExpectedFinding(type="EMAIL", span=(0, 4)),
            )
        },
    ),
]


@pytest.mark.parametrize(
    ("field", "edit"), HASHED_FIELDS, ids=[f"{f}-{i}" for i, (f, _) in enumerate(HASHED_FIELDS)]
)
def test_every_field_the_version_hashes_moves_it_and_the_doc_names_it(
    field: str, edit: dict[str, Any]
) -> None:
    """Both halves, because either alone is a hole.

    An earlier draft of this project's own plan described the version as a hash
    of `id` and `text`. It covers five things. A conformance doc that
    understates the digest tells a porter their corpus version may stay fixed
    while the measured numbers move, which is exactly the state a published
    baseline cannot survive.
    """
    assert _corpus(**edit).version != _corpus().version, (
        f"editing {field} did not move the corpus version"
    )
    assert f"`{field}`" in _section("## Corpus schema"), (
        f"the corpus schema section does not name {field}, which the version hashes"
    )


def test_the_corpus_schema_section_names_every_hashed_field() -> None:
    """The brief's own check, kept alongside the derived one above.

    That one proves a field moves the version; this one is a plain floor over
    the five names, so a refactor that broke the mutation helper could not take
    the coverage down with it silently.
    """
    schema = _section("## Corpus schema")
    for field in ("id", "text", "direction", "decision", "findings"):
        assert field in schema, f"corpus schema section does not mention {field}"


@pytest.mark.parametrize(
    ("field", "edit"),
    [("source", {"source": "third-party"}), ("license", {"license": "CC-BY-4.0"})],
)
def test_the_provenance_fields_are_outside_the_version(field: str, edit: dict[str, Any]) -> None:
    """The version identifies what was measured, not where it came from.

    Recorded as a test rather than as a sentence because the document states
    it: start hashing `source` and a reader is told something false, while
    every published baseline key silently changes at the same time.
    """
    assert _corpus(**edit).version == _corpus().version, (
        f"{field} moved the corpus version, which the conformance doc says it does not"
    )


def test_the_corpus_schema_section_publishes_the_version_length_the_code_produces() -> None:
    section = _section("## Corpus schema")
    stated = re.search(r"first (\d+) hex characters", section)
    assert stated is not None, "the corpus schema section does not state the version length"
    assert int(stated.group(1)) == len(_corpus().version)


def test_the_published_corpus_version_vector_is_what_this_implementation_computes(
    tmp_path: Path,
) -> None:
    """The worked example is loaded from the document and run.

    A porter copies that JSONL line and that digest. If either drifts from what
    `load_corpus` and `Corpus.version` do, the two halves of the example stop
    agreeing and this fails, rather than the porter discovering it against a
    corpus they cannot debug.
    """
    section = _section("## Corpus schema")
    blocks = [
        block.strip()
        for block in re.findall(r"```(?:json|jsonl)?\n(.*?)```", section, re.DOTALL)
        if block.strip().startswith("{")
    ]
    assert len(blocks) == 1, f"expected exactly one worked JSONL example, found {len(blocks)}"

    path = tmp_path / "worked.jsonl"
    path.write_text(blocks[0] + "\n", encoding="utf-8")
    version = load_corpus(path, name="example").version
    assert f"`{version}`" in section, (
        f"the doc's worked example loads to version {version}, which the doc does not publish"
    )


# ==========================================================================
# Third-party corpora: the licence chain and the dated guard.
# ==========================================================================


def _enforced_fng_domains() -> set[str]:
    """The domains the corpus screen actually rejects, read out of the screen.

    Read from the test source rather than imported, which keeps this file from
    depending on pytest's sys.path insertion for a sibling test module.
    """
    body = SCREEN.read_text(encoding="utf-8").split("FNG_DOMAINS = frozenset(", 1)[1]
    domains = set(re.findall(r'"([^"]+)"', body.split(")", 1)[0]))
    assert len(domains) >= 5, f"read {len(domains)} domains out of the screen; the parse is wrong"
    return domains


# Only paths that are meant to be local. The document also cites
# `src/datagen/pii/privy/privy/providers/english_us.py`, which lives in
# pixie-io/pixie and must not be looked for here.
_LOCAL_PATH = re.compile(
    r"`((?:corpora|tests|scripts|src/jamjet_guardrails|docs)/[^`]*|[A-Z]+\.md)`"
)


def test_every_repository_path_the_document_cites_exists() -> None:
    """A document that points at a renamed file sends a reader nowhere.

    The share-alike screen is cited by its full test id, which is the citation
    most likely to rot: renaming a test is a routine edit and nothing else in
    the repository reads that name. A trailing slash means a directory.
    """
    cited = sorted(set(_LOCAL_PATH.findall(_text())))
    assert cited, "no repository paths found in the doc; this check would prove nothing"
    missing = []
    for citation in cited:
        path_part, _, test_name = citation.partition("::")
        target = ROOT / path_part
        exists = target.is_dir() if path_part.endswith("/") else target.is_file()
        # Only read the file once it is known to be there, and treat a citation
        # with no `::` part as naming nothing to look for.
        defines_it = not test_name or (
            exists and f"def {test_name}(" in target.read_text(encoding="utf-8")
        )
        if not (exists and defines_it):
            missing.append(citation)
    assert missing == [], f"the doc cites paths that do not exist: {missing}"


def test_the_document_lists_every_domain_the_corpus_screen_rejects() -> None:
    """The fingerprint is only evidence if the published list is the enforced list.

    A domain added to the screen and not to the document leaves a reader
    checking a new corpus against a list that is one short.
    """
    section = _section("## Third-party corpora")
    missing = sorted(d for d in _enforced_fng_domains() if d not in section)
    assert missing == [], f"the doc omits domains the screen rejects: {missing}"


def test_the_domain_count_the_document_publishes_is_the_count_it_enforces() -> None:
    """The other direction: a domain removed from the screen, not from the doc."""
    section = _section("## Third-party corpora")
    stated = re.search(r"(\d+) house domains", section)
    assert stated is not None, "the doc does not state how many house domains the screen covers"
    assert int(stated.group(1)) == len(_enforced_fng_domains())


def test_the_expiry_date_both_files_publish_is_the_computed_boundary() -> None:
    """The issuer-digit guard's expiry, recomputed rather than repeated.

    The guard excludes epoch-millisecond timestamps because they currently
    begin with a 1. Epoch-ms carries a leading 2 from the instant the epoch
    second reaches 2,000,000,000, after which timestamps sit back inside the
    MII 2 to 6 range the guard admits. Both `docs/conformance.md` and
    `corpora/NOTICE.md` publish that date beside a precision figure whose
    meaning depends on it, so both are checked against the arithmetic.
    """
    boundary = datetime.fromtimestamp(2_000_000_000, tz=timezone.utc)
    stamp = boundary.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert stamp in _text(), f"the conformance doc does not publish the expiry {stamp}"
    assert stamp in NOTICE.read_text(encoding="utf-8"), f"the notice does not publish {stamp}"


def test_the_third_party_section_points_at_the_notice_it_does_not_replace() -> None:
    """CC BY 4.0 attribution lives in one file, so the two cannot fall out of step."""
    section = _section("## Third-party corpora")
    assert "corpora/NOTICE.md" in section


# ==========================================================================
# The injection-structural section: its types and its span vector, recomputed.
# ==========================================================================

INJECTION = "## The injection-structural constraint"
INJECTION_CORPUS = ROOT / "corpora" / "injection-structural" / "in-repo.jsonl"
# The case the document works its span vector out of.
SPAN_VECTOR_CASE = "inj-0001"


def _claim_region(phrase: str) -> str:
    """The document text a single claim owns: from its opening phrase to the next bullet.

    A whole-section search would let one bullet's ids answer for another's,
    which is precisely the mistake the tests below exist to catch -- the five
    claims cite twenty-nine cases between them and four of them share an id
    with a neighbour.
    """
    text = _text()
    assert phrase in text, f"the document makes no claim opening {phrase!r}"
    rest = text.split(phrase, 1)[1]
    stop = re.search(r"\n\s*-\s|\n#{2,3} ", rest)
    return rest[: stop.start()] if stop else rest


def _subsection(heading: str, sub: str) -> str:
    """The body of one H3, up to the next H3 or the end of its H2."""
    section = _section(heading)
    assert sub in section, f"{heading} has no {sub!r} subsection"
    return section.split(sub, 1)[1].split("\n### ", 1)[0]


def test_the_document_lists_every_type_this_check_can_produce() -> None:
    """Derived from `INJECTION_TYPES`, so a fourth signal is a red test here.

    A type is contract by the rule this document already states: a finding
    `type` has to match the label the corpus uses, because that is what a
    prediction is matched against. A type added to the check and left out of
    here is a piece of that a porter never learns, and both directions matter --
    a type listed here and not produced makes the document overstate the check.
    """
    documented = set(
        re.findall(r"`([A-Z][A-Z_]+)`", _subsection(INJECTION, "### Three finding types"))
    )
    assert documented == set(INJECTION_TYPES), (
        f"the document lists {sorted(documented)}; the check produces {sorted(INJECTION_TYPES)}"
    )


def test_the_span_vector_the_document_publishes_is_the_case_it_names() -> None:
    """All three renderings of one corpus span, recomputed from the case itself.

    The document tells a porter that spans count CODE POINTS, and backs it with
    one case rendered three ways. The numbers are the whole of the claim: quoted
    wrongly, a porter checks their UTF-16 implementation against a UTF-16 figure
    that was never measured and concludes it agrees.

    The three renderings are asserted to DIFFER as well. If they ever coincide
    the vector demonstrates nothing, and it would still be quotable.
    """
    corpus = load_corpus(INJECTION_CORPUS, name="injection-structural")
    case = next((c for c in corpus.cases if c.id == SPAN_VECTOR_CASE), None)
    assert case is not None, (
        f"{SPAN_VECTOR_CASE} is the document's span vector and is not in the corpus"
    )
    (finding,) = case.expect_findings
    assert finding.span is not None
    start, end = finding.span

    def utf8(index: int) -> int:
        return len(case.text[:index].encode("utf-8"))

    def utf16(index: int) -> int:
        return len(case.text[:index].encode("utf-16-le")) // 2

    spans = {
        "code points": (start, end),
        "UTF-8 bytes": (utf8(start), utf8(end)),
        "UTF-16 code units": (utf16(start), utf16(end)),
    }
    assert len(set(spans.values())) == 3, (
        f"the three renderings of {SPAN_VECTOR_CASE} coincide ({spans}), so the "
        "document's worked vector distinguishes nothing"
    )
    section = _section(INJECTION)
    for unit, (low, high) in spans.items():
        assert f"`[{low}, {high}]`" in section, (
            f"the document does not publish {SPAN_VECTOR_CASE} in {unit}, which is [{low}, {high}]"
        )


def test_the_injection_corpus_carries_both_directions_the_document_claims() -> None:
    """The document says this check runs in both directions and that the corpus
    now measures it. Both halves are checked here: the corpus really does carry
    output cases, and the document really does claim both.

    The previous version of this test asserted the opposite, that every case was
    `input`, and it was correct for 0.1.0. It is kept in this shape rather than
    deleted because the sentence it guards moved rather than went away: what the
    corpus can and cannot tell a port is still the point."""
    corpus = load_corpus(INJECTION_CORPUS, name="injection-structural")
    assert corpus.cases, "the injection corpus is empty; this check would prove nothing"
    directions = {case.direction for case in corpus.cases}
    assert directions == {"input", "output"}, f"the corpus carries directions {sorted(directions)}"
    section = _section(INJECTION)
    assert "runs on input and on output" in section
    assert "`direction: input`" not in section


# ==========================================================================
# The exemptions: the case lists in the document are RE-MEASURED, not read.
# ==========================================================================

INJECTION_MODULE = ROOT / "src" / "jamjet_guardrails" / "detectors" / "injection_structural.py"
SCORING = Context(direction="input", origin="model")

# Each entry: the phrase that opens the document's own claim, and the single
# source edit that switches that one exemption or exclusion off. The edits are
# textual on purpose. If one stops applying, this fails with "no longer applies"
# rather than silently measuring an unmutated module, and the right response is
# to re-measure the claim rather than to repair the patch.
_EXEMPTIONS: list[tuple[str, str, str]] = [
    (
        "Balanced bidi controls are allowed",
        "    for index, char in enumerate(content):\n        if char in _EMBED_OPEN:",
        (
            "    for index, char in enumerate(content):\n"
            "        if (\n"
            "            char in _EMBED_OPEN\n"
            "            or char in _ISOLATE_OPEN\n"
            "            or char == _EMBED_CLOSE\n"
            "            or char == _ISOLATE_CLOSE\n"
            "        ):\n"
            "            unbalanced.append(index)\n"
            "            continue\n"
            "        if char in _EMBED_OPEN:"
        ),
    ),
    (
        "The three RGI subdivision flag sequences are allowed",
        "    if start == 0 or ord(content[start - 1]) != _FLAG_BASE:",
        "    if True or start == 0 or ord(content[start - 1]) != _FLAG_BASE:",
    ),
    (
        "The joiner exemption is contextual, by script",
        "    char = content[index]\n    if char not in _CONTEXTUAL:",
        "    char = content[index]\n    return False\n    if char not in _CONTEXTUAL:",
    ),
    (
        "counting variation selectors denies",
        '        and _VARIATION_SELECTOR not in unicodedata.name(chr(point), "")\n',
        "",
    ),
    (
        "dropping the exclusion for the directional format characters denies",
        "        if not _is_directional(chr(point))\n        and ",
        "        if ",
    ),
]


def _allow_cases() -> list[Any]:
    return [
        c
        for c in load_corpus(INJECTION_CORPUS, name="injection-structural").cases
        if c.expect_decision == "allow"
    ]


def _flips(guardrail: Any, cases: list[Any]) -> set[str]:
    """Which of these `allow` cases the given guardrail does not allow."""
    return {c.id for c in cases if guardrail.check(c.text, SCORING).decision != "allow"}


def _without(old: str, new: str, cases: list[Any]) -> set[str]:
    """The `allow` cases one source edit costs, over and above the ones already failing.

    The module is exec'd from a patched copy of its own source rather than
    imported, which is what keeps this honest. Writing the patched file to disk
    and re-importing it is the obvious alternative and it silently lies: two
    patches applied in the same second that happen to produce the same file size
    make CPython reuse the first one's cached bytecode, and the second
    measurement comes back as the first one's. That happened while these numbers
    were being taken, and it made two different exemptions report the same seven
    cases. Nothing about the output said so.
    """
    source = INJECTION_MODULE.read_text(encoding="utf-8")
    assert source.count(old) == 1, (
        f"the mutation {old[:60]!r} no longer applies to the module "
        f"({source.count(old)} matches); re-measure the claim rather than repairing the patch"
    )
    namespace: dict[str, Any] = {"__name__": "injection_structural_under_mutation"}
    exec(compile(source.replace(old, new, 1), str(INJECTION_MODULE), "exec"), namespace)  # noqa: S102
    mutant = namespace["InjectionStructuralGuardrail"]()
    return _flips(mutant, cases) - _flips(build("injection-structural"), cases)


@pytest.mark.parametrize(
    ("phrase", "old", "new"), _EXEMPTIONS, ids=[e[0][:32].replace(" ", "-") for e in _EXEMPTIONS]
)
def test_every_case_list_the_exemptions_publish_is_the_list_the_measurement_gives(
    phrase: str, old: str, new: str
) -> None:
    """The most perishable numbers in the section, derived instead of asserted.

    Each bullet claims that switching one exemption off costs a named set of
    `allow` cases. Nothing else in this repository re-runs that, so the lists
    were true when they were written and would stay in the document unchanged
    while the detector moved underneath them.

    One of them was already wrong on the day it was written: the directional
    bullet named the exclusion of every directional FORMAT character and
    reported the result of counting only the three MARKS, which is one case
    short. It was measured on four hand-picked ids rather than on the corpus,
    and every id it named was correct, so nothing about it read as wrong.
    """
    region = _claim_region(phrase)
    cited = set(re.findall(r"inj-\d{4}", region))
    assert cited, f"the claim opening {phrase!r} cites no case ids"
    measured = _without(old, new, _allow_cases())
    assert cited == measured, (
        f"the document says switching off {phrase!r} costs {sorted(cited)}; "
        f"measured, it costs {sorted(measured)}"
    )


def test_the_count_of_allow_cases_riding_on_an_exemption_is_the_measured_union() -> None:
    """Both numbers in the section's opening claim, and the union is not the sum.

    `inj-0037` is held up by two of the five, so adding the bullet lists gives
    30 where the union is 29. The sentence this guards replaced one that said
    "most", which was true only if "turns on an exemption" is read as "contains
    a character some exemption is about". Measured, that reading gives 68 of the
    94 `allow` cases against 29 that depend on one, and NAMING THE SET IS THE
    WHOLE OF IT: 68 is the union of the five exemptions' own character sets --
    the nine bidi controls, the tag characters with U+1F3F4, the three
    contextual joiners, the variation selectors, and the characters
    `_is_directional` answers True for. A looser reading gives a different
    number and neither is wrong on its own: every `Cf` character, which is not
    what any exemption is about, gives 74. This note carried 73, which is
    neither set.
    """
    cases = _allow_cases()
    union: set[str] = set()
    for _phrase, old, new in _EXEMPTIONS:
        union |= _without(old, new, cases)
    section = _section(INJECTION)
    assert f"{len(union)} of its {len(cases)} `allow` cases" in section, (
        f"the document does not state the measured figures, which are {len(union)} of {len(cases)}"
    )
