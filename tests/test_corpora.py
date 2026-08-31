"""The committed corpora: the data every published number is measured on.

The brief's tests come first, verbatim apart from the annotations mypy strict
needs. Everything after them guards the corpus CONTENT, which is the half a
loader cannot check: that every type carries a positive, that the shapes an
independent review measured are all labelled, and that a licence obligation
which lives in a separate file is actually discharged there.

The failure direction for a corpus is not "malformed". It is "shaped like the
detector": a corpus that only holds cases the detector already handles publishes
a number that measures nothing, and no loader, formatter or gate downstream can
tell that from a good result.
"""

import re
from pathlib import Path

import pytest

from jamjet_guardrails.detectors import AVAILABLE, build
from jamjet_guardrails.detectors.injection_structural import _DEFAULT_IGNORABLE
from jamjet_guardrails.detectors.pii import PII_TYPES
from jamjet_guardrails.detectors.secrets import SECRET_TYPES
from jamjet_guardrails.eval.corpus import Corpus, load_corpus
from jamjet_guardrails.eval.metrics import evaluate
from jamjet_guardrails.eval.report import to_markdown
from jamjet_guardrails.types import Context

# Read from the detector's own test module rather than copied. tests/ is not a
# package, so pytest and mypy both see these as top-level modules; the import is
# what keeps this file in step with a list that another task owns and adds to.
from test_pii import _KNOWN_FALSE_POSITIVES, _NEW_FALSE_POSITIVES

ROOT = Path(__file__).resolve().parent.parent
CORPORA = ROOT / "corpora"

EXPECTED = [
    ("injection-structural", "in-repo"),
    ("pii", "in-repo"),
    ("pii", "third-party"),
    ("secrets", "in-repo"),
]


MINIMUM_CASES = {"in-repo": 30, "third-party": 20}


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_corpus_exists_and_loads(check: str, source: str) -> None:
    corpus = load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")
    floor = MINIMUM_CASES[source]
    assert len(corpus.cases) >= floor, (
        f"{check}/{source} has {len(corpus.cases)} cases; a corpus under {floor} "
        "cannot support a published number"
    )


def test_every_registered_check_has_a_corpus() -> None:
    """The symmetric half of the README's checks table, which IS enforced.

    `tests/test_readme.py` asserts that table equals `AVAILABLE`, so a check
    cannot be registered without appearing there. Nothing said the same about
    the evidence: `eval.cli.discover` globs `corpora/<check>/<source>.jsonl` and
    never consults `AVAILABLE`, so a registered check with no corpus directory
    is scored on nothing, publishes no row, and fails no gate. In a project
    whose headline is "measured, not asserted", an unmeasured check is the one
    absence that has to be loud.

    The directory, not a file: a check's corpora may be named anything, and the
    loader and the case floor above answer for what is in them.
    """
    missing = sorted(name for name in AVAILABLE if not (CORPORA / name).is_dir())
    assert missing == [], (
        f"{missing} are registered in AVAILABLE with no corpora/<check>/ directory, so "
        "nothing measures them and no published row would show they are missing"
    )


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_decision_and_findings_agree(check: str, source: str) -> None:
    """For a constraint, redact means findings and allow means none."""
    corpus = load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")
    for case in corpus.cases:
        if case.expect_decision == "allow":
            assert not case.expect_findings, f"{case.id}: allow with findings"
        else:
            assert case.expect_findings, f"{case.id}: {case.expect_decision} with no findings"


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_corpus_carries_positives(check: str, source: str) -> None:
    """A negatives-only corpus makes both ratios vacuous: a detector that denies
    everything scores 1.000 precision and 1.000 recall on it, because there is
    nothing to miss and nothing it failed to predict. Only the wrong-decision
    count would show the failure, and that one is not thresholded."""
    corpus = load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")
    positives = [c for c in corpus.cases if c.expect_decision != "allow"]
    assert len(positives) >= len(corpus.cases) // 4, "at least a quarter must be positives"


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_corpus_carries_negatives(check: str, source: str) -> None:
    """Precision is only meaningful if the corpus contains things that must NOT match."""
    corpus = load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")
    negatives = [c for c in corpus.cases if c.expect_decision == "allow"]
    assert len(negatives) >= len(corpus.cases) // 4, "at least a quarter must be negatives"


# NOTE: the two tests this block used to contain, "every case records a licence"
# and "every span slices inside its text", were deleted, not moved. Task 9's
# loader now REFUSES a case missing either, so both tests became unable to fail:
# they asserted a property the constructor guarantees. A test that cannot fail is
# not coverage, it is decoration that reads as coverage. What still needs
# asserting here is the corpus CONTENT (below), not its well-formedness.


def test_the_detectors_actually_score_on_their_corpora() -> None:
    """A smoke floor, not the published number. CI's gate is the real bar."""
    for check, source in EXPECTED:
        corpus = load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")
        ev = evaluate(build(check), corpus)
        assert ev.overall.precision > 0.5, f"{check}/{source} precision floor"
        assert ev.overall.recall > 0.5, f"{check}/{source} recall floor"


FNG_DOMAINS = frozenset(
    {
        "dayrep.com",
        "armyspy.com",
        "rhyta.com",
        "cuvox.de",
        "einrot.com",
        "fleckens.hu",
        "gustr.com",
        "jourrapide.com",
        "superrito.com",
        "teleworm.us",
    }
)


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_no_corpus_carries_share_alike_values(check: str, source: str) -> None:
    """Fake Name Generator identities are GPLv3 / CC-BY-SA-3.0-US and cannot be
    redistributed here. Two upstream datasets advertise MIT and carry them
    anyway, so the licence tag is not the thing to check. These ten house
    domains are diagnostic: Faker does not issue them.
    """
    corpus = load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")
    hits = [
        case.id for case in corpus.cases for domain in FNG_DOMAINS if domain in case.text.lower()
    ]
    assert hits == [], f"share-alike values in {check}/{source}: {hits[:5]}"


# Everything below this line is content, not shape. None of it is in the brief.


def _load(check: str, source: str) -> Corpus:
    return load_corpus(CORPORA / check / f"{source}.jsonl", name=f"{check}/{source}")


def _labelled_types(corpus: Corpus) -> set[str]:
    return {finding.type for case in corpus.cases for finding in case.expect_findings}


@pytest.mark.parametrize("pii_type", sorted(PII_TYPES))
def test_every_pii_type_carries_a_positive_in_the_in_repo_corpus(pii_type: str) -> None:
    """A type with no positive is a type whose recall is never measured.

    Parametrised per type rather than asserted as a set difference, so a missing
    type names itself instead of being one entry in a diff.
    """
    assert pii_type in _labelled_types(_load("pii", "in-repo"))


@pytest.mark.parametrize("secret_type", sorted(SECRET_TYPES))
def test_every_secret_type_carries_a_positive_in_the_in_repo_corpus(secret_type: str) -> None:
    assert secret_type in _labelled_types(_load("secrets", "in-repo"))


# Every shape the independent review measured, whether or not Task 18 closed it.
# The closed ones guard against regression; the open ones are the honest false
# negatives this project exists to publish, and they are the cases most likely to
# be quietly dropped later, because dropping them raises the published recall.
#
# Each shape carries the TYPE its finding must have. Without it a shape covered
# by a wrong-typed finding passed: "123 45 6789" labelled PHONE_NUMBER would have
# satisfied a test whose whole subject is a US_SSN the detector does not match.
#
# Written as escapes for the two accented forms, so that an editor normalising
# this file cannot turn the decomposed spelling into the composed one and leave
# two identical entries claiming to cover both.
_MEASURED_SHAPES = (
    ("4111111111111111", "CREDIT_CARD"),
    ("3782 822463 10005", "CREDIT_CARD"),
    ("4000000000000000006", "CREDIT_CARD"),
    ("(415) 555-2671", "PHONE_NUMBER"),
    ("415.555.2671", "PHONE_NUMBER"),
    ("+1-415-555-2671", "PHONE_NUMBER"),
    ("+14155552671", "PHONE_NUMBER"),
    ("jose\u0301@example.com", "EMAIL"),  # decomposed: e then U+0301
    ("jos\u00e9@example.com", "EMAIL"),  # composed: U+00E9
    ("alice@example.\u0440\u0444", "EMAIL"),  # the Cyrillic .rf ccTLD
    ("ssn-123-45-6789", "US_SSN"),
    ("123456789", "US_SSN"),
    ("123 45 6789", "US_SSN"),
    ("4111 1111 1111 1111-0000", "CREDIT_CARD"),
)


@pytest.mark.parametrize(("shape", "expected_type"), _MEASURED_SHAPES)
def test_every_measured_shape_is_labelled_as_a_finding(shape: str, expected_type: str) -> None:
    """Present, in a POSITIVE case, under a finding of the right type covering it.

    Presence alone is not enough: a shape sitting in a case labelled ``allow``,
    or in a case whose findings all point somewhere else, is a shape the recall
    number does not pay for.

    A null span does NOT satisfy this. Type-only matching is the weaker bar, and
    these fourteen shapes are exactly the ones whose localisation is the point,
    so weakening one of them to a null span has to fail here rather than pass
    quietly. That is also why the span is read after the None test rather than
    around it.
    """
    corpus = _load("pii", "in-repo")
    for case in corpus.cases:
        start = case.text.find(shape)
        if start < 0:
            continue
        end = start + len(shape)
        for finding in case.expect_findings:
            if finding.type != expected_type or finding.span is None:
                continue
            if finding.span[0] < end and start < finding.span[1]:
                return
    pytest.fail(
        f"{shape!r} is in no positive case of pii/in-repo, or no {expected_type} finding "
        "with a real span covers it"
    )


def test_the_two_spellings_of_the_accented_address_are_different_byte_sequences() -> None:
    """The pair is only worth two cases if they really are two.

    Task 18 found the brief's own test and pattern disagreed about which normal
    form they meant, which is invisible on screen: both render as jose@example
    with an accent. If a normalising editor collapses one into the other, the
    corpus silently stops covering the decomposed form while still carrying two
    cases that look right.
    """
    composed, decomposed = "jos\u00e9@example.com", "jose\u0301@example.com"
    assert composed != decomposed
    texts = [case.text for case in _load("pii", "in-repo").cases]
    assert any(composed in text for text in texts)
    assert any(decomposed in text for text in texts)


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_null_spans_stay_a_minority(check: str, source: str) -> None:
    """A null span asks for the weaker type-only match, so a corpus of them
    measures type detection and nothing about localisation. It is the honest
    label only where the boundary genuinely has no single right answer, which
    here is an UNTERMINATED PEM block and nothing else.
    """
    corpus = _load(check, source)
    findings = [finding for case in corpus.cases for finding in case.expect_findings]
    null_spans = [finding for finding in findings if finding.span is None]
    assert len(null_spans) * 4 <= len(findings), (
        f"{check}/{source} labels {len(null_spans)} of {len(findings)} findings with a "
        "null span; over a quarter is no longer a minority"
    )


NOTICE = CORPORA / "NOTICE.md"


@pytest.mark.parametrize(("check", "source"), EXPECTED)
def test_every_corpus_declares_its_provenance_in_the_notice(check: str, source: str) -> None:
    """Attribution is a licence condition, not a courtesy, and the failure
    direction is silence: a corpus added later with an unrecorded upstream is a
    licence violation with no symptom.

    The source and the licence have to appear on the SAME LINE, which here means
    the same row of the notice's table. Searching for them as independent
    substrings is what this test used to do, and it accepted the exact violation
    the sentence above describes: `Apache-2.0` and `in-repo` are both already in
    the file for the first-party corpora, so rewriting every third-party case's
    licence to `Apache-2.0`, or its source to `in-repo`, passed. A dataset named
    without its licence and a licence named without its dataset are each half an
    attribution, and two halves from different rows are not a whole one.
    """
    corpus = _load(check, source)
    licences = corpus.license.split(", ")
    rows = [
        line
        for line in NOTICE.read_text(encoding="utf-8").splitlines()
        if corpus.source in line and all(licence in line for licence in licences)
    ]
    assert rows, (
        f"no single line of {NOTICE} names source {corpus.source!r} together with "
        f"{licences}; an attribution split across two rows attributes nothing"
    )


def test_the_notice_carries_what_cc_by_asks_for() -> None:
    """Creator, title, URL and licence, which is what the licence text lists."""
    notice = NOTICE.read_text(encoding="utf-8")
    for required in (
        "Nemotron-PII",
        "NVIDIA",
        "https://huggingface.co/datasets/nvidia/Nemotron-PII",
        "CC-BY-4.0",
        "https://creativecommons.org/licenses/by/4.0/",
    ):
        assert required in notice, f"{required} missing from {NOTICE}"


def test_the_published_report_carries_the_attribution_pointer() -> None:
    """CC BY 4.0 asks for attribution wherever the material is used, and a
    published precision figure is a use. ``to_markdown`` renders BENCHMARKS.md,
    so the pointer has to come out of the formatter: a line added to the
    committed file by hand is overwritten the next time CI regenerates it.

    A pointer rather than the full notice, which the licence allows in as many
    words: the reader gets the dataset in the Source column and the rest one
    click away, and the formatter states nothing corpus-specific it cannot know.
    """
    evaluations = [evaluate(build(check), _load(check, source)) for check, source in EXPECTED]
    markdown = to_markdown(evaluations)

    assert "corpora/NOTICE.md" in markdown
    for ev in evaluations:
        assert ev.corpus_source in markdown, "the Source column has to name what it measured"


# The cases whose whole purpose is to record something the detector gets wrong.
# Their labels say what SHOULD happen, and the one edit that would quietly
# destroy them is relabelling to what the detector DOES: that scores a clean hit
# and erases the miss from the published numbers, which is the self-grading this
# corpus exists to prevent.
_RECORDS_A_MISS = {
    "pii": (
        "pii-0031",  # a bare nine-digit SSN, which the pattern deliberately skips
        "pii-0032",  # a space-separated SSN, the same deliberate gap
        "pii-0033",  # the shifted four-group window, leading group left standing
        "pii-0034",  # the same window shifting left, final group left standing
        "pii-0035",  # a TLD carrying a Devanagari spacing mark
        "pii-0036",  # punycode, whose final label the letters-only TLD class drops
    ),
    "secrets": (
        "sec-0020",  # github_pat_ fine-grained tokens are in no pattern here
        "sec-0021",  # xapp- Slack app-level tokens likewise
        "sec-0022",  # a JWT header over the 4096 bound, a complete miss
        "sec-0034",  # a PEM header in prose with no key body
    ),
}


@pytest.mark.parametrize(
    ("check", "case_id"),
    [(check, case_id) for check, ids in _RECORDS_A_MISS.items() for case_id in ids],
)
def test_a_case_that_records_a_miss_is_not_labelled_with_what_the_detector_does(
    check: str, case_id: str
) -> None:
    """The property stated directly, rather than by pinning a span.

    Deletion of one of these cases is already caught by the measured-shape test.
    The RELABEL is not, and it is the edit the brief named: labelling
    `4111 1111 1111 1111-0000` with the span the detector produces passes every
    other test in this file and moves published precision, because the miss
    stops being a miss without anything reading differently.

    This test fails if the detector is ever FIXED, which is the point. The fix
    is to move the case out of `_RECORDS_A_MISS` and let it be an ordinary
    positive, not to restore the old label. `tests/test_pii.py` keeps its
    false-positive records the same way and says so in the same words.
    """
    corpus = _load(check, "in-repo")
    case = next(c for c in corpus.cases if c.id == case_id)
    guardrail = build(check)
    verdict = guardrail.check(case.text, Context(direction=case.direction, origin="model"))

    labelled = sorted((f.type, f.span) for f in case.expect_findings)
    predicted = sorted((f.type, f.span) for f in verdict.findings)
    assert labelled != predicted or case.expect_decision != verdict.decision, (
        f"{case_id} is listed as recording a miss, but its label is exactly what "
        f"{check} produces on it. Either the label was moved onto the detector's "
        "output, which erases the miss, or the detector was fixed and this case "
        "should leave _RECORDS_A_MISS."
    )


@pytest.mark.parametrize(
    ("case_id", "card"),
    [("pii-0033", "4111 1111 1111 1111"), ("pii-0034", "4111 1111 1111 1111")],
)
def test_a_shifted_window_card_is_labelled_over_the_card(case_id: str, card: str) -> None:
    """The sharpest pair in the corpus, pinned exactly rather than by property.

    Both texts put a four-digit neighbour beside a card, and in both the detector
    matches a shifted four-group window: it keeps `4111 ` in one and the final
    `1111` in the other, while claiming a redaction it did not make. The label is
    the CARD, so the span is computed from the card's own position here rather
    than written down, for the same reason no span in these corpora is hand
    counted.
    """
    case = next(c for c in _load("pii", "in-repo").cases if c.id == case_id)
    start = case.text.index(card)

    assert [(f.type, f.span) for f in case.expect_findings] == [
        ("CREDIT_CARD", (start, start + len(card)))
    ]


@pytest.mark.parametrize(
    "recorded",
    [text for text, _ in _KNOWN_FALSE_POSITIVES] + [text for text, _, _ in _NEW_FALSE_POSITIVES],
)
def test_every_recorded_false_positive_is_in_the_corpus(recorded: str) -> None:
    """The claim "this corpus carries every recorded false positive", checked.

    `tests/test_pii.py` pins both records as CURRENT behaviour and says in as
    many words that they are listed so that this task scores them and Task 15
    publishes them. Until this test existed that was an unchecked promise, and
    the composition was unpinned in the flattering direction: deleting the
    sixteen dotted-tail negatives passed every test here and lifted published
    in-repo precision from 0.641 to 0.854.

    Reading the lists from `test_pii.py` rather than copying them is what makes
    this hold for the NEXT recorded false positive too. A guard that repeats the
    list it guards goes stale the first time somebody adds to one copy.
    """
    texts = [case.text for case in _load("pii", "in-repo").cases]
    assert any(recorded in text for text in texts), (
        f"{recorded!r} is recorded in tests/test_pii.py as a false positive the "
        "detector produces today, and no case in pii/in-repo carries it, so the "
        "published precision number does not pay for it"
    )


def test_the_notice_qualifies_the_two_numbers_that_need_qualifying() -> None:
    """Prose a published figure depends on, pinned so it cannot quietly go.

    Two things about these numbers are not visible in them. The in-repo PII
    corpus is a stress set rather than a sample of real text, so its precision
    is not a field figure; and the detector's issuer-digit guard on bare card
    runs stops working on a date, after which epoch-millisecond timestamps sit
    back inside the range it excludes.

    A substring pin on prose is a weak test, and it is here for the one failure
    it does catch: the qualifier being deleted while the figure it qualifies
    stays published.
    """
    notice = NOTICE.read_text(encoding="utf-8")
    assert "stress set" in notice
    assert "2033-05-18" in notice


# The twenty-four injection-structural cases a reader is most likely to argue
# with. Seventeen of them FAIL when the corpus is scored, on purpose: they are
# the check's known false positives and known false negatives, labelled with
# what SHOULD happen so that they cost precision and recall rather than being
# scored as successes. That convention is `corpora/pii/in-repo.jsonl`'s, and it
# is why that corpus publishes 0.631 rather than a number about its own labels.
#
# The other seven pass. Three are the balanced-override set and four are the
# stray closers with the chunked document they come from; every group is
# labelled with what should happen, and every one is here because a reader
# could reasonably expect the opposite.
#
# `corpora/NOTICE.md` names every one of these by id and says what it is. This
# test is what keeps the two together. The edit it exists to stop is the cheap
# one: flipping a label back to what the detector does turns a published failure
# into a published success and moves precision without reading as a change.
_INJECTION_DISCLOSED = {
    # False positives. Each denies; each is labelled allow and scores as an FP.
    "inj-0090": "allow",  # Thai line-break hints, four of them
    "inj-0091": "allow",  # FSI around a multi-line value, the idiom Unicode recommends
    "inj-0092": "allow",  # Persian ages written with ASCII digits
    "inj-0093": "allow",  # Urdu years written with ASCII digits
    "inj-0094": "allow",  # Persian suffixes on Latin acronyms
    "inj-0095": "allow",  # Persian decades written with ASCII digits
    "inj-0106": "allow",  # a 2,503-character page carrying four incidental ZWSPs
    "inj-0128": "allow",  # MathML extracted to plain text: four invisible operators
    "inj-0129": "allow",  # musical notation: two beam pairs are four format controls
    "inj-0134": "allow",  # Korean prose about jamo, four Hangul fillers
    "inj-0135": "allow",  # a Khmer dictionary entry, four inherent vowels
    "inj-0136": "allow",  # U+034F blocking a collation contraction
    "inj-0137": "allow",  # U+034F fixing point order in Biblical Hebrew
    "inj-0138": "allow",  # four UTF-8 files concatenated, each keeping its BOM
    # False negatives. Each allows; each is labelled deny and scores as an FN.
    "inj-0097": "deny",  # presence-and-absence encoding behind a Devanagari cover
    "inj-0098": "deny",  # the same encoding between variation selectors, nothing visible
    "inj-0099": "deny",  # a bitstream deperiodised with one spare cover every three bits
    # Not misses: the BOUNDARY of the bidi signal. Each of these three is a
    # balanced override whose scope reverses the order of what is inside it,
    # measured with GNU FriBidi 1.0.16, and the rule is imbalance rather than
    # presence because denying a balanced pair denies ordinary Arabic and
    # Hebrew, which `inj-0141` and `inj-0142` are in the file to show. Same
    # category as the `secrets` corpus's github_pat_ and xapp- cases.
    #
    # They are listed HERE, among the cases whose label a reader might argue
    # with, because nothing distinguishes them from each other: relabelling one
    # means relabelling all three, and this is what makes that visible. They
    # pass, so no number moves if one is deleted and nothing else would notice.
    "inj-0030": "allow",
    "inj-0038": "allow",
    "inj-0096": "allow",
    # Denied on a WEAKER ground than the rest of the bidi signal, and listed so
    # that the ground is visible. A stray terminator reorders nothing at all --
    # measured, `harmless<PDF> text` renders byte-identically to `harmless
    # text` -- so the "rendered order diverges" rationale does not reach it.
    # They are denied as a malformed control sequence, which is defensible on
    # its own but is not the same claim. `inj-0139` and `inj-0140` are the
    # realistic population: a document split across a balanced LRE ... PDF.
    "inj-0019": "deny",
    "inj-0020": "deny",
    # The same shape in its realistic setting, and they belong here for the
    # reason `inj-0030` and `inj-0038` do: a set that discloses two cases and
    # omits two more of the same shape is a disclosure a reader cannot rely on.
    "inj-0139": "deny",
    "inj-0140": "deny",
}


@pytest.mark.parametrize(("case_id", "decision"), sorted(_INJECTION_DISCLOSED.items()))
def test_a_disclosed_injection_shape_is_in_the_corpus_and_in_the_notice(
    case_id: str, decision: str
) -> None:
    """Present, labelled as the notice says, and named there by id.

    Both halves, because either one alone is half a disclosure. A case dropped
    from the file leaves the notice describing evidence that is not there; an id
    dropped from the notice leaves a case whose label reads as an ordinary
    expectation, which for these twenty-four it is not.
    """
    case = next(
        (c for c in _load("injection-structural", "in-repo").cases if c.id == case_id), None
    )
    assert case is not None, f"{case_id} is disclosed in {NOTICE} and is not in the corpus"
    assert case.expect_decision == decision, (
        f"{case_id} is labelled {case.expect_decision!r} and {NOTICE} describes it as {decision!r}"
    )
    assert case_id in NOTICE.read_text(encoding="utf-8"), (
        f"{case_id} carries a label a reader would disagree with and {NOTICE} does not name it"
    )


# Every file that cites a case id in prose. The detector and its tests cite ids
# to say "this input is that case"; the notice cites them to disclose.
_CITING = (
    ROOT / "src" / "jamjet_guardrails" / "detectors" / "injection_structural.py",
    ROOT / "tests" / "test_injection_structural.py",
    NOTICE,
    ROOT / "README.md",
)
_CASE_ID = re.compile(r"inj-\d{4}")

# Which signal each region of the detector is about, so an id cited inside a
# region can be checked against the case it names. The keys are the function
# definitions that open each region, in source order.
# The zero-width region opens at its CONSTANTS, not at `_zero_width_spans`: the
# default-ignorable table, the set builder and the context test all sit between
# `_bidi_spans` and the span function in source order, so a model that starts
# the region at the span function attributes their citations to the bidi signal.
# That is how this test first failed -- twice, on its own region model rather
# than on a wrong id.
_REGIONS = (
    ("def _is_valid_flag_sequence", "INVISIBLE_TAG_CHARS"),
    ("def _bidi_spans", "BIDI_OVERRIDE"),
    ("_DEFAULT_IGNORABLE: tuple", "ZERO_WIDTH_SMUGGLING"),
)


def _injection_cases() -> dict[str, object]:
    return {case.id: case for case in _load("injection-structural", "in-repo").cases}


def test_every_case_id_cited_in_prose_exists() -> None:
    """The cheapest half of the check, and it catches a typo in an id.

    Written because the alternative is what happened twice in one round: an id
    that reads plausibly, names a real case, and names the WRONG one.
    """
    known = set(_injection_cases())
    for path in _CITING:
        cited = set(_CASE_ID.findall(path.read_text(encoding="utf-8")))
        missing = sorted(cited - known)
        assert missing == [], f"{path.name} cites {missing}, which are not in the corpus"


def test_an_id_cited_as_a_labelled_case_carries_that_label() -> None:
    """Prose that says "labelled `allow`" beside an id has to be right about it.

    This is the half that catches the defect's usual shape, which is not a typo:
    the id written is the ATTACK and the id meant is the NEGATIVE, one digit
    apart. `tests/test_injection_structural.py` said `inj-0130` and `inj-0131`
    "carry these as owned false positives, labelled `allow`" when both are the
    attack bitstreams over the same characters and are labelled `deny`.

    The window is the sentence, not the file: an id and a label phrase have to
    be close enough that a reader would read them together.
    """
    cases = _injection_cases()
    for path in _CITING:
        text = path.read_text(encoding="utf-8")
        for phrase, decision in (("labelled `allow`", "allow"), ("labelled `deny`", "deny")):
            for hit in re.finditer(re.escape(phrase), text):
                window = text[max(0, hit.start() - 240) : hit.end() + 240]
                for case_id in _CASE_ID.findall(window):
                    case = cases[case_id]
                    assert case.expect_decision == decision, (  # type: ignore[attr-defined]
                        f"{path.name} says {phrase!r} near {case_id}, which is labelled "
                        f"{case.expect_decision!r}"  # type: ignore[attr-defined]
                    )


def test_an_id_cited_inside_a_signal_names_a_case_that_signal_is_about() -> None:
    """The half that catches an id naming a case from the wrong signal entirely.

    `_bidi_spans` said "`inj-0132` and `inj-0133` are those two chunks" when
    both are zero-width bitstreams carrying no bidi control at all, and the
    chunks are `inj-0139` and `inj-0140`. A label check cannot see that -- all
    four are labelled `deny` -- so the test is on the CONTENT.

    A POSITIVE cited in a region has to carry a finding of that region's type. A
    NEGATIVE cannot: it is cited precisely because the signal says nothing about
    it, so what is asked of it is that its text is in that signal's DOMAIN --
    a tag character, a bidi control or mark, a default-ignorable character. That
    distinction is what this test got wrong on its first two runs, and both
    times the failure was its own model rather than a wrong id, which is worth
    keeping in the docstring: a check on prose is itself prose until something
    exercises it.
    """
    module = ROOT / "src" / "jamjet_guardrails" / "detectors" / "injection_structural.py"
    source = module.read_text(encoding="utf-8")
    cases = _injection_cases()
    guardrail = build("injection-structural")
    ignorable = {chr(point) for low, high in _DEFAULT_IGNORABLE for point in range(low, high + 1)}
    domains = {
        "INVISIBLE_TAG_CHARS": lambda text: any(0xE0000 <= ord(c) <= 0xE007F for c in text),
        "BIDI_OVERRIDE": lambda text: any(
            c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c"
            for c in text
        ),
        "ZERO_WIDTH_SMUGGLING": lambda text: any(c in ignorable for c in text),
    }
    bounds = sorted((source.index(marker), kind) for marker, kind in _REGIONS)

    for index, (start, kind) in enumerate(bounds):
        end = bounds[index + 1][0] if index + 1 < len(bounds) else len(source)
        for case_id in sorted(set(_CASE_ID.findall(source[start:end]))):
            case = cases[case_id]
            text = case.text  # type: ignore[attr-defined]
            if case.expect_decision == "allow":  # type: ignore[attr-defined]
                assert domains[kind](text), (
                    f"the {kind} region cites {case_id}, a negative carrying nothing "
                    f"that signal reads"
                )
                continue
            verdict = guardrail.check(text, Context(direction="input", origin="model"))
            produced = {finding.type for finding in verdict.findings}
            labelled = {f.type for f in case.expect_findings}  # type: ignore[attr-defined]
            assert kind in produced | labelled, (
                f"the {kind} region of injection_structural.py cites {case_id}, which "
                f"neither produces nor expects a {kind} finding"
            )
