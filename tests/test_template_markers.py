"""Guards for the generated chat-template marker table.

The table is data nobody can check by reading it. Fifty-nine short strings
copied out of nine files across nine repositories look right whatever they say,
and the two ways they can be wrong are invisible on the page: a marker that was
never in any file, and a file that is no longer the one the recorded digest
describes.

So the guards are the same three `docs/specs` asks the vendored Unicode tables
for, in the same order, and for the same reason. Regeneration from the
committed raw files must reproduce the committed module byte for byte, so the
module cannot be edited by hand. The recorded digests must be the digests of
those raw files, so the module cannot describe files that are not there. And a
network-gated run re-fetches every pinned revision, so the day somebody bumps a
pin there is one command that says which files moved.

The first two need no network and run everywhere. The third needs
`JAMJET_GUARDRAILS_NETWORK=1` and is never set in CI: a test suite whose result
depends on the Hub being up is a test suite that goes red for reasons that have
nothing to do with this package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

from jamjet_guardrails.detectors import _template_markers as table

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "generate_template_markers.py"
MODULE = ROOT / "src" / "jamjet_guardrails" / "detectors" / "_template_markers.py"
DATA = ROOT / "template-data"
NOTICE = ROOT / "corpora" / "NOTICE.md"


def _generator() -> ModuleType:
    """Import the generator by path. `scripts/` is deliberately not a package.

    Registered in `sys.modules` BEFORE it executes, and that is not optional:
    `@dataclass` resolves its string annotations through
    `sys.modules[cls.__module__]`, so a module executed without being
    registered raises out of the decorator with a message about `KW_ONLY` that
    says nothing about the real cause.
    """
    name = "generate_template_markers"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, GENERATOR)
    assert spec is not None and spec.loader is not None, f"cannot load {GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_there_is_a_table_and_a_fixture_to_check() -> None:
    """Every guard below is derived from these three. Empty, they prove nothing.

    Written first because the failure it catches is silent: a generator that
    read no files writes an empty table, and a partition test over an empty
    table passes, and a digest test over no digests passes.
    """
    assert len(table.MARKERS) >= 40, f"only {len(table.MARKERS)} markers"
    assert len(table.SOURCES) >= 8, f"only {len(table.SOURCES)} sources"
    committed = sorted(p for p in DATA.rglob("*.json") if p.is_file())
    assert len(committed) >= 9, f"only {len(committed)} files under {DATA}"


def test_the_committed_module_is_what_the_generator_writes(tmp_path: Path) -> None:
    """Byte identity, the same mechanism `BENCHMARKS.md` is held to.

    The table is generated, so the only thing that makes it trustworthy is that
    it can be produced again from inputs a reader can inspect. Edited by hand it
    would still import, still type-check and still look exactly like a generated
    file, and no other test in this suite would notice: nothing else knows what
    the raw files say. This does, because it runs the reader over them again.

    No network. The generator reads `template-data/` and nothing else unless it
    is asked to fetch.
    """
    generated = tmp_path / "_template_markers.py"
    assert _generator().main(["--out", str(generated)]) == 0
    assert generated.read_bytes() == MODULE.read_bytes(), (
        "the committed table is not what the generator produces from "
        "template-data/. Rerun: .venv/bin/python scripts/generate_template_markers.py"
    )


def test_every_recorded_digest_is_the_digest_of_the_committed_file() -> None:
    """The other half of the byte-identity guard, and not the same claim.

    Byte identity says the module matches the files. This says the files match
    what the module says about them, which is what makes the provenance mean
    anything: a digest is a claim about bytes somebody can re-fetch, and a
    digest nothing compares is a hex string.

    Both directions are checked. A recorded digest whose file is missing fails,
    and a committed file no source claims fails too, because a raw file nobody
    generated from is a file nobody reviewed.
    """
    claimed: dict[Path, str] = {}
    for source in (*table.SOURCES, table.HTML_ELEMENT_SOURCE):
        for name, digest in source.files.items():
            claimed[DATA / source.key / name] = digest

    wrong = []
    for path, digest in claimed.items():
        if not path.is_file():
            wrong.append(f"{path.relative_to(ROOT)} is recorded and not committed")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            wrong.append(f"{path.relative_to(ROOT)} is {actual}, recorded as {digest}")
    assert wrong == [], wrong

    stray = sorted(
        str(p.relative_to(ROOT)) for p in DATA.rglob("*") if p.is_file() and p not in claimed
    )
    assert stray == [], f"files under template-data/ that no source records: {stray}"


def test_the_html_exclusion_is_the_element_index_and_not_two_strings() -> None:
    """Decision 9 of the phase 3 design, applied to the one exclusion here.

    `<s>` and `</s>` are what the rule removes today, and a rule spelled as
    those two strings would go on removing exactly those two the day a model
    adopts `<p>` or `<code>` as a delimiter. So the rule is membership in the
    element index of the HTML Standard, and this test re-derives the partition
    from the pinned index rather than from the answer: every excluded marker is
    an element name, and no marker that survived is one.

    The count is asserted through `EXCLUDED_AS_HTML` itself rather than as a
    literal, so a change to the rule shows up as a different partition and not
    as a number somebody edits to match.
    """
    generator = _generator()
    elements, _ = generator.read_html_elements()

    kept = [m for m in table.MARKERS if generator._is_html_element(m, elements)]
    assert kept == [], f"these markers are HTML element names and were not excluded: {kept}"

    not_elements = [
        m for m in table.EXCLUDED_AS_HTML if not generator._is_html_element(m, elements)
    ]
    assert not_elements == [], f"excluded as HTML and not an element name: {not_elements}"
    assert table.EXCLUDED_AS_HTML, "the exclusion removed nothing; the rule would be untested"


def test_no_marker_is_a_reserved_vocabulary_slot() -> None:
    """The other property filter, held the same way.

    A tokenizer that allocates a block of a thousand numbered slots would
    otherwise put a thousand strings no template emits into a table whose whole
    claim is that its strings delimit turns.
    """
    generator = _generator()
    slots = [m for m in table.MARKERS if generator._is_reserved_slot(m)]
    assert slots == [], f"reserved slots reached the table: {slots}"
    assert table.RESERVED_SLOTS_DROPPED > 0, (
        "no reserved slot was dropped, so the rule that drops them is untested"
    )


def test_every_marker_came_out_of_a_recorded_source() -> None:
    """A marker with no source is a marker somebody typed in.

    `MARKERS` and `MARKER_SOURCES` are two views of one table and the table is
    only worth anything if they agree: the check reads `MARKERS`, and a reader
    asking where a string came from reads `MARKER_SOURCES`.
    """
    keys = {source.key for source in table.SOURCES}
    assert sorted(table.MARKERS) == sorted(table.MARKER_SOURCES), (
        "MARKERS and MARKER_SOURCES describe different tables"
    )
    orphans = {
        marker: sorted(set(sources) - keys)
        for marker, sources in table.MARKER_SOURCES.items()
        if set(sources) - keys
    }
    assert orphans == {}, f"markers attributed to repositories SOURCES does not list: {orphans}"
    empty = [marker for marker, sources in table.MARKER_SOURCES.items() if not sources]
    assert empty == [], f"markers with no source at all: {empty}"


def test_every_gated_source_names_the_mirror_its_markers_were_read_from() -> None:
    """A gated model is recorded, and recorded means traceable.

    The three gated repositories cannot be fetched without accepting a licence,
    so their markers come from a mirror. A mirror with no record of what it
    mirrors is the same as no provenance: nothing downstream could tell whether
    `[INST]` came from Meta's tokenizer or from somebody's guess at it.
    """
    for source in table.SOURCES:
        if source.upstream_gated:
            assert source.upstream, f"{source.key} is marked gated and names no upstream"
            assert re.fullmatch(r"[0-9a-f]{40}", source.upstream_revision), (
                f"{source.key} records upstream revision {source.upstream_revision!r}, "
                "which is not a commit"
            )
            assert source.repository != source.upstream, (
                f"{source.key} claims to have fetched the gated repository itself"
            )
    gated = [s.key for s in table.SOURCES if s.upstream_gated]
    assert gated, "no source is marked gated; this guard would prove nothing"


def test_every_source_is_pinned_to_a_commit_and_not_to_a_branch() -> None:
    """The defect this whole script exists to prevent, asserted rather than described.

    A revision that reads `main` produces a table that changes under a release
    nobody here made, and the byte-identity test above would then report the
    Hub's edits as this package's.
    """
    unpinned = [
        source.key
        for source in (*table.SOURCES, table.HTML_ELEMENT_SOURCE)
        if not re.fullmatch(r"[0-9a-f]{40}", source.revision)
    ]
    assert unpinned == [], f"these sources are not pinned to a commit: {unpinned}"


def test_the_notice_records_every_source_repository_with_its_revision() -> None:
    """The licence obligation, held the way `corpora/NOTICE.md` holds the others.

    Two of these repositories carry licences with use restrictions and one is
    read through a mirror. A notice that named seven of the nine would be a
    notice that looked complete, so the domain is read from `SOURCES` rather
    than listed here: a repository added to the table and not to the notice
    fails until somebody writes the attribution.

    Repository and revision on ONE line, the shape `tests/test_corpora.py`
    already requires of the corpus attributions, because an attribution split
    across two rows attributes nothing.
    """
    lines = NOTICE.read_text(encoding="utf-8").splitlines()
    missing = [
        f"{source.repository}@{source.revision[:12]}"
        for source in (*table.SOURCES, table.HTML_ELEMENT_SOURCE)
        if not any(source.repository in line and source.revision in line for line in lines)
    ]
    assert missing == [], f"corpora/NOTICE.md does not record: {missing}"


def test_the_counts_the_notice_publishes_are_the_ones_the_table_holds() -> None:
    """A number in prose that counts something in code is a claim about the code.

    The notice states how many markers ship and how many the HTML rule removed.
    Both are derived here from the table itself, so the sentence cannot drift
    away from the thing it describes; that drift is what `docs/specs` calls a
    published number nobody can rely on.
    """
    # Whitespace-flattened, because the notice is wrapped prose and a count
    # that happens to land at the end of a line is the same claim as one that
    # does not. A raw substring test made the guard depend on where the text
    # wrapped, which is a property of nothing.
    notice = " ".join(NOTICE.read_text(encoding="utf-8").split())
    for label, value in (
        ("markers", len(table.MARKERS)),
        ("HTML element names", len(table.EXCLUDED_AS_HTML)),
        ("reserved slots", table.RESERVED_SLOTS_DROPPED),
    ):
        assert f"{value} {label}" in notice, (
            f"corpora/NOTICE.md does not say '{value} {label}'; the table holds {value}"
        )


def test_nothing_that_ships_imports_the_table_yet() -> None:
    """The module says so, so something has to hold it.

    `template-integrity` is not registered: it has no corpus, no published row,
    no conformance section, and `tests/test_completeness.py` demands all three
    of every name in `AVAILABLE`. Until the check lands, an import of this table
    from anywhere under `src/` means a shipped module reads a table that no
    published number covers.

    When the check does land, this test fails, and the fix is one line in the
    generator's docstring plus deleting this test. That is the intended
    lifetime: the claim in the docstring is what is being held, not the absence.
    """
    from jamjet_guardrails.detectors import AVAILABLE

    assert "template-integrity" not in AVAILABLE, (
        "template-integrity is registered; delete this test and the docstring claim "
        "in scripts/generate_template_markers.py that nothing imports the table"
    )
    importers = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "src").rglob("*.py"))
        if path != MODULE and "_template_markers" in path.read_text(encoding="utf-8")
    ]
    assert importers == [], f"these shipped modules already read the table: {importers}"


# --------------------------------------------------------------------------
# Everything above reads committed files and needs nothing. The one below
# re-fetches every pinned revision, which is what turns the recorded digests
# from strings into verified ones on the day a pin is bumped.
# --------------------------------------------------------------------------

requires_network = pytest.mark.skipif(
    os.environ.get("JAMJET_GUARDRAILS_NETWORK") != "1",
    reason="re-downloads every pinned revision; set JAMJET_GUARDRAILS_NETWORK=1",
)


@requires_network
def test_every_pinned_revision_still_serves_the_bytes_that_were_recorded() -> None:
    """The pin, checked against the artifact it claims to pin.

    A Hub revision is a commit and should be immutable, and this is what says
    so rather than assuming it. It is also the tool for a deliberate bump: point
    a source at a newer commit, run this, and it names every file whose contents
    moved, which is the list that decides whether the table changes.

    A revision that no longer resolves is reported here rather than raised. A
    bare `HTTPError` out of the fetch fails the test too, and it fails it with a
    traceback about urllib instead of the one line that matters, which is which
    pin stopped resolving.
    """
    generator = _generator()
    moved = []
    for source in (*generator.SOURCES, generator.HTML_ELEMENTS):
        for name in source.files:
            local = DATA / source.key / name
            expected = hashlib.sha256(local.read_bytes()).hexdigest()
            try:
                fetched = hashlib.sha256(generator._fetch(source.url(name))).hexdigest()
            except urllib.error.HTTPError as exc:
                moved.append(f"{source.repository}@{source.revision}/{name}: HTTP {exc.code}")
                continue
            if fetched != expected:
                moved.append(
                    f"{source.repository}@{source.revision}/{name}: "
                    f"serves {fetched}, committed is {expected}"
                )
    assert moved == [], f"pinned revisions no longer serve the committed bytes: {moved}"
