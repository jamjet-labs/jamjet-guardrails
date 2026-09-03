"""The counts the published files print about the vendored Unicode data.

A number in prose that COUNTS a thing in this repository is a claim, and this
repository holds about thirty of them to a derivation. This one was not held,
and it went stale the moment it mattered: the notice said "The four files under
`unicode-data/16.0.0/`" while `IdentifierStatus.txt` was committed beside them
for the `confusables` check, and the whole suite stayed green. The sentence was
corrected by hand afterwards, which is exactly the state this file exists to
make impossible.

The table under that sentence has the same property one level down. It is the
provenance record for material this repository REDISTRIBUTES rather than merely
names, which the notice itself calls the strongest obligation it carries, so a
file present on disk and absent from the table is a redistribution with no
attribution attached to it.

Both are derived from the directory listing. Nothing here is a list of
filenames: a fifth file was added once and a sixth will be added when the pin
moves, and neither should need anybody to remember this file exists.

AND THE SAME COUNTS SIT IN `pyproject.toml`, WHICH GOT NONE OF THIS. The commit
that derived the notice's count added the derivation for that file alone, while
the SPDX decomposition block three files over went on saying "the four files
published by Unicode, Inc." and "the two modules generated from it" with five
files and three modules on disk. That block ships in the sdist and states its
own purpose as naming WHAT EACH TERM COVERS, "because an expression nobody can
decompose is a longer way of saying nothing". A guard written for the file its
author had open, one file short of the file with the same defect, is the shape
this repository produces more than any other.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICE = ROOT / "corpora" / "NOTICE.md"
PYPROJECT = ROOT / "pyproject.toml"
GENERATED = ROOT / "src" / "jamjet_guardrails" / "_unicode"
DATA = ROOT / "unicode-data" / "16.0.0"

#: The count words this notice actually uses, in the range a directory of
#: published Unicode data plausibly reaches. Spelled out rather than digits
#: because that is how the sentence reads, and matched case-insensitively
#: because it may open a sentence.
_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _committed_files() -> list[str]:
    """Every file under the pinned Unicode data directory, sorted."""
    return sorted(path.name for path in DATA.iterdir() if path.is_file())


def test_there_is_something_to_count() -> None:
    """The vacuity guard. An empty directory would make both tests below pass
    over a notice describing nothing."""
    files = _committed_files()
    assert len(files) >= 4, f"expected the pinned Unicode data, found {files}"


def test_the_notice_counts_the_files_it_says_it_counts() -> None:
    """The sentence names how many files are under that directory.

    Mutation-checked: adding or removing a file under `unicode-data/16.0.0/`
    fails this, and so does editing the word in the sentence.
    """
    text = NOTICE.read_text(encoding="utf-8")
    expected = _NUMBER_WORDS[len(_committed_files())]
    sentence = re.search(
        r"The (\w+) files under\s*\n?`unicode-data/16\.0\.0/`",
        text,
    )
    assert sentence is not None, (
        "corpora/NOTICE.md no longer says how many files are under "
        "unicode-data/16.0.0/; the sentence this guard reads has moved or gone"
    )
    assert sentence.group(1).lower() == expected, (
        f"the notice says {sentence.group(1)!r} files under unicode-data/16.0.0/ and "
        f"there are {len(_committed_files())}, which is {expected!r}"
    )


def test_every_redistributed_unicode_file_is_in_the_provenance_table() -> None:
    """And every row of the table names a file that is there.

    Both directions, because they fail differently. A file missing from the
    table is material redistributed with no attribution beside it, which is the
    obligation the notice says is the strongest one it carries. A row naming a
    file that is gone is a provenance record for something nobody ships, which
    is the kind of claim a reader cannot check and therefore stops checking.

    Mutation-checked: deleting a row fails the first assertion, and adding a row
    for a file that does not exist fails the second.
    """
    text = NOTICE.read_text(encoding="utf-8")
    # The rows are markdown links whose text is the bare filename in backticks,
    # inside a table whose other columns are the byte count and the digest.
    tabled = set(re.findall(r"\|\s*\[`([A-Za-z]+\.txt)`\]\(https://www\.unicode\.org/", text))
    committed = set(_committed_files())

    unattributed = sorted(committed - tabled)
    assert unattributed == [], (
        f"{unattributed} are committed under unicode-data/16.0.0/ and have no row in "
        "the provenance table in corpora/NOTICE.md"
    )
    phantom = sorted(tabled - committed)
    assert phantom == [], (
        f"corpora/NOTICE.md carries a provenance row for {phantom}, which is not "
        "committed under unicode-data/16.0.0/"
    )


def _generated_modules() -> list[str]:
    """The modules `scripts/generate_unicode_tables.py` writes, sorted.

    `__init__.py` is the package's own front door and is hand-written, so it is
    not one of them: it is excluded by reading each module's first line for the
    marker the generator stamps, rather than by naming it here. A fourth
    generated module would be counted with no edit to this helper.
    """
    return sorted(
        path.name
        for path in GENERATED.glob("*.py")
        if "GENERATED; do not edit" in path.read_text(encoding="utf-8")[:400]
    )


def test_there_are_generated_modules_to_count() -> None:
    """The vacuity guard for the two tests below."""
    assert len(_generated_modules()) >= 2, f"found {_generated_modules()}"


def test_the_licence_block_counts_the_files_it_redistributes() -> None:
    """`pyproject.toml` says how many Unicode files it carries, in two places.

    Both said four while five were committed. The block's stated purpose is to
    decompose the SPDX expression, so a term that undercounts what it covers is
    the one failure this block exists to prevent, and the file it sits in is
    what a licence scanner and a distribution reviewer read.
    """
    # Comment continuations rejoined first. The two sentences wrap differently
    # and the second carries a `#` in the middle of the phrase, so a pattern
    # written against one of them silently covers one site and reads as covering
    # both -- which is how the block came to have two copies of this count.
    text = re.sub(r"\n#[ \t]*", " ", PYPROJECT.read_text(encoding="utf-8"))
    expected = _NUMBER_WORDS[len(_committed_files())]
    sentences = re.findall(r"the (\w+) (?:files published by Unicode|published files)", text)
    assert len(sentences) >= 2, (
        f"pyproject.toml states this count in two places and {len(sentences)} were found; "
        "the sentences this guard reads have moved or gone"
    )
    wrong = [word for word in sentences if word.lower() != expected]
    assert wrong == [], (
        f"pyproject.toml says {wrong} Unicode files and there are "
        f"{len(_committed_files())}, which is {expected!r}"
    )


def test_the_licence_block_counts_the_modules_generated_from_them() -> None:
    """The same claim one level down, and it drifted the same way.

    The block said "unicode-data/16.0.0/ and the two modules generated from it"
    while three ship: `scripts.py`, `confusables.py` and `identifiers.py`. Those
    modules are what `Unicode-3.0` covers in the WHEEL, which is the artifact
    most consumers receive, so undercounting them understates what the term is
    there for.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    expected = _NUMBER_WORDS[len(_generated_modules())]
    sentence = re.search(r"the (\w+) modules\n#\s*generated from it", text)
    assert sentence is not None, (
        "pyproject.toml no longer counts the modules generated from the Unicode data"
    )
    assert sentence.group(1).lower() == expected, (
        f"pyproject.toml says {sentence.group(1)!r} generated modules and there are "
        f"{len(_generated_modules())}, which is {expected!r}: {_generated_modules()}"
    )
