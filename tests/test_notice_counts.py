"""The counts `corpora/NOTICE.md` prints about the vendored Unicode files.

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
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICE = ROOT / "corpora" / "NOTICE.md"
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
