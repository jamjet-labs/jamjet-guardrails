"""`docs/performance.md` publishes one answer per check per size, and no more.

The page carried the `rules` latency TWICE, old and new, in one table with no
column distinguishing them: 0.135 ms and 0.699 ms at a kilobyte, 142.608 ms and
730.951 ms at a megabyte, a factor of 5.1 apart. The commit that added the
post-`fold_confusables` block never removed the pre-`fold_confusables` one, so a
reader sent here from `README.md` to decide whether an in-process check fits
their latency budget got two answers and no way to tell which was current. Half
the published rows for that check were unreproducible by the command the page
prints, because the configuration they were measured under no longer exists.

The same edit left the prose split in two: a stale bullet quoting the old ratios
and a replacement tail appended, with no bullet marker, onto the end of the
`script-constraint` bullet, so it read as a run-on continuation of a different
check and attributed `rules`' fixture to that one.

Nothing read this page. `tests/test_readme.py` and `tests/test_conformance_doc.py`
each guard their own document and `tests/test_published_docs.py` guards the
citations and the em dash in all of them; the numbers table here was
hand-maintained and unread. These guards are derived from the registry and from
the measurement script's own `SIZES`, so a check added later is covered without
anyone remembering to come here, which is the failure mode that produced the
duplicate in the first place.

There is deliberately NO assertion about the values. A timing figure is not
reproducible in CI and `docs/performance.md` says so at the top. What is checked
is the SHAPE: that every registered check has exactly one row per size, and that
the provenance table names every check. Both are facts about the document rather
than about a clock.
"""

from __future__ import annotations

import re
from pathlib import Path

from jamjet_guardrails.detectors import AVAILABLE

ROOT = Path(__file__).resolve().parent.parent
PERFORMANCE = ROOT / "docs" / "performance.md"

# Read from the script that produced the rows rather than restated here. A page
# measured at sizes the script no longer uses is the same defect one column over.
_SIZES: tuple[int, ...]


def _sizes() -> tuple[int, ...]:
    """`SIZES` out of `scripts/measure_throughput.py`, without importing it.

    The script is not a package and importing it drags in the whole benchmark
    harness for one tuple. Read as text for the same reason the corpus guards
    read corpora as text: the question is what the artifact says.
    """
    source = (ROOT / "scripts" / "measure_throughput.py").read_text(encoding="utf-8")
    match = re.search(r"^SIZES:[^=]*=\s*\(([^)]*)\)", source, re.MULTILINE)
    assert match is not None, "scripts/measure_throughput.py no longer declares SIZES"
    return tuple(
        int(part.strip().replace("_", "")) for part in match.group(1).split(",") if part.strip()
    )


def _rows() -> list[tuple[str, int]]:
    """Every `(check, chars)` pair the numbers table publishes, in page order.

    Duplicates are KEPT, because a duplicate is the thing being looked for.
    """
    rows: list[tuple[str, int]] = []
    for line in PERFORMANCE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*([a-z-]+)\s*\|\s*([\d ]+)\s*\|\s*[\d ]+\s*\|", line)
        if match is None:
            continue
        rows.append((match.group(1), int(match.group(2).replace(" ", ""))))
    return rows


def test_there_is_something_to_check() -> None:
    """The guard on every test below. An empty parse would make them vacuous."""
    assert PERFORMANCE.is_file()
    assert len(_rows()) >= len(AVAILABLE) * len(_sizes())
    assert len(_sizes()) >= 4


def test_no_check_is_published_twice_at_one_size() -> None:
    """The defect: two `rules` blocks, 5.1x apart, in one unlabelled table.

    A table with two rows for one `(check, size)` publishes two latencies for
    one call and marks neither. Every other guard on this page would pass with
    both blocks present, because each block is internally consistent and each
    reads correctly on its own.
    """
    seen: dict[tuple[str, int], int] = {}
    for row in _rows():
        seen[row] = seen.get(row, 0) + 1
    duplicated = sorted(row for row, count in seen.items() if count > 1)
    assert duplicated == [], (
        f"docs/performance.md publishes more than one row for {duplicated}; "
        "a table with two answers for one call has no answer"
    )


def test_every_registered_check_is_published_at_every_size() -> None:
    """Derived from the registry and from the script's own `SIZES`.

    A check that ships with no row here is a check whose cost the page does not
    state, and the page is what `README.md` sends a reader to for that cost.
    """
    expected = {(check, size) for check in AVAILABLE for size in _sizes()}
    published = set(_rows())
    assert published == expected, (
        f"missing rows: {sorted(expected - published)}; unexpected rows: {sorted(published - expected)}"
    )


def test_the_measurement_table_names_every_registered_check() -> None:
    """The provenance table replaces a sentence that counted checks in prose.

    It said "`script-constraint` was measured on 2026-09-03 and the other four
    when this page was first written" while nine checks were registered and two
    had been re-measured since. A count of checks written into a sentence is a
    claim about the registry, and this is that claim derived.
    """
    text = PERFORMANCE.read_text(encoding="utf-8")
    start = text.index("### When each row was measured")
    end = text.index("\n**A row IS re-measured", start)
    section = text[start:end]
    named = set(re.findall(r"^\|\s*`([a-z-]+)`\s*\|", section, re.MULTILINE))
    assert named == set(AVAILABLE), (
        f"the measurement table names {sorted(named)}; the registry has {sorted(AVAILABLE)}"
    )
