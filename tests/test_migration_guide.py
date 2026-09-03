"""The migration guide's arithmetic, recomputed from its own table.

`docs/migrating-from-llm-guard.md` publishes counts about another project's
source tree, which CI cannot reach. What CI can do is refuse to let the page
disagree with itself, and that is what every test here does: the summary table
at the top, the counting table below it and the one sentence about models are
each rederived from the 37 rows of the mapping table, so a row edited, added or
deleted without moving the totals fails the build.

That shape was chosen because the alternative failed here before. A number in
prose that counts something is a claim, and the claim this page rests on is
arithmetic rather than a percentage: the design that produced the page rejected
an "80 percent coverage" line outright, on the grounds that the number would be
either false or vacuous, so a percent sign anywhere on the page is refused too.

The one thing these tests can check against the code rather than against the
page is the half of the comparison that is ours: every replacement the table
names has to be a check `build` can actually construct, and the zero-dependency
claim is read out of the installed distribution metadata.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from importlib.metadata import requires
from pathlib import Path

from jamjet_guardrails.detectors import AVAILABLE

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "migrating-from-llm-guard.md"

DIST = "jamjet-guardrails"

STATUSES = ("mapped", "partial", "gap")
DIRECTIONS = ("input", "output")

# One row of the mapping table: a backticked scanner class, a direction, a
# status, the replacement cell and the sentence. Anchored to the line so a
# sentence in prose that happens to contain pipes cannot be read as a row.
_ROW = re.compile(
    r"^\| `([A-Za-z]+)` \| (input|output) \| (mapped|partial|gap) \| (.+?) \| (.+) \|$",
    re.MULTILINE,
)

# The summary table at the top of the page.
_SUMMARY = re.compile(
    r"^\| (mapped|partial|gap|all scanners) \| (\d+) \| (\d+) \| (\d+) \|$",
    re.MULTILINE,
)

# The row of the counting table that states the basis this page uses.
_BASIS = re.compile(r"\*\*(\d+)\*\* \((\d+) input, (\d+) output\)")

# "Of those 37 scanner classes, 23 need a model and 14 do not."
_MODELS = re.compile(r"Of those (\d+) scanner classes, (\d+) need a model and (\d+) do not")

# "migrating that scanner means carrying its 28 phrases yourself"
_PHRASE_COUNT = re.compile(r"its (\d+) phrases yourself")


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Whitespace collapsed, so a rewrapped sentence still matches."""
    return " ".join(text.split())


def _rows() -> list[tuple[str, str, str, str, str]]:
    rows = _ROW.findall(_text())
    assert rows, "no rows parsed out of the mapping table; every test here would prove nothing"
    return rows


def _summary() -> dict[str, tuple[int, int, int]]:
    parsed = {label: (int(a), int(b), int(c)) for label, a, b, c in _SUMMARY.findall(_text())}
    assert set(parsed) == {*STATUSES, "all scanners"}, (
        f"the summary table does not carry one row per status plus a total; found {sorted(parsed)}"
    )
    return parsed


def test_the_guide_exists_and_is_the_one_the_readme_links() -> None:
    """The guard on every test below, and on the README's link.

    A file that moved would make the parses return nothing, and an empty parse
    is the failure mode this whole module is written to avoid.
    """
    assert GUIDE.is_file()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/migrating-from-llm-guard.md" in readme, (
        "the README does not link the migration guide, so the page is published and unreachable"
    )


def test_the_summary_table_is_the_count_of_the_rows_below_it() -> None:
    """The two-tables-must-agree case, which is the one this page is built on.

    A status changed in one row and not in the summary leaves both tables
    internally consistent and the page wrong, and it is the edit most likely to
    happen: a `gap` closes when a check ships.
    """
    rows = _rows()
    summary = _summary()
    counted = Counter((status, direction) for _, direction, status, _, _ in rows)
    for status in STATUSES:
        expected = (
            counted[(status, "input")],
            counted[(status, "output")],
            counted[(status, "input")] + counted[(status, "output")],
        )
        assert summary[status] == expected, (
            f"the summary table says {status} is {summary[status]}, the rows say {expected}"
        )


def test_the_total_row_is_the_number_of_rows_in_the_mapping_table() -> None:
    rows = _rows()
    by_direction = Counter(direction for _, direction, _, _, _ in rows)
    total = _summary()["all scanners"]
    assert total == (by_direction["input"], by_direction["output"], len(rows)), (
        f"the summary total is {total} and the table has {len(rows)} rows, "
        f"{by_direction['input']} input and {by_direction['output']} output"
    )


def test_the_statuses_add_up_to_the_total() -> None:
    """The arithmetic the page publishes instead of a percentage."""
    summary = _summary()
    for column in range(3):
        parts = sum(summary[status][column] for status in STATUSES)
        assert parts == summary["all scanners"][column], (
            f"column {column} of the summary sums to {parts}, not {summary['all scanners'][column]}"
        )


def test_the_counting_table_states_the_same_basis_the_rows_use() -> None:
    """The page explains that four ways of counting scanners disagree, then
    picks one. The one it picks has to be the one it shipped."""
    match = _BASIS.search(_text())
    assert match is not None, "the counting table does not state the basis in the expected shape"
    total, inputs, outputs = (int(value) for value in match.groups())
    rows = _rows()
    by_direction = Counter(direction for _, direction, _, _, _ in rows)
    assert (total, inputs, outputs) == (len(rows), by_direction["input"], by_direction["output"])


def test_the_model_split_covers_every_row_and_no_more() -> None:
    """23 plus 14 is a claim about the same 37 rows the table carries."""
    match = _MODELS.search(_flat(_text()))
    assert match is not None, "the page does not state the model-backed split in the expected shape"
    total, backed, free = (int(value) for value in match.groups())
    assert total == len(_rows())
    assert backed + free == total, f"{backed} plus {free} is not {total}"
    assert backed > 0 and free > 0


def test_every_replacement_the_table_names_is_a_check_this_library_can_build() -> None:
    """The half of the comparison that is ours, and the one that can go stale.

    A row naming a check that does not exist is a forecast published as
    coverage. Derived from the registry so a check renamed, or one written into
    the page before it ships, fails here rather than in a reader's terminal.
    """
    named: set[str] = set()
    for scanner, direction, status, replacement, _ in _rows():
        found = set(re.findall(r"`([a-z][a-z0-9-]*)`", replacement))
        if replacement.strip() == "none":
            assert status == "gap", f"{scanner} ({direction}) names no replacement and is {status}"
            assert not found
        else:
            assert status != "gap", f"{scanner} ({direction}) is a gap and names {sorted(found)}"
            assert found, f"{scanner} ({direction}) is {status} and names no replacement"
        named |= found
    unknown = sorted(named - set(AVAILABLE))
    assert unknown == [], f"the guide names {unknown} as replacements, and `build` refuses them"
    assert named, "no replacement named anywhere; this guard would prove nothing"


def test_no_scanner_appears_twice_in_one_direction() -> None:
    """Six class names appear on both sides of llm-guard and are two rows.

    A duplicated row would inflate the total while every other test here still
    passed, because they all count the same rows.
    """
    duplicated = [
        key
        for key, count in Counter(
            (scanner, direction) for scanner, direction, _, _, _ in _rows()
        ).items()
        if count > 1
    ]
    assert duplicated == [], f"the mapping table repeats {duplicated}"


def test_the_guide_publishes_no_percentage() -> None:
    """Decision 8 of the phase 3 design, held mechanically.

    A coverage percentage over these rows would be either false or vacuous,
    because the scanners are not interchangeable units: 23 of the 37 are a model
    making a judgment. The page says so in a sentence; this refuses the
    character, because "80% coverage" is the line somebody adds to a migration
    guide without thinking about it, and a percent sign has no other use here.
    """
    text = _text()
    assert "%" not in text, "the migration guide publishes a percentage"
    assert "percentage" in text, "the page no longer explains why it publishes no percentage"


def test_the_refusal_list_is_as_long_as_the_page_says_it_is() -> None:
    """The one llm-guard fact a reader can check on the page itself.

    The 28 phrases are copied out of `output_scanners/no_refusal.py` at v0.3.16,
    which CI cannot reach, so the count is published beside the list rather than
    asserted about the other repository. What is held here is that the two
    agree: a phrase dropped while editing leaves the sentence claiming a list
    the page does not carry.
    """
    match = _PHRASE_COUNT.search(_flat(_text()))
    assert match is not None, "the page does not state how many refusal phrases it carries"
    stated = int(match.group(1))

    blocks = re.findall(r"```py\n(.*?)```", _text(), re.DOTALL)
    tuples = [
        node.value
        for block in blocks
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "REFUSALS" for t in node.targets)
    ]
    assert len(tuples) == 1, "expected exactly one REFUSALS assignment in the page's code blocks"
    phrases = ast.literal_eval(tuples[0])
    assert len(phrases) == stated, f"the page says {stated} phrases and lists {len(phrases)}"
    assert len(set(phrases)) == len(phrases), "the refusal list repeats a phrase"


def test_every_worked_replacement_on_the_page_runs_and_does_what_it_claims() -> None:
    """The snippets are executed, not read.

    A migration guide is a page of code somebody pastes, and this project has a
    standing rule that a published snippet is run against the installed package
    before it ships. Both blocks are executed in one namespace, in order, and
    then required to produce the two behaviours the surrounding prose promises:
    the pattern block denies a ticket id, and the refusal block denies a
    refusal. Asserted by behaviour rather than by position, so reordering the
    page or adding a third block does not silently stop checking either one.
    """
    blocks = re.findall(r"```py\n(.*?)```", _text(), re.DOTALL)
    assert len(blocks) >= 2, f"expected the worked replacements, found {len(blocks)} code blocks"

    namespace: dict[str, object] = {}
    guards: list[object] = []
    for block in blocks:
        exec(compile(block, str(GUIDE), "exec"), namespace)  # noqa: S102
        built = namespace.get("guard")
        if built is not None and built not in guards:
            guards.append(built)

    context = namespace["Context"](direction="input", origin="user")  # type: ignore[operator]
    decisions = [
        (
            guard.check("see JIRA-1234 today", context).decision,  # type: ignore[attr-defined]
            guard.check("I'm sorry, I cannot help with that", context).decision,  # type: ignore[attr-defined]
            guard.check("here is the answer you wanted", context).decision,  # type: ignore[attr-defined]
        )
        for guard in guards
    ]
    assert ("deny", "allow", "allow") in decisions, (
        f"no block on the page denies a ticket id and allows ordinary text; got {decisions}"
    )
    assert ("allow", "deny", "allow") in decisions, (
        f"no block on the page denies a refusal and allows ordinary text; got {decisions}"
    )


def test_the_zero_dependency_claim_is_what_the_installed_metadata_says() -> None:
    """Both directions, as `tests/test_readme.py` holds the same claim.

    This is the strongest line on the page and the reason anybody reads it, so
    a dependency added later has to break the sentence rather than the sentence
    quietly outliving it.
    """
    declared = requires(DIST) or []
    runtime = [item for item in declared if "extra ==" not in item]
    assert runtime == [], f"{DIST} declares runtime dependencies: {runtime}"
    assert "declares zero runtime dependencies" in _flat(_text())
