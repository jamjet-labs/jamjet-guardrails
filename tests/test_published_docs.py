"""Guards that apply to every published Markdown file, not one file each.

`tests/test_readme.py` and `tests/test_conformance_doc.py` each resolve the
repository paths their own document cites. Neither covers `corpora/NOTICE.md`
or `BENCHMARKS.md`, which are published too and cite paths too: two guards were
written, each for the file its author had open, and the other two published
documents got neither. That is the defect this project produced more than any
other, so the guard here is derived from what git tracks rather than from a
list, and a Markdown file added later is covered without anyone remembering to
add it.
"""

from __future__ import annotations

import re
from pathlib import Path

from _tracked import tracked as shipped

ROOT = Path(__file__).resolve().parent.parent

# The same shape `tests/test_conformance_doc.py` uses, and deliberately not a
# looser one. It anchors on real top-level directories, and on
# `src/jamjet_guardrails` rather than bare `src`, which is what keeps a foreign
# path out: `docs/conformance.md` cites a file inside pixie-io's repository
# when it traces the Fake Name Generator licence chain, and that path starts
# `src/datagen/`. A heuristic about URLs on the same line does NOT work, which I
# found by writing one: the repository that owns the path is named on the line
# above it.
#
# `unicode-data` was added when `corpora/NOTICE.md` began citing the four
# pinned Unicode files, `template-data` when the marker table began citing its
# own, and adding each is the whole lesson this module's docstring states: a
# directory this pattern does not name is a directory whose citations nothing
# checks, and every path in the new section would have been unverified while the
# guard sat right here looking thorough.
#
# `packages/` gets the same treatment one level deeper, and this tree supplied
# the reason before the directory existed: `docs/conformance.md` uses
# `packages/media/` as the ILLUSTRATION of what an unanchored gitignore pattern
# matches, so a bare `packages` prefix turns a worked example about paths into a
# dangling citation. Anchoring on `packages/jamjet-guardrails-`, which is what
# both adapter distributions are actually called, covers every path either
# adapter README can cite and matches nothing that is merely prose about paths.
_CITED = re.compile(
    r"`((?:benchmarks|corpora|tests|scripts|src/jamjet_guardrails|docs"
    r"|unicode-data|template-data|\.github)/[^`]*"
    r"|packages/jamjet-guardrails-[^`]*"
    r"|[A-Z]+\.md)`"
)


def _published_markdown() -> list[Path]:
    """Every tracked Markdown file, minus the ones that are not published prose.

    Read from git, not from a glob: what ships in the sdist is what git tracks,
    so this is the same set a reader can see.
    """
    tracked = shipped("*.md")
    return [ROOT / name for name in tracked]


def test_there_is_something_to_check() -> None:
    """A derived list that came back empty would make every test below vacuous."""
    found = _published_markdown()
    assert len(found) >= 4, f"expected the published docs, found {found}"
    names = {p.name for p in found}
    assert {"README.md", "BENCHMARKS.md", "NOTICE.md", "conformance.md"} <= names


def _defines(text: str, node: str) -> bool:
    """Whether `text` defines a function called exactly `node`.

    Matched to the end of the name, not by substring. `f"def {node}" in text`
    was the earlier spelling, and it says yes to any definition whose name
    merely starts with the cited one: renaming
    `test_no_corpus_carries_share_alike_values` to
    `..._share_alike_values_anywhere` leaves the citation dangling and the
    guard green. There are five such prefix pairs among the test names on this
    branch, so it is not an exotic edit.

    `async` and leading indentation are allowed because neither changes whether
    the name exists, and a guard that missed a method would send someone
    hunting for a test that is right there.
    """
    return (
        re.search(rf"^[ \t]*(?:async[ \t]+)?def[ \t]+{re.escape(node)}[ \t]*\(", text, re.MULTILINE)
        is not None
    )


def test_a_cited_name_is_matched_whole_and_not_as_a_prefix() -> None:
    """The guard on the guard above, over text rather than over the tree.

    Renaming a cited test to a strict superstring of its old name is the edit
    that slipped past: the citation is dangling, the file still parses, and the
    substring is still there. Checked here against a string so the check is a
    property of `_defines` rather than of whichever names happen to exist today.
    """
    assert _defines("def test_share_alike_values():\n    pass\n", "test_share_alike_values")
    assert _defines("    async def test_share_alike_values(self):\n", "test_share_alike_values")
    assert not _defines(
        "def test_share_alike_values_anywhere():\n    pass\n", "test_share_alike_values"
    )
    assert not _defines("def test_share_alike_value():\n", "test_share_alike_values")
    assert not _defines("# def test_share_alike_values() in a comment\n", "test_share_alike_values")
    assert not _defines("call(test_share_alike_values)\n", "test_share_alike_values")


def test_every_repository_path_any_published_doc_cites_exists() -> None:
    """The union of the two per-file guards, plus the two files neither covered."""
    missing: list[str] = []
    checked = 0
    for doc in _published_markdown():
        text = doc.read_text(encoding="utf-8")
        links = [
            target
            for target in re.findall(r"\]\(([^)]+)\)", text)
            if "://" not in target and not target.startswith(("#", "mailto:"))
        ]
        cited = set(_CITED.findall(text)) | set(links)
        for target in sorted(cited):
            # A citation may name a test as `path::test_name`; resolve the file
            # and require the test to be defined, since a renamed test makes the
            # citation as wrong as a deleted file would.
            path, _, node = target.partition("::")
            # Root-relative OR doc-relative. A doc in a subdirectory cites both
            # ways: `corpora/NOTICE.md` names `tests/test_pii.py` from the root
            # and its own siblings from beside it. My first version resolved
            # against the doc's directory alone, because I wrote it thinking of
            # README, which sits at the root and cannot tell the two apart.
            candidates = [ROOT / path, doc.parent / path]
            checked += 1
            resolved = next(
                (c for c in candidates if (c.is_dir() if path.endswith("/") else c.is_file())),
                None,
            )
            if resolved is None:
                missing.append(f"{doc.name} -> {target}")
            elif node and not _defines(resolved.read_text(encoding="utf-8"), node):
                missing.append(f"{doc.name} -> {target} (file exists, test does not)")
    assert checked > 0, "no citations found; this guard would prove nothing"
    assert missing == [], f"published docs cite things that do not exist: {missing}"


def test_no_published_document_uses_an_em_dash() -> None:
    """The house rule, over every published document rather than over README.

    `tests/test_readme.py::test_the_readme_has_no_em_dash` held this for one
    file. Four community health documents were added later, and under a
    file-scoped guard they would have got the same treatment the other published
    documents got when the path and case-id guards were written for whichever
    file their author had open: nothing. Derived from git for that reason, so a
    Markdown file added after this one is covered without anyone remembering.

    THE CHARACTER ONLY, and the narrowing is deliberate rather than an
    oversight. `tests/test_readme.py` also refuses ` -- `, and that half is left
    scoped to README because it is not met: `corpora/NOTICE.md`,
    `docs/conformance.md`, `training/README.md` and one measurement note use the
    construction 53 times between them. Widening the rule to reach them would
    mean rewriting 53 sentences, several of which other tests template claims
    out of word for word, to gain nothing the character check does not already
    give. Widening it and exempting those four files would be worse: an
    exemption list is the thing the next document gets added to. So this guard
    states what is actually true everywhere and holds it there, and the stricter
    rule stays where it is enforced.
    """
    offenders = [
        f"{doc.relative_to(ROOT)} contains an em dash"
        for doc in _published_markdown()
        if "—" in doc.read_text(encoding="utf-8")
    ]
    assert offenders == [], offenders


def test_every_test_a_shipped_source_file_cites_is_defined() -> None:
    """The same citation rule, applied to the source the sdist ships.

    `test_every_repository_path_any_published_doc_cites_exists` resolves
    `path::test_name` citations in Markdown, and this repository cites tests
    from comments and docstrings at least as often as from prose: a comment
    that names the test holding its claim is the house convention. Nothing read
    those, and two of them named a test that has never existed under the name
    they gave, while two other sites in the same tree cited the same test
    correctly. The repository disagreed with itself about the name of a test
    across four files and every one of them was green.

    A dangling citation in source is worse placed than one in Markdown. It is
    the pointer from a defence to the evidence for it, and a reader who cannot
    find the test concludes the defence is unheld rather than that the name
    moved.

    ONLY A BACKTICKED CITATION COUNTS, and the delimiters are what separate a
    citation from a string that merely has the shape of one. `scripts/mutate.py`
    builds node ids with an f-string and carries a worked example of a
    MALFORMED one; `tests/test_completeness.py` asserts on node ids as data.
    None of the three is a claim about a test that exists, and a guard that
    read them would be met by renaming things nobody cites. The opening and
    closing delimiters must match in length, which is what keeps the malformed
    example out: its node part carries a slash where the closing backtick
    would be.

    Derived from what git tracks, like every other guard in this module, so a
    module added later is covered without anyone remembering.
    """
    missing: list[str] = []
    checked = 0
    cited = re.compile(
        r"(?<!`)(`{1,2})(tests/[A-Za-z0-9_./-]+\.py)::([A-Za-z_][A-Za-z0-9_]*)\1(?!`)"
    )
    for name in shipped("*.py"):
        source = ROOT / name
        text = source.read_text(encoding="utf-8")
        for _, path, node in set(cited.findall(text)):
            checked += 1
            target = ROOT / path
            if not target.is_file():
                missing.append(f"{name} -> {path}::{node} (no such file)")
            elif not _defines(target.read_text(encoding="utf-8"), node):
                missing.append(f"{name} -> {path}::{node} (file exists, test does not)")
    assert checked > 0, "no citations found in the shipped source; this guard would prove nothing"
    assert missing == [], f"shipped source cites tests that do not exist: {missing}"
