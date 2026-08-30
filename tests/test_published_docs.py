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
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The same shape `tests/test_conformance_doc.py` uses, and deliberately not a
# looser one. It anchors on real top-level directories, and on
# `src/jamjet_guardrails` rather than bare `src`, which is what keeps a foreign
# path out: `docs/conformance.md` cites a file inside pixie-io's repository
# when it traces the Fake Name Generator licence chain, and that path starts
# `src/datagen/`. A heuristic about URLs on the same line does NOT work, which I
# found by writing one: the repository that owns the path is named on the line
# above it.
_CITED = re.compile(
    r"`((?:corpora|tests|scripts|src/jamjet_guardrails|docs|\.github)/[^`]*|[A-Z]+\.md)`"
)


def _published_markdown() -> list[Path]:
    """Every tracked Markdown file, minus the ones that are not published prose.

    Read from git, not from a glob: what ships in the sdist is what git tracks,
    so this is the same set a reader can see.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / name for name in tracked]


def test_there_is_something_to_check() -> None:
    """A derived list that came back empty would make every test below vacuous."""
    found = _published_markdown()
    assert len(found) >= 4, f"expected the published docs, found {found}"
    names = {p.name for p in found}
    assert {"README.md", "BENCHMARKS.md", "NOTICE.md", "conformance.md"} <= names


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
            elif node and f"def {node}" not in resolved.read_text(encoding="utf-8"):
                missing.append(f"{doc.name} -> {target} (file exists, test does not)")
    assert checked > 0, "no citations found; this guard would prove nothing"
    assert missing == [], f"published docs cite things that do not exist: {missing}"
