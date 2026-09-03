"""What this distribution contains, asked of git or of the filesystem.

Seven test modules ask the same question: which files does this project ship.
Every one of them asked it by shelling out to `git ls-files` with `check=True`,
which is the right source in a checkout and raises in the one place the answer
matters most.

**`pyproject.toml` states the sdist's purpose three times.** "The sdist ships the
tests, and the tests ARE the corpora for anything that scores." "The sdist is the
evidence." "It is what a distribution reviewer, a licence scanner or a rebuild
reads to see what this package IS." An unpacked sdist carries `.gitignore` and
`.github/` and no `.git`, so `git ls-files` exits non-zero there, and
`tests/test_workflows.py` made that call at MODULE level inside a parametrize
argument, which raises during COLLECTION. The result: running the suite inside an
unpacked sdist executed zero of its tests. The evidence could not be run, and
three sentences said it was the evidence.

So the question is asked of git where git is there and of the filesystem where it
is not, and that is not a weakening. In a checkout, git tracking is the right
answer because it is what the sdist will be built from. In an unpacked sdist,
walking what is present is the same question answered from the artifact itself:
the files that are there are exactly the files that shipped. The guards that read
this stay guards in both places, and the one place they could not run is the one
they were written to describe.

The fallback is NOT a general "if anything goes wrong, walk the tree". It is
taken only when this is not a git working tree at all. A `git ls-files` that
fails for any other reason still raises, because a broken git in a checkout is a
fault worth stopping for and not a reason to quietly measure something else.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Directories that exist in a checkout, never ship, and would otherwise be
#: walked by the fallback. Derived from `.gitignore`'s own top-level entries
#: rather than listed here would be better; they are named because the fallback
#: only ever runs where `.gitignore` is not being honoured by anything.
_NEVER_SHIPPED = {
    ".git",
    ".venv",
    ".venv-training",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    "data",
    ".claude",
    ".superpowers",
}


@lru_cache(maxsize=1)
def _is_git_worktree() -> bool:
    """Whether ROOT is inside a git working tree.

    Asked once. `rev-parse --is-inside-work-tree` is the question git itself
    answers for this, and it is cheap; the alternative of testing for a `.git`
    entry gets a worktree wrong, and this repository is developed in worktrees.
    """
    done = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0 and done.stdout.strip() == "true"


def tracked(*patterns: str) -> list[str]:
    """Every shipped path matching the patterns, as repository-relative strings.

    Patterns are `git ls-files` pathspecs. Under the fallback they are matched
    with `Path.rglob` over the same shapes this repository actually uses: a
    directory (`unicode-data`), a suffix glob (`*.md`), or a directory plus one
    (`.github/workflows`). A pattern outside those raises rather than silently
    matching nothing, because a guard that quietly measures an empty set is the
    failure every caller of this module was written to prevent.
    """
    if _is_git_worktree():
        done = subprocess.run(
            ["git", "ls-files", *patterns],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return sorted(done.stdout.split())

    found: set[str] = set()
    for pattern in patterns or ("",):
        candidates: Iterable[Path]
        if pattern in ("", "."):
            candidates = ROOT.rglob("*")
        elif pattern.startswith("*."):
            candidates = ROOT.rglob(pattern)
        elif "*" not in pattern:
            base = ROOT / pattern
            candidates = (
                base.rglob("*") if base.is_dir() else iter([base] if base.is_file() else [])
            )
        else:
            raise AssertionError(
                f"tests/_tracked.py cannot match the pathspec {pattern!r} without git; "
                "add the shape here rather than letting a guard measure an empty set"
            )
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if _NEVER_SHIPPED.intersection(relative.parts):
                continue
            found.add(str(relative))
    return sorted(found)
