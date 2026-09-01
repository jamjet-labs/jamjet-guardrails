"""Corpus sources, fetched once and verified against a recorded hash.

A corpus that changes under a published number is the same failure as a model
that changes under one: the figure stops describing the artifact and nothing
says so. The hash is what makes a re-fetch either identical or loud.

The manifest reader is written against the standard library rather than PyYAML,
and the reason is that `tests/test_training_data.py` imports this module. That
test is the licence and contamination screen, it runs in CI over committed
data, and CI installs `.[dev]` and nothing else. A PyYAML import here would
either add a dependency to a `[project]` table -- which the whole point of the
`training/` tree is to avoid -- or turn the screen into a test that skips on
every CI leg, which is the same as not having it.

So the reader implements the subset of YAML `training/sources.yaml` actually
uses: a top-level sequence of mappings, plain and quoted scalars, and folded
(`>`) blocks for the notes. Anything outside that subset raises rather than
being guessed at, and `test_the_manifest_reader_agrees_with_the_yaml_library`
compares the result against PyYAML wherever PyYAML is installed.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, get_args
from urllib.parse import urlsplit

Role = Literal["train", "eval", "excluded"]

# Derived from the Literal rather than restated, so the two cannot drift.
ROLES: tuple[Role, ...] = get_args(Role)

#: What a source records in `sha256` when there is no content to hash. Only an
#: `excluded` source may carry it, and `fetch` refuses it outright: a download
#: with nothing to check it against is not a pinned source.
NO_DIGEST = "unavailable"

#: What a recorded digest has to look like. Part of the manifest contract,
#: so `tests/test_training_data.py` imports this rather than restating the
#: pattern: two copies of a rule drift and both sides look right alone.
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

_REQUIRED = ("name", "url", "license", "sha256", "role")
_OPTIONAL = ("note",)

# The three line shapes the manifest is allowed to use. A key is lower-case and
# snake-cased because those are the dataclass's field names; a manifest that
# reads `Name:` is a manifest whose author expected something this does not do.
_ITEM = re.compile(r"\A- ([a-z][a-z0-9_]*):[ ]?(.*)\Z")
_KEY = re.compile(r"\A {2}([a-z][a-z0-9_]*):[ ]?(.*)\Z")
# Exactly four spaces, not "four or more". YAML keeps a more-indented line
# inside a folded block literal instead of folding it, so accepting deeper
# indentation here would agree with PyYAML on the shape and disagree with it on
# the text. Rejecting it means the two never quietly differ.
_FOLD = re.compile(r"\A {4}(\S.*)\Z")

# Characters that open a YAML construct this reader does not implement. Listed
# so an unsupported manifest fails on the line that introduced it, rather than
# being read as a plain string that happens to start with a sigil.
_UNSUPPORTED_OPENERS = ">|[{&*!%@`"


class SourceError(Exception):
    """A source is missing, malformed, or does not hash to what was recorded."""


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    url: str
    license: str
    sha256: str
    role: Role
    note: str = ""


def load_sources(path: Path) -> list[Source]:
    """Every source in the manifest, validated at the boundary.

    Validation lives here rather than in each caller because the rules are
    about the manifest as a whole: a role outside the three, a digest that is
    neither hex nor the recorded absence of one, an absence paired with a role
    that gets measured on. A caller that had to re-check those would be a
    caller that could forget to.
    """
    entries = _read_manifest(path)
    if not entries:
        raise SourceError(f"{path} lists no sources")
    sources = [_to_source(entry, path, lineno) for entry, lineno in entries]
    names = [source.name for source in sources]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise SourceError(f"{path} names {repeated} more than once")
    return sources


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_sha: str) -> None:
    actual = sha256_of(path)
    if actual != expected_sha:
        raise SourceError(
            f"{path.name} sha256 is {actual}, recorded {expected_sha}; the corpus moved "
            "under the numbers measured on it"
        )


def fetch(source: Source, into: Path) -> Path:
    """Download a source if it is not already present, and verify it either way.

    The verification is outside the download branch on purpose. A file left in
    `data/` by an interrupted run, or by an earlier revision of the manifest,
    is exactly the case a hash exists to catch, and checking only what this
    call downloaded would skip it.
    """
    if source.sha256 == NO_DIGEST:
        raise SourceError(
            f"{source.name} records no digest, so there would be nothing to check a "
            "download against; it cannot be fetched"
        )
    into.mkdir(parents=True, exist_ok=True)
    target = into / _filename(source)
    if not target.exists():
        # The URL comes from the manifest, which `load_sources` has already
        # validated, and never from anything a caller passes in.
        urllib.request.urlretrieve(source.url, target)
    verify(target, source.sha256)
    return target


def _filename(source: Source) -> str:
    """The local name for a source, keeping whatever extension the URL has.

    The extension is read from the URL rather than fixed at `.jsonl`. The
    sources pinned so far are CSV, and a CSV written to a `.jsonl` path is a
    file whose name tells the next stage to parse it the wrong way.
    """
    suffix = PurePosixPath(urlsplit(source.url).path).suffix
    return f"{source.name.replace('/', '__')}{suffix}"


def _to_source(entry: dict[str, str], path: Path, lineno: int) -> Source:
    where = f"{path}:{lineno}"
    missing = [key for key in _REQUIRED if key not in entry]
    if missing:
        raise SourceError(f"{where}: source is missing {missing}")
    unknown = sorted(set(entry) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise SourceError(f"{where}: source carries unknown keys {unknown}")

    role = entry["role"]
    if role not in ROLES:
        raise SourceError(f"{where}: role is {role!r}, expected one of {list(ROLES)}")

    sha256 = entry["sha256"]
    if HEX64.match(sha256) is None:
        if sha256 != NO_DIGEST:
            raise SourceError(
                f"{where}: sha256 is {sha256!r}, which is neither 64 lower-case hex "
                f"digits nor {NO_DIGEST!r}"
            )
        if role != "excluded":
            raise SourceError(
                f"{where}: {entry['name']} records no digest and its role is {role!r}. "
                "Only an excluded source may go unpinned, because it is the only role "
                "nothing is measured on or trained from"
            )
        if not entry.get("note", "").strip():
            raise SourceError(
                f"{where}: {entry['name']} records no digest and no note saying why; "
                "an unpinned source without a reason is one nobody can re-check"
            )

    return Source(
        name=entry["name"],
        url=entry["url"],
        license=entry["license"],
        sha256=sha256,
        # No cast. `ROLES` is annotated `tuple[Role, ...]`, so the membership
        # test above narrows `role` from `str` to the Literal on its own, and
        # mypy rejects a cast here as redundant. A `cast` would have accepted
        # any string the day the membership test was loosened.
        role=role,
        note=entry.get("note", ""),
    )


def _read_manifest(path: Path) -> list[tuple[dict[str, str], int]]:
    """The manifest as (mapping, line the item opened on) pairs, or an error.

    The line number travels with the mapping so that a validation failure names
    the entry a reader can go and look at, rather than an index into a list.
    """
    entries: list[tuple[dict[str, str], int]] = []
    entry: dict[str, str] = {}
    entry_line = 0
    fold_key = ""
    fold_lines: list[str] = []

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if fold_key:
            folded = _FOLD.match(line)
            if folded is not None:
                fold_lines.append(folded.group(1).rstrip())
                continue
            # A line that is not part of the block ends it. A blank line inside
            # a folded block is a paragraph break in YAML and this reader does
            # not implement that: the block ends here, and any further indented
            # line then fails to match `- ` or a key and raises below.
            entry[fold_key] = " ".join(fold_lines)
            fold_key = ""
            fold_lines = []

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        item = _ITEM.match(line)
        key = _KEY.match(line)
        if item is not None:
            if entry_line:
                entries.append((entry, entry_line))
            entry = {}
            entry_line = lineno
            name, raw = item.group(1), item.group(2)
        elif key is not None:
            if not entry_line:
                raise SourceError(f"{path}:{lineno}: a key appears before any `- ` item")
            name, raw = key.group(1), key.group(2)
        else:
            raise SourceError(
                f"{path}:{lineno}: this is not a `- key: value` item, a two-space "
                "`key: value`, or a four-space folded continuation"
            )

        if name in entry:
            raise SourceError(f"{path}:{lineno}: duplicate key {name!r}")
        if raw.strip() == ">":
            fold_key = name
            fold_lines = []
        else:
            entry[name] = _scalar(raw, path, lineno)

    if fold_key:
        entry[fold_key] = " ".join(fold_lines)
    if entry_line:
        entries.append((entry, entry_line))
    return entries


def _scalar(raw: str, path: Path, lineno: int) -> str:
    where = f"{path}:{lineno}"
    value = raw.strip()
    if not value:
        raise SourceError(f"{where}: empty value; every field a source declares must carry one")
    if value[0] in _UNSUPPORTED_OPENERS:
        raise SourceError(
            f"{where}: {value[0]!r} opens a YAML construct this reader does not implement"
        )
    if value[0] in "\"'":
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise SourceError(f"{where}: a value opened with {quote!r} is not closed")
        inner = value[1:-1]
        if quote in inner or "\\" in inner:
            raise SourceError(
                f"{where}: a quoted value containing {quote!r} or a backslash needs escape "
                "handling this reader does not implement"
            )
        return inner
    if " #" in value:
        raise SourceError(
            f"{where}: ' #' starts a comment inside a plain YAML scalar; quote the value "
            "if the '#' is part of it"
        )
    return value
