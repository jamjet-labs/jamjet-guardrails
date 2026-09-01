"""Corpus sources, fetched once and verified against a recorded hash.

A corpus that changes under a published number is the same failure as a model
that changes under one: the figure stops describing the artifact and nothing
says so. The hash is what makes a re-fetch either identical or loud.

The manifest is read with `yaml.safe_load`. PyYAML sits in
`[project.optional-dependencies].dev`, beside pytest, ruff and mypy, which is
where a test-only dependency belongs. The property this tree exists to protect
is `[project].dependencies = []`, and the two are distinguished mechanically
rather than by care: `tests/test_packaging.py` reads the BUILT metadata and
filters out `extra ==` markers, so a dev extra cannot become a runtime
requirement without that test saying so. CI installs `.[dev]`, so the licence
and contamination screens run on every leg, and nothing under `training/` is in
the wheel either way.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, get_args
from urllib.parse import urlsplit

import yaml

Role = Literal["train", "eval", "excluded"]

# Derived from the Literal rather than restated, so the two cannot drift.
ROLES: tuple[Role, ...] = get_args(Role)

#: The repository root. `training/` sits at the root rather than inside the
#: package, so this file's grandparent is it.
ROOT = Path(__file__).resolve().parent.parent

#: Where a download lands unless a caller says otherwise. `.gitignore` carries
#: `/data/`, anchored to the repository root, and this is that directory.
DATA = ROOT / "data"

#: The URL schemes `fetch` will open. `urllib.request.urlretrieve` also honours
#: `ftp://` and, depending on the build, more than that, so the set is stated
#: rather than inherited. `file` is here because the fetch tests need a local
#: origin to stay hermetic with no network; every URL in the manifest is
#: `https`, and `test_every_url_in_the_manifest_is_https` holds it there.
FETCHABLE_SCHEMES = frozenset({"file", "https"})

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
    sources = [_to_source(entry, path, index) for index, entry in enumerate(entries)]
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


def fetch(source: Source, into: Path = DATA) -> Path:
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
    scheme = urlsplit(source.url).scheme
    if scheme not in FETCHABLE_SCHEMES:
        raise SourceError(
            f"{source.name} is served over {scheme!r}; this fetches "
            f"{sorted(FETCHABLE_SCHEMES)} and nothing else"
        )
    _refuse_a_committed_destination(into)
    into.mkdir(parents=True, exist_ok=True)
    target = into / _filename(source)
    if not target.exists():
        # The URL comes from the manifest, which `load_sources` has already
        # validated, and its scheme has just been checked against the
        # allowlist above.
        urllib.request.urlretrieve(source.url, target)
    verify(target, source.sha256)
    return target


def _refuse_a_committed_destination(into: Path) -> None:
    """A download must not land where git would commit it.

    `.gitignore` anchors `/data/` to the repository root, so `training/data/`
    is NOT ignored -- and `training/data/` is the natural thing for a later
    script to write, because every other file this tree owns lives under
    `training/`. `training/README.md` says where downloads go; this is the part
    that survives someone not reading it.

    Lexical, through `os.path.abspath`, rather than `Path.resolve()`:
    `resolve()` raises `RuntimeError` on a symlink loop up to 3.12 and this
    package's floor is 3.10, so a guard written with it turns a bad destination
    into a traceback instead of a refusal. A destination outside the repository
    is nothing this rule has an opinion about.
    """
    target = Path(os.path.abspath(into))
    if target != ROOT and ROOT not in target.parents:
        return
    if target == DATA or DATA in target.parents:
        return
    raise SourceError(
        f"{target} is inside the repository and outside {DATA}, the one directory "
        ".gitignore keeps out of a commit; a corpus written there would be committed"
    )


def _filename(source: Source) -> str:
    """The local name for a source, keeping whatever extension the URL has.

    The extension is read from the URL rather than fixed at `.jsonl`. The
    sources pinned so far are CSV, and a CSV written to a `.jsonl` path is a
    file whose name tells the next stage to parse it the wrong way.

    The result is checked with `PureWindowsPath`, which knows both separators,
    so a name that would split into a path is refused on every platform rather
    than only on the one where it would do damage. Replacing `/` alone is a
    POSIX answer to a question that has two spellings.
    """
    suffix = PurePosixPath(urlsplit(source.url).path).suffix
    name = f"{source.name.replace('/', '__')}{suffix}"
    if name in (".", "..") or PureWindowsPath(name).name != name:
        raise SourceError(
            f"{source.name} and its URL give the local name {name!r}, which is a path "
            "rather than a filename"
        )
    return name


def _to_source(entry: object, path: Path, index: int) -> Source:
    where = f"{path}[{index}]"
    if not isinstance(entry, dict):
        raise SourceError(f"{where}: source is a {type(entry).__name__}, expected a mapping")

    missing = [key for key in _REQUIRED if key not in entry]
    if missing:
        raise SourceError(f"{where}: source is missing {missing}")
    unknown = sorted(str(key) for key in set(entry) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise SourceError(f"{where}: source carries unknown keys {unknown}")

    fields: dict[str, str] = {}
    for key, value in entry.items():
        # YAML types values, and the types it picks are not the ones a manifest
        # means: `role: yes` is a bool, an all-digit digest is an int, and a
        # key whose value is a nested list is a list. Each of those reaches a
        # field the screens read as text, so each is refused here rather than
        # coerced. A source with a silently retyped field is one every screen
        # below walks straight past.
        text = value.strip() if isinstance(value, str) else ""
        if not isinstance(value, str) or not text:
            raise SourceError(
                f"{where}: {key} is {value!r}; every field a source declares must carry one "
                "non-empty piece of text"
            )
        fields[str(key)] = text

    role = fields["role"]
    if role not in ROLES:
        raise SourceError(f"{where}: role is {role!r}, expected one of {list(ROLES)}")

    sha256 = fields["sha256"]
    if HEX64.match(sha256) is None:
        if sha256 != NO_DIGEST:
            raise SourceError(
                f"{where}: sha256 is {sha256!r}, which is neither 64 lower-case hex "
                f"digits nor {NO_DIGEST!r}"
            )
        if role != "excluded":
            raise SourceError(
                f"{where}: {fields['name']} records no digest and its role is {role!r}. "
                "Only an excluded source may go unpinned, because it is the only role "
                "nothing is measured on or trained from"
            )
        if not fields.get("note", ""):
            raise SourceError(
                f"{where}: {fields['name']} records no digest and no note saying why; "
                "an unpinned source without a reason is one nobody can re-check"
            )

    return Source(
        name=fields["name"],
        url=fields["url"],
        license=fields["license"],
        sha256=sha256,
        # No cast. `ROLES` is annotated `tuple[Role, ...]`, so the membership
        # test above narrows `role` from `str` to the Literal on its own, and
        # mypy rejects a cast here as redundant. A `cast` would have accepted
        # any string the day the membership test was loosened.
        role=role,
        note=fields.get("note", ""),
    )


def _read_manifest(path: Path) -> list[object]:
    """The manifest as a list of whatever YAML made of each entry.

    A parse failure is re-raised as a `SourceError` so that every way this file
    can be wrong -- unreadable, the wrong shape, or a field that does not hold
    up -- reaches a caller as one exception type. A caller catching
    `yaml.YAMLError` separately is a caller that will forget to.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SourceError(f"{path} is not readable as YAML: {error}") from error
    if document is None:
        raise SourceError(f"{path} lists no sources")
    if not isinstance(document, list):
        raise SourceError(
            f"{path} holds a {type(document).__name__}, and a manifest is a list of sources"
        )
    return document
