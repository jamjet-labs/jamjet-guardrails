"""Corpus sources, fetched once and verified against a recorded hash.

A corpus that changes under a published number is the same failure as a model
that changes under one: the figure stops describing the artifact and nothing
says so. The hash is what makes a re-fetch either identical or loud.

The manifest is read with PyYAML, through `ManifestLoader` below, which is
`yaml.SafeLoader` with the strictness the file depends on added back. PyYAML
sits in
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
from typing import Any, Literal, get_args
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

#: What a source `name` may look like, and the only place this manifest's own
#: pinning syntax is written down.
#:
#: A dataset identifier is a base id -- `<org>/<name>`, or a bare `name` -- followed
#: by any number of suffixes introduced by `@` or `:`. Both spellings are
#: already in use here: `corpora/NOTICE.md` records a source as
#: `nvidia/Nemotron-PII@b70ffaf`, and ProtectAI's own licence summary spells a
#: dataset `natolambert/xstest-v2-copy:1_full_compliance`. Every source in this
#: tree is required to be pinned, so a suffixed name is the normal case rather
#: than the exotic one.
#:
#: The contamination and attribution screens compare `base_id(name)`, which is
#: this pattern's own `base` group. Writing the grammar down once is the point:
#: a suffix syntax added here is a suffix the screens strip on the same edit,
#: where a list of separators written out inside a screen would go on matching
#: nothing and reporting a pass. A name outside this grammar is refused at
#: load, so there is no third spelling that both loads and evades the screens.
_ATOM = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_BASE = rf"(?P<base>{_ATOM}(?:/{_ATOM})?)"
_PINS = rf"(?P<pins>(?:[@:]{_ATOM})*)"

SOURCE_NAME = re.compile(rf"\A{_BASE}{_PINS}\Z")

#: The same grammar, unanchored and requiring the `/`, for finding identifiers
#: inside prose. Built from the same atoms as `SOURCE_NAME` rather than written
#: out again, because the two drifting apart is how a screen comes to look at a
#: different set of names from the one the manifest can hold.
QUALIFIED_ID = re.compile(rf"{_ATOM}/{_ATOM}(?:[@:]{_ATOM})*")

_REQUIRED = ("name", "url", "license", "sha256", "role")
_OPTIONAL = ("note",)


class SourceError(Exception):
    """A source is missing, malformed, or does not hash to what was recorded."""


class ManifestLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` with the strictness the manifest depends on put back.

    `safe_load` closes the dangerous half of YAML: `!!python/object/apply` is an
    unconstructible tag rather than a call, and that stays true here. It leaves
    open three things the hand-rolled reader this replaced refused outright, and
    each of them changes what a source says with nothing reporting it, which
    against a file whose whole purpose is a recorded content hash is the wrong
    way round:

    - a repeated key inside one entry is last-wins, so a second `sha256:` line
      pins the corpus to whichever happens to come second;
    - an anchor or an alias lets one field take another's text, so the file can
      say something no reader of it would expect;
    - a ` #` inside a plain scalar starts a comment, so an unquoted URL carrying
      a fragment loses it silently.

    All three are closed here, in one class, rather than at each place a
    manifest is read: strictness spread across callers is strictness one caller
    forgets. `load_sources` is the only thing that reads a manifest and it is
    the only thing that should have to know.
    """

    def _next_event(self) -> yaml.events.Event:
        """`peek_event` with a type. types-PyYAML leaves it unannotated."""
        event: yaml.events.Event = self.peek_event()  # type: ignore[no-untyped-call]
        return event

    def compose_node(self, parent: yaml.nodes.Node | None, index: int) -> yaml.nodes.Node | None:
        """Anchors and aliases, refused where they are introduced.

        Refusing the alias alone would leave the anchor definition legal and
        inert, which is a manifest carrying syntax that means nothing. Both ends
        go, so a `<<:` merge key has nothing to merge from either; the merge key
        itself is refused separately, because `<<: {inline: mapping}` needs no
        anchor.

        How the two arms divide, because two mutations were needed to find out.
        A *defined* anchor is refused at its definition, so an alias to it is
        never composed; the only thing that reaches the alias arm is an alias to
        an anchor that does not exist. An `AliasEvent` carries the name in
        `.anchor` too, so the anchor arm below would refuse that input as well.
        The safety property is therefore held by either arm alone, and what the
        alias arm buys is that an alias is reported as an alias: a refusal that
        calls it an anchor sends the reader looking for an `&` that is not
        there. Each arm is pinned by the wording of its own message, because
        that is the only thing that distinguishes them.

        `index` is annotated `int` because the stub says so. PyYAML itself
        passes the key node here when it composes a mapping value, and this only
        forwards it, so the annotation is the stub's inaccuracy rather than
        this class's.
        """
        event = self._next_event()
        where = f"line {event.start_mark.line + 1}"
        if self.check_event(yaml.events.AliasEvent):
            raise SourceError(
                f"{where}: an alias (`*name`) takes another field's text, and a source has "
                "to say what it says"
            )
        anchor = getattr(event, "anchor", None)
        if anchor is not None:
            raise SourceError(
                f"{where}: an anchor (`&{anchor}`) is only useful to an alias, and aliases "
                "are refused here"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Duplicate keys, merge keys, and a comment eating a plain value.

        Checked over the node's own key and value pairs before the base class
        flattens anything, because flattening is what makes a merge key
        invisible and `dict` construction is what makes a duplicate key
        invisible. By the time either has run there is nothing left to see.
        """
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.nodes.ScalarNode):
                # A non-scalar key is unhashable and the base class says so
                # with a better message than anything this could add.
                continue
            key = key_node.value
            where = f"line {key_node.start_mark.line + 1}"
            if key == "<<":
                raise SourceError(
                    f"{where}: a merge key silently gives one source another's fields, and "
                    "a source that inherits a digest is a source pinned to a corpus nobody "
                    "recorded for it"
                )
            if key in seen:
                raise SourceError(
                    f"{where}: duplicate key {key!r}. YAML keeps the last one and says "
                    "nothing, so two `sha256:` lines would pin the corpus to whichever was "
                    "written second"
                )
            seen.add(key)
            if isinstance(value_node, yaml.nodes.ScalarNode) and value_node.style is None:
                self._refuse_a_comment_that_truncates(key, value_node)
        return super().construct_mapping(node, deep=deep)

    @staticmethod
    def _refuse_a_comment_that_truncates(key: str, node: yaml.nodes.ScalarNode) -> None:
        """A ` #` after a plain value is a comment, and the value loses the rest.

        `url: https://example.invalid/corpus.csv #real` loads as the URL without
        ` #real`, which is spec-correct YAML and a silently different source.
        Quoted values are untouched, because a `#` inside quotes is content and
        the manifest already quotes every URL.

        Read out of the mark's own buffer, which PyYAML fills in only when the
        document was handed to it as text. If it is absent the check cannot run,
        and this raises rather than passing: a guard that cannot see is not a
        guard that agrees.
        """
        mark = node.end_mark
        if mark.buffer is None:
            raise SourceError(
                "this manifest was not read as text, so a comment truncating a plain value "
                "could not be checked for; read the file with `read_text` and pass the string"
            )
        rest_of_line = mark.buffer[mark.pointer :].split("\n", 1)[0]
        if "#" in rest_of_line:
            raise SourceError(
                f"line {node.start_mark.line + 1}: `{key}` is a plain value followed by "
                f"`{rest_of_line.strip()}`, which YAML reads as a comment and drops. Quote "
                "the value if the text after it belongs to it"
            )


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


def base_id(name: str) -> str:
    """A source name with its pins and configs removed, folded for comparison.

    What the screens compare, and the reason they compare anything but the
    string itself: `hackaprompt/hackaprompt-dataset@abc1234` and
    `hackaprompt/hackaprompt-dataset` are one corpus, and a screen that reads
    them as two returns "not contaminated" for a corpus the reference model was
    trained on. Case is folded because the Hugging Face hub resolves an id
    without regard to it, so `VMware/open-instruct` and `vmware/open-instruct`
    fetch the same rows.

    Nothing else is folded. `_` and `-` are different characters in a hub id
    and stay different here, because a normaliser that collides two real
    corpora is the same defect pointing the other way.

    Raises rather than passing the name through unchanged when it does not
    match `SOURCE_NAME`. `load_sources` has already refused such a name, so the
    only way to reach this is a `Source` built by hand; a screen that quietly
    compared an unparseable name against a set of parsed ones would report no
    match, which is the answer that means "safe".
    """
    match = SOURCE_NAME.match(name)
    if match is None:
        raise SourceError(
            f"{name!r} is not a dataset identifier this manifest can read, so it cannot be "
            "screened; expected `<org>/<name>` or `name`, optionally followed by `@revision` "
            "or `:config`"
        )
    return match.group("base").casefold()


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

    name = fields["name"]
    if SOURCE_NAME.match(name) is None:
        raise SourceError(
            f"{where}: name is {name!r}, which is not `<org>/<name>` or `name` optionally "
            "followed by `@revision` or `:config`. The screens compare the base id, and a "
            "name they cannot parse is a source they cannot screen"
        )

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
        name=name,
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
        document = yaml.load(path.read_text(encoding="utf-8"), ManifestLoader)
    except yaml.YAMLError as error:
        raise SourceError(f"{path} is not readable as YAML: {error}") from error
    except SourceError as error:
        # Raised by ManifestLoader, which knows the line but not the file.
        raise SourceError(f"{path}:{error}") from error
    if document is None:
        raise SourceError(f"{path} lists no sources")
    if not isinstance(document, list):
        raise SourceError(
            f"{path} holds a {type(document).__name__}, and a manifest is a list of sources"
        )
    return document
