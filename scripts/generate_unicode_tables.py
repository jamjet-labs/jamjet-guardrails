"""Generate `src/jamjet_guardrails/_unicode/` from the pinned Unicode data files.

    ./.venv/bin/python scripts/generate_unicode_tables.py             # generate, offline
    ./.venv/bin/python scripts/generate_unicode_tables.py --download  # refetch, network

Two modes, and they are separate on purpose. Generation reads only the files
committed under `unicode-data/16.0.0/` and touches no network, so
`tests/test_unicode.py::test_the_generated_modules_are_byte_identical_to_a_regeneration`
can run the same code path on every CI leg. Downloading is the explicit,
occasional act of moving the pin, and it writes the raw files and nothing else:
a mode that downloaded and regenerated in one step would let a fetch that
returned a redirect page, a truncated body or next year's revision rewrite the
tables with nobody looking at the diff of the inputs first.

A dev tool. It is not in the wheel, which packages `src/jamjet_guardrails`
only, and nothing under `src/` imports it.

**The output is already `ruff format` clean, and that is a constraint on how it
is emitted, not a happy accident.** `ruff format --check .` runs over `src/` on
every CI leg, so a generated module the formatter would rewrite makes the build
red with a fix that then breaks byte identity, and the two guards deadlock. The
emission rules that keep it clean, each one measured against ruff 0.16.5 rather
than assumed:

- every collection carries a magic trailing comma, so the formatter expands it
  one element per line and never reflows it back;
- no element that can exceed the 100-column line length contains a nested
  collection. A long Script_Extensions set written inline as
  `(0x00B7, 0x00B7, ("Avestan", ...))` is exploded by the formatter across four
  lines; the same set interned into `EXTENSION_SETS` as one string is left
  alone, because a string literal is the one thing the formatter cannot split.
  That is why the extension sets are interned and referenced by index rather
  than written where they are used;
- every string literal is pure ASCII, one `\\uXXXX` or `\\UXXXXXXXX` escape per
  code point. `corpora/NOTICE.md` gives the reason for the injection corpus and
  it is stronger here: `confusables.txt` maps Arabic ligatures to strings that
  begin with a right-to-left character, and a reviewer reading a raw diff of
  those would see the line reordered on screen by the terminal's bidi
  algorithm. What is escaped is what the loader decodes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from collections.abc import Hashable, Iterator
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parent.parent

_Value = TypeVar("_Value", bound=Hashable)

UNICODE_VERSION = "16.0.0"

DATA_DIR = ROOT / "unicode-data" / UNICODE_VERSION
TARGET_DIR = ROOT / "src" / "jamjet_guardrails" / "_unicode"

_UCD = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd"
_SECURITY = f"https://www.unicode.org/Public/security/{UNICODE_VERSION}"

# Every file this generator reads, and where it came from. The URLs carry the
# version literally rather than `latest`, because `latest` is the one URL whose
# content changes under a pin.
SOURCES: dict[str, str] = {
    "IdentifierStatus.txt": f"{_SECURITY}/IdentifierStatus.txt",
    "PropertyValueAliases.txt": f"{_UCD}/PropertyValueAliases.txt",
    "ScriptExtensions.txt": f"{_UCD}/ScriptExtensions.txt",
    "Scripts.txt": f"{_UCD}/Scripts.txt",
    "confusables.txt": f"{_SECURITY}/confusables.txt",
}

# Which module is generated from which inputs. Each module records the digests
# of ITS OWN inputs and no others, so a `confusables.txt` bump does not make the
# script tables look stale and send someone regenerating a file nothing
# happened to.
SCRIPTS_INPUTS = ("PropertyValueAliases.txt", "ScriptExtensions.txt", "Scripts.txt")
CONFUSABLES_INPUTS = ("confusables.txt",)
IDENTIFIERS_INPUTS = ("IdentifierStatus.txt",)

# Read as `utf-8-sig`, because four of the five files begin with a UTF-8 BOM.
# Read as plain `utf-8` the BOM becomes U+FEFF on the front of the first line,
# the first data line of Scripts.txt parses as the code point `﻿0000`, and
# `int(..., 16)` raises from inside a parser whose error would name the line
# rather than the encoding. `IdentifierStatus.txt` carries no BOM and decodes
# identically either way, so one encoding covers all five.
_ENCODING = "utf-8-sig"


def digest(path: Path) -> str:
    """SHA-256 of the file's BYTES.

    Bytes, not the decoded text: the BOM and the line endings are part of what
    was published, and a digest taken over text normalises both away, so the
    network-gated test would pass against a file that had been rewritten by an
    editor.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fields(text: str) -> Iterator[list[str]]:
    """Data lines of a UCD-format file, comment stripped and split on `;`.

    The comment is cut at the FIRST `#` and before the split, which matters for
    `confusables.txt`: its comments carry a second `#` and the character names
    behind it contain no `;` only by convention. Cutting first makes that
    convention irrelevant.
    """
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        yield [field.strip() for field in line.split(";")]


def _range(field: str) -> tuple[int, int]:
    """`XXXX` or `XXXX..YYYY` as an inclusive pair."""
    start, _, end = field.partition("..")
    return int(start, 16), int(end or start, 16)


def script_aliases(text: str) -> dict[str, str]:
    """Short script code to long property value, from `PropertyValueAliases.txt`.

    `ScriptExtensions.txt` writes `Latn`; every name this package hands a
    caller is `Latin`, because a check's `allowed_scripts` option and a corpus
    label are both read by people. The two spellings must not both be in
    circulation: a constraint written as `{"Latin"}` that silently failed to
    match a code point whose extensions resolved to `Latn` would deny nothing
    and report nothing.

    Later fields on an `sc` line are further aliases (`Qaac` for Coptic,
    `Qaai` for Inherited) and are deliberately not indexed. They are the
    obsolete spellings, and admitting them would put a second name for one
    script into the table a `script-constraint` option is validated against.
    """
    aliases: dict[str, str] = {}
    for fields in _fields(text):
        if fields[0] != "sc":
            continue
        short, long = fields[1], fields[2]
        if short in aliases:
            raise ValueError(f"PropertyValueAliases.txt declares {short!r} twice")
        aliases[short] = long
    if not aliases:
        raise ValueError("PropertyValueAliases.txt yielded no `sc` rows")
    return aliases


def _merged(ranges: list[tuple[int, int, _Value]]) -> list[tuple[int, int, _Value]]:
    """Sort by code point and join ranges that touch and carry the same value.

    Scripts.txt splits one script's run by general category, so `0021..0023`,
    `0024` and `0025..0027` are three lines there and one range here. The
    general category is not what this table answers and `unicodedata` answers
    it on every interpreter, so keeping the split would triple the table to
    encode a property nothing reads from it.

    `sorted` first rather than trusting the file's order, and the overlap check
    below is what makes the merge safe: two ranges that OVERLAP rather than
    touch would silently lose whichever value came second.
    """
    ordered = sorted(ranges, key=lambda entry: entry[0])
    out: list[tuple[int, int, _Value]] = []
    for start, end, value in ordered:
        if out and start <= out[-1][1]:
            raise ValueError(f"ranges overlap at {start:04X}: {out[-1]} and {(start, end, value)}")
        if out and start == out[-1][1] + 1 and value == out[-1][2]:
            out[-1] = (out[-1][0], end, value)
        else:
            out.append((start, end, value))
    return out


def script_ranges(text: str, aliases: dict[str, str]) -> list[tuple[int, int, str]]:
    """`(start, end, script)` from `Scripts.txt`, merged, long names throughout.

    Scripts.txt already writes long names, so the aliases are used here as a
    CHECK rather than a translation: a value that is not a long property value
    means the file's format moved under the pin, and failing here is how that
    is discovered rather than by a caller getting a script name no
    `allowed_scripts` option would ever list.
    """
    known = set(aliases.values())
    ranges: list[tuple[int, int, str]] = []
    for fields in _fields(text):
        start, end = _range(fields[0])
        script = fields[1]
        if script not in known:
            raise ValueError(f"Scripts.txt names {script!r}, which is no `sc` long value")
        ranges.append((start, end, script))
    if not ranges:
        raise ValueError("Scripts.txt yielded no ranges")
    return _merged(ranges)


def extension_ranges(text: str, aliases: dict[str, str]) -> list[tuple[int, int, tuple[str, ...]]]:
    """`(start, end, scripts)` from `ScriptExtensions.txt`, merged, long names.

    Sorted within each set. The file says the ordering of the values in a set
    is not material and gives them alphabetically by SHORT code, which is a
    different order from alphabetically by long name; sorting here rather than
    keeping the file's order is what makes two runs of this generator produce
    the same bytes when the file's presentation order changes under a pin.
    """
    ranges: list[tuple[int, int, tuple[str, ...]]] = []
    for fields in _fields(text):
        start, end = _range(fields[0])
        scripts = []
        for short in fields[1].split():
            if short not in aliases:
                raise ValueError(f"ScriptExtensions.txt names {short!r}, which is no `sc` code")
            scripts.append(aliases[short])
        ranges.append((start, end, tuple(sorted(scripts))))
    if not ranges:
        raise ValueError("ScriptExtensions.txt yielded no ranges")
    return _merged(ranges)


def identifier_ranges(text: str) -> list[tuple[int, int, bool]]:
    """`(start, end, True)` for every Allowed run in `IdentifierStatus.txt`.

    The file lists the Allowed code points and nothing else; its own `@missing`
    line makes every code point outside them Restricted, so what is emitted is
    the Allowed set and the absence of a range IS the answer for everything
    else. Refusing any other status value is what discovers a revision that
    starts writing `Restricted` rows explicitly, which would otherwise load as
    Allowed ranges and turn the whole table inside out.

    The third element is a constant `True`, carried only so `_merged` can join
    adjacent runs the way it does for the script tables: two Allowed ranges that
    touch are one range, and merging them here rather than emitting both keeps
    the table the size of the data rather than the size of its presentation.
    """
    ranges: list[tuple[int, int, bool]] = []
    for fields in _fields(text):
        status = fields[1]
        if status != "Allowed":
            raise ValueError(f"IdentifierStatus.txt names status {status!r}, not 'Allowed'")
        start, end = _range(fields[0])
        ranges.append((start, end, True))
    if not ranges:
        raise ValueError("IdentifierStatus.txt yielded no Allowed ranges")
    return _merged(ranges)


def prototypes(text: str) -> dict[str, str]:
    """Confusable source character to prototype string, from `confusables.txt`.

    Refuses a multi-character SOURCE rather than skipping one. UTS #39 section 4
    maps each source CHARACTER, and `_fold.fold` is a per-character contract, so
    a table that grew a two-character key would be silently ignored by the fold
    and the skeleton would stop closing a hole nobody was told about. All 6,355
    rows at 16.0.0 have a single-character source; this is what discovers the
    day one does not.

    Refuses a type other than `MA` for the same reason. Older revisions of this
    file carried SA, SL and ML tables, and this generator has never been run
    against one: it would map single-script confusables into the same table and
    change what every published number means without changing a line of code.
    """
    mapping: dict[str, str] = {}
    for fields in _fields(text):
        source, target, kind = fields[0], fields[1], fields[2]
        if kind != "MA":
            raise ValueError(f"confusables.txt row {source!r} is type {kind!r}, not MA")
        codes = source.split()
        if len(codes) != 1:
            raise ValueError(f"confusables.txt maps a {len(codes)}-character source {source!r}")
        character = chr(int(codes[0], 16))
        prototype = "".join(chr(int(code, 16)) for code in target.split())
        if not prototype:
            raise ValueError(f"confusables.txt maps {source!r} to nothing")
        if character in mapping and mapping[character] != prototype:
            raise ValueError(f"confusables.txt maps {source!r} two different ways")
        mapping[character] = prototype
    if not mapping:
        raise ValueError("confusables.txt yielded no mappings")
    return mapping


def _literal(text: str) -> str:
    """One ASCII Python string literal, one escape per code point.

    EVERY code point, including the printable ASCII ones. A rule that escaped
    only the non-ASCII would have to decide what to do about the quote, the
    backslash and the code points confusables maps to a bare space, and each of
    those decisions is a way to emit a literal that does not parse. Escaping
    everything makes the emitter total.
    """
    out = []
    for char in text:
        code = ord(char)
        out.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
    return '"' + "".join(out) + '"'


def _digests_block(names: tuple[str, ...], data: Path) -> str:
    lines = ["SOURCE_DIGESTS: dict[str, str] = {"]
    for name in names:
        lines.append(f'    "{name}": "{digest(data / name)}",')
    lines.append("}")
    return "\n".join(lines)


def render_scripts(data: Path) -> str:
    """The text of `_unicode/scripts.py`."""
    aliases = script_aliases((data / "PropertyValueAliases.txt").read_text(encoding=_ENCODING))
    ranges = script_ranges((data / "Scripts.txt").read_text(encoding=_ENCODING), aliases)
    extensions = extension_ranges(
        (data / "ScriptExtensions.txt").read_text(encoding=_ENCODING), aliases
    )

    sets = sorted({scripts for _, _, scripts in extensions})
    index = {scripts: position for position, scripts in enumerate(sets)}

    lines = [
        '"""Script and Script_Extensions ranges. GENERATED; do not edit.',
        "",
        "Written by `scripts/generate_unicode_tables.py` from the three files committed",
        f"under `unicode-data/{UNICODE_VERSION}/`. Regenerate rather than edit: a test",
        "rewrites this module from those files and compares bytes, so a hand edit fails",
        "the build with no way for the next reader to tell which of the two was meant.",
        "",
        "**Why the data is vendored at all.** `unicodedata` exposes no Script property on",
        "any interpreter from 3.10 to 3.14, and the Unicode version behind it runs from",
        "13.0 to 16.0 across this project's CI matrix. A check that derived script from",
        "`unicodedata.name()` prefixes would answer differently on different legs of one",
        "test suite, and a corpus label written on one leg would be wrong on another. The",
        "tables here are 16.0.0 everywhere, so a code point assigned after 13.0 has a",
        "script on 3.10 that that interpreter's own `unicodedata` cannot name.",
        "",
        "**Ranges are merged where `Scripts.txt` splits one script's run by general",
        "category.** `0021..0023`, `0024` and `0025..0027` are three lines there and one",
        "range here. The general category is not what this table answers, `unicodedata`",
        "answers it on every interpreter, and keeping the split would have tripled the",
        "table to encode a property nothing reads from it.",
        "",
        "**Extension sets are interned into `EXTENSION_SETS` and referenced by index.**",
        "Written inline, the 16-script set on U+00B7 runs past 100 columns and",
        "`ruff format` explodes the row across four lines, which the byte-identity test",
        "then reports as a difference the generator cannot fix. A string literal is the",
        "one thing the formatter will not split.",
        "",
        (
            f"Contents: {len(ranges):,} script ranges naming "
            f"{len({s for _, _, s in ranges}):,} scripts, and {len(extensions):,} "
            "Script_Extensions"
        ),
        f"ranges over {len(sets):,} distinct sets. Both counts are held to this table by",
        "tests/test_unicode.py, because a number in prose that counts a thing in code is",
        "a claim.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'UNICODE_VERSION = "{UNICODE_VERSION}"',
        "",
        "# SHA-256 of each file this module was generated from, as published. Checked",
        "# against the committed copies offline, and against unicode.org under",
        "# JAMJET_GUARDRAILS_NETWORK=1, by tests/test_unicode.py.",
        _digests_block(SCRIPTS_INPUTS, data),
        "",
        "# Sorted by start, disjoint, and looked up by bisection. Every code point outside",
        "# every range has Script=Unknown, which is `Scripts.txt`'s own @missing rule and",
        "# is what an unassigned code point resolves to.",
        "SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (",
    ]
    lines += [f'    (0x{start:04X}, 0x{end:04X}, "{script}"),' for start, end, script in ranges]
    lines += [
        ")",
        "",
        "# The distinct Script_Extensions sets, space separated, sorted. Referenced by",
        "# index from EXTENSION_RANGES below.",
        "EXTENSION_SETS: tuple[str, ...] = (",
    ]
    lines += [f'    "{" ".join(scripts)}",' for scripts in sets]
    lines += [
        ")",
        "",
        "# Sorted by start, disjoint, looked up by bisection. A code point inside one of",
        "# these takes the set as its resolved scripts; a code point outside every one of",
        "# them takes its Script value, per UTS #39 section 5.1.",
        "EXTENSION_RANGES: tuple[tuple[int, int, int], ...] = (",
    ]
    lines += [
        f"    (0x{start:04X}, 0x{end:04X}, {index[scripts]})," for start, end, scripts in extensions
    ]
    lines += [")", ""]
    return "\n".join(lines)


def render_confusables(data: Path) -> str:
    """The text of `_unicode/confusables.py`."""
    mapping = prototypes((data / "confusables.txt").read_text(encoding=_ENCODING))
    identities = sum(1 for source, target in mapping.items() if source == target)

    lines = [
        '"""Confusable prototypes for UTS #39 skeletons. GENERATED; do not edit.',
        "",
        "Written by `scripts/generate_unicode_tables.py` from the file committed under",
        f"`unicode-data/{UNICODE_VERSION}/`. Regenerate rather than edit: a test rewrites this",
        "module from that file and compares bytes, so a hand edit fails the build with no",
        "way for the next reader to tell which of the two was meant.",
        "",
        "**One table, `MA`.** That is the whole of `confusables.txt` at 16.0.0: the",
        "single-script and mixed-script-lowercase tables of older revisions are gone, and",
        "the generator refuses a row of any other type rather than folding it in here,",
        "because doing so would change what every published number means without changing",
        "a line of code.",
        "",
        "**Every literal is escaped.** `corpora/NOTICE.md` argues this for the injection",
        "corpus and it is stronger here: the Arabic ligature rows map to strings that open",
        "with a right-to-left character, and a raw diff of those is reordered on screen by",
        "the reader's bidi algorithm. What is escaped is what the loader decodes.",
        "",
        f"**Every row changes something.** At this revision the file carries {identities} rows",
        "mapping a character to itself, so nothing here is a no-op the fold pays for and",
        "nothing was dropped to make that true. The table is a transcription: a reader",
        "comparing it against the published file finds every row where they left it.",
        "",
        (
            f"Contents: {len(mapping):,} prototypes, the longest of them "
            f"{max(len(value) for value in mapping.values())} code points"
        ),
        "(U+FDFA, the Arabic ligature for a whole phrase). The count is held to this",
        "table by tests/test_unicode.py, because a number in prose that counts a thing in",
        "code is a claim.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'UNICODE_VERSION = "{UNICODE_VERSION}"',
        "",
        "# SHA-256 of the file this module was generated from, as published.",
        _digests_block(CONFUSABLES_INPUTS, data),
        "",
        "# Keyed by the character rather than by its integer code point, so the fold that",
        "# reads it is `PROTOTYPES.get(character, character)` with no `ord` per character",
        "# on a megabyte of content.",
        "PROTOTYPES: dict[str, str] = {",
    ]
    lines += [
        f"    {_literal(source)}: {_literal(target)}," for source, target in sorted(mapping.items())
    ]
    lines += ["}", ""]
    return "\n".join(lines)


def render_identifiers(data: Path) -> str:
    """The text of `_unicode/identifiers.py`."""
    ranges = identifier_ranges((data / "IdentifierStatus.txt").read_text(encoding=_ENCODING))
    covered = sum(end - start + 1 for start, end, _ in ranges)

    lines = [
        '"""The UTS #39 Identifier Profile. GENERATED; do not edit.',
        "",
        "Written by `scripts/generate_unicode_tables.py` from the file committed under",
        f"`unicode-data/{UNICODE_VERSION}/`. Regenerate rather than edit: a test rewrites this",
        "module from that file and compares bytes, so a hand edit fails the build with no",
        "way for the next reader to tell which of the two was meant.",
        "",
        "**What the property is for.** Identifier_Status=Allowed is Unicode's own answer",
        "to which characters are recommended for identifiers, and UTS #39 section 5.2",
        "defines every restriction level above Unrestricted over exactly this set. The",
        "`confusables` check reads it to decide whether a confusable prototype names a",
        "string anybody could be reading: Cyrillic small a maps to Latin `a`, which is a",
        "character brands and hostnames are written in, and Cyrillic small em maps to",
        "U+028D LATIN SMALL LETTER TURNED W, which is a phonetic letter no brand or",
        "hostname is written in. Both prototypes are Latin; only the first is a spoof.",
        "",
        "**Absence is the answer.** `IdentifierStatus.txt` lists the Allowed code points",
        "and nothing else, and its own @missing line makes every other code point",
        "Restricted, so a code point inside no range here is Restricted rather than",
        "unknown. The generator refuses a row carrying any other status for that reason.",
        "",
        f"Contents: {len(ranges):,} Allowed ranges covering {covered:,} code points.",
        "Both counts are held to this table by tests/test_unicode.py, because a number in",
        "prose that counts a thing in code is a claim.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'UNICODE_VERSION = "{UNICODE_VERSION}"',
        "",
        "# SHA-256 of the file this module was generated from, as published.",
        _digests_block(IDENTIFIERS_INPUTS, data),
        "",
        "# Sorted by start, disjoint, and looked up by bisection. A code point inside one",
        "# of these has Identifier_Status=Allowed; every other code point is Restricted.",
        "ALLOWED_RANGES: tuple[tuple[int, int], ...] = (",
    ]
    lines += [f"    (0x{start:04X}, 0x{end:04X})," for start, end, _ in ranges]
    lines += [")", ""]
    return "\n".join(lines)


def download(data: Path) -> int:
    """Refetch every pinned file. The ONLY thing here that touches the network."""
    data.mkdir(parents=True, exist_ok=True)
    for name, url in sorted(SOURCES.items()):
        with urllib.request.urlopen(url) as response:
            body = response.read()
        (data / name).write_bytes(body)
        print(f"{name}: {len(body):,} bytes, sha256 {digest(data / name)}")
    return 0


def generate(data: Path, target: Path) -> int:
    """Rewrite all three modules from the committed files. No network."""
    target.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("scripts.py", render_scripts(data)),
        ("confusables.py", render_confusables(data)),
        ("identifiers.py", render_identifiers(data)),
    ):
        # newline="\n" and never the platform default. `scripts/sample_nemotron.py`
        # pins its write for the same reason: a CRLF checkout would regenerate a
        # file that differs from the committed one in every line, and the
        # byte-identity test would name none of the real difference.
        (target / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"{target.name}/{name}: {len(text.encode('utf-8')):,} bytes")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="refetch the pinned raw files from unicode.org and exit, generating nothing",
    )
    args = parser.parse_args(argv[1:])
    if args.download:
        return download(DATA_DIR)
    return generate(DATA_DIR, TARGET_DIR)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
