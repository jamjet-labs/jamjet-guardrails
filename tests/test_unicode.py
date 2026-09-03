"""The vendored Unicode tables, their generator, and the two functions over them.

Three separate things are guarded here and they fail for different reasons.

**The tables are what Unicode published.** A vendored table is a copy, and a
copy nobody checks is a place where a value can be edited into the package with
no source to contradict it. So the raw files carry recorded digests, the
generated modules are reproduced from those files byte for byte, and a
network-gated test compares the committed files against unicode.org for the day
the pin moves.

**The functions answer what the tables say.** A bisection over merged ranges is
exactly the shape that is right at every value someone thinks to type and wrong
at a boundary, so the sweep below tests every range's edges and the code points
either side of them, against an oracle that re-reads the raw files and searches
them linearly.

**The offsets come back.** `skeleton` is three folds deep, one of which
reorders, and a span that comes back one character short leaves the attacker's
character standing inside content this library reported as rewritten.

Every test below was watched to FAIL against a mutation of the thing it
guards, with `__pycache__` cleared between runs. The mutations, in order:

- one raw file untracked, so the guard on the tree is not vacuous;
- one range value flipped in the committed `scripts.py`;
- `render_scripts` made to fetch its input instead of reading it;
- one recorded digest corrupted;
- the two tables pinned to different Unicode versions;
- a stated range count and a stated prototype count each moved by one;
- a stated module size moved from 45 KiB to 40;
- `jamjet_guardrails/__init__.py` made to import the confusables table;
- the confusables table imported at the top of `_unicode` instead of inside
  `skeleton`;
- `Latin` written back as `Latn` in the generated table;
- the Script lookup put in front of the Script_Extensions lookup;
- an empty set returned for `Common` and `Inherited`;
- an unassigned code point failing open to `Common`;
- the one-code-point refusal removed;
- `code <= end` narrowed to `code < end`;
- the confusables map skipped inside `skeleton`;
- `fold` giving every product of one source character its own index;
- `compose` keeping the inner view's offsets;
- canonical ordering dropped from `_nfd`;
- `span` reading the first and last entries of the map instead of its minimum
  and maximum;
- the second NFD pass dropped from `skeleton`;
- the empty content refused instead of folded;
- a raw file tampered with, against the network-gated test.

One of those mutations survived at first and is why
`test_the_second_normalisation_pass_is_not_redundant` is written the way it
is: the original probe used U+011B, whose own decomposition means the map
never reaches its row, so the test passed with the third step deleted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import ModuleType

import pytest

from jamjet_guardrails._unicode import UNKNOWN, script_set, skeleton
from jamjet_guardrails._unicode import confusables as confusables_table
from jamjet_guardrails._unicode import scripts as scripts_table

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "unicode-data" / "16.0.0"
GENERATED = ROOT / "src" / "jamjet_guardrails" / "_unicode"
GENERATOR = ROOT / "scripts" / "generate_unicode_tables.py"

NETWORK = pytest.mark.skipif(
    os.environ.get("JAMJET_GUARDRAILS_NETWORK") != "1",
    reason="set JAMJET_GUARDRAILS_NETWORK=1 to re-download the pinned Unicode files",
)


def _generator() -> ModuleType:
    """Import the generator by path. `scripts/` is deliberately not a package.

    The same mechanism `tests/test_benchmarks.py` uses on `benchmarks/`, and for
    the same reason: a directory of dev tools that is importable by name is a
    top-level module name this project would then own.
    """
    spec = importlib.util.spec_from_file_location("generate_unicode_tables", GENERATOR)
    assert spec is not None and spec.loader is not None, f"cannot load {GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# The data is there, and the guards below are therefore not vacuous.
# ==========================================================================


def test_the_pinned_files_and_the_generator_are_committed() -> None:
    """Every test in this module is vacuous if the tree has moved under it."""
    assert GENERATOR.is_file(), f"{GENERATOR} is missing"
    for name in ("PropertyValueAliases.txt", "ScriptExtensions.txt", "Scripts.txt"):
        assert (DATA / name).is_file(), f"{DATA / name} is missing"
    assert (DATA / "confusables.txt").is_file()
    for name in ("__init__.py", "scripts.py", "confusables.py"):
        assert (GENERATED / name).is_file(), f"{GENERATED / name} is missing"

    tracked = subprocess.run(
        ["git", "ls-files", "unicode-data"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert len(tracked) == 4, (
        f"unicode-data/ tracks {tracked}; the sdist ships what git tracks, so an "
        "untracked raw file is evidence nobody downstream receives"
    )


# ==========================================================================
# Guard 1: the generated modules are reproducible from the committed files.
# ==========================================================================


def test_the_generated_modules_are_byte_identical_to_a_regeneration() -> None:
    """The same mechanism `BENCHMARKS.md` is held by, and no network.

    A generated module is source a reviewer reads and a reader trusts, and
    nothing about it announces that it was generated except a docstring anyone
    can edit. Without this, a value tuned by hand to make one test pass sits in
    the table forever and no diff against `unicode-data/` would ever be run.

    Byte identity rather than an equivalence: the modules must be REGENERABLE,
    which means the generator is the only thing that ever writes them, which
    means the next pin bump is a mechanical act rather than an archaeology.
    """
    generator = _generator()
    for name, rendered in (
        ("scripts.py", generator.render_scripts(DATA)),
        ("confusables.py", generator.render_confusables(DATA)),
    ):
        committed = (GENERATED / name).read_text(encoding="utf-8")
        assert rendered == committed, (
            f"_unicode/{name} differs from what the generator produces from "
            f"unicode-data/16.0.0/. Rerun scripts/generate_unicode_tables.py"
        )


def test_the_generator_touches_no_network_when_it_generates() -> None:
    """The byte-identity test above runs on every CI leg, so it must be offline.

    Asserted by taking the network away rather than by reading the source for
    an import: `urllib` is imported at the top of the generator for the
    download mode, so a grep for it proves nothing about which mode uses it.
    Every socket constructor raises inside the child, so any attempt to open
    one fails the run rather than reaching out.
    """
    code = (
        "import socket, sys\n"
        "class Refused(socket.socket):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise AssertionError('the generator opened a socket')\n"
        "socket.socket = Refused\n"
        "socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(\n"
        "    AssertionError('the generator opened a connection'))\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('gen', {str(GENERATOR)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.render_scripts(module.DATA_DIR)\n"
        "module.render_confusables(module.DATA_DIR)\n"
        "print('offline')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "offline"


# ==========================================================================
# Guard 2: the digests each module records match the committed raw files.
# ==========================================================================


def test_each_generated_module_records_the_digest_of_every_file_it_read() -> None:
    """Byte identity alone does not say WHICH files the tables came from.

    The regeneration test reads whatever sits in `unicode-data/16.0.0/` today,
    so a raw file replaced with a different revision and the modules
    regenerated from it passes that test cleanly: both sides moved together.
    The recorded digests are the half that does not move on its own, so a
    replaced input has to be a deliberate edit of a hash a reviewer can see in
    the diff.

    Taken over the BYTES. All four files carry a UTF-8 BOM and are read as
    `utf-8-sig` by the generator; a digest over the decoded text would drop
    the BOM and match a file that had been rewritten by an editor.
    """
    generator = _generator()
    recorded = {
        "scripts.py": (scripts_table.SOURCE_DIGESTS, generator.SCRIPTS_INPUTS),
        "confusables.py": (confusables_table.SOURCE_DIGESTS, generator.CONFUSABLES_INPUTS),
    }
    for module_name, (digests, inputs) in recorded.items():
        assert set(digests) == set(inputs), (
            f"_unicode/{module_name} records digests for {sorted(digests)}, "
            f"but is generated from {sorted(inputs)}"
        )
        for name, expected in sorted(digests.items()):
            actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            assert actual == expected, (
                f"_unicode/{module_name} records {expected} for {name}, "
                f"which now hashes to {actual}"
            )


def test_the_two_modules_agree_about_the_unicode_version_they_are_pinned_to() -> None:
    """A table from 16.0.0 beside one from 15.1.0 is a spoof nothing would catch.

    The scripts of a confusable's prototype are the whole of the mixed-script
    rule in the `confusables` check, so the two tables are read together on
    every token. Two revisions would answer coherently on almost every input
    and incoherently on exactly the code points a revision changed.
    """
    assert scripts_table.UNICODE_VERSION == confusables_table.UNICODE_VERSION
    assert scripts_table.UNICODE_VERSION == DATA.name


@NETWORK
def test_the_pinned_files_still_match_what_unicode_org_publishes() -> None:
    """For the day the pin moves, and never in CI.

    Not a guard on the current state so much as the tool that says whether a
    revision is still there and whether the committed bytes are still the
    published ones. `JAMJET_GUARDRAILS_NETWORK=1` is never set on any CI leg:
    a test that reaches the internet turns an outage into a red build and
    trains everyone to ignore it.
    """
    import urllib.request

    generator = _generator()
    for name, url in sorted(generator.SOURCES.items()):
        with urllib.request.urlopen(url) as response:
            published = hashlib.sha256(response.read()).hexdigest()
        committed = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert published == committed, (
            f"{name} at {url} now hashes to {published}; the committed copy is {committed}"
        )


# ==========================================================================
# The numbers in the generated docstrings count things in the tables.
# ==========================================================================


def test_the_counts_each_generated_docstring_states_match_its_tables() -> None:
    """A number in prose that counts a thing in code is a claim.

    These particular numbers survive a regeneration only because the generator
    computes them, so what this guards is a table edited by hand: the docstring
    and the tuple below it would then disagree, and this is what notices.
    """
    scripts_doc = scripts_table.__doc__ or ""
    found = re.search(
        r"([\d,]+) script ranges naming ([\d,]+) scripts, and ([\d,]+) Script_Extensions\s+"
        r"ranges over ([\d,]+) distinct sets",
        scripts_doc,
    )
    assert found is not None, f"scripts.py no longer states its counts: {scripts_doc[:200]!r}"
    ranges, names, extensions, sets = (int(value.replace(",", "")) for value in found.groups())
    assert ranges == len(scripts_table.SCRIPT_RANGES)
    assert names == len({script for _, _, script in scripts_table.SCRIPT_RANGES})
    assert extensions == len(scripts_table.EXTENSION_RANGES)
    assert sets == len(scripts_table.EXTENSION_SETS)

    confusables_doc = confusables_table.__doc__ or ""
    found = re.search(
        r"([\d,]+) prototypes, the longest of them ([\d,]+) code points", confusables_doc
    )
    assert found is not None, (
        f"confusables.py no longer states its count: {confusables_doc[:200]!r}"
    )
    count, longest = (int(value.replace(",", "")) for value in found.groups())
    assert count == len(confusables_table.PROTOTYPES)
    assert longest == max(len(value) for value in confusables_table.PROTOTYPES.values())


def test_the_module_sizes_the_package_docstring_records_are_the_real_ones() -> None:
    """The import-cost paragraph is a measurement, and half of it is checkable.

    The timings cannot be gated: byte-identical artifacts and wall-clock
    measurements do not mix, which is the phase 3 performance posture. The
    SIZES can, because the byte-identity test above already fixes both files
    exactly, so a size that drifts from the prose is a table that grew while
    the sentence about what it costs stayed still.

    Rounded to KiB, which is how the docstring states them, so the assertion
    reads the same way the sentence does.
    """
    import jamjet_guardrails._unicode as package

    stated = dict(re.findall(r"`(\w+\.py)` is (\d+) KiB", package.__doc__ or ""))
    assert set(stated) == {"scripts.py", "confusables.py"}, (
        f"the cost paragraph names {sorted(stated)}, not both generated modules"
    )
    for name, kib in sorted(stated.items()):
        actual = round((GENERATED / name).stat().st_size / 1024)
        assert int(kib) == actual, f"{name} is {actual} KiB, and the docstring says {kib}"


# ==========================================================================
# The tables cost nothing until something asks for them.
# ==========================================================================


def test_importing_the_package_loads_neither_unicode_table() -> None:
    """`import jamjet_guardrails` must stay cheap, and 222 KiB is not cheap.

    A fresh interpreter, because this test module has already imported both
    tables at the top and `sys.modules` in THIS process proves nothing. The
    child imports the package the way a consumer does and reports what that
    pulled in.

    The rule the child enforces is what the checks have to keep doing: import
    `jamjet_guardrails._unicode` inside the method that needs it, never at the
    top of a detector module, because `detectors/__init__.py` is imported from
    the package root and would drag both tables in behind it.
    """
    code = (
        "import sys\n"
        "import jamjet_guardrails\n"
        "loaded = sorted(name for name in sys.modules if '_unicode' in name)\n"
        "print(loaded)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]", (
        f"importing jamjet_guardrails loaded {done.stdout.strip()}; the Unicode tables "
        "must be imported inside the checks that read them"
    )


def test_asking_for_a_script_does_not_load_the_confusables_table() -> None:
    """The half of the laziness that is inside this package rather than above it.

    `script-constraint` never asks for a skeleton, and `confusables.py` is the
    expensive one: 176 KiB against 45. Importing it at the top of
    `_unicode/__init__.py` would be invisible in every test, because every test
    process ends up importing both anyway.
    """
    code = (
        "import sys\n"
        "from jamjet_guardrails._unicode import script_set\n"
        "script_set('a')\n"
        "print('jamjet_guardrails._unicode.confusables' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "False", (
        "resolving a script loaded the confusables table; skeleton() imports it, "
        "and nothing above skeleton() should"
    )


# ==========================================================================
# script_set
# ==========================================================================


def test_a_script_comes_back_under_its_long_property_name() -> None:
    """`Latin`, never `Latn`. An option and a corpus label are read by people.

    `ScriptExtensions.txt` is written in four-letter codes and `Scripts.txt` in
    long names, so a table built from both without resolving the aliases would
    hand a caller two spellings of one script. A `{"Latin"}` constraint would
    then pass Latin letters and deny the apostrophe, whose extensions are
    written `Latn` in the file.
    """
    assert script_set("A") == frozenset({"Latin"})
    assert script_set("\u0430") == frozenset({"Cyrillic"})  # CYRILLIC SMALL A
    assert script_set("あ") == frozenset({"Hiragana"})
    assert "Latn" not in {name for _, _, name in scripts_table.SCRIPT_RANGES}
    assert "Latin" in script_set("\u02bc"), "U+02BC resolves through the code Latn"


def test_script_extensions_win_over_the_script_property() -> None:
    """UTS #39 section 5.1's order, and reversing it is a different answer.

    U+00B7 MIDDLE DOT has Script=Common, which every caller treats as a
    wildcard, and Script_Extensions naming 16 scripts, which is a constraint.
    A Script-first lookup would report a middle dot in Georgian text as passing
    every constraint there is.

    U+30FC, the Katakana-Hiragana prolonged sound mark, is the case the
    `script-constraint` fixture exists to prove: Script=Common, extensions
    {Hiragana, Katakana}, and ordinary Japanese text is full of it.
    """
    middle_dot = script_set("·")
    assert "Common" not in middle_dot
    assert {"Georgian", "Latin", "Greek"} <= middle_dot
    assert script_set("ー") == frozenset({"Hiragana", "Katakana"})


def test_common_and_inherited_come_back_as_themselves() -> None:
    """Not as an empty set and not as every script. The caller decides.

    A comma is Common and a combining mark with no extensions is Inherited.
    Returning an empty set for either would make them indistinguishable from a
    code point with no scripts at all, and `script-constraint` would have to
    guess which of the two it was looking at.
    """
    assert script_set(",") == frozenset({"Common"})
    assert script_set(" ") == frozenset({"Common"})
    assert script_set("5") == frozenset({"Common"})
    # U+030F COMBINING DOUBLE GRAVE ACCENT: Script=Inherited and no
    # Script_Extensions entry, so the Script value is what comes back. Its
    # neighbour U+0305 is deliberately NOT the example: that one HAS extensions
    # naming eight scripts, which is the section 5.1 order under test next
    # door rather than the Inherited case under test here.
    assert script_set("\u030f") == frozenset({"Inherited"})


def test_an_unassigned_code_point_resolves_to_unknown() -> None:
    """Fail closed. `Scripts.txt`'s own @missing rule, and a disclosed residual.

    A code point unassigned at 16.0.0 fires under `script-constraint` rather
    than passing, because a constraint that let through everything the pinned
    tables had never heard of would be widest exactly where the data is
    oldest. The fix for a real code point caught this way is a pin bump.
    """
    assert script_set("\U000e0000") == frozenset({UNKNOWN})
    assert script_set("\U0010fffe") == frozenset({UNKNOWN})


def test_script_set_refuses_a_string_that_is_not_one_code_point() -> None:
    """Silently reading the first character is the failure this replaces.

    A caller that passed two characters meant something by it, and the answer
    for the first one is not it.
    """
    for value in ("", "ab", "a\u0301"):  # the last is TWO code points, not one
        with pytest.raises(ValueError, match="one code point"):
            script_set(value)


def _oracle() -> tuple[list[tuple[int, int, str]], list[tuple[int, int, frozenset[str]]]]:
    """The raw files parsed again, unmerged, for a linear search.

    An independent oracle rather than a second call into the same code: the
    thing under test bisects a MERGED table, and every defect that shape has
    (an off-by-one on a range end, a merge that joined two ranges that only
    looked adjacent, a bisection that lands one slot low) is invisible to a
    test that looks the answer up the same way.
    """
    generator = _generator()
    aliases = generator.script_aliases(
        (DATA / "PropertyValueAliases.txt").read_text(encoding="utf-8-sig")
    )
    scripts: list[tuple[int, int, str]] = []
    for fields in generator._fields((DATA / "Scripts.txt").read_text(encoding="utf-8-sig")):
        start, end = generator._range(fields[0])
        scripts.append((start, end, fields[1]))
    extensions: list[tuple[int, int, frozenset[str]]] = []
    for fields in generator._fields(
        (DATA / "ScriptExtensions.txt").read_text(encoding="utf-8-sig")
    ):
        start, end = generator._range(fields[0])
        extensions.append((start, end, frozenset(aliases[code] for code in fields[1].split())))
    return scripts, extensions


def test_every_range_edge_resolves_the_way_the_raw_files_say() -> None:
    """The sweep. Boundaries first, because that is where this shape breaks.

    Five code points per range in each file: one below the start, the start,
    the middle, the end, and one above the end. That is where a merge that ran
    one code point too far, a bisection that returned the preceding range, or
    an `end` compared with `<` instead of `<=` shows up, and nowhere else.

    The oracle searches the unmerged ranges linearly and applies UTS #39
    section 5.1 itself, so the two sides share the raw files and nothing else.
    """
    scripts, extensions = _oracle()

    def resolve(code: int) -> frozenset[str]:
        for start, end, value in extensions:
            if start <= code <= end:
                return value
        for start, end, name in scripts:
            if start <= code <= end:
                return frozenset({name})
        return frozenset({UNKNOWN})

    probes: set[int] = set()
    for table in (scripts, extensions):
        for start, end, _ in table:
            probes.update({start - 1, start, (start + end) // 2, end, end + 1})
    probes = {code for code in probes if 0 <= code <= 0x10FFFF}
    assert len(probes) > 4000, f"only {len(probes)} probes; this sweep would prove little"

    wrong = [
        (f"U+{code:04X}", sorted(script_set(chr(code))), sorted(resolve(code)))
        for code in sorted(probes)
        if script_set(chr(code)) != resolve(code)
    ]
    assert wrong == [], f"the tables disagree with the raw files at {wrong[:5]}"


# ==========================================================================
# skeleton
# ==========================================================================


def test_a_substituted_cyrillic_letter_folds_onto_its_latin_prototype() -> None:
    """The whole reason the confusables table is here.

    A banned substring check over the raw content misses `p<U+0430>ypal` with one
    Cyrillic a. Over the skeleton it does not, and no list of letters to watch
    for is involved.
    """
    view = skeleton("p\u0430ypal")  # CYRILLIC SMALL A in place of the second letter
    assert view.text == "paypal"
    assert view.span(0, 6) == (0, 6)


def test_a_multi_character_prototype_maps_back_to_its_one_source_character() -> None:
    """U+FDFA is one character whose prototype is 18, and the span must be (0, 1).

    This is the property that keeps spans closed over the source run. If any of
    those 18 view characters carried an index of its own, a match covering part
    of the expansion would come back as a span covering part of a character,
    and `_spans._rewrite` would cut a code point in half.
    """
    view = skeleton("\ufdfa")  # ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM
    assert len(view.text) == 18
    assert set(view.origin) == {0}
    assert view.span(0, 18) == (0, 1)
    assert view.span(4, 6) == (0, 1)


def test_a_prototype_expansion_leaves_the_characters_around_it_where_they_were() -> None:
    """The expansion shifts every later view index, and the map absorbs that.

    Without the map, a match after the expansion would be reported at an offset
    that is right in the skeleton and wrong in the content, which is a
    redaction of the wrong bytes.
    """
    source = "a\u2100b"  # U+2100 ACCOUNT OF, whose prototype is "a/c"
    view = skeleton(source)
    assert view.text == "aa/cb"
    start = view.text.index("b")
    assert view.span(start, start + 1) == (2, 3)
    assert source[slice(*view.span(start, start + 1))] == "b"


def test_the_decomposition_agrees_with_the_interpreters_own_nfd() -> None:
    """A normalisation this package computes itself has one way to be right.

    `_nfd` is a per-character fold plus a canonical ordering pass, written that
    way because `unicodedata.normalize` over a whole string gives no offsets. A
    reimplementation that disagreed with the standard one would produce
    skeletons no other UTS #39 implementation produces, while looking correct
    in every test written against it.

    Swept over the BMP and the supplementary planes that carry decomposable
    characters, plus every multi-mark sequence below.
    """
    from jamjet_guardrails._unicode import _nfd

    wrong = []
    for code in range(0x11000):
        character = chr(code)
        if _nfd(character).text != unicodedata.normalize("NFD", character):
            wrong.append(f"U+{code:04X}")
    assert wrong == [], f"_nfd disagrees with unicodedata at {wrong[:5]}"

    for source in (
        "a\u0301\u0316b",  # acute (230) then below-comma (220): reordered
        "a\u0316\u0301b",  # the same marks already in canonical order
        "\u1e0b\u0323",  # d-with-dot-above plus dot-below, the UAX #15 example
        "q\u0307\u0323",  # dot-above (230) before dot-below (220)
        "\u0f77",  # a Tibetan vowel sign with a decomposition
        "\uac00\u0301",  # a Hangul syllable, decomposed algorithmically
    ):
        assert _nfd(source).text == unicodedata.normalize("NFD", source), source


def test_canonical_ordering_moves_a_mark_and_the_span_still_covers_the_run() -> None:
    """The defect `_Folded.span` was widened to close.

    NFD reorders combining marks by class, so the offset map is the first one
    in this package that is not non-decreasing: for `a` + acute (class 230) +
    below-comma (class 220) the map reads (0, 2, 1). Reading the first and last
    entries of the matched range returns (0, 2), which stops one character
    short: a redaction over it would leave the acute accent standing in content
    reported as rewritten. Taking the minimum and the maximum returns (0, 3).
    """
    from jamjet_guardrails._unicode import _nfd

    source = "a\u0301\u0316"
    view = _nfd(source)
    assert view.origin == (0, 2, 1), "this test proves nothing unless the map is reordered"
    assert view.span(0, 3) == (0, 3)
    assert source[slice(*view.span(0, 3))] == source
    # The narrower match, over the two marks alone, still covers both of them.
    assert view.span(1, 3) == (1, 3)


def test_the_skeleton_matches_the_algorithm_uts_39_states() -> None:
    """NFD, map, NFD, checked against the three steps written out plainly.

    The implementation composes three folds so that it can carry the offsets;
    this is the same computation with no offsets at all, which is what every
    other implementation of UTS #39 section 4 does. They must agree, or this
    package's skeletons are its own.
    """
    prototypes = confusables_table.PROTOTYPES

    def reference(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        mapped = "".join(prototypes.get(character, character) for character in decomposed)
        return unicodedata.normalize("NFD", mapped)

    for source in (
        "",
        "paypal",
        "p\u0430ypal",
        "\u1e9a",  # its prototype is precomposed, so the second NFD splits it
        "\u2251",  # its prototype is out of canonical order, so the second NFD sorts it
        "\ufdfa",
        "\u2100",
        "Straße",
        "a\u0301\u0316",
        "\u0430\u0435\u043e\u0440\u0441",  # Cyrillic letters with Latin prototypes
        "\U0001d400\U0001d401\U0001d402",
        "パスワード",
        "hello\u200bworld",
    ):
        assert skeleton(source).text == reference(source), repr(source)


def test_the_second_normalisation_pass_is_not_redundant() -> None:
    """UTS #39 section 4 normalises TWICE, and 32 rows are why.

    The rows that need it are the ones whose SOURCE survives the first NFD and
    whose prototype does not, because a source that decomposes never reaches
    its own row: the map runs over the decomposed text, so `\u011b` is already
    `e\u030c` before any lookup happens and the row keyed on `\u011b` is dead.
    That is the mistake this test was written with, and it made the whole
    guard vacuous: it passed with the third step deleted.

    U+1E9A maps to a precomposed character, and U+2251 maps to three code
    points that are not in canonical ORDER, so between them the third step has
    to decompose and to reorder.

    The count is read out of the docstring that states it and derived from the
    table, so neither the sentence nor the table can move alone.
    """
    needing = [
        source
        for source, prototype in confusables_table.PROTOTYPES.items()
        if unicodedata.normalize("NFD", source) == source
        and unicodedata.normalize("NFD", prototype) != prototype
    ]
    stated = re.search(r"(\d+) rows of the [\d.]+ table have a", skeleton.__doc__ or "")
    assert stated is not None, "skeleton's docstring no longer states how many rows need it"
    assert len(needing) == int(stated.group(1)), (
        f"{len(needing)} rows need the second pass, and the docstring says {stated.group(1)}"
    )

    # Decomposition on the second pass. U+1E9A -> U+1EA3 -> "a" + U+0309.
    assert skeleton("\u1e9a").text == "a\u0309"
    assert skeleton("\u1e9a").text == skeleton("a\u0309").text
    # The span still comes back as the one source character, even though the
    # second pass produced a second view character from what the map produced.
    assert skeleton("\u1e9a").span(0, 2) == (0, 1)

    # Reordering on the second pass. U+2251 -> "=" + U+0307 (230) + U+0323
    # (220), which canonical ordering swaps.
    assert skeleton("\u2251").text == "=\u0323\u0307"


def test_the_empty_content_folds_to_an_empty_view_rather_than_raising() -> None:
    """A caller finds no match rather than being handed an exception."""
    view = skeleton("")
    assert view.text == ""
    assert view.origin == ()
    assert view.source_length == 0


def test_every_view_character_carries_a_source_index_inside_the_source() -> None:
    """The invariant every span depends on, over content that exercises all
    three folds at once: a precomposed character that decomposes, a confusable
    that expands, marks that reorder, and an astral character."""
    source = "\u00e9\u2100p\u0430ypal a\u0301\u0316 \U0001d400"
    view = skeleton(source)
    assert len(view.origin) == len(view.text)
    assert view.source_length == len(source)
    assert all(0 <= index < len(source) for index in view.origin)
    assert min(view.origin) == 0
    assert max(view.origin) == len(source) - 1
