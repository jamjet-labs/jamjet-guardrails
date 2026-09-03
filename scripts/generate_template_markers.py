"""Generate `src/jamjet_guardrails/detectors/_template_markers.py`.

    ./.venv/bin/python scripts/generate_template_markers.py             # offline
    ./.venv/bin/python scripts/generate_template_markers.py --fetch     # download first
    ./.venv/bin/python scripts/generate_template_markers.py --verify-remote

The table is the set of delimiter strings that chat templates use to mark a
turn boundary or a role. They are read out of `tokenizer_config.json`,
`special_tokens_map.json` and the chat template of the model repositories
listed in `SOURCES`, at PINNED revisions, and the raw files are committed under
`template-data/` so the generation can be repeated and diffed with no network.

Pinned by revision, never by `main`, and that is the whole reason this script
exists rather than a hand-written list. A repository's `main` moves: Meta
added a `chat_template` key to the Llama 2 tokenizer config months after the
model shipped, and Llama 3.1 turned reserved token slots into named tool
markers. A table generated from `main` would change under a release nobody in
this repository made, and the byte-identity guard in
`tests/test_template_markers.py` would be reporting the Hub's edits as this
package's.

A dev tool. It is not in the wheel, which packages `src/jamjet_guardrails`
only, and nothing under `src/` imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "template-data"
OUT = ROOT / "src" / "jamjet_guardrails" / "detectors" / "_template_markers.py"


@dataclass(frozen=True)
class Source:
    """One repository the table is read out of, pinned to one commit."""

    key: str
    repository: str
    revision: str
    files: tuple[str, ...]
    licence: str
    licence_url: str
    host: str = "huggingface"
    #: The canonical repository, when `repository` is a mirror of it. Recorded
    #: even though it cannot be fetched, because a marker whose provenance
    #: stops at a mirror is a marker nobody can trace to the model that defines
    #: it.
    upstream: str = ""
    upstream_revision: str = ""
    #: Whether `upstream` refuses an anonymous fetch. Measured, not assumed:
    #: `--verify-remote` reports the status code every pinned file returns, and
    #: the three marked here answer 401 to a request with no token.
    upstream_gated: bool = False
    note: str = ""

    def url(self, path: str) -> str:
        if self.host == "github":
            return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{path}"
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{path}"

    def local(self, path: str) -> Path:
        return DATA / self.key / path


# The recorded list. Every revision below was resolved once, by hand, from the
# Hub API and written here; the script never asks for `main`, so re-running it
# cannot silently move a pin.
#
# Three of the eight models are gated: an anonymous GET of any file in
# `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Meta-Llama-3-8B-Instruct` or
# `google/gemma-2-9b-it` answers 401. Their markers come from a NAMED non-gated
# mirror, recorded here beside the gated repository it mirrors, because "we got
# it from somewhere" is not provenance.
SOURCES: tuple[Source, ...] = (
    Source(
        key="llama-2-chat",
        repository="unsloth/llama-2-7b-chat",
        revision="a6d63d7c9ac31fd7e6d31e66ee0d1c784a489fcf",
        files=("tokenizer_config.json", "special_tokens_map.json"),
        licence="LLAMA 2 Community License",
        licence_url="https://ai.meta.com/llama/license/",
        upstream="meta-llama/Llama-2-7b-chat-hf",
        upstream_revision="f5db02db724555f92da89c216ac04704f23d4590",
        upstream_gated=True,
        note=(
            "The mirror declares apache-2.0 on the Hub. The stricter upstream "
            "licence is recorded instead, because a mirror cannot relicense "
            "Meta's material. NousResearch/Llama-2-7b-chat-hf is the other "
            "non-gated mirror and was rejected: its tokenizer config predates "
            "the chat_template key, so it carries no [INST] or <<SYS>>."
        ),
    ),
    Source(
        key="llama-3-instruct",
        repository="NousResearch/Meta-Llama-3-8B-Instruct",
        revision="53346005fb0ef11d3b6a83b12c895cca40156b6c",
        files=("tokenizer_config.json", "special_tokens_map.json"),
        licence="Meta Llama 3 Community License",
        licence_url="https://llama.meta.com/llama3/license/",
        upstream="meta-llama/Meta-Llama-3-8B-Instruct",
        upstream_revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        upstream_gated=True,
    ),
    Source(
        key="qwen-2.5-instruct",
        repository="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        files=("tokenizer_config.json",),
        licence="Apache-2.0",
        licence_url="https://www.apache.org/licenses/LICENSE-2.0",
        note="No special_tokens_map.json exists at this revision; 404.",
    ),
    Source(
        key="mistral-instruct",
        repository="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        files=("tokenizer_config.json", "special_tokens_map.json"),
        licence="Apache-2.0",
        licence_url="https://www.apache.org/licenses/LICENSE-2.0",
    ),
    Source(
        key="gemma-2-instruct",
        repository="unsloth/gemma-2-9b-it",
        revision="fc7d4737cda11c3a19af2b722319e846670b4d89",
        files=("tokenizer_config.json", "special_tokens_map.json"),
        licence="Gemma Terms of Use",
        licence_url="https://ai.google.dev/gemma/terms",
        upstream="google/gemma-2-9b-it",
        upstream_revision="11c9b309abf73637e4b6f9a3fa1e92e615547819",
        upstream_gated=True,
    ),
    Source(
        key="phi-3-instruct",
        repository="microsoft/Phi-3-mini-4k-instruct",
        revision="f39ac1d28e925b323eae81227eaba4464caced4e",
        files=("tokenizer_config.json", "special_tokens_map.json"),
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
    ),
    Source(
        key="deepseek-v3",
        repository="deepseek-ai/DeepSeek-V3",
        revision="e815299b0bcbac849fa540c768ef21845365c9eb",
        files=("tokenizer_config.json",),
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
        note=(
            "LICENSE-CODE in the repository is MIT and covers the repository's "
            "code and configuration; LICENSE-MODEL covers the weights, which "
            "are not read here. No special_tokens_map.json exists; 404."
        ),
    ),
    Source(
        key="gpt-2",
        repository="openai-community/gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        files=("tokenizer_config.json", "onnx/special_tokens_map.json"),
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
        note=(
            "The root tokenizer_config.json at this revision is 26 bytes and "
            "names no token. The end-of-text token is read from the ONNX "
            "export's special_tokens_map.json, committed in the same "
            "repository at the same revision; there is no root "
            "special_tokens_map.json to read it from."
        ),
    ),
)

# The HTML element names, which decide one exclusion and nothing else.
#
# A file, pinned by commit, rather than a literal set in this script. The
# exclusion has to derive from a property with a source, per decision 9 of the
# phase 3 design: "exemptions derive from properties" and never from a hand
# list. `<s>` and `</s>` are what it removes today, and writing those two
# strings down here is exactly the hand list the rule exists to replace, since
# nothing would then notice the day a model adopts `<p>` or `<code>`.
#
# w3c/webref publishes a machine-readable extraction of the element index of
# the HTML Standard itself, regenerated from the spec, so this pins the spec's
# own list at a commit rather than somebody's transcription of it.
HTML_ELEMENTS = Source(
    key="html-elements",
    repository="w3c/webref",
    revision="f3b81966c45f34f62df20e7f8d6f66d5b5ba9279",
    files=("ed/elements/html.json",),
    licence="MIT",
    licence_url="https://opensource.org/license/mit",
    host="github",
    note="A curated extraction of the element index of the WHATWG HTML Standard.",
)


# ==========================================================================
# Fetching
# ==========================================================================


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "jamjet-guardrails-generator"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body: bytes = response.read()
    return body


def download(sources: Sequence[Source]) -> None:
    """Write every pinned file under `template-data/`, overwriting."""
    for source in sources:
        for path in source.files:
            target = source.local(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            body = _fetch(source.url(path))
            target.write_bytes(body)
            print(f"{target.relative_to(ROOT)}  {len(body)} bytes  {sha256(body)}")


def verify_remote(sources: Sequence[Source]) -> int:
    """Re-fetch every pinned file and compare it with what is committed.

    The day a pin is bumped, this is what says which files moved. It is also
    the only place the gating claim is measured rather than restated: a gated
    upstream is requested here too, and its status code is printed.
    """
    bad = 0
    for source in sources:
        for path in source.files:
            local = source.local(path)
            expected = sha256(local.read_bytes()) if local.is_file() else "(not committed)"
            try:
                got = sha256(_fetch(source.url(path)))
            except urllib.error.HTTPError as exc:
                got = f"HTTP {exc.code}"
            status = "ok" if got == expected else "DIFFERS"
            if status == "DIFFERS":
                bad += 1
            print(
                f"{status:8} {source.key}/{path}\n         local={expected}\n         remote={got}"
            )
        if source.upstream:
            probe = (
                f"https://huggingface.co/{source.upstream}/resolve/"
                f"{source.upstream_revision}/{source.files[0]}"
            )
            try:
                _fetch(probe)
                code = "200 (no longer gated)"
            except urllib.error.HTTPError as exc:
                code = f"HTTP {exc.code}"
            print(f"         upstream {source.upstream} -> {code}")
    return bad


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


# ==========================================================================
# Reading candidate strings out of the fetched files
# ==========================================================================

#: The keys whose value is one special token, in either file. `special` in
#: `added_tokens_decoder` and `additional_special_tokens` are handled beside
#: them; every other key in a tokenizer config is a setting, not a string the
#: model emits.
_TOKEN_KEYS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "sep_token",
    "cls_token",
    "mask_token",
)

#: A Jinja construct. Everything outside one is literal template text.
_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)

#: A quoted literal inside a Jinja construct. The markers of every template in
#: `SOURCES` live inside these rather than in the literal text: Llama 2 writes
#: `{{ bos_token + '[INST] ' + content }}`, so scanning only the text between
#: constructs would find nothing at all.
_QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", re.DOTALL)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}

#: A bracketed run with no whitespace in it. Applied to strings that are
#: already known to be either a declared special token or a literal from a chat
#: template, never to the template's Jinja code: `messages[0]` and `[-1]` are
#: expressions, and a scan over the raw template would collect them.
#:
#: The doubled form comes first so `<<SYS>>` is one candidate rather than the
#: inner `<SYS>` with a stray angle bracket on each side.
_CANDIDATE = re.compile(r"<<[^\s<>]+>>|<[^\s<>]+>|\[[^\s\[\]]+\]")

#: A reserved slot: a name whose whole tail is a number. Tokenizers allocate
#: blocks of these to be renamed later, which is what happened between Llama 3
#: and 3.1, where `<|reserved_special_token_2|>` became `<|python_tag|>`. They
#: are placeholders in the vocabulary, not delimiters in a template, and
#: keeping them would add 1,000-odd `[control_N]` and `<unusedN>` strings that
#: no template ever emits.
#:
#: A property, not a list: a block added by a model this table has never seen
#: is dropped by the same rule, and a slot that is later given a real name
#: stops matching it and arrives on the next regeneration.
_RESERVED_SLOT = re.compile(r"[A-Za-z][A-Za-z_]*_?\d+")

#: A start or end tag with a bare element name and no attributes. Only a
#: candidate of this exact shape is tested against the HTML element set, so
#: `<|s|>` cannot be mistaken for `<s>` after the decoration is stripped.
_BARE_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)>")

_MIN_LENGTH = 3
_MAX_LENGTH = 64


def _contents(value: object) -> Iterator[str]:
    """The string a special-token entry carries, in either of its two shapes."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            yield content


def _special_tokens(doc: object) -> Iterator[str]:
    if not isinstance(doc, dict):
        return
    for key in _TOKEN_KEYS:
        yield from _contents(doc.get(key))
    extra = doc.get("additional_special_tokens")
    if isinstance(extra, list):
        for entry in extra:
            yield from _contents(entry)
    decoder = doc.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for entry in decoder.values():
            # `special: true` is the tokenizer's own statement that the string
            # is a control token rather than a word it learned. Reading every
            # added token instead would pull in merges and user vocabulary.
            if isinstance(entry, dict) and entry.get("special") is True:
                yield from _contents(entry)


def _unescape(literal: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), literal)


def _templates(doc: object) -> Iterator[str]:
    """Every chat template in a tokenizer config.

    A repository may carry several under `chat_template`, as a list of
    `{"name": ..., "template": ...}`, which is how tool-calling variants are
    shipped. Both shapes are read; a variant is where a tool marker such as
    `[AVAILABLE_TOOLS]` appears.
    """
    if not isinstance(doc, dict):
        return
    template = doc.get("chat_template")
    if isinstance(template, str):
        yield template
    elif isinstance(template, list):
        for entry in template:
            if isinstance(entry, dict) and isinstance(entry.get("template"), str):
                yield str(entry["template"])


def _template_literals(template: str) -> Iterator[str]:
    """The strings a template can put into its own output.

    The literal text between Jinja constructs and the quoted literals inside
    them, and nothing else. Handing the raw template to the candidate scan
    would collect `[0]`, `[1:]` and `['role']` from its expressions.
    """
    cursor = 0
    for tag in _JINJA.finditer(template):
        yield template[cursor : tag.start()]
        for single, double in _QUOTED.findall(tag.group()):
            yield _unescape(single or double)
        cursor = tag.end()
    yield template[cursor:]


def _candidates(text: str) -> Iterator[str]:
    for match in _CANDIDATE.finditer(text):
        candidate = match.group()
        if _MIN_LENGTH <= len(candidate) <= _MAX_LENGTH and any(c.isalpha() for c in candidate):
            yield candidate


def _inner(marker: str) -> str:
    """A marker's name, with its brackets, pipes and closing slash removed.

    `</s>` and `<s>` both give `s`, `<|reserved_special_token_0|>` gives
    `reserved_special_token_0`, and `<<SYS>>` gives `SYS`. The fullwidth
    vertical bar is stripped beside the ASCII one because DeepSeek writes its
    delimiters with it, and a rule that saw `｜User｜` where it expected `User`
    would treat every DeepSeek marker as a name nobody could classify.
    """
    name = marker.strip("<>[]").strip("|｜").lstrip("/")
    return name


def _is_reserved_slot(marker: str) -> bool:
    return _RESERVED_SLOT.fullmatch(_inner(marker)) is not None


def _is_html_element(marker: str, elements: frozenset[str]) -> bool:
    tag = _BARE_TAG.fullmatch(marker)
    return tag is not None and tag.group(1).casefold() in elements


def read_html_elements(source: Source = HTML_ELEMENTS) -> tuple[frozenset[str], str]:
    """The element names, and the digest of the file they were read from."""
    local = source.local(source.files[0])
    if not local.is_file():
        raise SystemExit(f"missing {local}; run with --fetch")
    body = local.read_bytes()
    doc = json.loads(body.decode("utf-8"))
    names = {str(element["name"]).casefold() for element in doc["elements"]}
    if not names:
        raise SystemExit(f"{local} lists no elements; the exclusion would remove nothing")
    return frozenset(names), sha256(body)


@dataclass
class Table:
    """What one generation produced, before it is written out."""

    markers: dict[str, list[str]] = field(default_factory=dict)
    digests: dict[str, dict[str, str]] = field(default_factory=dict)
    html_excluded: dict[str, list[str]] = field(default_factory=dict)
    reserved_dropped: int = 0
    html_digest: str = ""


def collect(sources: Iterable[Source], elements: frozenset[str]) -> Table:
    """Read every pinned file and partition what it yields.

    Three populations come out, and each is recorded rather than only the one
    that ships: the markers, the ones an HTML element name removed, and how
    many reserved slots were dropped. An exclusion nobody counted is an
    exclusion nobody can argue with.
    """
    table = Table()
    reserved: set[str] = set()
    for source in sources:
        for path in source.files:
            local = source.local(path)
            if not local.is_file():
                raise SystemExit(f"missing {local}; run with --fetch")
            body = local.read_bytes()
            table.digests.setdefault(source.key, {})[path] = sha256(body)
            doc = json.loads(body.decode("utf-8"))
            texts = list(_special_tokens(doc))
            for template in _templates(doc):
                texts += _template_literals(template)
            for text in texts:
                for candidate in _candidates(text):
                    if _is_reserved_slot(candidate):
                        reserved.add(candidate)
                    elif _is_html_element(candidate, elements):
                        _claim(table.html_excluded, candidate, source.key)
                    else:
                        _claim(table.markers, candidate, source.key)
    table.reserved_dropped = len(reserved)
    return table


def _claim(into: dict[str, list[str]], marker: str, key: str) -> None:
    keys = into.setdefault(marker, [])
    if key not in keys:
        keys.append(key)


# ==========================================================================
# Emitting the module
# ==========================================================================

_HEADER = '''"""Chat-template markers, read from pinned model repositories. GENERATED.

Do not edit. Regenerate with:

    ./.venv/bin/python scripts/generate_template_markers.py

`tests/test_template_markers.py` regenerates this module from the raw files
committed under `template-data/` and requires the result to be byte-identical
to what is committed here, so an edit made by hand is a failing test rather
than a table nobody can reproduce.

**What a marker is.** A bracketed run with no whitespace in it, three to
sixty-four characters long and containing at least one letter, that a listed
repository declares as a special token or that its chat template writes into
its own output. `<|im_start|>`, `[INST]`, `<<SYS>>`, `<start_of_turn>` and
`<｜User｜>` are the shapes this covers. These strings exist to delimit
turns, which is why one of them arriving inside untrusted content is a claim to
a role that content does not have.

**What is not a marker.** Two populations are removed, each by a property
rather than by a list, and each counted here so the cost of removing it can be
argued with.

Reserved vocabulary slots, whose name ends in a number, are dropped:
`<|reserved_special_token_0|>`, `[control_8]`, `<unused12>`. A tokenizer
allocates these in blocks to be named later, and Llama 3.1 did exactly that
when it renamed `<|reserved_special_token_2|>` to `<|python_tag|>`. No chat
template emits one. `RESERVED_SLOTS_DROPPED` records how many went: dropped
here, {reserved_dropped}. A detector that later wants them back can reach
them with the same rule rather than with a longer table.

Markers that are also HTML element names are excluded and kept in
`EXCLUDED_AS_HTML`, where a reader can see what the rule costs. Such a marker
is a boundary token rather than a role claim, and denying it would deny a
strikethrough tag in any ordinary HTML document. The element names come from
the element index of the HTML Standard itself, pinned at a commit and recorded
in `HTML_ELEMENT_SOURCE`, so the rule is a property of that index rather than
of the strings it happens to remove today. `corpora/NOTICE.md` records the same
exclusion. Excluded here, {html_count}.

**The weakest two entries are named rather than quietly kept.**
`<function-name>` and `<args-json-object>` are written by the Qwen 2.5
tool-calling template into the system prompt it builds, as placeholders inside
a JSON example. They were read out of a real template and are kept for that
reason, but they are the two entries most likely to appear in ordinary
developer prose, and they are the first place to look if the check's precision
row disappoints. `corpora/NOTICE.md` says the same.

**Markers are stored as they were read.** No normalisation, no case folding,
no confusable skeleton. The detector that consumes this table matches over the
folded view described in the phase 3 design, and folds the table entries the
same way at load; folding them here would bake one interpreter's Unicode
version into a generated file and lose the string a source actually declares.

Nothing imports this yet. The `template-integrity` check lands separately, and
until it does this table is private, unregistered and carries no corpus.

Markers: {marker_count}. Model repositories read: {source_count}.
"""

from __future__ import annotations

from typing import NamedTuple


class Source(NamedTuple):
    """One repository the table was read out of, pinned to one commit."""

    key: str
    """The name `MARKER_SOURCES` uses for this repository."""

    repository: str
    """What was actually fetched. A mirror, where the upstream is gated."""

    revision: str
    """The commit the files below were read at. Never a branch name."""

    licence: str
    licence_url: str

    upstream: str
    """The canonical repository this mirrors, or the empty string."""

    upstream_revision: str
    upstream_gated: bool
    """Whether an anonymous fetch of `upstream` is refused."""

    files: dict[str, str]
    """Path in the repository to the SHA-256 of the bytes that were read."""

    note: str


'''


def _q(value: str) -> str:
    """A string literal in the form `ruff format` leaves alone: double quotes.

    `repr` prefers SINGLE quotes and only switches when the string contains
    one, which is the opposite of the formatter's rule. Emitting `repr` output
    directly produced a file that `ruff format --check` rewrote on sight, and a
    rewritten generated file is a byte-identity test that fails on a clean
    checkout.
    """
    literal = repr(value)
    if literal.startswith("'") and '"' not in value:
        return '"' + literal[1:-1].replace("\\'", "'") + '"'
    return literal


def _lines_for(name: str, entries: dict[str, list[str]], doc: str) -> list[str]:
    """A mapping of marker to source keys, one entry per line where it fits.

    An entry too long for the line length is exploded with a trailing comma,
    which is the form `ruff format` produces for it and, because of the magic
    trailing comma, the form it then leaves alone. A one-element tuple is
    written flat, because its comma is syntax rather than a magic trailing
    comma and the formatter collapses the exploded form back.
    """
    out = [f"{name}: dict[str, tuple[str, ...]] = {{"]
    for marker, keys in sorted(entries.items()):
        inner = ", ".join(_q(key) for key in keys)
        flat = (
            f"    {_q(marker)}: ({inner},)," if len(keys) == 1 else f"    {_q(marker)}: ({inner}),"
        )
        if len(flat) <= 100:
            out.append(flat)
        else:
            out.append(f"    {_q(marker)}: (")
            out += [f"        {_q(key)}," for key in keys]
            out.append("    ),")
    out.append("}")
    out.append(f'"""{doc}"""')
    return out


def _source_lines(source: Source, digests: dict[str, str], indent: str) -> list[str]:
    lines = [f"{indent}Source("]
    for field_name, value in (
        ("key", source.key),
        ("repository", source.repository),
        ("revision", source.revision),
        ("licence", source.licence),
        ("licence_url", source.licence_url),
        ("upstream", source.upstream),
        ("upstream_revision", source.upstream_revision),
    ):
        lines.append(f"{indent}    {field_name}={_q(value)},")
    lines.append(f"{indent}    upstream_gated={source.upstream_gated!r},")
    lines.append(f"{indent}    files={{")
    for path, digest in digests.items():
        lines.append(f"{indent}        {_q(path)}: {_q(digest)},")
    lines.append(f"{indent}    }},")
    lines.append(f"{indent}    note={_q(source.note)},")
    lines.append(f"{indent}),")
    return lines


def render(table: Table, sources: Sequence[Source], html: Source) -> str:
    body = _HEADER.format(
        reserved_dropped=table.reserved_dropped,
        html_count=len(table.html_excluded),
        marker_count=len(table.markers),
        source_count=len(sources),
    )
    lines: list[str] = ["SOURCES: tuple[Source, ...] = ("]
    for source in sources:
        lines += _source_lines(source, table.digests[source.key], "    ")
    lines.append(")")
    lines.append('"""Every model repository the markers were read out of."""')
    lines.append("")

    element_lines = _source_lines(html, {html.files[0]: table.html_digest}, "")
    lines.append("HTML_ELEMENT_SOURCE: Source = " + element_lines[0])
    lines += element_lines[1:-1]
    lines.append(")")
    lines.append('"""The element index that decides which markers are HTML tags."""')
    lines.append("")

    lines.append("MARKERS: tuple[str, ...] = (")
    lines += [f"    {_q(marker)}," for marker in sorted(table.markers)]
    lines.append(")")
    lines.append('"""Every marker, sorted. The table the check matches against."""')
    lines.append("")
    lines += _lines_for(
        "MARKER_SOURCES",
        table.markers,
        "Every marker, to the keys of the SOURCES that declare it.",
    )
    lines.append("")
    lines += _lines_for(
        "EXCLUDED_AS_HTML",
        table.html_excluded,
        "Candidates an HTML element name removed, and what declared them.",
    )
    lines.append("")
    lines.append(f"RESERVED_SLOTS_DROPPED: int = {table.reserved_dropped}")
    lines.append('"""Reserved vocabulary slots the name-ends-in-a-number rule dropped."""')
    lines.append("")
    return body + "\n".join(lines)


def _shown(path: Path) -> str:
    """A path relative to the repository where it is inside one.

    `relative_to` RAISES for a path outside the tree, and the byte-identity
    test writes into a temporary directory, so a bare call made the generator
    unusable from the one place that proves it still works.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download every pinned file first")
    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help="re-fetch every pinned file and compare digests; writes nothing",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    every = (*SOURCES, HTML_ELEMENTS)
    if args.verify_remote:
        return 1 if verify_remote(every) else 0
    if args.fetch:
        download(every)

    elements, table_html_digest = read_html_elements()
    table = collect(SOURCES, elements)
    table.html_digest = table_html_digest
    text = render(table, SOURCES, HTML_ELEMENTS)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"wrote {_shown(args.out)}: {len(table.markers)} markers, "
        f"{len(table.html_excluded)} excluded as HTML element names, "
        f"{table.reserved_dropped} reserved slots dropped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
