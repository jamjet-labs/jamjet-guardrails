"""Where a finding sits in a file, and what an invisible character looks like.

Two column counts, because two consumers disagree. Text and JSON reports count
code points, which is what a person moving a cursor through the line counts.
SARIF counts UTF-16 code units and says so with ``columnKind``, because that is
the kind GitHub's own SARIF example declares, and a tag character is astral, so
the two counts diverge on exactly the lines a scan reports.

Everything a scan prints is untrusted: the path is whatever a pull request
named a file, and the excerpt is the content being reported. ``safe_line``
is the one function that turns either into something a log can carry.
"""

from __future__ import annotations

import unicodedata

_TAG_LO, _TAG_HI = 0xE0000, 0xE007F
_KEEP = frozenset("\t\n\r")
# Cs is here for file names: a name that is not valid UTF-8 reaches Python as
# lone surrogates, which cannot be written to a UTF-8 stream at all.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc", "Cs", "Zl", "Zp"})


def _is_selector(cp: int) -> bool:
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


def _selector_byte(cp: int) -> int:
    return cp - 0xFE00 if cp <= 0xFE0F else cp - 0xE0100 + 16


def _check_offset(text: str, offset: int) -> None:
    if offset < 0 or offset > len(text):
        raise ValueError(f"offset {offset} outside text of length {len(text)}")


def line_col(text: str, offset: int) -> tuple[int, int]:
    """1-based line and column of ``offset``, the column in code points.

    ``\\n`` ends a line, so CRLF is one line end: the CR is the last character
    of the line it ends and never a column of the next one.
    """
    _check_offset(text, offset)
    line = text.count("\n", 0, offset) + 1
    start = text.rfind("\n", 0, offset) + 1
    return line, offset - start + 1


def utf16_col(text: str, offset: int) -> int:
    """1-based column of ``offset`` in UTF-16 code units, for SARIF."""
    _check_offset(text, offset)
    start = text.rfind("\n", 0, offset) + 1
    return 1 + sum(2 if ord(c) > 0xFFFF else 1 for c in text[start:offset])


def _invisible(ch: str) -> bool:
    if ch in _KEEP:
        return False
    cp = ord(ch)
    if _TAG_LO <= cp <= _TAG_HI or _is_selector(cp):
        return True
    return unicodedata.category(ch) in _INVISIBLE_CATEGORIES


def _is_tag(ch: str) -> bool:
    return _TAG_LO <= ord(ch) <= _TAG_HI


def reveal_tokens(text: str) -> list[str]:
    """``text`` as a list of pieces, each a visible character or one marker.

    A run of tag characters is one ``<TAG xN>`` marker: each would otherwise
    print as nine characters and a short cap would show a dozen of them, and
    the decoded text is reported beside the excerpt anyway. A run of one
    repeated invisible character is one ``<U+XXXX xN>`` marker.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not _invisible(ch):
            out.append(ch)
            i += 1
            continue
        j = i + 1
        if _is_tag(ch):
            while j < n and _is_tag(text[j]):
                j += 1
            out.append(f"<TAG x{j - i}>")
        else:
            while j < n and text[j] == ch:
                j += 1
            code = f"U+{ord(ch):04X}"
            out.append(f"<{code}>" if j - i == 1 else f"<{code} x{j - i}>")
        i = j
    return out


def reveal(text: str) -> str:
    """``text`` with every invisible or format character made visible.

    Tab, line feed and carriage return are kept; ``safe_line`` escapes those.
    """
    return "".join(reveal_tokens(text))


def excerpt(text: str, limit: int = 120) -> str:
    """``reveal(text)`` cut at ``limit`` characters without splitting a marker."""
    out: list[str] = []
    used = 0
    for token in reveal_tokens(text):
        if used + len(token) > limit:
            out.append("...")
            break
        out.append(token)
        used += len(token)
    return "".join(out)


def decode_hidden(text: str) -> str | None:
    """The text carried by tag characters or by a variation-selector run.

    Tag characters U+E0020..U+E007E mirror ASCII one for one. Variation
    selectors carry one byte each, U+FE00..U+FE0F as 0..15 and
    U+E0100..U+E01EF as 16..255, the encoding published for emoji smuggling.
    A single selector is a variation sequence, not a payload, so selectors are
    decoded only when there are at least two. ``None`` when nothing decodes.
    """
    tags = [chr(ord(c) - _TAG_LO) for c in text if 0xE0020 <= ord(c) <= 0xE007E]
    if tags:
        return "".join(tags)
    selectors = [_selector_byte(ord(c)) for c in text if _is_selector(ord(c))]
    if len(selectors) >= 2:
        return bytes(selectors).decode("utf-8", errors="replace")
    return None


def safe_line(text: str) -> str:
    """``text`` made safe to print inside one log line.

    Every invisible, bidirectional or format character is revealed, CR and LF
    are revealed too so the text cannot start a line of its own, and every
    ``::`` becomes ``:<U+003A>`` so no part of it can be read by GitHub Actions
    as a workflow command.
    """
    revealed = reveal(text).replace("\r", "<U+000D>").replace("\n", "<U+000A>")
    while "::" in revealed:
        revealed = revealed.replace("::", ":<U+003A>")
    return revealed
