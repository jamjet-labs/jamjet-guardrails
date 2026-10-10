"""Where a scan finding sits, and how an invisible character is printed.

Every invisible character in this file is written as an escape. A literal one
can be lost by any tool between the author and the interpreter, and a test of
"silent on a ZWJ sequence" that lost its ZWJ passes for the wrong reason.
"""

from __future__ import annotations

import pytest

from jamjet_guardrails.scan.positions import (
    decode_hidden,
    line_col,
    reveal,
    safe_line,
    utf16_col,
)


def _tags(s: str) -> str:
    return "".join(chr(0xE0000 + ord(c)) for c in s)


def test_line_col_first_char() -> None:
    assert line_col("abc", 0) == (1, 1)


def test_line_col_counts_code_points_across_lines() -> None:
    text = "ab\nc\U0001f600d\ne"
    assert line_col(text, text.index("d")) == (2, 3)
    assert line_col(text, text.index("e")) == (3, 1)


def test_line_col_crlf_is_one_line_end() -> None:
    text = "a\r\nb"
    assert line_col(text, text.index("b")) == (2, 1)


def test_line_col_accepts_the_end_of_the_text_and_nothing_past_it() -> None:
    assert line_col("ab", 2) == (1, 3)
    with pytest.raises(ValueError):
        line_col("ab", 3)
    with pytest.raises(ValueError):
        line_col("ab", -1)


def test_utf16_col_counts_an_astral_character_as_two_units() -> None:
    text = "x\n\U0001f600" + _tags("a") + "z"
    assert utf16_col(text, text.index("z")) == 5
    assert utf16_col(text, 0) == 1


def test_reveal_collapses_a_tag_run_to_one_marker() -> None:
    assert reveal("ok" + _tags("hi there")) == "ok<TAG x8>"


def test_reveal_collapses_runs_of_one_character() -> None:
    assert reveal("x" + "\u200b" * 3) == "x<U+200B x3>"
    assert reveal("a\u200bb\u200cc") == "a<U+200B>b<U+200C>c"


def test_reveal_shows_variation_selectors_although_they_are_not_format_characters() -> None:
    # U+FE0F and U+E0100 are category Mn. A reveal that keyed on Cf alone would
    # print a selector payload as nothing at all.
    assert reveal("a\ufe0f") == "a<U+FE0F>"
    assert reveal("a\U000e0100\U000e0101") == "a<U+E0100><U+E0101>"


def test_reveal_keeps_ordinary_whitespace_and_letters() -> None:
    assert reveal("a\tb\nc caf\u00e9") == "a\tb\nc caf\u00e9"


def test_decode_hidden_recovers_tag_ascii() -> None:
    assert decode_hidden("Summarise." + _tags("ignore all")) == "ignore all"


def test_decode_hidden_recovers_variation_selector_bytes() -> None:
    def vs(b: int) -> str:
        return chr(0xFE00 + b) if b < 16 else chr(0xE0100 + b - 16)

    payload = "".join(vs(b) for b in b"exfiltrate")
    assert decode_hidden("\U0001f600" + payload) == "exfiltrate"


def test_decode_hidden_is_none_when_nothing_is_hidden() -> None:
    assert decode_hidden("plain text") is None
    # A single VS16 after an emoji is presentation, not a payload.
    assert decode_hidden("\u2764\ufe0f") is None


def test_safe_line_cannot_start_a_workflow_command() -> None:
    assert not safe_line("::stop-commands::x/SKILL.md").startswith("::")
    assert "::" not in safe_line("a::b")


def test_safe_line_escapes_line_breaks_and_bidi_controls() -> None:
    out = safe_line("a\nb\rc\u202ed\u2028e")
    assert "\n" not in out and "\r" not in out and "\u202e" not in out
    assert "\u2028" not in out
    assert out == "a<U+000A>b<U+000D>c<U+202E>d<U+2028>e"


def test_safe_line_reveals_a_lone_surrogate_from_an_undecodable_file_name() -> None:
    name = b"bad\xffname".decode("utf-8", "surrogateescape")
    out = safe_line(name)
    out.encode("utf-8")
    assert out == "bad<U+DCFF>name"
