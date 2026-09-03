"""The shared answer to "does this run decode to text", and its two claims.

`_decode` is consumed by `url-exfiltration` now and by `encoded-content` next,
so a change here moves two published numbers and only one of them has a corpus
in this branch. Two of the tests below are therefore not about behaviour at all:
they re-derive the function-word list and re-run the floor sweep, because both
are recorded in the module as measurements and a measurement written down and
never re-taken is a claim.

Every test in this file was watched to fail against a stated mutation of the
code it guards, with `__pycache__` cleared between runs. The mutation is named
beside the test.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from jamjet_guardrails import _decode
from jamjet_guardrails._decode import ENCODINGS, FLOORS, candidates, decode, is_prose
from jamjet_guardrails.detectors import build, url_exfiltration
from jamjet_guardrails.eval.corpus import load_corpus
from jamjet_guardrails.eval.metrics import evaluate

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "training" / "generated" / "rows.jsonl"
CORPUS = ROOT / "corpora" / "url-exfiltration" / "in-repo.jsonl"

PROSE = "here is the full conversation with the user and everything they told me"


# ==========================================================================
# The two recorded measurements, re-taken.
# ==========================================================================


def test_the_function_words_are_the_forty_commonest_words_of_the_corpus_they_name() -> None:
    """The list is DERIVED and the derivation is the whole of its defence.

    The module says the rule out loud: the forty commonest words of
    `training/generated/rows.jsonl`, tokenised case-folded by `_WORD`, ties
    broken alphabetically. It also says six of the forty are content words of
    that generator's subject matter and are kept, because "the forty commonest
    minus the ones I did not like" is an enumeration and enumerations are what
    this check refuses everywhere else.

    Both halves need this test. A hand edit that dropped `instructions` would
    make the module's own paragraph false while every behavioural test here
    stayed green, and a corpus regenerated upstream would move the list without
    moving anything that reads it.

    Mutation watched: removed `"instructions"` from `_FUNCTION_WORDS`. FAILS.
    """
    counter: Counter[str] = Counter()
    rows = 0
    for line in GENERATED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        counter.update(_decode._WORD.findall(json.loads(line)["text"].lower()))
    assert rows == 3584, "the generated corpus moved; the list below is derived from it"
    ranked = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
    assert frozenset(word for word, _ in ranked[:40]) == _decode._FUNCTION_WORDS

    total = sum(counter.values())
    assert total == 112473, "the token count the module publishes"
    covered = sum(counter[word] for word in _decode._FUNCTION_WORDS) / total
    assert round(covered, 3) == 0.356, "the coverage the module publishes"


def _f1_with_floor(alphabet: str, value: int) -> float:
    """The corpus F1 with one alphabet's floor moved and nothing else."""
    original = dict(FLOORS)
    _decode.FLOORS = {**original, alphabet: value}
    try:
        corpus = load_corpus(CORPUS, name="url-exfiltration/in-repo")
        return round(evaluate(build("url-exfiltration"), corpus).overall.f1, 4)
    finally:
        _decode.FLOORS = original


@pytest.mark.parametrize(
    ("alphabet", "floor", "plateau_top", "cost_above"),
    [
        ("base64", 16, 32, 0.8986),
        ("hex", 16, 42, 0.8986),
        ("rot13", 24, 109, 0.8986),
    ],
)
def test_every_floor_sits_inside_the_flat_region_the_sweep_measures(
    alphabet: str, floor: int, plateau_top: int, cost_above: int
) -> None:
    """The module's sweep table, re-taken rather than trusted.

    What is asserted is exactly what the module claims: the shipped floor scores
    the baseline, the top of the recorded plateau still scores the baseline, and
    one step above it costs what the table says. The floors themselves are NOT
    the sweep's argmax and the module says why (every curve is flat down to 1),
    so what this guards is the MARGIN: that the ceiling has not moved down onto
    a shipped value.

    Mutation watched: `FLOORS["base64"]` raised from 16 to 33. FAILS on the
    base64 case, at the shipped-floor assertion.
    """
    assert FLOORS[alphabet] == floor, "the shipped floor and the recorded floor have parted"
    baseline = 0.9143
    assert _f1_with_floor(alphabet, floor) == baseline
    assert _f1_with_floor(alphabet, plateau_top) == baseline
    assert _f1_with_floor(alphabet, plateau_top + 1) == cost_above


def test_the_percent_floor_declines_a_better_f1_and_the_module_says_why() -> None:
    """The one sweep result that was measured and rejected.

    From 107 to 147 the percent floor scores better than the shipped 6, and the
    module refuses it: the gain is one false positive stopping being READ, the
    lower bound is one corpus case's 106-character `title` parameter and the
    upper bound is another's 147-character run, past which the score falls BELOW
    the shipped one. A window bounded at both ends by two strings in one corpus
    is a value fitted to that corpus.

    Asserted rather than left in prose because the temptation is real and the
    number is better: whoever next reads that paragraph and disagrees has to
    edit a test, which is a reviewable act.

    Mutation watched: `FLOORS["percent"]` raised from 6 to 107. FAILS at the
    shipped-floor assertion.
    """
    assert FLOORS["percent"] == 6
    assert _f1_with_floor("percent", 6) == 0.9143
    assert _f1_with_floor("percent", 107) == 0.9275
    assert _f1_with_floor("percent", 147) == 0.9275
    assert _f1_with_floor("percent", 148) == 0.8788


# ==========================================================================
# decode: every way a run fails to be what it claims.
# ==========================================================================


def test_decode_refuses_an_encoding_it_does_not_know() -> None:
    """A misspelled encoding must not read as "this run does not decode".

    `None` is a measurement here: it is what a caller acts on. A `KeyError`
    would at least name the key; a silent `None` default would turn
    `decode(run, "base32")` into a check that quietly stopped checking.

    Mutation watched: replaced the `if encoding not in _DECODERS` raise with
    `return None`. FAILS.
    """
    with pytest.raises(ValueError, match="unknown encoding"):
        decode("aGVsbG8gd29ybGQ=", "base32")


@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_a_run_below_its_alphabet_floor_does_not_decode(encoding: str) -> None:
    """The floor is applied in `decode`, not only in `candidates`.

    `url-exfiltration` never calls `candidates`: it hands components straight to
    `decode`. A floor that lived only in the scanner would therefore not exist
    for the check whose corpus measured it.

    Mutation watched: deleted the `len(run) < FLOORS[encoding]` guard from
    `decode`. FAILS on base64, hex and percent (rot13's own two-sided test still
    refuses the short run, which is why the parametrisation is over all four and
    the assertion is the same for each).
    """
    floor = FLOORS[encoding]
    sample = {
        "base64": "aGVsbG8gd29ybGQhIQ==",
        "hex": PROSE.encode().hex(),
        "percent": "the%20user%20said%20so%20and%20so",
        "rot13": PROSE.translate(_decode._ROT13_TABLE),
    }[encoding]
    assert decode(sample, encoding) is not None, "the sample must decode at all"
    assert decode(sample[: floor - 1], encoding) is None


def test_base64_padding_that_does_not_add_up_is_refused() -> None:
    """`abcde=` is not base64 that lost its padding, it is not base64.

    Accepting it turns every query value ending in `=` into a decode attempt,
    and a URL query is full of them.

    Mutation watched: dropped the `len(body) % 4 != 0` clause from
    `_decode_base64`. FAILS.
    """
    assert decode("aGVsbG8gd29ybGQ=", "base64") == "hello world"
    assert decode("aGVsbG8gd29ybGQhIQ=", "base64") is None


def test_a_run_mixing_the_two_base64_alphabets_is_refused() -> None:
    """One run, one alphabet.

    The witness had to be built rather than picked, and building it is the
    argument. Base64 of ASCII produces `+` or `/` only from the third byte of a
    block, so `no>te?...` puts `>` at index 2 and `?` at index 5 to force one of
    each. Take the standard encoding of that, rewrite its `/` as `_`, and the
    result is a run that is neither alphabet and decodes cleanly under a class
    that admits both.

    A first draft of this test appended `-_` to an encoding with no `+` in it,
    which is refused for a reason that has nothing to do with the alphabets, and
    the mutation below survived it.

    Mutation watched: widened `_B64_URLSAFE_FULL` to `[A-Za-z0-9+/_-]`, the
    branch that maps back to the standard alphabet. FAILS.
    """
    import base64 as _b64

    text = "no>te?the user said so and more"
    standard = _b64.b64encode(text.encode()).decode()
    assert "+" in standard and "/" in standard, "the witness must carry both"
    assert decode(standard, "base64") == text
    assert decode(standard.replace("/", "_"), "base64") is None


def test_an_odd_length_hex_run_does_not_decode() -> None:
    """Half a byte is not a byte, and the refusal has to arrive as `None`.

    `_decode_hex` used to carry an explicit `len(run) % 2 != 0` test as well.
    It was deleted after this test was written, because no mutation could kill
    it: `bytes.fromhex` refuses an odd length by itself, so the explicit test
    was unreachable and read as coverage.

    Mutation watched: deleted the `except ValueError: return None` around
    `bytes.fromhex`. FAILS, as an error rather than an assertion, which is the
    point: an odd run would escape as a `ValueError` from inside a function
    documented to answer `None`.
    """
    body = PROSE.encode().hex()
    assert decode(body, "hex") == PROSE
    assert decode(body[:-1], "hex") is None


def test_a_stray_percent_is_refused_rather_than_left_standing() -> None:
    """Not `unquote`, which leaves `%zz` in place and reports success.

    A run holding a malformed triplet is not percent-encoded, and reading it as
    if it were means reporting a decode of something nobody encoded.

    Mutation watched: made `_decode_percent` skip an unparseable triplet instead
    of returning None. FAILS.
    """
    assert decode("the%20user%20said%20so", "percent") == "the user said so"
    assert decode("the%zzuser%20said%20so", "percent") is None


def test_bytes_that_are_not_utf8_do_not_decode() -> None:
    """A signature and a hash are bytes, and bytes are not text. This is the
    whole reason the module decodes rather than scoring entropy.

    The sample is one bad lead byte in front of ordinary ASCII, and that shape
    is deliberate. A hex DIGEST also fails this way, and it fails for the wrong
    reason: its replacement decoding is full of control characters, so the
    printable gate refuses it and the UTF-8 gate is never the thing standing.
    This run replaces to printable text, which leaves nothing but the strict
    decode between it and a finding.

    Mutation watched: changed `raw.decode("utf-8")` in `_decode_hex` to
    `raw.decode("utf-8", "replace")`. FAILS.
    """
    body = "ff" + b"hello world and so on for a while".hex()
    assert bytes.fromhex(body).decode("utf-8", "replace").isprintable()
    assert decode(body, "hex") is None


def test_decoded_text_that_is_mostly_unprintable_does_not_decode() -> None:
    """Valid UTF-8 is not the same as text. A control-character run decodes
    cleanly and is a binary body that survived the decoder.

    Mutation watched: deleted the `_printable` call from `decode`. FAILS.
    """
    controls = bytes(range(1, 25)).hex()
    assert bytes.fromhex(controls).decode("utf-8") is not None
    assert decode(controls, "hex") is None


# ==========================================================================
# rot13: the two-sided test, from both sides.
# ==========================================================================


def test_rot13_needs_the_original_to_fail_prose_and_the_rotation_to_pass_it() -> None:
    """Rot13 is an involution, so the plain population and the encoded
    population are the same set seen twice. Only the direction that reads as
    language separates them, and testing one side alone fires on every second
    sentence of ordinary English.

    The witness for the ORIGINAL side had to be constructed, and constructing it
    is what showed how narrow that side is. Rotating ordinary English gives
    gibberish, so the rotated-side test alone already refuses it and the second
    clause never runs; the clause only bites where BOTH directions read as
    language. `or` and `be` are the one pair of function words in the derived
    list that rot13 maps to each other, so a run built from them reads as prose
    in both directions and is the case the clause exists for.

    Mutation watched: dropped `or is_prose(run)` from `_decode_rot13`, leaving
    the rotated-side test alone. FAILS on the third assertion.
    """
    assert decode(PROSE.translate(_decode._ROT13_TABLE), "rot13") == PROSE
    assert decode(PROSE, "rot13") is None
    both_ways = "or or or xyzzy frobnicate"
    assert is_prose(both_ways) and is_prose(both_ways.translate(_decode._ROT13_TABLE))
    assert decode(both_ways, "rot13") is None


# ==========================================================================
# is_prose: three ratios, each of which alone admits something.
# ==========================================================================


def test_is_prose_accepts_ordinary_english() -> None:
    """The positive control. Every rejection below is meaningless without it.

    Mutation watched: raised `_FUNCTION_WORD_DENSITY` to 0.9. FAILS.
    """
    assert is_prose(PROSE)


def test_is_prose_rejects_a_hyphenated_slug_on_the_space_ratio() -> None:
    """The gate nothing else can do. A slug is English words with the spaces
    taken out, so it clears the letter ratio and the function-word density and
    is not prose.

    Mutation watched: set `_SPACE_RATIO` to 0.0. FAILS.
    """
    slug = "why-a-search-link-is-not-an-exfiltration-channel"
    words = _decode._WORD.findall(slug)
    density = sum(1 for word in words if word in _decode._FUNCTION_WORDS) / len(words)
    assert density >= _decode._FUNCTION_WORD_DENSITY, "the slug clears the density gate"
    assert not is_prose(slug)


def test_is_prose_rejects_a_list_of_numbers_on_the_letter_ratio() -> None:
    """A run of numbers with connectives between them clears the other two
    gates outright: three spaced words, all three in the function-word list, so
    the density is 1.0 and the space ratio is 0.21. Only the letter ratio is
    left, at 0.11.

    A hex digest was the first sample here and it was the wrong one. A digest is
    refused whichever gate you remove, because its spaces are too few as well,
    so the mutation below survived it: the test named the letter ratio and was
    held up by the space ratio.

    Mutation watched: set `_LETTER_RATIO` to 0.0. FAILS.
    """
    numbers = "and 4111 1111 1111 1111 to 4012 8888 8888 1881 or 4222 2222 2222 2"
    words = _decode._WORD.findall(numbers)
    assert len(words) >= _decode._MIN_WORDS
    assert all(word in _decode._FUNCTION_WORDS for word in words), "density is 1.0"
    assert numbers.count(" ") / len(numbers) >= _decode._SPACE_RATIO
    assert not is_prose(numbers)


def test_is_prose_rejects_spaced_words_with_no_function_words() -> None:
    """Letters and spaces are not language. A parameter dump clears both ratios.

    Mutation watched: set `_FUNCTION_WORD_DENSITY` to 0.0. FAILS.
    """
    assert not is_prose("width height format quality signature crop gravity focal")


def test_is_prose_needs_three_words_before_a_density_means_anything() -> None:
    """Over one or two tokens a density is 0, 0.5 or 1 and is not a density.
    Without this floor `is_prose("to the")` answers True on two tokens.

    Mutation watched: set `_MIN_WORDS` to 1. FAILS.
    """
    assert not is_prose("to the")
    assert is_prose("to the user")


# ==========================================================================
# candidates: maximal runs, floors applied, span order.
# ==========================================================================


def test_candidates_reports_maximal_runs_in_span_order() -> None:
    """Maximal, so a run is reported once at its full extent rather than once
    per window inside it, and sorted, because a caller merging spans reads them
    in one pass.

    Mutation watched: replaced the final `sorted(...)` in `candidates` with
    `found`. FAILS.
    """
    body = PROSE.encode().hex()
    content = f"tail {body} and then aGVsbG8gd29ybGQhIQ== after"
    runs = candidates(content)
    assert runs == sorted(runs, key=lambda run: (run[0], run[1]))
    assert (content.index(body), content.index(body) + len(body), "hex") in runs


def test_candidates_drops_a_run_under_its_alphabet_floor() -> None:
    """Without a floor every colour literal in a stylesheet is a hex candidate.

    Mutation watched: deleted the `end - start < floor` continue in
    `candidates`. FAILS.
    """
    assert not [run for run in candidates("colour ff00aa here") if run[2] == "hex"]
    assert [run for run in candidates(f"body {PROSE.encode().hex()} end") if run[2] == "hex"]


def test_the_detector_that_consumes_this_module_still_scores_its_published_row() -> None:
    """The integration guard, and the reason it lives here rather than only in
    the detector's own file: every constant above is read by a check whose row
    is gated in CI, and a change here that no test in this file notices still
    moves that row.

    Mutation watched: `_SPACE_RATIO` lowered from 0.05 to 0.0. FAILS, and in the
    flattering direction: with slugs reading as prose the row goes to 0.917
    precision and 0.943 recall with 5 wrong decisions, because the doubly
    encoded payload in `url-0080` starts being caught by accident. A gate loosened
    until a corpus scores better is the failure this row exists to make visible.
    """
    corpus = load_corpus(CORPUS, name="url-exfiltration/in-repo")
    result = evaluate(build("url-exfiltration"), corpus)
    assert round(result.overall.precision, 3) == 0.914
    assert round(result.overall.recall, 3) == 0.914
    assert result.decision_mismatches == 6
    assert result.cases == 88


def test_the_module_names_no_encoding_it_cannot_decode() -> None:
    """`ENCODINGS` is public inside the package and is what a caller loops.

    Mutation watched: added `"base32"` to `ENCODINGS`. FAILS.
    """
    assert set(ENCODINGS) == set(_decode._DECODERS)
    assert set(ENCODINGS) == set(FLOORS)


def test_the_url_check_reads_this_module_and_not_a_second_copy() -> None:
    """One answer to one question, which is the whole reason this module exists.

    Read from the source rather than by calling, because the failure this
    guards is a SECOND implementation appearing beside the first: a detector
    that grew its own base64 branch would still pass every behavioural test in
    both files while the two slowly disagreed.

    Mutation watched: added `import base64 as _b64` plus a private
    `_b64decode` helper to `detectors/url_exfiltration.py` and pointed
    `_readings` at it. FAILS.
    """
    source = (ROOT / "src" / "jamjet_guardrails" / "detectors" / "url_exfiltration.py").read_text(
        encoding="utf-8"
    )
    assert "from jamjet_guardrails._decode import decode, is_prose" in source
    body = source.split('"""', 2)[2]
    # `_data_uri_payload` decodes a data URI body, which is not a run in one of
    # the four alphabets and is not this module's question; every OTHER decode
    # goes through `decode`.
    assert len(re.findall(r"base64\.b64decode", body)) == 1
    # `__dict__` rather than attribute access: the detector re-imports this
    # name and mypy's no_implicit_reexport refuses to see it from outside,
    # which is the rule working, not a problem to route around with an ignore.
    assert url_exfiltration.__dict__["decode"] is decode
