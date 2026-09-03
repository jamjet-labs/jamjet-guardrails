"""The options each opt-in check's published row is measured under.

Two checks cannot be built without options, because a rules engine with no
rules and a script constraint with no allowed scripts each check nothing and
are refused at construction. The evaluation harness builds every check by name,
so without this table those checks could carry no published number at all, and
a check with no number is a check nobody can compare.

WHAT A ROW MEASURED THROUGH THIS TABLE PROMISES, and it is narrower than the
other rows: it measures the ENGINE on these options. It says nothing about
rules a user writes. `docs/conformance.md` prints the fixture beside the row so
the two cannot come apart, and a reader who sees the number sees the
configuration that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping

from jamjet_guardrails.authoring import Limits

# Keyed by registry name. A key naming no registered check fails a test in
# tests/test_completeness.py rather than sitting here describing nothing.
FIXTURES: Mapping[str, Mapping[str, object]] = {
    "rules": {
        "patterns": {
            "TICKET_ID": r"\bJIRA-\d{4,}\b",
            "INTERNAL_HOST": r"\b[a-z0-9][a-z0-9-]*\.corp\.example\b",
        },
        "banned": {"PROJECT_CODENAME": ("project bluebird",)},
        "limits": Limits(max_chars=2000),
        "on_match": "redact",
    },
    # Chosen so the row measures the two things that separate this check from a
    # naive one, on text where both occur naturally. Japanese needs all three
    # of Hiragana, Katakana and Han in one sentence, so the row exercises a
    # multi-script constraint rather than a single-script one; and Japanese
    # punctuation is where Script_Extensions resolution stops being theoretical,
    # because the prolonged sound mark and the ideographic comma are not
    # `Common` to this check and pass only because their extensions name a
    # script the fixture allows. Latin rides alongside because every one of
    # those sentences also carries a product name written in it. What the
    # fixture then LOCATES is a Cyrillic, Greek, Arabic or Devanagari run
    # inside that text, which is the shape the check exists for.
    "script-constraint": {"allowed_scripts": frozenset({"Latin", "Hiragana", "Katakana", "Han"})},
}


def options_for(name: str) -> Mapping[str, object]:
    """The fixture for a check, or nothing for a check that needs none.

    A function rather than a bare `FIXTURES.get`, so every caller spells the
    default the same way and a check that later gains a fixture needs no change
    at any call site.
    """
    return FIXTURES.get(name, {})
