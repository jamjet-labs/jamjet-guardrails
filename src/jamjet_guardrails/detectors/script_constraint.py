"""A constraint on which writing systems content may be written in.

Every other check here decides for itself what is wrong with a piece of text.
This one does not decide anything: the deployment states which scripts it
expects to see, and the check locates exactly what lies outside that statement.
A support desk that answers in English and Japanese has said something true and
checkable about its traffic, and a paragraph of Cyrillic arriving in a
retrieved page is a fact worth a decision even when nothing about it is
individually suspicious.

That is also why there is no default. A default here would be this library
choosing a language for somebody, and the only honest defaults are "everything"
(a check that checks nothing) and "Latin" (a check that denies most of the
world by shipping). `build("script-constraint")` refuses, and every other shape
of a constraint that would deny or permit everything by accident is refused at
construction beside it.

**What passes is as much of the design as what fires.** Punctuation, digits,
currency, mathematical symbols, whitespace, emoji, combining marks, ZWJ, ZWNJ
and variation selectors are `Common` or `Inherited` and pass under EVERY
constraint. A `{"Latin"}` constraint that denied a comma would be switched off
inside a day, and a guardrail that gets switched off has precision zero in the
only sense that matters. `injection-structural` learned the same lesson from
the other end, where a rule on zero-width characters that ignored their
neighbours denied conjunct Devanagari and every emoji family.

Cost, and the shape of it. Linear in the content length: two bisections over the
vendored tables per code point, allocating nothing per character. 256 ms median
for one megabyte of the seeded input recorded in `docs/performance.md`, on an
Apple M3 Max under CPython 3.14.5, with the median rising 3.94x to 4.06x per 4x
of input across the whole range from 1 KB. It is the slowest check in that table,
and the reason is that it is the only one with no way to skip a character: a
regular expression rejects most start positions on their first byte, and this
check has to resolve every code point it is given. Measured under the fixture in
`jamjet_guardrails.eval.fixtures`, which is the configuration the published row
uses, and on an input that reports NO findings, so the figure is the pass path
alone. `scripts/measure_throughput.py` reruns it, and `docs/performance.md`
states the machine, the input and the method.
"""

from __future__ import annotations

from collections.abc import Collection
from functools import lru_cache

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

SCRIPT_CONSTRAINT_TYPES = frozenset({"DISALLOWED_SCRIPT"})

_VERSION = "0.1.0"

# The two script values UTS #39 section 5.1 calls wildcards, and the whole of
# this check's false-positive defence. They are the resolved script of every
# punctuation mark, digit, symbol, space, emoji and combining mark that is not
# tied to one writing system, so they pass whatever the caller allowed. Named
# rather than spelled at the one place that reads them, because two of the
# refusals below also have to reason about them and a second literal is a
# second thing that can drift.
_WILDCARDS = frozenset({"Common", "Inherited"})

# What `jamjet_guardrails._unicode.script_set` returns for a code point the
# pinned tables do not assign. Spelled here rather than imported, so that
# refusing it as an `allowed_scripts` value costs no import: the module-level
# cost of this file has to stay zero, per the note on the lazy imports below.
_UNKNOWN = "Unknown"

# How much of a caller-supplied value a refusal message may quote. The same
# ceiling `chain._refusal` and `detectors.build` hold, and for the same measured
# reason: a value that arrives from a configuration file can BE the content, and
# a message that quotes it whole puts the content into whatever log wraps the
# configuration seam and produces a two-million-character error out of a
# two-million-character mistake. What a reader needs is enough to recognise
# which value was refused.
_VALUE_LIMIT = 120


def _quoted(value: object) -> str:
    """`repr(value)`, truncated to `_VALUE_LIMIT` and marked where it was cut."""
    shown = repr(value)
    return shown if len(shown) <= _VALUE_LIMIT else shown[:_VALUE_LIMIT] + "... (truncated)"


@lru_cache(maxsize=1)
def _script_names() -> frozenset[str]:
    """Every long script name the pinned tables can produce, `Unknown` aside.

    Derived from the tables rather than written out. A hand-written list of 170
    names is a list that goes out of date on the first pin bump, silently and
    in the direction that hurts: a script added by a later Unicode version
    would be refused as unknown while the tables resolve code points to it, so
    a caller could not name the script their own text is written in.

    Cached, and safe to cache for a reason rather than by habit: the tables are
    generated from data pinned at one Unicode version and committed, and a test
    requires them to be byte-identical to a regeneration from that data, so this
    set cannot change inside a process. Without the cache every construction
    walked 979 script ranges and 110 extension sets to build the same 170 names,
    which a deployment building one chain per configuration pays once and a test
    suite building many pays many times. Found by a Copilot review on the pull
    request.

    `maxsize=1` because the function takes no argument: there is one answer and
    the cache holds it.
    """
    from jamjet_guardrails._unicode.scripts import EXTENSION_SETS, SCRIPT_RANGES

    names = {script for _, _, script in SCRIPT_RANGES}
    # Script_Extensions names too, and not only Script names. Every extension
    # name is also a Script name in 16.0.0, so this union is currently a no-op;
    # it is written because that is a property of one version of the data and
    # not of the format, and the failure it prevents is the same silent one the
    # docstring describes.
    names |= {name for entry in EXTENSION_SETS for name in entry.split()}
    return frozenset(names)


class ScriptConstraintGuardrail:
    """Denies runs of text written in a script the deployment did not allow."""

    # Annotated with the Literal types rather than left as bare assignments,
    # for the reason `injection_structural` records: a bare `kind =
    # "constraint"` infers `str`, and protocol attribute matching is invariant,
    # so it would not satisfy `kind: Kind`.
    name: str = "script-constraint"
    version: str = _VERSION
    kind: Kind = "constraint"
    # Both directions, and neither is decorative. On input this locates a
    # language the deployment does not handle, which is where a retrieved page
    # smuggles a payload nobody on the team can read. On output it locates a
    # model answering in a script the caller cannot render or moderate, which
    # is the same fact seen from the far end of the call.
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(
        self,
        allowed_scripts: Collection[str] | None = None,
        on_match: Decision = "deny",
    ) -> None:
        """Refuse every configuration that would not constrain what it claims to.

        The refusals follow the doctrine `jamjet_guardrails.detectors.build`
        states: a check that is configured and would check less than the
        configuration says is refused before any content exists.
        ``GuardrailUnavailableError`` is what a caller wraps their
        configuration seam in, so it carries the mistakes that arrive FROM a
        configuration file; a value that is simply outside the domain keeps the
        ``ValueError`` that seam does not catch, which is the same split
        ``build("pii", types=frozenset())`` and ``build("pii",
        types={"PASSPORT"})`` already make.
        """
        if allowed_scripts is None:
            # There is deliberately no default, and this is the refusal that
            # says so. The two candidate defaults are both worse than raising:
            # every script is a check that reports nothing on any input, which
            # is the silent-and-configured shape every refusal in this package
            # exists to prevent, and a Latin default is this library deciding
            # which languages are ordinary. A config key present but empty
            # (`script-constraint:` with no value) arrives here as None, which
            # is why the message names that shape.
            raise GuardrailUnavailableError(
                "script-constraint has no default and needs allowed_scripts, a "
                "collection of long Unicode script names such as "
                "{'Latin', 'Hiragana', 'Katakana', 'Han'}. There is no default because "
                "the two candidates are a check that permits every script and reports "
                "nothing, and a check that decides for you which languages are ordinary. "
                "A configuration key that is present but empty arrives here as None."
            )
        if isinstance(allowed_scripts, (str, bytes, bytearray)):
            # The mistake `build_chain` refuses in the same words: a scalar
            # config value (`allowed_scripts: Latin` rather than
            # `allowed_scripts: [Latin]`) is an iterable of its own characters,
            # so it would otherwise be read as the five script names 'L', 'a',
            # 't', 'i' and 'n'. The unknown-name refusal further down would
            # catch it, and would report five mysteries instead of the one
            # mistake actually made.
            raise GuardrailUnavailableError(
                f"allowed_scripts must be a collection of script names, not the "
                f"{type(allowed_scripts).__name__} {_quoted(allowed_scripts)}; a string "
                f"is an iterable of its own characters, so this would select its "
                f"{len(set(str(allowed_scripts)))} distinct characters as script names. "
                f"Pass a list such as ['Latin']"
            )
        try:
            entries = list(allowed_scripts)
        except TypeError as exc:
            # The try wraps only the iteration, the discipline `build` applies
            # to its own membership test. An int, a float or any other
            # non-iterable arrives from a configuration file as readily as a
            # list does, and without this it escapes as a bare TypeError past
            # every `except GuardrailUnavailableError` the caller wrote around
            # the seam that produced it.
            raise GuardrailUnavailableError(
                f"allowed_scripts must be a collection of script names; a "
                f"{type(allowed_scripts).__name__} cannot be iterated ({exc})"
            ) from exc
        not_names = [entry for entry in entries if not isinstance(entry, str)]
        if not_names:
            # Before the set and the sort below, and that ordering is the
            # point. `sorted(set(...))` over mixed types raises `TypeError: '<'
            # not supported between instances of 'int' and 'str'` and an
            # unhashable entry raises another, both from inside the try above,
            # which would report a list of names as something that cannot be
            # iterated. `build` records the same failure in its own error path.
            raise ValueError(
                f"allowed_scripts must hold script names as strings; "
                f"{len(not_names)} of its entries are not strings, the first being "
                f"the {type(not_names[0]).__name__} {_quoted(not_names[0])}"
            )
        requested = sorted(set(entries))
        if not requested:
            # An empty collection is the same configuration mistake as an empty
            # `guardrails:` list, arriving through a different key, and it is
            # refused for the same reason: `allowed_scripts: []` is a key that
            # lost its value, not a request. Its behaviour makes that worse
            # rather than better -- every run of letters in every language on
            # earth becomes a finding, so the check denies its own deployment's
            # traffic on the first request.
            raise GuardrailUnavailableError(
                "allowed_scripts is empty, so no script is allowed and every run of "
                "letters in every language is a finding; a check that denies all "
                "content is one that gets switched off. Name the scripts this "
                "deployment expects, such as {'Latin'}"
            )
        known = _script_names()
        if _UNKNOWN in requested:
            # `Unknown` IS what `script_set` returns, so it would pass the
            # membership test below if it were in the table, and it is not.
            # Refused with its own message because the mistake it represents is
            # not a misspelling: a caller who writes it is asking the check to
            # permit exactly the code points the pinned tables cannot identify,
            # which is the one class this check denies on purpose.
            raise ValueError(
                f"{_UNKNOWN!r} is not a script; it is what the pinned tables return for "
                "a code point they do not assign, and script-constraint denies those on "
                "purpose so that a code point this build cannot name cannot pass "
                "unexamined"
            )
        unknown = [name for name in requested if name not in known]
        if unknown:
            # The whole list, not a nearest-match suggestion. A suggestion is a
            # guess, and what a caller needs here is the exact spelling: the
            # names are the LONG property values with underscores
            # (`Old_Permic`, `Canadian_Aboriginal`), never the four-letter
            # codes `ScriptExtensions.txt` is written in, so `Latn`, `Cyrl` and
            # `Jpan` are all near misses that no amount of guessing turns into
            # the right answer. There is no ISO 15924 alias table here to
            # translate them, deliberately: two spellings in circulation is a
            # constraint that matches nothing while reporting nothing.
            raise ValueError(
                f"allowed_scripts names {unknown}, which the pinned Unicode "
                f"{self._unicode_version()} tables do not know. Names are the long "
                f"property values, not the four-letter codes: {sorted(known)}"
            )
        self._allowed = frozenset(requested)
        # The wildcards are folded in ONCE, here, rather than added to the test
        # in the scan below. The scan runs per code point over a megabyte of
        # content; this runs once per guardrail.
        #
        # A caller naming only `Common` or `Inherited` is permitted and is not
        # refused, though the result is the same check an empty collection
        # would have produced: the wildcards already pass unconditionally, so
        # naming them adds nothing and every letter of every script fires. It
        # stays permitted because the refusal above is about a value that
        # arrived empty from a configuration file, which is a mistake nobody
        # makes on purpose, and this is a set somebody typed.
        self._effective = self._allowed | _WILDCARDS
        if on_match not in ("redact", "deny"):
            raise ValueError(
                f"on_match must be 'redact' or 'deny', got {on_match!r}. A check "
                "configured to allow on a match is a check that runs and cannot act"
            )
        # Deny by default, and unlike `secrets` the redact option is the
        # unusual one. Deleting the Cyrillic from a Russian sentence leaves
        # content that is no longer what anybody wrote, and a model handed the
        # remains will answer about the placeholders. The option exists because
        # the spans are exact and a caller who wants the run stripped rather
        # than the message dropped can have it.
        self._on_match: Decision = on_match

    @staticmethod
    def _unicode_version() -> str:
        """The version of the pinned tables, for the message above.

        Imported inside the function for the reason the scan below is, and
        `staticmethod` so the refusal path can reach it before `self` is fully
        built.
        """
        from jamjet_guardrails._unicode import UNICODE_VERSION

        return UNICODE_VERSION

    @property
    def allowed_scripts(self) -> frozenset[str]:
        """What the caller allowed, without the wildcards folded in.

        The wildcards are an implementation of "punctuation is not a language",
        not part of what the caller asked for, and a reader of an audit record
        who saw `Common` in a set they never typed would reasonably conclude
        the configuration had been rewritten under them.
        """
        return self._allowed

    def _runs(self, content: str) -> list[tuple[int, int]]:
        """Maximal runs of code points no allowed script covers, in text order.

        A code point passes if ANY member of its resolved script set is
        allowed, which is what makes a code point shared between scripts pass
        under a constraint naming any one of them: the ideographic full stop
        `U+3002` has no `Common` in its resolved set at all and passes under a
        `{"Han"}` constraint because `Han` is in its Script_Extensions. Asking
        for the intersection to be CONTAINED in the allowed set instead would
        deny it, and would deny the middle dot `U+00B7` under every constraint
        that does not name all sixteen of its scripts.

        A run is maximal over DISALLOWED-ness and not over one script, so
        Greek immediately followed by Cyrillic is one finding and not two. The
        finding says a stretch of content is outside the constraint; which
        scripts it is in is not something the type or the span carries, and
        splitting on a script boundary would report two facts where there is
        one and produce two placeholders where a caller expects one.

        `script_set` is imported HERE and not at the top of this module.
        `detectors/__init__.py` is imported from the package root, so a
        top-level import would drag 45 KiB of tables into every
        `import jamjet_guardrails`, including every one that never builds this
        check.
        `tests/test_unicode.py::test_importing_the_package_loads_neither_unicode_table`
        holds that.
        """
        from jamjet_guardrails._unicode import script_set

        allowed = self._effective
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for index, character in enumerate(content):
            if script_set(character) & allowed:
                if start is not None:
                    spans.append((start, index))
                    start = None
            elif start is None:
                start = index
        if start is not None:
            # The run that reaches the end of the content. Without this a
            # constraint violation in the last characters of a message produced
            # no finding at all, which is the half of the input an exfiltrated
            # payload is appended to.
            spans.append((start, len(content)))
        return spans

    def check(self, content: str, context: Context) -> Verdict:
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        # Already in text order and already disjoint, because the scan walks
        # left to right and closes each run before opening the next, so the
        # sort every other detector applies before `_merge` would be a no-op
        # here. `_merge` still runs inside `_rewrite`, which is what keeps the
        # placeholder arithmetic identical to every other check's.
        found = [("DISALLOWED_SCRIPT", span) for span in self._runs(content)]
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        findings = [Finding(type=name, span=span) for name, span in found]
        if self._on_match == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
