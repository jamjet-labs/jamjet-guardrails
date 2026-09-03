# Changelog

Notable changes to this package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published precision or recall figure is part of the interface here. A release
that moves one says so in its entry, with the old value and the new one, because
a number that changes quietly is a number nobody can rely on.

## [Unreleased]

### Security

- `GuardrailChain` no longer trusts anything a guardrail returns. It reads each
  field of a returned `Verdict` once, checks it against what the chain itself
  knows, and then BUILDS a new verdict from those reads: `saw` is the chain's own
  digest over the content it passed to `check`, `provenance` carries the identity
  the chain holds for that guardrail, and the findings are fresh objects with
  their own strings and plain integer spans. Nothing a detector returned reaches
  `ChainResult.verdicts`, the combined decision or the rewrite AS THE OBJECT it
  arrived in. Previously a `check` that returned successfully could report any
  detector name, version and kind, hash text it was never given, and carry a
  finding's span past the end of the content, and all of it was recorded
  unexamined. Verifying that verdict and then keeping it was not enough either: a
  `str` subclass whose `__eq__` returns True passes every comparison, an object
  whose `__class__` says `Verdict` passes `isinstance` and can answer a property
  honestly while it is being checked and falsely while it is being recorded, and
  an `int` subclass passes `0 <= start < end <= len(content)` and then slices as
  a negative number, which emitted a prefix of the ORIGINAL content in front of a
  redaction placeholder under a `redact` decision. Every strictness check reads
  `type(x) is T`, never `isinstance`, because a subclass with a lying `__eq__` or
  `__str__` walks straight through the comparison or the coercion `isinstance`
  would still allow.
- A finding's `type` is bounded to the same length every other caller-chosen
  string in an error message already was. It is still, by design, the one
  detector-chosen string that reaches `ChainResult.content`: a redaction
  placeholder has to name what claimed the region, so `EMAIL` in
  `[REDACTED:EMAIL]` is a finding's `type`. Unbounded, a `redact` whose
  finding's type IS the content it redacted reproduced that content, verbatim,
  inside a string this library calls safe to forward, and a type running to
  millions of characters inflated the audit record by the same amount for one
  finding. The bound stops the second failure, not the first: a short type
  equalling a short secret is not new, and is not what the bound closes.
- `provenance.threshold` and a finding's `confidence` are checked for
  finiteness, not only type. Both are `float | None`, and NaN and infinity are
  both a `float`, so the type check alone let either one reach the audit
  record. A non-finite value is now a contract violation like any other.
- A malformed finding span no longer abandons the run. A span of `(1, 2, 3)`,
  `("a", "b")` or `5` raised out of `run` from validation that sat outside the
  try, losing the whole run's audit record and every guardrail after the one
  that misbehaved. It is now a synthesised `deny` like any other failed check.
- An error message never repeats a value a detector chose: not a claimed
  provenance, not a finding's type, not the class name of whatever was returned,
  not a span. A detector's returned values are picked after it has seen the
  content, so a detector whose class name or finding type IS the content used to
  write the content into the audit record through the message. Messages name the
  clause that failed, and name the guardrail from the chain's own bounded copy
  of the name it declared; a 2,000,000-character name produced a
  4,000,073-character `error` before that bound.
- `GuardrailChain.__init__` reads each guardrail's `name`, `version`, `kind` and
  `directions` once, checks them, and uses those copies for every verdict
  afterwards, refusing a guardrail that cannot declare a usable identity with
  `GuardrailUnavailableError` -- the error `build` and `build_chain` already
  raise for a configuration that would check less than it claims. A `name` that
  answered one way while it was being validated and another way while it was
  being recorded is no longer read twice. A guardrail declaring a `kind` this
  library does not know is refused at construction rather than raising
  `ValueError` from inside `run`: because provenance is stamped from the
  declared kind, that guardrail used to turn the chain's own fail-closed path
  into an exception, so an honest verdict or a raising `check` took the entire
  run down with it.

### Changed

- A `redact` carrying no content, and a `redact` whose span does not index into
  the content, no longer raise `GuardrailChainError` out of `run`. Both are
  verdicts the chain will not rebuild, so both become a synthesised `deny` and
  the run keeps its audit record. One shape still abandons a run: a `redact` the
  chain cannot locate, meaning no findings or a finding without a span.
- A `Verdict` returned with `error` set is refused. `error` is the chain's own
  field, set on a verdict the chain synthesised, and a detector writing its own
  prose there put an unbounded, detector-chosen string into the audit record.

## [0.2.0]

### Added

- `PatternGuardrail` and `Limits`, exported from the package root. A check is
  now typed regular expressions, banned substrings and size limits, and the
  spans, the merging and the refusals come with it. `docs/conformance.md`
  specifies what a published row measured through it does and does not promise.
- `rules`, a check that takes your own patterns, banned substrings and size
  limits. Size limits are characters, bytes and lines; there is no token limit,
  because counting tokens needs a tokenizer this library does not carry.
- `TYPES`, beside `AVAILABLE`, naming the finding types each check can report.
- `scripts/new_check.py`, which writes the files a new check needs and names
  the four it leaves to you.

### Changed

- `injection-structural` runs on output as well as input, and its detector
  version moves to 0.2.0. A chain that previously skipped this check on model
  output now runs it there and can deny. A model emitting tag characters into
  its own output is smuggling to whatever reads that output next.
- Its published precision moved from 0.971 to 0.972 and its recall from 0.870
  to 0.873, because the corpus gained eight output cases and grew from 146 to
  154. The detector's behaviour on every case it already scored is unchanged.
- `rules` is new and is published at 1.000 precision and 1.000 recall on 40
  cases. That row measures the engine under one fixed configuration, printed
  in `docs/conformance.md`, and says nothing about a rule you write yourself.

### Security

- `GuardrailUnavailableError` is now also raised from `PatternGuardrail.check`,
  not only at construction, when a caller passes a context whose direction the
  guardrail does not declare. This is the same error as the constructor raises
  when a check is misconfigured, because both mistakes mean a check was asked
  to evaluate something it was not set up for, and answering would falsely
  report that content had been examined. This replaces the previous silent allow
  on content with no matches and a bare `KeyError` on content with matches. A
  chain never triggers this, since it only calls a guardrail whose declared
  directions contain the context's direction; the error is for a caller holding
  a single guardrail directly.

### Notes

- Still no runtime dependencies. The authoring primitive is standard library
  only, and `dependencies = []` is still checked against the built metadata.

## [0.1.0]

First release.

### Added

- `build_chain` and `GuardrailChain`, returning a decision, the findings behind
  it, and a provenance record naming the detector and its version for every
  check that ran. Each verdict carries `saw`, the SHA-256 of the exact string
  that check inspected.
- Three deterministic checks. `injection-structural` (input) reports
  `BIDI_OVERRIDE`, `INVISIBLE_TAG_CHARS` and `ZERO_WIDTH_SMUGGLING`. `pii`
  (input and output) reports `CREDIT_CARD`, `EMAIL`, `PHONE_NUMBER` and
  `US_SSN`. `secrets` (input and output) reports `ANTHROPIC_KEY`,
  `AWS_ACCESS_KEY`, `GITHUB_TOKEN`, `JWT`, `OPENAI_KEY`, `PRIVATE_KEY` and
  `SLACK_TOKEN`.
- Published precision and recall per check and per type, measured on corpora
  committed in `corpora/` and gated in CI against `corpora/baselines.json`. The
  scores at this release are in [BENCHMARKS.md](BENCHMARKS.md).
- `jamjet-guardrails`, a console script that scores the corpora, writes
  `benchmarks.json` and `BENCHMARKS.md`, and gates a run against recorded
  baselines.
- [docs/conformance.md](docs/conformance.md), specifying the verdict fields, the
  combination order, the single-pass rewriting rule, the `saw` hash and the
  corpus schema, so the checks can be reimplemented in another language and
  scored on the same corpora.
- PEP 561 typing marker. The wheel ships `py.typed` and the release workflow
  asserts it is inside the built artifact.

### Security

- Decisions combine restrictively, `deny` > `redact` > `allow`, and no path can
  weaken a decision another check has already made.
- Every check inspects the content the chain was given, and redactions from all
  checks are merged and applied in one pass. Rewriting sequentially let a
  personal-data redaction cut a credential in half so the next check matched
  only the stump, and the rest of the credential survived into content the chain
  reported as redacted.
- A check that raises becomes `deny`, never `allow`, and the error message is
  withheld from the verdict because a detector's message may quote the content.
- A check named in configuration that is not installed raises
  `GuardrailUnavailableError` at construction rather than running unguarded. An
  empty list of checks is refused for the same reason.

### Notes

- No runtime dependencies, no network calls, no model downloads. The installed
  distribution declaring none is checked against the built metadata.
- The distribution declares `Apache-2.0 AND CC-BY-4.0`. The code is Apache-2.0;
  the source distribution also carries `corpora/pii/third-party.jsonl`, derived
  from `nvidia/Nemotron-PII` under CC-BY-4.0. Attribution and the list of
  changes are in [corpora/NOTICE.md](corpora/NOTICE.md). The wheel contains code
  only.

[Unreleased]: https://github.com/jamjet-labs/jamjet-guardrails/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jamjet-labs/jamjet-guardrails/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamjet-labs/jamjet-guardrails/releases/tag/v0.1.0
