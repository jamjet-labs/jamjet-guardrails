# Changelog

Notable changes to this package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published precision or recall figure is part of the interface here. A release
that moves one says so in its entry, with the old value and the new one, because
a number that changes quietly is a number nobody can rely on.

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/jamjet-labs/jamjet-guardrails/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jamjet-labs/jamjet-guardrails/releases/tag/v0.1.0
