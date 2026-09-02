# Security policy

## Reporting a vulnerability

Report privately through GitHub, not in a public issue:

**<https://github.com/jamjet-labs/jamjet-guardrails/security/advisories/new>**

That opens a private security advisory visible only to you and the maintainers.
If the form is unavailable to you, open a public issue saying only that you have
a security report and asking for a private channel. Do not put the details in
it.

Please include the input that triggers the problem, the checks and the direction
you ran, the version of the package, and what you expected instead. A minimal
reproduction against the published wheel is worth more than a description.

We aim to acknowledge a report within three working days and to say within ten
whether we consider it a vulnerability, with our reasoning either way. This is a
single-maintainer project and those are targets, not a contract.

## What counts as a vulnerability here

This library publishes its own error rates, so "the detector missed something"
is usually a measurement, not a security bug. The line we draw:

**Not a vulnerability on its own.** A missed detection or a false alarm within
the rates published in [BENCHMARKS.md](BENCHMARKS.md), or a limit already
disclosed in [corpora/NOTICE.md](corpora/NOTICE.md) and the README. The
balanced bidi override that this package deliberately allows, the invisible
character families it does not count, and the four scattered zero-width
characters it lets through are all named there with the reasoning. A case of one
of them is a corpus contribution, and a welcome one. Open an issue.

**A vulnerability.** Anything that makes the library report a safer outcome than
the content deserves:

- Content that comes back `allow` or `redact` while a value the chain reported
  as handled is still present in the returned content. Composition bugs of this
  shape are the reason every check inspects the same input and redactions are
  merged into one pass.
- A decision that weakens: any path where a later check turns a `deny` into a
  `redact` or an `allow`.
- A failure that opens rather than closes. A raising detector must become
  `deny`, and a check named in configuration that is not installed must refuse
  at construction.
- A `saw` hash, a span or a provenance record that does not describe the text
  the decision was actually made about, since the audit record is the product.
- Input that makes a detector consume unbounded time or memory. The patterns are
  written and measured against backtracking; a case that defeats that is a
  report.
- A published number that cannot be reproduced from the committed corpus and the
  code. The numbers are the claim.

**Out of scope.** Anything about how a caller uses the result. This is a
library: it returns a decision and provenance, and forwarding content on a
`deny` is the caller's bug. `ChainResult.content` on a `deny` is an audit
record, not something to send, and the README says so.

## Supported versions

The latest released version on PyPI is the only supported one. There are no
maintained release branches, and a fix ships as a new release.

## Disclosure

We will credit you in the advisory and the changelog unless you ask us not to.
Please give us a chance to release a fix before publishing, and tell us the date
you intend to publish so we can work to it.
