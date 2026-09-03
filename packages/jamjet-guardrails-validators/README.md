# jamjet-guardrails-validators

[Guardrails AI](https://github.com/guardrails-ai/guardrails) validators backed by
[jamjet-guardrails](https://github.com/jamjet-labs/jamjet-guardrails), a
zero-dependency library of deterministic content checks with published precision
and recall.

Installed from PyPI, not from the Guardrails Hub, which is closed. Every
validator here is an ordinary pip install.

## Install

```
pip install jamjet-guardrails-validators
```

Requires Python 3.10 to 3.13, the range `guardrails-ai` itself supports.

## Use the composite

```python
from guardrails import Guard
from jamjet_guardrails_validators import JamJetChain

guard = Guard().use(JamJetChain(checks=["pii", "secrets"], on_fail="fix"))

outcome = guard.validate("SLACK_BOT_TOKEN=xoxb-0000000000-2000000000008-EXAMPLEONLYnotarealtoken")
outcome.validated_output
# 'SLACK_BOT_TOKEN=[REDACTED:CREDIT_CARD+SLACK_TOKEN]'
```

## Why one validator and not two

Stacking `jamjet/pii` and `jamjet/secrets` as two validators is not the same
thing, and the difference is a leak rather than a preference. Measured on
guardrails-ai 0.11.0, which has two validator services and picks between them
without asking.

**The sequential service** (`GUARDRAILS_RUN_SYNC=true`) hands each validator the
previous validator's `fix_value`. Over the token above, with `jamjet/pii` first,
the guard reports `validation_passed=True` and returns:

```
SLACK_BOT_TOKEN=[REDACTED:SLACK_TOKEN][REDACTED:CREDIT_CARD]-EXAMPLEONLYnotarealtoken
```

The 13-digit middle segment is Luhn-valid and begins with a 2, so the PII check
reads it as a payment card and redacts it first. That placeholder splits the
token in half. The secrets check then matches only the 16-character prefix, and
the 24-character secret tail survives into a value the guard calls valid.

**The default async service** runs every validator over the original value and
combines the fixes with a three-way merge. That closes the stump above and opens
something worse: a `FailResult` with no `fix_value`, which is what a deny is, is
outvoted by one that carries a fix. Stacking `jamjet/injection-structural` with
`jamjet/pii` under `on_fail="fix"`, over a string carrying both an address and a
tag-character injection payload, returns `validation_passed=False` in one order
and `validation_passed=True` with the payload intact in the other.

`JamJetChain` has neither failure because it does not compose fixes at all. It
runs one chain, which hands every check the same original string, merges every
redaction in a single pass over that string, and combines decisions restrictively
so a later check cannot talk an earlier deny back down to a pass. The host gets
one `FailResult` with one `fix_value`, and the placeholder names every check that
claimed the region: `[REDACTED:CREDIT_CARD+SLACK_TOKEN]` rather than one of the
two.

The per-check validators exist for a caller composing a single check. If you want
two, use the composite.

## Which checks you can name

`checks` takes any check the installed `jamjet-guardrails` registers. That list
grows with the core library, so read it from the library rather than from this
page:

```python
from jamjet_guardrails_validators import VALIDATORS
sorted(VALIDATORS)
```

`VALIDATORS` is generated from `jamjet_guardrails.detectors.AVAILABLE` at import,
so a check added to the core is a validator here with no edit. A single check:

```python
from jamjet_guardrails_validators import validator_for

guard = Guard().use(validator_for("pii")(on_fail="fix"))
```

A check that takes options is configured through `options`, keyed by check name:

```python
JamJetChain(checks=["secrets"], options={"secrets": {"on_match": "deny"}}, on_fail="exception")
```

## Choosing `on_fail`

| Chain decision | `FailResult` carries | `on_fail="fix"` | `on_fail="exception"` |
|---|---|---|---|
| `allow` | nothing, it passes | value unchanged | value unchanged |
| `redact` | `fix_value` and the spans | the redacted value | raises |
| `deny` | the spans, no `fix_value` | `validation_passed` is False and `validated_output` is None | raises |

Measured on guardrails-ai 0.11.0: `on_fail="fix"` over a `FailResult` with no
`fix_value` does not raise and does not pass the original through. It sets
`validation_passed` to False and `validated_output` to None. That is the right
answer and a quiet one, so **put deny-class checks under
`on_fail="exception"`**. There is no fix for a deny, and a caller that reads
`validated_output` without reading `validation_passed` gets `None` rather than a
signal.

The chain's own content on a deny is never offered as a `fix_value`.
`GuardrailChain`'s docstring is explicit that on a deny that string exists for
the audit record and is not safe to send.

## What the failure carries

`error_message` names the validator, the decision and the finding types, for
example:

```
jamjet/chain[pii,secrets]: redact (pii:CREDIT_CARD, secrets:SLACK_TOKEN)
```

`error_spans` carries one `ErrorSpan` per located finding, with `start` and `end`
indexing the value that was validated and `reason` naming the detector and the
finding type.

Neither ever quotes the content. If a check itself fails, the message carries the
exception's type name and the SHA-256 of what was inspected, and not the
exception's message: an exception message routinely quotes the value that caused
it, and that value is the content.

## Direction and origin

Every validator takes `direction` (`"input"` or `"output"`, default `"output"`)
and `origin` (default `"model"`). They reach the check as its `Context`, they are
recorded on every verdict, and some checks decide differently per direction.
Validating a user's message rather than a model's reply means
`JamJetChain(checks=[...], direction="input", origin="user")`.

Any `metadata` a Guard passes is forwarded to the check as `Context.metadata`.

## Licence

Apache-2.0.
