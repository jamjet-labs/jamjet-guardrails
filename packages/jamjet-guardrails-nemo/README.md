# jamjet-guardrails-nemo

[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) input and output
rails backed by [jamjet-guardrails](https://github.com/jamjet-labs/jamjet-guardrails),
a zero-dependency library of deterministic content checks with published
precision and recall.

Two system actions, two Colang 1.0 flows, and one line in your config folder's
`config.py`. Every check that your config names is built when the rails load, so
a check that is not installed fails the load rather than the first message. Every
call writes an audit record naming which detector decided, at which version,
what it decided and where it found something, and never the content.

## Install

```
pip install jamjet-guardrails-nemo
```

Requires Python 3.10 to 3.13, the range `nemoguardrails` itself supports.

## Wire it up

Copy the two shipped flow files into your rails config folder:

```
cp "$(python -c 'import jamjet_guardrails_nemo as m; print(m.flows_path())')"/*.co my_config/
```

`my_config/config.py`:

```python
from jamjet_guardrails_nemo import init  # noqa: F401
```

`my_config/config.yml`:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows: [jamjet guardrails check input]
  output:
    flows: [jamjet guardrails check output]

custom_data:
  jamjet_guardrails:
    input: [injection-structural, secrets]
    output: [pii, secrets]
    options:
      secrets: {on_match: deny}
```

Your config also needs a refusal message, which is the same one NeMo's own rails
use. Put it in any `.co` file in the folder:

```
define bot refuse to respond
  "I am sorry, I cannot respond to that."
```

It is deliberately not shipped inside the flow files, so a config that already
defines one keeps its own wording.

## Which checks you can name

`input` and `output` take any check the installed `jamjet-guardrails` registers.
That list grows with the core library, so read it from the library rather than
from this page:

```
python -c "from jamjet_guardrails.detectors import AVAILABLE; print(sorted(AVAILABLE))"
```

The configuration above is an illustration, not the full set. `options` is per
check and is passed to that check's constructor, so `secrets: {on_match: deny}`
turns the secrets check from a redaction into a refusal. A check that takes no
options needs no entry.

## Configuration lives under `custom_data`

`RailsConfig` is a pydantic model with the default `extra="ignore"` behaviour.
Measured on nemoguardrails 0.24.0: a top-level `jamjet_guardrails:` key in
`config.yml` is parsed, discarded and unreachable from the loaded config, with
no error anywhere. `custom_data` is NeMo's own field for configuration it does
not model itself, so that is where the block goes. A config that puts it at the
top level instead fails the load with a message showing the right shape, because
a guardrail that silently did not run is the failure this library exists to
prevent.

## What the rails do

| Chain decision | Action returns | What happens |
|---|---|---|
| `allow` | true | the turn continues, message untouched |
| `redact` | true | the message is replaced with the redacted one, then the turn continues |
| `deny` | false | the flow runs `bot refuse to respond` and stops |
| `check` raised | false | same as deny; the audit record names the failure's type |

On a redact, the rewritten string is what the rest of the turn sees: the model is
prompted with it on input, and it is what `generate` returns on output. On a deny
the chain's merged content is never written back. `GuardrailChain`'s own
docstring is explicit that on a deny that string exists for the audit record and
is not safe to send.

Every check in a direction inspects the same content. No check ever sees a string
another check has already rewritten, and all the redactions are merged and
applied in one pass. That is a leak fix rather than a simplification: rewriting
in sequence lets one detector's placeholder split a credential so that the next
detector matches only a stump and reports success over a tail that is still
there. The core library's `GuardrailChain` documents the measured case.

## The audit record

Every call writes a JSON record to the rail context under `jamjet_guardrails`.
This is the record a real run produces for `mail alice@example.com please`
checked with `pii` on the output rail:

```json
{
  "decision": "redact",
  "direction": "output",
  "saw": "e0a1e7a20fda68f59025ebc1bca79c3aa6bb6e202edf48828b5db5b08fdc9dfe",
  "verdicts": [
    {
      "decision": "redact",
      "detector": "pii",
      "kind": "constraint",
      "spans": [[5, 22]],
      "types": ["EMAIL"],
      "version": "0.1.0"
    }
  ]
}
```

`saw` is the SHA-256 of exactly the string that was inspected. `types` and
`spans` are parallel by index. An `error` key appears where a check failed, and
carries the exception's type name only, never its message: an exception message
routinely quotes the value that caused it, and that value is the content.

The record never carries the content, the rewritten content, or any substring of
either. Read it with `jamjet_guardrails_nemo.parse_audit_record`.

The context key holds the record of the most recent call, so on a turn that runs
both rails the output record replaces the input one. Both are retained in NeMo's
own event log, and each record names its own `direction`.

## What this adapter does not do yet

Colang 2.0 flows. The shipped flows are Colang 1.0.

## Licence

Apache-2.0.

## The framework this adapter installs makes network calls

`jamjet-guardrails` says on its own front page: no dependencies, no network
calls, no model downloads. That is true of the core and it stops being true of
your environment the moment you install this adapter, so it is said here rather
than left for you to find.

Measured on `nemoguardrails` 0.24.0: it posts usage statistics to
`https://events.telemetry.data.nvidia.com/v1.1/events/json`, with a heartbeat,
and reads the destination from `NEMO_GUARDRAILS_USAGE_STATS_SERVER`. It is on by
default.

It turns itself off under several conditions, and two of them are yours to set:

```sh
export NEMO_GUARDRAILS_NO_USAGE_STATS=1   # this framework only
export DO_NOT_TRACK=1                     # the cross-vendor convention
```

It also suppresses itself when `CI` is truthy, when `PYTEST_CURRENT_TEST` is
set, when `pytest` is in `sys.modules`, and when a do-not-track file exists. That
is why this package's own test suite never reaches the network and why a green
CI run is not evidence about your deployment.

Nothing in this adapter sends anything anywhere. The audit record it writes goes
into the rail context and stays in your process.
