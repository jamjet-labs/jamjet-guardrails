# LiteLLM proxy

This recipe runs a jamjet-guardrails chain inside a LiteLLM proxy, on every
request before the model sees it and on every reply before the client does. A
tag-smuggled instruction is refused with a 400, and a credential or an email
address is redacted in place in the request or the reply.

Verified against `litellm` 1.104.2, on Python 3.12, by a CI job that runs
[tests/test_recipes.py](../../tests/test_recipes.py) against that release. The
tests call the proxy's own dispatch for this kind of guardrail and load the
config below through the proxy's own initialiser, with no network and no API
key.

```console
pip install jamjet-guardrails "litellm[proxy]==1.104.2"
```

## The code

LiteLLM's [custom guardrail](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail)
interface asks for one method, `apply_guardrail`. The proxy pulls the text out
of a request or a reply, passes it in as `inputs["texts"]`, and maps whatever
comes back onto the original request or reply.

<!-- file: docs/recipes/litellm_guardrail.py -->
```python
"""jamjet-guardrails as a LiteLLM proxy guardrail.

One class for both directions: LiteLLM calls `apply_guardrail` with the texts
of each request before the model sees them, and with the texts of each reply
before the client does. Verified against litellm 1.104.2.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.utils import GenericGuardrailAPIInputs

from jamjet_guardrails import ChainResult, Context, build_chain

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

AUDIT_LOG = logging.getLogger("jamjet_guardrails.litellm")

DEFAULT_CHECKS = ("injection-structural", "pii", "secrets")


def audit_record(result: ChainResult) -> dict[str, Any]:
    """Which check, at which version, decided what about which text.

    `saw` is the SHA-256 of the exact string each check inspected, so the record
    ties a decision to its text without storing the text.
    """
    return {
        "decision": result.decision,
        "checks": [
            {
                "detector": verdict.provenance.detector,
                "version": verdict.provenance.version,
                "decision": verdict.decision,
                "saw": verdict.saw,
                "findings": [{"type": f.type, "span": f.span} for f in verdict.findings],
            }
            for verdict in result.verdicts
        ],
    }


class JamjetGuardrail(CustomGuardrail):
    def __init__(self, jamjet_checks: Any = DEFAULT_CHECKS, **kwargs: Any) -> None:
        # litellm 1.104.2 catches ValueError and TypeError from a guardrail's
        # constructor, logs them, and starts the proxy WITHOUT that guardrail.
        # A typo in config.yaml must stop the proxy instead, so both are
        # re-raised as an error it does not catch.
        try:
            if isinstance(jamjet_checks, str) or not jamjet_checks:
                raise ValueError(f"jamjet_checks must be a non-empty list, got {jamjet_checks!r}")
            self.chain = build_chain(list(jamjet_checks))
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"jamjet-guardrails is misconfigured: {error}") from error
        super().__init__(**kwargs)

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type == "request":
            context = Context(direction="input", origin="user")
        else:
            context = Context(direction="output", origin="model")
        texts = list(inputs.get("texts", []))
        denied: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            result = self.chain.run(text, context)
            record = audit_record(result)
            AUDIT_LOG.info(json.dumps(record))
            if result.decision == "deny":
                denied.append(record)
            else:
                texts[index] = result.content
        if denied:
            # Names what fired, never the content: the content is what was denied.
            fired = sorted(
                {c["detector"] for r in denied for c in r["checks"] if c["decision"] == "deny"}
            )
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"denied by jamjet-guardrails ({', '.join(fired)})",
                status_code=400,
                blocked_content=True,
            )
        inputs["texts"] = texts
        return inputs
```

Save it as `litellm_guardrail.py` beside the proxy's config:

<!-- file: docs/recipes/litellm_config.yaml -->
```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: jamjet
    litellm_params:
      # litellm_guardrail.py sits beside this file.
      guardrail: litellm_guardrail.JamjetGuardrail
      mode: [pre_call, post_call]
      default_on: true
      jamjet_checks: [injection-structural, pii, secrets]
```

```console
litellm --config litellm_config.yaml
```

## How decisions map

| Decision | On a request (`pre_call`) | On a reply (`post_call`) |
|---|---|---|
| `allow` | the text is unchanged | the text is unchanged |
| `redact` | the message is rewritten before the model sees it | the reply is rewritten before the client sees it |
| `deny` | the request is refused with status 400 | the reply is refused with status 400 |

LiteLLM allows rewriting in both directions, so `redact` here does what it says.
A `deny` raises `GuardrailRaisedException` with `blocked_content=True`, which is
the type LiteLLM itself counts as a guardrail block rather than a guardrail
failure. Its message names the checks that denied and never quotes the content,
because the content is what was denied.

Each text is checked on its own, and the texts of a chat request include the
system prompt and every earlier turn. So an earlier message the chain would
deny is denied again on every later request in that conversation.

**A misconfigured guardrail must stop the proxy.** LiteLLM 1.104.2 catches a
`ValueError` or a `TypeError` raised while it builds a guardrail from the
config, logs it, and starts the proxy without that guardrail.
`test_litellm_starts_without_a_guardrail_whose_constructor_raises_value_error`
recomputes that against the pinned release. So the constructor turns both into
a `RuntimeError`, which LiteLLM does not catch, and a typo in `jamjet_checks`
is a proxy that fails to start rather than one running with no guardrail. The
option is called `jamjet_checks` and not `checks` because `checks` is already a
LiteLLM guardrail option with its own type, and a list there fails LiteLLM's
validation, which is the same silent skip.

Only `pre_call` and `post_call` are configured. LiteLLM's documentation
describes `during_call`, which runs beside the model call, as able to block but
not to rewrite.

## What the provenance gives you

Every text the guardrail checks produces one JSON line on the
`jamjet_guardrails.litellm` logger at INFO: the combined decision, and for each
check its name, its version, its own decision, the spans it matched, and `saw`,
the SHA-256 of the exact text it inspected. The record holds no content, so it
can go to an ordinary log. Send that logger wherever your proxy's logs go and
you can say afterwards which check at which version made a decision about which
text.

## Where this does not reach

- Streaming. LiteLLM's documentation says `post_call` guardrails on a streaming
  response are audit-only: they see the whole reply but cannot stop its
  delivery. The streamed path is not tested here.
- Only `inputs["texts"]`. Images and the arguments of tool calls the model
  makes arrive in other fields, and this recipe does not read them.

## What the checks do not catch

The recipe runs `injection-structural`, `pii` and `secrets`, the same three as
the README's quickstart.

- None of them reads meaning. "Ignore all previous instructions" written in
  plain visible words passes all three, and
  `test_a_plainly_worded_injection_passes_the_recipe_checks` keeps that
  sentence true. `injection-structural` catches instructions hidden in the
  encoding: tag characters, bidirectional controls and zero-width runs.
- Each check has published misses. The precision, recall and wrong-decision
  counts are in the "Measured, not asserted" table in the
  [README](../../README.md), and every disclosed miss is named by case id in
  [corpora/NOTICE.md](../../corpora/NOTICE.md). The README also names the
  credential prefixes `secrets` does not match.

To add a check, add its name to `jamjet_checks` in the config. The README lists
them all, with what each one costs in false positives.
