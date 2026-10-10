# LiteLLM proxy

This recipe runs a jamjet-guardrails chain inside a LiteLLM proxy, on requests
to the chat completions, `/v1/messages` and `/v1/responses` endpoints before the
model sees them, and on their non-streamed replies before the client does. A
tag-smuggled instruction is refused with a 400, and a credential or an email
address is redacted in place in the request or the reply. The endpoints and
fields it does not reach are listed under "Where this does not reach".

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
comes back onto the original request or reply. On a request it also passes
`inputs["structured_messages"]`, the conversation as chat messages, and that is
the only place some content appears: on `/v1/responses`, tool outputs
(`function_call_output`) and `instructions` are not in `texts`. So the recipe
checks both, and writes redactions back through `structured_messages`.

<!-- file: docs/recipes/litellm_guardrail.py -->
```python
"""jamjet-guardrails as a LiteLLM proxy guardrail.

One class for both directions: LiteLLM calls `apply_guardrail` with the content
of a request before the model sees it, and with the text of a reply before the
client does. Verified against litellm 1.104.2.
"""

from __future__ import annotations

import copy
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


def decoded(arguments: str) -> str:
    """Tool call arguments with their JSON escapes undone.

    A model may write a hidden character as `\\udb40\\udc69` or a letter as
    `\\u0073`, and the checks read characters, not escapes.
    """
    try:
        value = json.loads(arguments)
    except ValueError:
        return arguments
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class JamjetGuardrail(CustomGuardrail):
    def __init__(
        self, jamjet_checks: Any = DEFAULT_CHECKS, allow_streaming: Any = False, **kwargs: Any
    ) -> None:
        # litellm 1.104.2 catches ValueError and TypeError from a guardrail's
        # constructor, logs them, and starts the proxy WITHOUT that guardrail.
        # A typo in config.yaml must stop the proxy instead, so both are
        # re-raised as an error it does not catch.
        try:
            if isinstance(jamjet_checks, str) or not jamjet_checks:
                raise ValueError(f"jamjet_checks must be a non-empty list, got {jamjet_checks!r}")
            if not isinstance(allow_streaming, bool):
                raise TypeError(f"allow_streaming must be true or false, got {allow_streaming!r}")
            self.chain = build_chain(list(jamjet_checks))
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"jamjet-guardrails is misconfigured: {error}") from error
        self.allow_streaming = allow_streaming
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # litellm 1.104.2 routes a guardrail to `apply_guardrail` only when the
        # method is defined on the guardrail's own class. A subclass that only
        # inherits it is dispatched to the default hooks, which do nothing, so
        # the proxy would run with no checks and no error.
        super().__init_subclass__(**kwargs)
        if "apply_guardrail" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define apply_guardrail itself: litellm only calls "
                "apply_guardrail when it is defined on the class, not inherited. Re-declare "
                "it and call super().apply_guardrail(...)."
            )

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type == "request":
            if request_data.get("stream") and not self.allow_streaming:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(
                        "jamjet-guardrails refuses stream: true, because reply checks cannot "
                        "rewrite or stop a streamed reply; send the request without streaming"
                    ),
                    status_code=400,
                    blocked_content=True,
                )
            context = Context(direction="input", origin="user")
        else:
            context = Context(direction="output", origin="model")
        results: dict[str, ChainResult] = {}

        def check(text: str) -> str:
            # The same text can arrive twice, as a text and inside a structured
            # message, so each distinct text is checked and logged once.
            if text not in results:
                results[text] = self.chain.run(text, context)
                AUDIT_LOG.info(json.dumps(audit_record(results[text])))
            return results[text].content

        texts = [check(text) for text in inputs.get("texts", [])]
        # Responses API tool outputs (`function_call_output`) and `instructions`
        # reach the model through `structured_messages` only, never `texts`.
        original = inputs.get("structured_messages")
        messages: list[Any] | None = copy.deepcopy(list(original)) if original else None
        changed = False
        for message in messages or []:
            fields: list[tuple[Any, Any]] = []
            content = message.get("content")
            if isinstance(content, str):
                fields.append((message, "content"))
            elif isinstance(content, list):
                fields += [(p, "text") for p in content if isinstance(p, dict)]
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    fields.append((call["function"], "arguments"))
            for holder, key in fields:
                if isinstance(holder.get(key), str):
                    text = decoded(holder[key]) if key == "arguments" else holder[key]
                    checked = check(text)
                    if checked != text:
                        holder[key] = checked
                        changed = True
        denied = [audit_record(r) for r in results.values() if r.decision == "deny"]
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
        if changed and messages is not None:
            # A new list tells LiteLLM to write these messages back; the same
            # object tells it nothing changed.
            inputs["structured_messages"] = messages
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

| Decision | On a request (`pre_call`) | On a reply (`post_call`), not streamed | On a streamed reply, with `allow_streaming: true` |
|---|---|---|---|
| `allow` | the text is unchanged | the text is unchanged | the text is unchanged |
| `redact` | the message is rewritten before the model sees it | the reply is rewritten before the client sees it | logged only; the client already has the text |
| `deny` | the request is refused with status 400 | the reply is refused with status 400 | logged only; the client already has the text |

LiteLLM allows rewriting in both directions for a reply that is not streamed,
so `redact` there does what it says.
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

**Streaming is refused unless you allow it.** It is the client that chooses to
stream, by sending `stream: true`, and a streamed reply is one the reply checks
cannot rewrite or stop. So by default the recipe refuses such a request on the
request pass, with a 400 that says why.
`test_litellm_a_streamed_reply_reaches_the_client_unredacted` recomputes the
reason against the pinned release: with streaming allowed, a credential in a
streamed reply reaches the client as the model wrote it. Set
`allow_streaming: true` beside `jamjet_checks` in the config to accept streamed
requests anyway. Then the reply checks on a streamed reply are audit-only: they
log a decision about text the client already has.

**A subclass must define `apply_guardrail` itself.** LiteLLM 1.104.2 sends a
guardrail through `apply_guardrail` only when the method is defined on the
guardrail's own class, so a subclass that inherits it runs with no checks and
no error. The recipe's `__init_subclass__` refuses such a subclass with a
`TypeError`. Re-declare the method and call `super().apply_guardrail(...)`.

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

These endpoints have no guardrail translation in litellm 1.104.2, so the
guardrail is never given their content:

- Assistants threads (`/v1/threads/{id}/messages`).
- `/v1/realtime`.
- Image edits (`/v1/images/edits`).
- Moderations (`/v1/moderations`).

On the endpoints it does cover:

- A `/v1/responses` request made of tool outputs only is never checked. That is
  the usual shape of a tool loop that uses `previous_response_id`: the client
  sends just the `function_call_output` items. LiteLLM returns before calling
  any guardrail when a Responses request carries no message text, so the tool
  output reaches the model unchecked. A tool output sent beside a message is
  checked.
- Content held by OpenAI under `previous_response_id` is never in the request,
  so it is not checked here either.
- Tool definitions are not checked. A smuggled `tools[].description` passes.
- The tool calls in a reply, the arguments the model writes for a tool, are not
  checked. Neither are images.
- If LiteLLM cannot tell which kind of call a reply belongs to, it logs a
  warning and skips `post_call` for it.
- An admin can switch this guardrail off for a key or a team, even with
  `default_on: true`, by setting `disable_global_guardrails: true` or listing
  `jamjet` in `opted_out_global_guardrails` in that key's or team's metadata.
  A caller cannot do this from the request body.
- Base64 sent as text is checked as text. A data URL inside a tool output or a
  message string can contain a run that `secrets` reads as a credential, such
  as one starting `AKIA`, and redacting it corrupts the image. Send images as
  image parts (`image_url` or `input_image`), which this recipe does not read
  as text, rather than as strings inside a message or a tool output. Do not
  exempt data URLs from the checks instead: an exempted field is one an
  attacker can write text into.

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
