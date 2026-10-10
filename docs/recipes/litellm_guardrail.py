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
        messages = copy.deepcopy(original) if original else None
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
                    checked = check(holder[key])
                    changed = changed or checked != holder[key]
                    holder[key] = checked
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
