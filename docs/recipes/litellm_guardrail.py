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
