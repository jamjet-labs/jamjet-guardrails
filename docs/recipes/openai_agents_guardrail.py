"""jamjet-guardrails in the request path of an OpenAI Agents SDK agent.

Three guardrails on one chain: the user's input, every tool output, and the
agent's final output. Verified against openai-agents 0.23.1.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolOutputGuardrailData,
    TResponseInputItem,
    function_tool,
    input_guardrail,
    output_guardrail,
    tool_output_guardrail,
)

from jamjet_guardrails import ChainResult, Context, build_chain

CHAIN = build_chain(["injection-structural", "pii", "secrets"])


def audit_record(result: ChainResult) -> dict[str, Any]:
    """What to log: which check, at which version, decided what about which text.

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


class Blocked(Exception):
    """The chain denied the content. `record` says which check and why."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(f"blocked by jamjet-guardrails: {record['decision']}")
        self.record = record


def screen_input(text: str) -> str:
    """Run before `Runner.run`: returns the text to send, redacted where needed.

    Input guardrails can stop a run but cannot change its input, so redaction
    has to happen here, before the run starts.
    """
    result = CHAIN.run(text, Context(direction="input", origin="user"))
    if result.decision == "deny":
        raise Blocked(audit_record(result))
    return result.content


def _texts(value: str | list[TResponseInputItem]) -> list[str]:
    """Every piece of text in an agent input, skipping nothing.

    Message text is read as text. Any other item, such as a tool output carried
    in a conversation history, is read as its JSON, with `ensure_ascii=False` so
    invisible characters stay characters rather than becoming `\\u` escapes the
    checks cannot see.
    """
    if isinstance(value, str):
        return [value]
    texts: list[str] = []
    for item in value:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list) and all(
            isinstance(part, dict) and isinstance(part.get("text"), str) for part in content
        ):
            texts.extend(part["text"] for part in content)
        else:
            texts.append(json.dumps(item, ensure_ascii=False, default=str))
    return texts


@input_guardrail(run_in_parallel=False)
async def jamjet_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Trips on anything but `allow`, because it cannot rewrite what it is given.

    `run_in_parallel=False` makes the SDK finish this check before the model is
    called. Its default runs the two side by side, so a tripped input may
    already have been sent.
    """
    results = [CHAIN.run(text, Context(direction="input", origin="user")) for text in _texts(input)]
    return GuardrailFunctionOutput(
        output_info=[audit_record(result) for result in results],
        tripwire_triggered=any(result.decision != "allow" for result in results),
    )


@tool_output_guardrail
def jamjet_tool_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """A tool's result on its way into the model, which is where a retrieved page arrives.

    The one place the SDK lets a guardrail replace content: `reject_content`
    gives the model a message in place of the tool's result, and on `redact`
    that message is the redacted result.
    """
    text = (
        data.output
        if isinstance(data.output, str)
        else json.dumps(data.output, ensure_ascii=False, default=str)
    )
    result = CHAIN.run(text, Context(direction="input", origin="tool"))
    record = audit_record(result)
    if result.decision == "deny":
        return ToolGuardrailFunctionOutput.raise_exception(output_info=record)
    if result.decision == "redact":
        return ToolGuardrailFunctionOutput.reject_content(
            message=result.content, output_info=record
        )
    return ToolGuardrailFunctionOutput.allow(output_info=record)


@output_guardrail
async def jamjet_output(
    ctx: RunContextWrapper[Any], agent: Agent[Any], output: Any
) -> GuardrailFunctionOutput:
    """Trips on anything but `allow`. On `redact` the redacted reply rides along
    in `output_info`, for `answer` below to send instead."""
    text = (
        output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
    )
    result = CHAIN.run(text, Context(direction="output", origin="model"))
    record = audit_record(result)
    if result.decision == "redact":
        record["redacted"] = result.content
    return GuardrailFunctionOutput(
        output_info=record, tripwire_triggered=result.decision != "allow"
    )


def guarded_agent(
    name: str, instructions: str, tools: list[Callable[..., Any]], model: Any = None
) -> Agent[Any]:
    """An agent with all three guardrails attached."""
    return Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=[function_tool(tool, tool_output_guardrails=[jamjet_tool_output]) for tool in tools],
        input_guardrails=[jamjet_input],
        output_guardrails=[jamjet_output],
    )


async def answer(agent: Agent[Any], text: str) -> str:
    """One turn: screen the input, run the agent, send the redacted reply on `redact`.

    A denied input raises `Blocked`. A denied reply or tool output re-raises the
    SDK's own tripwire exception, which carries the audit record.
    """
    try:
        result = await Runner.run(agent, screen_input(text))
    except OutputGuardrailTripwireTriggered as tripped:
        redacted = tripped.guardrail_result.output.output_info.get("redacted")
        if redacted is None:
            raise
        return str(redacted)
    return str(result.final_output)
