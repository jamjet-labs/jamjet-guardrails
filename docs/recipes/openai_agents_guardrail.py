"""jamjet-guardrails in the request path of an OpenAI Agents SDK agent.

Four guardrails on one chain: the user's input, the arguments the model writes
for each tool, every tool output, and the agent's final output. Verified
against openai-agents 0.23.1.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable, Sequence
from typing import Any

from agents import (
    Agent,
    AgentOutputSchemaBase,
    FunctionTool,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    TResponseInputItem,
    function_tool,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.mcp import MCPServer
from pydantic import BaseModel, TypeAdapter, ValidationError

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


def _plain(value: Any) -> Any:
    """`value` as JSON-ready data, with every string exactly as it was."""
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # A repr escapes invisible characters into text no check can see, so
    # anything else is refused rather than checked as its repr.
    raise TypeError(f"cannot check a {type(value).__name__} without losing characters")


def as_text(value: Any) -> str:
    """The text a check reads: a string as itself, anything else as its JSON.

    `ensure_ascii=False` keeps invisible characters as characters rather than
    `\\u` escapes the checks cannot see.
    """
    return value if isinstance(value, str) else json.dumps(_plain(value), ensure_ascii=False)


def decoded(arguments: str) -> str:
    """Tool arguments with their JSON escapes undone.

    A model may write a hidden character as `\\udb40\\udc69` or a letter as
    `\\u0073`, and the checks read characters, not escapes.
    """
    try:
        return as_text(json.loads(arguments))
    except ValueError:
        return arguments


def _texts(value: str | list[TResponseInputItem]) -> list[str]:
    """Every piece of text in an agent input, skipping nothing.

    Message text is read as text. Any other item, such as a tool output carried
    in a conversation history, is read as its JSON.
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
            texts.append(as_text(item))
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


@tool_input_guardrail
def jamjet_tool_input(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """The arguments the model wrote for a tool, before the tool runs.

    This is the direction data leaves in: a model that puts a credential into
    `send_email(body=...)`. The SDK cannot change the arguments, so on `redact`
    the tool does not run and the model is told why, without the content.
    """
    arguments = decoded(data.context.tool_arguments)
    result = CHAIN.run(arguments, Context(direction="output", origin="model"))
    record = audit_record(result)
    if result.decision == "deny":
        return ToolGuardrailFunctionOutput.raise_exception(output_info=record)
    if result.decision == "redact":
        fired = sorted(c["detector"] for c in record["checks"] if c["decision"] == "redact")
        return ToolGuardrailFunctionOutput.reject_content(
            message=f"refused by jamjet-guardrails: the arguments carried {', '.join(fired)}",
            output_info=record,
        )
    return ToolGuardrailFunctionOutput.allow(output_info=record)


@tool_output_guardrail
def jamjet_tool_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """A tool's result on its way into the model, which is where a retrieved page arrives.

    The one place the SDK lets a guardrail replace content: `reject_content`
    gives the model a message in place of the tool's result, and on `redact`
    that message is the redacted result.
    """
    result = CHAIN.run(as_text(data.output), Context(direction="input", origin="tool"))
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
    result = CHAIN.run(as_text(output), Context(direction="output", origin="model"))
    record = audit_record(result)
    if result.decision == "redact":
        record["redacted"] = result.content
    return GuardrailFunctionOutput(
        output_info=record, tripwire_triggered=result.decision != "allow"
    )


def guard_tool(tool: Callable[..., Any] | FunctionTool) -> FunctionTool:
    """A function tool with the tool guardrails attached.

    A plain function is decorated here. An existing `FunctionTool`, such as one
    from `@function_tool` or `agent.as_tool()`, is copied with ours added to its
    own guardrails, so the caller's object is left as it was.
    """
    if isinstance(tool, FunctionTool):
        guarded = copy.copy(tool)
        guarded.tool_input_guardrails = [*(tool.tool_input_guardrails or []), jamjet_tool_input]
        guarded.tool_output_guardrails = [*(tool.tool_output_guardrails or []), jamjet_tool_output]
        return guarded
    if not callable(tool) or not hasattr(tool, "__name__"):
        raise TypeError(
            f"{type(tool).__name__} cannot be guarded: the SDK runs tool guardrails on "
            "function tools only, so pass a function, a FunctionTool or an MCP server"
        )
    return function_tool(
        tool,
        tool_input_guardrails=[jamjet_tool_input],
        tool_output_guardrails=[jamjet_tool_output],
    )


def guard_mcp_server(server: MCPServer) -> MCPServer:
    """An MCP server whose every tool gets the tool guardrails.

    The same as passing `tool_input_guardrails` and `tool_output_guardrails`
    to the server's constructor, for a server built elsewhere.
    """
    server.tool_input_guardrails = [*(server.tool_input_guardrails or []), jamjet_tool_input]
    server.tool_output_guardrails = [*(server.tool_output_guardrails or []), jamjet_tool_output]
    return server


def guarded_agent(
    name: str,
    instructions: str,
    tools: Sequence[Callable[..., Any] | FunctionTool],
    model: Any = None,
    output_type: Any = None,
    mcp_servers: Sequence[MCPServer] = (),
) -> Agent[Any]:
    """An agent with every guardrail attached."""
    return Agent(
        name=name,
        instructions=instructions,
        model=model,
        output_type=output_type,
        tools=[guard_tool(tool) for tool in tools],
        mcp_servers=[guard_mcp_server(server) for server in mcp_servers],
        input_guardrails=[jamjet_input],
        output_guardrails=[jamjet_output],
    )


async def answer(agent: Agent[Any], text: str) -> Any:
    """One turn: screen the input, run the agent, send the redacted reply on `redact`.

    A denied input raises `Blocked`. A denied reply or tool output re-raises the
    SDK's own tripwire exception, which carries the audit record. With an
    `output_type`, a redacted reply comes back as that type.
    """
    try:
        result = await Runner.run(agent, screen_input(text))
    except OutputGuardrailTripwireTriggered as tripped:
        if tripped.guardrail_result.guardrail is not jamjet_output:
            raise
        redacted = tripped.guardrail_result.output.output_info.get("redacted")
        if redacted is None:
            raise
        output_type = agent.output_type
        if output_type is None or output_type is str:
            return redacted
        if isinstance(output_type, AgentOutputSchemaBase):
            # A schema object decides its own parsing; the reply stays on the tripwire.
            raise
        try:
            return TypeAdapter(output_type).validate_json(redacted)
        except ValidationError:
            raise tripped from None
    return result.final_output
