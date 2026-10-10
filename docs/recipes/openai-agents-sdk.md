# OpenAI Agents SDK

This recipe puts one jamjet-guardrails chain on the places content enters or
leaves an OpenAI Agents SDK agent: the user's input, the arguments the model
writes for each function tool and MCP tool, each of those tools' output, and
the agent's final output. A tag-smuggled instruction stops the run before the
model sees it, a credential or an email address in a tool result or a reply is
redacted before it goes any further, and a tool call whose arguments carry one
does not run.

Verified against `openai-agents` 0.23.1, on Python 3.12, by a CI job that runs
[tests/test_recipes.py](../../tests/test_recipes.py) against that release. The
tests drive the SDK's real `Runner` with a scripted model that records what it
was sent, so no network and no API key are involved.

```console
pip install jamjet-guardrails "openai-agents==0.23.1"
```

## The code

Copy the file into your project. `guarded_agent` builds an agent with every
guardrail attached, and `answer` runs one turn. It takes plain functions or
`FunctionTool` objects, such as those from `@function_tool` or
`agent.as_tool()`, and MCP servers through `mcp_servers`.

<!-- file: docs/recipes/openai_agents_guardrail.py -->
```python
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
        if agent.output_type in (None, str):
            return redacted
        try:
            return TypeAdapter(agent.output_type).validate_json(redacted)
        except ValidationError:
            raise tripped from None
    return result.final_output
```

## How decisions map

The SDK gives each kind of guardrail different powers, so the same decision
does different things in each place.

| Where | `allow` | `redact` | `deny` |
|---|---|---|---|
| Input (`jamjet_input`) | passes | trips the tripwire | trips the tripwire |
| Tool arguments (`jamjet_tool_input`) | the tool runs | the tool does not run; the model is told it was refused | raises `ToolInputGuardrailTripwireTriggered` |
| Tool output (`jamjet_tool_output`) | passes | the model gets the redacted result instead | raises `ToolOutputGuardrailTripwireTriggered` |
| Final output (`jamjet_output`) | passes | trips; `answer` returns the redacted reply | trips; `answer` re-raises |

**Input guardrails cannot rewrite.** `GuardrailFunctionOutput` has two fields,
`output_info` and `tripwire_triggered`, and nothing that replaces the input. So
`jamjet_input` trips on `redact` as well as on `deny`: passing the input on
unchanged would send the model the very text the chain said to redact. The
redaction happens before the run instead, in `screen_input`, which `answer`
calls first. Text that `screen_input` has redacted comes back as `allow` from
the input guardrail, so the two do not fight. The input guardrail stays as the
check on whatever reaches `Runner.run` without going through `screen_input`.

**Input guardrails run beside the model by default.** The SDK's default for
`input_guardrail` is `run_in_parallel=True`, which starts the model call while
the check is still running, so by the time a tripwire fires the model may
already have the input. The recipe sets `run_in_parallel=False`, and
`test_agents_input_guardrail_runs_before_the_model` holds it there.

**Tool outputs can be replaced.** A tool output guardrail can return
`reject_content(message=...)`, and the model receives that message in place of
the tool's result. On `redact` the message is the redacted result, so this is
the one place the SDK lets the chain rewrite content in flight. It matters
because a tool result is how a retrieved page reaches the model.

**Tool arguments cannot be rewritten.** A tool input guardrail can let the call
run, refuse it with a message to the model, or halt the run. So on `redact` the
tool does not run, and the model is told which checks fired but not what they
found. This is the direction data leaves in, a model writing a credential into
`send_email(body=...)`. It costs something: `pii` refuses a tool whose
arguments legitimately carry an email address, such as a `to` field. The
arguments are checked with their JSON escapes undone, because a model can
write a hidden character as a `\u` escape pair.

**Final outputs cannot be rewritten either.** An output guardrail can only trip.
On `redact` the redacted reply travels in `output_info`, and `answer` catches
`OutputGuardrailTripwireTriggered` and returns it, parsed back into the agent's
`output_type` when it has one. On `deny` there is no `redacted` key and
`answer` re-raises. A tripwire from any other output guardrail is re-raised
untouched.

**Structured values are checked as JSON, never as a repr.** A pydantic model
is turned into data with `model_dump(mode="json")` and a dataclass with
`dataclasses.asdict`, then written with `json.dumps(..., ensure_ascii=False)`.
A repr escapes a tag character into the six letters `\U000e0069`, which no
check reads as a tag character, so a value the recipe cannot turn into JSON
raises instead of being checked as its repr.

**MCP tools get the same tool guardrails.** `guard_mcp_server` adds them to an
MCP server's `tool_input_guardrails` and `tool_output_guardrails`, which the
SDK applies to every tool that server exposes. To build the server yourself,
pass them to its constructor instead:
`MCPServerStdio(params, tool_input_guardrails=[jamjet_tool_input], tool_output_guardrails=[jamjet_tool_output])`.
`test_agents_mcp_tool_outputs_are_checked` runs this through the SDK's MCP
path with an in-memory server.

## What the provenance gives you

`audit_record` turns a chain result into a JSON-serialisable record: the
combined decision, and for each check its name, its version, its own decision,
the spans it matched, and `saw`, the SHA-256 of the exact text it inspected.
Every guardrail puts that record in `output_info`, so it is on the exception
when a tripwire fires and on the run result when one does not. Log it and you
can say afterwards which check at which version made a decision about which
text, without having stored the text.

## Where this does not reach

The SDK decides where guardrails run, and its
[guardrails documentation](https://openai.github.io/openai-agents-python/guardrails/)
sets out the boundaries this recipe inherits:

- Input guardrails run only for the first agent in a run, and output guardrails
  only for the agent that produces the final output. Handoffs in between are
  not checked by either.
- Tool guardrails run only on function tools and MCP tools. Hosted tools such
  as `WebSearchTool` and `FileSearchTool` do not go through them, so a page a
  hosted search tool retrieves reaches the model unchecked by this recipe.
  `guarded_agent` refuses a hosted tool with a `TypeError` rather than attach
  it unguarded.
- Streamed replies are not protected by the output guardrail. `answer` uses
  `Runner.run`. With `Runner.run_streamed`, the reply reaches your stream as
  it is generated, and the output guardrail runs on the final output after
  that, so a `redact` or `deny` on the reply fires when the client already has
  the text. The streamed path is not tested here.

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
- `pii` false positives cost more here than in the quickstart, because on the
  input side a `redact` trips the run when the input did not go through
  `screen_input`.

To add a check, change the list passed to `build_chain`. The README lists them
all, with what each one costs in false positives.
