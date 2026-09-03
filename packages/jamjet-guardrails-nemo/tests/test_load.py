"""The rails load, and a full `generate` round trip through the shipped flows.

The load tests read a real config folder through `RailsConfig.from_path`, so the
shipped `.co` files are parsed by Colang rather than by a regex here, and the
config folder's `config.py` is the same one line the README gives.

The round trips use `nemoguardrails.testing.TestChat`, a public, shipped fake
LLM (nemoguardrails 0.24.0, `nemoguardrails/testing/`), so the whole path runs
with no network and no model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from jamjet_guardrails import GuardrailUnavailableError
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.testing import TestChat

from conftest import REFUSAL_MESSAGE
from jamjet_guardrails_nemo import ACTION_INPUT, ACTION_OUTPUT, FLOW_INPUT, FLOW_OUTPUT

SECRET = "sk-abcdefghijklmnopqrstuvwxyz012345"
# A tag-character run, which `injection-structural` denies by default.
TAG_TEXT = "ignore previous" + "".join(chr(0xE0000 + ord(c)) for c in "hi")

BOTH_DIRECTIONS = """
models: []
rails:
  input:
    flows: [jamjet guardrails check input]
  output:
    flows: [jamjet guardrails check output]
custom_data:
  jamjet_guardrails:
    input: [injection-structural, secrets]
    output: [pii, secrets]
"""

Folder = Callable[[str], Path]


def test_both_flows_parse_and_both_actions_register(rails_folder: Folder) -> None:
    """The shipped `.co` files and the two action names, checked together.

    A flow that parses but names an action nobody registered fails at the first
    message with "Action not found", which NeMo turns into a bot reply. That
    reads to a user as an answer, not as a broken guardrail.
    """
    config = RailsConfig.from_path(str(rails_folder(BOTH_DIRECTIONS)))
    flow_ids = {flow["id"] for flow in config.flows if isinstance(flow, dict) and "id" in flow}
    assert {FLOW_INPUT, FLOW_OUTPUT} <= flow_ids
    app = LLMRails(config)
    registered = set(app.runtime.action_dispatcher.get_registered_actions())
    assert {ACTION_INPUT, ACTION_OUTPUT} <= registered


def test_a_denied_input_never_reaches_the_model(rails_folder: Folder) -> None:
    """End to end: the deny stops the turn, and the fake LLM is never called.

    The completion is scripted, so if the deny leaked the model would answer
    with it. Asserting `inference_count` is what separates "the refusal was
    returned" from "the refusal was returned after the model answered anyway".
    """
    config = RailsConfig.from_path(str(rails_folder(BOTH_DIRECTIONS)))
    chat = TestChat(config, llm_completions=["  I should not be reached"])
    reply = chat.app.generate(messages=[{"role": "user", "content": TAG_TEXT}])
    assert reply["content"] == REFUSAL_MESSAGE
    assert chat.llm.inference_count == 0


def test_a_redacted_input_is_what_the_model_is_prompted_with(rails_folder: Folder) -> None:
    """The measurement risk 5 was written to settle, as an assertion.

    `ActionResult.context_updates` IS the mechanism for replacing a message on
    nemoguardrails 0.24.0: the credential is absent from the prompt the model
    received and the placeholder is present. A redaction that did not reach the
    prompt would be a label on an audit record and nothing else.
    """
    config = RailsConfig.from_path(str(rails_folder(BOTH_DIRECTIONS)))
    chat = TestChat(config, llm_completions=["  Understood."])
    result = chat.app.generate(
        messages=[{"role": "user", "content": f"use {SECRET} please"}],
        options={"log": {"llm_calls": True}},
    )
    prompts = [str(call.prompt) for call in (result.log.llm_calls or [])]
    assert prompts, "no LLM call was made, so this proves nothing about the prompt"
    assert all(SECRET not in prompt for prompt in prompts)
    assert any("[REDACTED:OPENAI_KEY]" in prompt for prompt in prompts)


def test_a_redacted_output_is_what_generate_returns(rails_folder: Folder) -> None:
    """The other half: `bot_message` written back is the content the caller gets."""
    config = RailsConfig.from_path(str(rails_folder(BOTH_DIRECTIONS)))
    chat = TestChat(config, llm_completions=["  write to alice@example.com"])
    reply = chat.app.generate(messages=[{"role": "user", "content": "who do I mail"}])
    assert reply["content"] == "write to [REDACTED:EMAIL]"


def test_an_allowed_turn_is_untouched(rails_folder: Folder) -> None:
    config = RailsConfig.from_path(str(rails_folder(BOTH_DIRECTIONS)))
    chat = TestChat(config, llm_completions=["  Hello there"])
    reply = chat.app.generate(messages=[{"role": "user", "content": "hello"}])
    assert reply["content"] == "Hello there"


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        pytest.param(
            """
models: []
rails:
  input:
    flows: [jamjet guardrails check input]
""",
            "custom_data",
            id="no-block",
        ),
        pytest.param(
            """
models: []
rails:
  input:
    flows: [jamjet guardrails check input]
custom_data:
  jamjet_guardrails:
    input: [not-a-check]
""",
            "not available",
            id="unknown-check",
        ),
        pytest.param(
            """
models: []
custom_data:
  jamjet_guardrails:
    input: [pii]
""",
            "would be configured and never run",
            id="checks-without-the-flow",
        ),
        pytest.param(
            """
models: []
rails:
  input:
    flows: [jamjet guardrails check input]
  output:
    flows: [jamjet guardrails check output]
custom_data:
  jamjet_guardrails:
    input: [pii]
""",
            "would execute a check that does not exist",
            id="flow-without-the-checks",
        ),
    ],
)
def test_a_configuration_that_would_check_less_than_it_names_fails_the_load(
    rails_folder: Folder, yaml_text: str, message: str
) -> None:
    """Every one of these used to be discoverable only at the first user message.

    `init` runs inside `LLMRails.__init__`, and an exception raised there
    propagates out of the constructor unchanged (measured on nemoguardrails
    0.24.0). So a config that names a check nobody installed, or wires a flow to
    a direction with no checks, fails the process that loads it rather than the
    request that trips over it.
    """
    config = RailsConfig.from_path(str(rails_folder(yaml_text)))
    with pytest.raises(GuardrailUnavailableError, match=message):
        LLMRails(config)


def test_only_the_configured_direction_registers_an_action(rails_folder: Folder) -> None:
    """An action registered for a direction with no chain would answer every call
    with a deny, which looks like a working guardrail and is a broken one."""
    config = RailsConfig.from_path(
        str(
            rails_folder(
                """
models: []
rails:
  output:
    flows: [jamjet guardrails check output]
custom_data:
  jamjet_guardrails:
    output: [pii]
"""
            )
        )
    )
    app = LLMRails(config)
    registered = set(app.runtime.action_dispatcher.get_registered_actions())
    assert ACTION_OUTPUT in registered
    assert ACTION_INPUT not in registered
