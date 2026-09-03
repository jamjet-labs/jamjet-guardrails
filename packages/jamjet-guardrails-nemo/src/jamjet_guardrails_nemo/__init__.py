"""NeMo Guardrails input and output rails backed by jamjet-guardrails.

A rails config folder's `config.py` is one line:

    from jamjet_guardrails_nemo import init  # noqa: F401

nemoguardrails calls `init(app)` while `LLMRails` is being constructed, so
everything this adapter can refuse is refused at rails load, with content
nowhere in sight. See `init`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jamjet_guardrails import Direction, GuardrailChain, GuardrailUnavailableError

from jamjet_guardrails_nemo import _config
from jamjet_guardrails_nemo._audit import parse as parse_audit_record
from jamjet_guardrails_nemo.actions import (
    ACTION_INPUT,
    ACTION_OUTPUT,
    AUDIT_KEY,
    JamJetRails,
    MissingRailMessage,
    UnconfiguredDirection,
)

__version__ = "0.1.0"

# The Colang 1.0 flow names in the shipped `.co` files. A user lists these under
# `rails.input.flows` and `rails.output.flows`; `init` refuses a configuration
# where a flow and its direction's check list disagree, so the two constants and
# the two files cannot drift apart in silence.
FLOW_INPUT = "jamjet guardrails check input"
FLOW_OUTPUT = "jamjet guardrails check output"

_FLOW_FOR: Mapping[Direction, str] = {"input": FLOW_INPUT, "output": FLOW_OUTPUT}


def flows_path() -> Path:
    """The directory holding the shipped `.co` files.

    A rails config folder needs the flow definitions themselves, not a
    reference to them, so the install step is a copy:

        cp "$(python -c 'import jamjet_guardrails_nemo as m; print(m.flows_path())')"/*.co my_config/
    """
    return Path(__file__).resolve().parent / "flows"


def init(app: Any) -> JamJetRails:
    """Build both chains and register both actions, or fail the rails load.

    This is the whole of the adapter's failure posture. `build_chain` refuses a
    check that is named and not installed, an option table that selects nothing,
    and three shapes of malformed list; `_config.read` refuses a misspelled
    direction key and an option for a check no direction runs. All of it happens
    here, inside `LLMRails.__init__`, where an exception aborts the load.
    Measured on nemoguardrails 0.24.0: an exception raised from `init`
    propagates out of `LLMRails(config)` unchanged.

    The alternative is what a lazily built chain gives you, and it is worse than
    it sounds: the first user message is the one that discovers the config names
    a check that is not installed, and by then a request is in flight and
    somebody has to decide whether to fail it.

    **A flow and its direction must agree.** A direction that lists checks while
    its flow is not in `rails.<direction>.flows` is configured and silent: the
    action is registered, nothing executes it, and the config reads as though
    the checks run. A flow that is enabled for a direction that lists no checks
    is the mirror image: the flow executes an action that has no chain. Both are
    refused here, because both are a configuration that does not mean what it
    looks like, which is the one thing this library refuses everywhere else.
    """
    configured = _config.read(getattr(app.config, "custom_data", {}) or {})
    listed: Mapping[Direction, tuple[str, ...]] = {
        "input": configured.input,
        "output": configured.output,
    }
    chains: dict[Direction, GuardrailChain] = {}
    for direction, names in listed.items():
        flow = _FLOW_FOR[direction]
        enabled = flow in (getattr(app.config.rails, direction).flows or [])
        if names and not enabled:
            raise GuardrailUnavailableError(
                f"custom_data.{_config.CONFIG_KEY}.{direction} names {list(names)} but "
                f"rails.{direction}.flows does not include {flow!r}, so those checks would "
                f"be configured and never run"
            )
        if enabled and not names:
            raise GuardrailUnavailableError(
                f"rails.{direction}.flows includes {flow!r} but "
                f"custom_data.{_config.CONFIG_KEY}.{direction} names no checks, so the flow "
                f"would execute a check that does not exist"
            )
        if names:
            chains[direction] = _config.chain_for(names, configured.options)
    rails = JamJetRails(chains)
    if "input" in chains:
        app.register_action(rails.check_input, ACTION_INPUT)
    if "output" in chains:
        app.register_action(rails.check_output, ACTION_OUTPUT)
    return rails


__all__ = [
    "ACTION_INPUT",
    "ACTION_OUTPUT",
    "AUDIT_KEY",
    "FLOW_INPUT",
    "FLOW_OUTPUT",
    "JamJetRails",
    "MissingRailMessage",
    "UnconfiguredDirection",
    "__version__",
    "flows_path",
    "init",
    "parse_audit_record",
]
