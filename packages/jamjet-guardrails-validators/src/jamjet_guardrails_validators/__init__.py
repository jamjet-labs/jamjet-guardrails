"""Guardrails AI validators backed by jamjet-guardrails.

    from guardrails import Guard
    from jamjet_guardrails_validators import JamJetChain

    guard = Guard().use(JamJetChain(checks=["pii", "secrets"], on_fail="fix"))

`JamJetChain` is the one to reach for. The per-check validators in `VALIDATORS`
exist for a caller composing a single check; stacking two of them in one Guard
is a measured leak, and `JamJetChain`'s docstring records both shapes of it.
"""

from __future__ import annotations

from jamjet_guardrails_validators._validators import (
    VALIDATORS,
    JamJetChain,
    validator_for,
)

__version__ = "0.1.0"

__all__ = ["VALIDATORS", "JamJetChain", "__version__", "validator_for"]
