"""Content guardrails for LLM applications, with provenance on every decision.

Every bundled check is a constraint: a pattern with published precision and
recall, not a model. Which check decided, and over exactly what text, is on
every verdict.
"""

__version__ = "0.3.0"

from jamjet_guardrails.authoring import Limits, PatternGuardrail
from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors import build, build_chain
from jamjet_guardrails.errors import GuardrailChainError, GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import (
    ChainResult,
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Origin,
    Provenance,
    Verdict,
    combine,
)

# Sorted, and checked to be sorted, so a later addition has exactly one correct
# place. Under mypy's no_implicit_reexport a name imported above but missing
# here is invisible to consumers while looking present to anyone reading the
# module, which is the half-wired state tests/test_registry.py checks both
# directions of.
#
# The detector classes are deliberately absent: `build("pii")` is the front
# door, and a name that resolves through the registry is a name the registry can
# refuse. They remain importable from jamjet_guardrails.detectors.
#
# `PatternGuardrail` and `Limits` are here for the opposite reason. They are not
# a bundled check; they are how a caller BUILDS one, at runtime or in a
# contribution, and there is no registry name for a check that does not exist
# yet. `build("rules", ...)` is the registered instantiation of the same class.
__all__ = [
    "ChainResult",
    "Context",
    "Decision",
    "Direction",
    "Finding",
    "Guardrail",
    "GuardrailChain",
    "GuardrailChainError",
    "GuardrailUnavailableError",
    "Kind",
    "Limits",
    "Origin",
    "PatternGuardrail",
    "Provenance",
    "Verdict",
    "build",
    "build_chain",
    "combine",
    "saw",
]
