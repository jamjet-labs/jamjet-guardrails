"""Errors that are part of the contract."""

from __future__ import annotations


class GuardrailChainError(Exception):
    """Base for chain construction and execution problems."""


class GuardrailUnavailableError(GuardrailChainError):
    """A guardrail was named in configuration but is not installed.

    Raised at construction and never swallowed. A configuration that silently
    means "this check is not running" is the failure this library exists to
    prevent.
    """
