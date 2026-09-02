"""Errors that are part of the contract."""

from __future__ import annotations


class GuardrailChainError(Exception):
    """Base for chain construction and execution problems."""


class GuardrailUnavailableError(GuardrailChainError):
    """A guardrail is configured and would not check.

    Raised at construction, most often when a guardrail is not installed,
    cannot be built, or would be skipped in every context. Also raised by
    ``PatternGuardrail.check`` when called with a ``Context`` whose direction
    it does not declare. Both are the same mistake: a configuration that
    cannot possibly inspect the content, so answering allow would report a
    check that did not run. A configuration that silently means "this check
    is not running" is the failure this library exists to prevent.
    """
