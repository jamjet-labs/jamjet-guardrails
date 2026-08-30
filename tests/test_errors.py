import pytest

from jamjet_guardrails.errors import GuardrailChainError, GuardrailUnavailableError


def test_unavailable_is_a_chain_error() -> None:
    """The hierarchy is the contract.

    Nothing raises GuardrailUnavailableError until the registry lands, but callers
    will catch the base class, and a missing guardrail must be caught by that catch
    rather than escaping as a bare Exception.
    """
    assert issubclass(GuardrailUnavailableError, GuardrailChainError)
    assert issubclass(GuardrailChainError, Exception)

    with pytest.raises(GuardrailChainError):
        raise GuardrailUnavailableError("pii not installed")
