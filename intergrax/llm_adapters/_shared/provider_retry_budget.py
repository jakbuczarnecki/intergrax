# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional injectable provider retry budget port (W2-C)."""

from __future__ import annotations

from intergrax.contracts.retry_budget import RetryBudgetPort

_port: RetryBudgetPort | None = None


def set_llm_provider_retry_budget_port(port: RetryBudgetPort | None) -> None:
    global _port
    if port is not None and not isinstance(port, RetryBudgetPort):
        raise TypeError("port must implement RetryBudgetPort or be None")
    _port = port


def get_llm_provider_retry_budget_port() -> RetryBudgetPort | None:
    return _port
