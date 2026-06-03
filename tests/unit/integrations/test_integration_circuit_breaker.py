# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations._shared.circuit_breaker import (
    CircuitState,
    IntegrationCircuitBreaker,
    IntegrationCircuitBreakerConfig,
)
from intergrax.integrations.contracts.base import IntegrationDependencyError

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_circuit_opens_after_failure_threshold() -> None:
    breaker = IntegrationCircuitBreaker(
        "test-backend",
        IntegrationCircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=60.0),
    )
    calls = 0

    def failing() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("backend down")

    with pytest.raises(IntegrationDependencyError):
        breaker.call(failing)
    assert breaker.state == CircuitState.CLOSED

    with pytest.raises(IntegrationDependencyError):
        breaker.call(failing)
    assert breaker.state == CircuitState.OPEN
    assert calls == 2

    with pytest.raises(IntegrationDependencyError, match="circuit"):
        breaker.call(failing)
    assert calls == 2


def test_circuit_resets_after_success() -> None:
    breaker = IntegrationCircuitBreaker("ok-backend")
    flag = {"fail_once": True}

    def flaky() -> str:
        if flag["fail_once"]:
            flag["fail_once"] = False
            raise RuntimeError("transient")
        return "ok"

    with pytest.raises(IntegrationDependencyError):
        breaker.call(flaky)
    assert breaker.call(flaky) == "ok"
    assert breaker.state == CircuitState.CLOSED
