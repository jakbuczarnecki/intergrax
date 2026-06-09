# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.resilience_policy import (
    FailureClass,
    FailureResponse,
    ResiliencePolicy,
)
from intergrax.runtime.nexus.retry.retry_engine import _retry_decision_from_resilience_policy
from intergrax.runtime.resilience.policy_resolver import resolve_failure_action

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_quality_failure_retries_with_alternate_agent() -> None:
    decision = _retry_decision_from_resilience_policy(
        policy=ResiliencePolicy(max_attempts=3),
        attempt=0,
        alternate_agent_id="alt_agent",
    )
    assert decision.should_retry is True
    assert decision.alternate_agent_id == "alt_agent"


def test_max_attempts_escalates() -> None:
    resolution = resolve_failure_action(
        FailureClass.QUALITY_ERROR,
        policy=ResiliencePolicy(max_attempts=1),
        attempt=1,
    )
    assert resolution.response is FailureResponse.ESCALATE
