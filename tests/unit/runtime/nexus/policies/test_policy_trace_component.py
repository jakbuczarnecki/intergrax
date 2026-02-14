# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from tests._support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


def test_policy_trace_component_is_recorded() -> None:
    """
    Contract test:
    Runtime policies must be able to emit trace events using TraceComponent.POLICY.
    This is a foundational observability requirement for retry/timeout/HITL logic.
    """

    state = build_runtime_state_for_tests(run_id="test-run-policy-trace")

    state.trace_event(
        component=TraceComponent.POLICY,
        step="retry_attempt",
        level=TraceLevel.INFO,
        message="Retry attempt 1 due to transient failure",
    )

    assert len(state.trace_events) == 1

    event = state.trace_events[0]

    assert event.component == TraceComponent.POLICY
    assert event.step == "retry_attempt"
    assert event.level == TraceLevel.INFO
    assert "Retry attempt" in event.message
