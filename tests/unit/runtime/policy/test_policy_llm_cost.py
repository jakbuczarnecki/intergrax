# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.replay.policy import PolicyDecisionType

pytestmark = pytest.mark.gate


def test_evaluate_llm_cost_allow_when_no_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = PolicyEngine()
    _cost, decision = engine.evaluate_llm_cost_on_task_completed(
        tenant_id="t1",
        run_id="r1",
    )
    assert decision.decision == PolicyDecisionType.ALLOW
