# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from legal_agent.config.legal_agent_config import LegalAgentConfig
from legal_agent.config.legal_failure_policy import (
    LegalFailurePolicy,
    LegalFailureScenarioContract,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


def test_legal_failure_policy_product_default_is_frozen_snapshot() -> None:
    policy = LegalFailurePolicy.product_default()
    expected_keys = {
        "no_retrieval_hits",
        "nexus_layer_disabled_vs_plan",
        "organization_tool_plan_clamp",
        "optional_tool_plan_governance_hook",
        "nexus_tools_runtime_failure",
        "noop_or_empty_tools",
        "decision_low_confidence_no_early_exit",
        "decision_escalate_or_blocking_issues_no_early_exit",
        "policy_violations_no_early_exit",
        "legal_run_evaluator_llm_failure",
        "finalize_empty_llm_answer",
        "response_governance_hook",
    }
    assert set(LegalFailurePolicy.model_fields.keys()) == expected_keys
    for name in expected_keys:
        spec = getattr(policy, name)
        assert isinstance(spec, LegalFailureScenarioContract)
        assert spec.user_facing.strip()
        assert spec.pipeline.strip()
        assert spec.telemetry.strip()


def test_legal_agent_config_includes_failure_policy_by_default() -> None:
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
    )
    assert isinstance(cfg.failure_policy, LegalFailurePolicy)
    # Contract anchors (regression if copy drifts from implementation terminology)
    assert "no_hits" in cfg.failure_policy.no_retrieval_hits.pipeline
    assert "[ERROR] Empty legal finalize answer" in cfg.failure_policy.finalize_empty_llm_answer.user_facing
    assert "legal_loop_early_exit_min_confidence" in cfg.failure_policy.decision_low_confidence_no_early_exit.pipeline
    assert "LegalToolPlanGovernanceClampDiagV1" in cfg.failure_policy.organization_tool_plan_clamp.telemetry


def test_legal_failure_policy_round_trip_json_schema_stability() -> None:
    policy = LegalFailurePolicy.product_default()
    d = policy.model_dump()
    restored = LegalFailurePolicy.model_validate(d)
    assert restored == policy
