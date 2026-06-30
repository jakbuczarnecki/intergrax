# © Artur Czarnecki. All rights reserved.

"""Regression: Tier-2 reference_harness strict mode after application-boundary refactor."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_config,
    build_lab_agent_runtime_context,
    lab_harness_context_from_modality_tooling,
)
from intergrax.runtime.modality.modality_profile import lab_default_modality_profile
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.runtime.wiring.harness_governance import LabAllowGovernanceService
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-strict",
        user_id="user-strict",
        session_id="session-strict",
        agent_id="echo",
        message="strict probe",
    )


def test_strict_harness_builds_production_context_with_governance(tmp_path: Path) -> None:
    trace_path = tmp_path / "strict_trace"
    harness = LabHarnessContext(
        policy_bundle=RuntimePolicyBundle(),
        strict_harness=True,
        trace_db_path=trace_path,
    )
    ctx = build_lab_agent_runtime_context(
        request=_request(),
        llm_adapter=FakeLLMAdapter(),
        harness=harness,
    )
    assert ctx.config.production_mode is True
    assert ctx.config.trace_db_path == str(trace_path)
    assert isinstance(ctx.governance_service, LabAllowGovernanceService)


def test_strict_harness_without_trace_db_raises_on_context_build() -> None:
    harness = LabHarnessContext(
        policy_bundle=RuntimePolicyBundle(),
        strict_harness=True,
        trace_db_path=None,
    )
    with pytest.raises(ValueError, match="trace_db_path must be set"):
        build_lab_agent_runtime_context(
            request=_request(),
            llm_adapter=FakeLLMAdapter(),
            harness=harness,
        )


def test_strict_harness_preserves_policy_bundle_and_tool_scope(tmp_path: Path) -> None:
    scope = StaticToolScopePolicy(allowed_tools={"websearch.query"})
    bundle = RuntimePolicyBundle(tool_access=scope)
    harness = LabHarnessContext(
        policy_bundle=bundle,
        strict_harness=True,
        trace_db_path=tmp_path / "strict_trace",
    )
    config = build_lab_agent_runtime_config(
        request=_request(),
        llm_adapter=FakeLLMAdapter(),
        harness=harness,
    )
    assert config.policy_bundle is bundle
    assert config.tool_scope_policy is scope


def test_lab_harness_context_from_modality_tooling_defaults_modality_when_strict() -> None:
    ctx = lab_harness_context_from_modality_tooling(
        policy_bundle=RuntimePolicyBundle(),
        strict_harness=True,
    )
    assert ctx.modality_profile == lab_default_modality_profile()
