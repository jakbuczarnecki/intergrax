# © Artur Czarnecki. All rights reserved.

"""COST-1/2: Cost governance runtime bridge and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.cost_assembly_resolver import (
    CostAssemblyError,
    assert_cost_assembly_valid,
    validate_cost_wiring,
)
from intergrax.applications._shared.cost_runtime_bridge import (
    apply_cost_profile_to_runtime_config,
    resolve_cost_wiring_options,
)
from intergrax.applications._shared.cost_wiring import wire_application_cost
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CostProfile,
)
from intergrax.runtime.nexus.budget.budget_models import BudgetEnforcementMode, BudgetPolicy
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_cost_wiring_options_maps_profile_fields() -> None:
    profile = CostProfile(
        budget_enforcement_enabled=True,
        enforcement_mode="hitl",
        max_total_tokens=10_000,
        max_llm_calls=8,
        max_tool_calls=16,
        quota_degrade_threshold_ratio=0.85,
    )
    options = resolve_cost_wiring_options(profile)
    assert options.enforcement_mode == "hitl"
    assert options.max_total_tokens == 10_000
    assert options.max_llm_calls == 8
    assert options.quota_degrade_threshold_ratio == 0.85


def test_wire_application_cost_builds_budget_policy_and_run_budget() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cost.wire")
    wiring = wire_application_cost(env)
    assert isinstance(wiring.budget_policy, BudgetPolicy)
    assert wiring.run_budget is not None
    assert wiring.run_budget.max_llm_calls == 64
    assert wiring.run_budget.max_tool_calls == 128


def test_apply_cost_profile_to_runtime_config_sets_budget_fields() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter())
    profile = CostProfile(max_total_tokens=5_000, enforcement_mode="abort")
    updated = apply_cost_profile_to_runtime_config(config, profile)
    assert isinstance(updated.budget_policy, BudgetPolicy)
    assert updated.budget_policy.enforcement_mode == BudgetEnforcementMode.ABORT
    assert updated.run_budget is not None
    assert updated.run_budget.max_total_tokens == 5_000


def test_assert_cost_assembly_valid_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cost.valid")
    wiring = wire_application_cost(env)
    assert_cost_assembly_valid(wiring, env)


def test_validate_cost_wiring_requires_limits_when_enforcement_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cost.limits")
    env.cost_profile = CostProfile(
        budget_enforcement_enabled=True,
        max_total_tokens=None,
        max_llm_calls=None,
        max_tool_calls=None,
        max_planner_iterations=None,
    )
    env.context_profile = env.context_profile.model_copy(update={"budget_policy": None})
    wiring = wire_application_cost(env)
    result = validate_cost_wiring(wiring, env)
    assert not result.valid
    assert any("explicit cost limits or context budget_policy" in error for error in result.errors)


def test_validate_cost_wiring_rejects_budget_policy_when_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cost.reject")
    env.cost_profile = CostProfile(budget_enforcement_enabled=False)
    wiring = wire_application_cost(env)
    wiring = type(wiring)(
        profile=wiring.profile,
        options=wiring.options,
        budget_policy=BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT),
        run_budget=wiring.run_budget,
        domain_fragments=wiring.domain_fragments,
    )
    with pytest.raises(CostAssemblyError):
        assert_cost_assembly_valid(wiring, env)


def test_materialize_runtime_config_applies_cost_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cost.runtime")
    request = RuntimeRequest(
        message="hello",
        tenant_id="t1",
        agent_id="echo",
        user_id="user-1",
        session_id="session-1",
    )
    manifest = build_lab_manifest(LabApplicationSettings.from_env())
    build_ctx = ApplicationBuildContext.for_manifest(manifest, environment=env)
    config = materialize_runtime_config(request, build_ctx, env)
    assert config.budget_policy is not None
    assert config.run_budget is not None
    assert config.run_budget.max_llm_calls == 64


def test_build_harness_host_runtime_wires_cost_policy_bundle() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    assert runtime.cost.budget_policy is not None
    assert runtime.env_wiring.policy_bundle.budget == runtime.cost.budget_policy
    assert "cost_governance" in runtime.env_wiring.policy_bundle.domain_fragments
