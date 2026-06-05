# © Artur Czarnecki. All rights reserved.

"""EVAL-1/2: Evaluation runtime bridge and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.evaluation_assembly_resolver import (
    EvaluationAssemblyError,
    assert_evaluation_assembly_valid,
    validate_evaluation_wiring,
)
from intergrax.applications._shared.evaluation_runtime_bridge import (
    apply_evaluation_profile_to_runtime_config,
    resolve_evaluation_wiring_options,
)
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    EvaluationProfile,
)
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_evaluation_wiring_options_maps_profile_fields() -> None:
    profile = EvaluationProfile(
        shadow_eval_enabled=True,
        online_registry_enabled=True,
        offline_eval_runner_enabled=True,
        trend_comparison_enabled=False,
        require_baseline_for_release=True,
        evaluation_assets_ref="assets/v1",
    )
    options = resolve_evaluation_wiring_options(profile)
    assert options.offline_eval_runner_enabled is True
    assert options.trend_comparison_enabled is False
    assert options.evaluation_assets_ref == "assets/v1"


def test_wire_application_evaluation_builds_registry_and_bridge() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.wire")
    wiring = wire_application_evaluation(env)
    assert wiring.registry is not None
    assert wiring.governance_bridge is not None
    assert wiring.domain_fragments["evaluation_governance"]["shadow_eval_enabled"] is True


def test_apply_evaluation_profile_to_runtime_config_sets_fields() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter())
    profile = EvaluationProfile(shadow_eval_enabled=False, online_registry_enabled=False)
    registry = InMemoryOnlineEvaluationRegistry()
    updated = apply_evaluation_profile_to_runtime_config(
        config,
        profile,
        registry=registry,
    )
    assert updated.evaluation_profile == profile
    assert updated.evaluation_registry is registry


def test_assert_evaluation_assembly_valid_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.valid")
    wiring = wire_application_evaluation(env)
    assert_evaluation_assembly_valid(wiring, env)


def test_validate_evaluation_wiring_requires_registry_when_online_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.registry")
    env.evaluation_profile = EvaluationProfile(
        shadow_eval_enabled=True,
        online_registry_enabled=True,
    )
    wiring = wire_application_evaluation(env)
    wiring = type(wiring)(
        profile=wiring.profile,
        options=wiring.options,
        registry=None,
        governance_bridge=wiring.governance_bridge,
        domain_fragments=wiring.domain_fragments,
    )
    result = validate_evaluation_wiring(wiring, env)
    assert not result.valid
    assert any("online_registry_enabled requires evaluation registry" in error for error in result.errors)


def test_validate_evaluation_wiring_rejects_baseline_without_trend() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.baseline")
    env.evaluation_profile = EvaluationProfile(
        require_baseline_for_release=True,
        trend_comparison_enabled=False,
        online_registry_enabled=True,
    )
    wiring = wire_application_evaluation(env)
    with pytest.raises(EvaluationAssemblyError):
        assert_evaluation_assembly_valid(wiring, env)


def test_materialize_runtime_config_applies_evaluation_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.runtime")
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
    assert config.evaluation_profile is not None
    assert config.evaluation_profile.shadow_eval_enabled is True
    assert config.evaluation_registry is not None


def test_build_harness_host_runtime_wires_evaluation_policy_bundle() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    assert runtime.evaluation.registry is not None
    assert runtime.evaluation.governance_bridge is not None
    assert "evaluation_governance" in runtime.env_wiring.policy_bundle.domain_fragments
