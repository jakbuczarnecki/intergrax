# © Artur Czarnecki. All rights reserved.

"""EVAL-1/2: Evaluation runtime bridge and assembly validation."""

from __future__ import annotations

from dataclasses import replace

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
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    EvaluationProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.nexus.config import RuntimeConfig
from testing_support.application_harness_test_support import build_harness_host_runtime_for_tests
from testing_support.builder import FakeLLMAdapter, build_runtime_request_for_tests

from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


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


def test_validate_evaluation_wiring_strict_online_registry_requires_tenant() -> None:
    """G-09: PRODUCT/STRICT hosts must not leave the evaluation registry unscoped."""
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.g09")
    env = env.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env = env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(
                update={"online_registry_enabled": True, "shadow_eval_enabled": True}
            )
        }
    )
    wiring = wire_application_evaluation(env, tenant_id=None)
    result = validate_evaluation_wiring(wiring, env)
    assert not result.valid
    assert any("G-09" in error for error in result.errors)
    with pytest.raises(EvaluationAssemblyError):
        assert_evaluation_assembly_valid(wiring, env)


def test_validate_evaluation_wiring_product_profile_online_registry_requires_tenant() -> None:
    """G-09 follow-up: PRODUCT application profile is fail-closed without tenant scope."""
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="eval.g09.product")
    env = env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(
                update={"online_registry_enabled": True, "shadow_eval_enabled": True}
            )
        }
    )
    wiring = wire_application_evaluation(env, tenant_id=None)
    result = validate_evaluation_wiring(wiring, env)
    assert not result.valid
    assert any("G-09" in error for error in result.errors)


def test_validate_evaluation_wiring_strict_online_registry_whitespace_tenant_rejected() -> None:
    """G-09 follow-up: whitespace-only tenant_id is treated as unscoped."""
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.g09.ws")
    env = env.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env = env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(
                update={"online_registry_enabled": True, "shadow_eval_enabled": True}
            )
        }
    )
    wiring = replace(
        wire_application_evaluation(env, tenant_id="tenant-placeholder"),
        tenant_id="   ",
    )
    result = validate_evaluation_wiring(wiring, env)
    assert not result.valid
    assert any("G-09" in error for error in result.errors)


def test_validate_evaluation_wiring_strict_online_registry_accepts_scoped_tenant() -> None:
    """G-09 follow-up: STRICT hosts pass when the evaluation registry is tenant-scoped."""
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.g09.ok")
    env = env.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env = env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(
                update={"online_registry_enabled": True, "shadow_eval_enabled": True}
            )
        }
    )
    wiring = wire_application_evaluation(env, tenant_id="tenant-eval-1")
    assert_evaluation_assembly_valid(wiring, env)


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
    request = build_runtime_request_for_tests(
        seed="eval-runtime",
        tenant_id="t1",
        agent_id="echo",
        user_id="user-1",
        session_id="session-1",
        message="hello",
    )
    manifest = build_lab_manifest(LabApplicationSettings.from_env())
    build_ctx = ApplicationBuildContext.for_manifest(manifest, environment=env)
    config = materialize_runtime_config(request, build_ctx, env, llm_adapter=FakeLLMAdapter())
    assert config.evaluation_profile is not None
    assert config.evaluation_profile.shadow_eval_enabled is True
    assert config.evaluation_registry is not None


def test_build_harness_host_runtime_wires_evaluation_policy_bundle() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime_for_tests(manifest, env, settings=settings)
    assert runtime.evaluation.registry is not None
    assert runtime.evaluation.governance_bridge is not None
    assert "evaluation_governance" in runtime.env_wiring.policy_bundle.domain_fragments
