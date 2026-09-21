# © Artur Czarnecki. All rights reserved.

"""U3 — agent runtime governance grant materialization (declarative, no agent ctor)."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    AgentRuntimeGovernanceMaterializationError,
    capability_grants_from_application_manifest,
    validated_capabilities_for_binding,
)
from intergrax.applications._shared.application_composition_context import (
    composition_for_factory_context,
)
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from testing_support.builder import FakeLLMAdapter, build_runtime_request_for_tests
from testing_support.u3_factory_only_agent import (
    FactoryOnlyAgent,
    build_factory_only_agent,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CAP = "factory-only.run"


def _runtime_request(**overrides: str) -> RuntimeRequest:
    return build_runtime_request_for_tests(
        seed=overrides.pop("seed", "gov-mat"),
        tenant_id=overrides.pop("tenant_id", "tenant-a"),
        agent_id=overrides.pop("agent_id", "echo"),
        user_id=overrides.pop("user_id", "user"),
        session_id=overrides.pop("session_id", "session"),
        message=overrides.pop("message", "probe"),
        task_id=overrides.pop("task_id", None),
        run_id=overrides.pop("run_id", None),
        metadata=overrides.pop("metadata", None),
    )


def _echo_manifest(*agents: AgentBinding) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="gov_mat",
        name="Gov Mat",
        route_prefix="/v1/gov_mat",
        env_prefix="GOV_MAT_",
        agents=list(agents),
    )


def _registry_for_manifest(
    manifest: ApplicationManifest,
    *,
    builders: dict[type[Agent], object] | None = None,
) -> object:
    ctx = ApplicationBuildContext.for_manifest(manifest)
    composition = composition_for_factory_context(
        ctx,
        policy_bundle=RuntimePolicyBundle(),
    )
    return build_application_registry(manifest, ctx, builders=builders, composition=composition)


def test_capability_grants_do_not_instantiate_agent_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(
            FactoryOnlyAgent,
            contract_id="factory-only",
            factory=build_factory_only_agent,
            capabilities=[_CAP],
        ),
    )
    registry = _registry_for_manifest(
        manifest,
        builders={FactoryOnlyAgent: build_factory_only_agent},
    )

    def _forbid_resolve(_binding: AgentBinding) -> type[object]:
        raise AssertionError("governance materialization must not resolve agent types")

    monkeypatch.setattr(
        "intergrax.applications._shared.agent_resolution.resolve_agent_type_from_binding",
        _forbid_resolve,
    )

    grants = capability_grants_from_application_manifest(
        manifest,
        tenant_id="tenant-a",
        agent_registry=registry,
    )
    assert grants[0].agent_id == "factory-only"
    assert grants[0].tenant_id == "tenant-a"
    assert _CAP in grants[0].allowed_capabilities


def test_factory_only_agent_participates_via_registry_contract() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(
            FactoryOnlyAgent,
            contract_id="factory-only",
            factory=build_factory_only_agent,
            capabilities=[_CAP],
        ),
    )
    registry = _registry_for_manifest(
        manifest,
        builders={FactoryOnlyAgent: build_factory_only_agent},
    )
    binding = manifest.enabled_agents()[0]
    caps = validated_capabilities_for_binding(binding, registry)
    assert caps == frozenset({_CAP})


def test_manifest_capability_mismatch_fails_closed() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(
            EchoAgent,
            contract_id="echo",
            capabilities=["echo.wrong"],
        ),
    )
    registry = _registry_for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="do not match"):
        validated_capabilities_for_binding(binding, registry)


def test_missing_registry_agent_fails_closed() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"]),
    )
    registry = _registry_for_manifest(manifest)
    orphan = AgentBinding.mount(EchoAgent, contract_id="missing", capabilities=["echo.basic"])
    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="not registered"):
        validated_capabilities_for_binding(orphan, registry)


def test_missing_tenant_id_fails_closed() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"]),
    )
    registry = _registry_for_manifest(manifest)
    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="tenant_id"):
        capability_grants_from_application_manifest(
            manifest,
            tenant_id="   ",
            agent_registry=registry,
        )


def test_production_materialize_requires_agent_registry_on_build_context() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"]),
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gov_mat.strict")
    env = env.model_copy(update={"execution_mode": "strict"})
    build_ctx = ApplicationBuildContext.for_manifest(
        manifest,
        strict_harness=True,
    )
    composition = composition_for_factory_context(
        build_ctx,
        policy_bundle=RuntimePolicyBundle(),
    )
    request = _runtime_request(tenant_id="tenant-prod", agent_id="echo")
    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="agent_registry"):
        materialize_runtime_config(
            request,
            build_ctx,
            env,
            llm_adapter=FakeLLMAdapter(),
            composition=composition,
        )


def test_production_materialize_wires_grants_from_manifest_and_registry() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"]),
    )
    registry = _registry_for_manifest(manifest)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gov_mat.strict")
    env = env.model_copy(update={"execution_mode": "strict"})
    build_ctx = ApplicationBuildContext.for_manifest(
        manifest,
        strict_harness=True,
    )
    composition = composition_for_factory_context(
        build_ctx,
        policy_bundle=RuntimePolicyBundle(),
        agent_registry=registry,
    )
    request = _runtime_request(tenant_id="tenant-prod", agent_id="echo")
    config = materialize_runtime_config(
        request,
        build_ctx,
        env,
        llm_adapter=FakeLLMAdapter(),
        composition=composition,
    )
    assert config.agent_runtime_governance is not None
    assert isinstance(config.agent_runtime_governance, AgentRuntimeGovernanceBoundary)


def test_agent_id_aligns_with_runtime_execution_context_lookup() -> None:
    manifest = _echo_manifest(
        AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"]),
    )
    registry = _registry_for_manifest(manifest)
    grants = capability_grants_from_application_manifest(
        manifest,
        tenant_id="tenant-align",
        agent_registry=registry,
    )
    grant = grants[0]
    assert grant.agent_id == "echo"
    assert grant.agent_id == registry.get_contract("echo").id

    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=True,
        tenant_id="tenant-align",
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=grants,
        ),
    )
    assert config.tenant_id == grant.tenant_id
