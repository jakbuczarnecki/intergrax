# © Artur Czarnecki. All rights reserved.

"""Enterprise durable agent lifecycle E2E proof (EA-01/EA-02)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_manager_models import AgentManagerDerivedStatus
from intergrax.agent_distribution.errors import RuntimeActivationConflict
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from testing_support.agent_platform_admin_harness import admin_test_principal
from testing_support.canonical_agent_lifecycle_composition import (
    default_stage15_proof_config,
)
from testing_support.enterprise_agent_lifecycle_composition import (
    EnterpriseAgentLifecycleProofStack,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def _assert_checkpoint(stack: EnterpriseAgentLifecycleProofStack, result) -> None:
    config = stack.config
    bundle = stack.durable_runtime.distribution_store_bundle
    serving = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving.traffic_serving_revision_id == result.traffic_serving_revision_id
    installation = bundle.installation_store.get_installation(config.installation_id)
    assert installation is not None
    binding = bundle.binding_store.get_binding(config.application_binding_id)
    assert binding is not None
    revision = bundle.revision_store.get_revision(result.runtime_revision_id)
    assert revision is not None
    assert revision.revision_state is RuntimeRevisionState.ACTIVE
    materialization = bundle.materialization_store.get_by_revision(
        result.runtime_revision_id,
    )
    assert materialization is not None
    assert (
        materialization.materialization_artifact_digest
        == result.materialization_artifact_digest
    )
    manager_entry = next(
        item
        for item in stack.agent_manager_query.list_agents(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        ).items
        if item.lifecycle.logical_agent_id == config.logical_agent_id
    )
    assert manager_entry.derived_status is AgentManagerDerivedStatus.SERVING


def test_enterprise_durable_lifecycle_happy_path_and_restart(tmp_path: Path) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "enterprise-lifecycle.db"
    shared_root = tmp_path / "shared-artifacts"
    stack_a = EnterpriseAgentLifecycleProofStack.build(shared_root, db_path=db_path)
    result = stack_a.run_happy_path()
    _assert_checkpoint(stack_a, result)
    revision_id = result.runtime_revision_id
    expected_agent_id = result.execution_agent_id
    expected_answer = result.execution_answer
    pointer_revision = stack_a.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    ).serving_pointer_revision

    del stack_a
    stack_b = EnterpriseAgentLifecycleProofStack.reopen(shared_root, db_path, config)
    serving = stack_b.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving.traffic_serving_revision_id == revision_id
    assert serving.serving_pointer_revision == pointer_revision
    projection = stack_b.resolve_serving_projection()
    assert projection.evidence.runtime_revision_id == revision_id
    execution_agent_id, execution_answer = asyncio.run(stack_b.execute_canonical())
    assert execution_agent_id == expected_agent_id
    assert execution_answer == expected_answer
    _assert_checkpoint(stack_b, result)


def test_enterprise_restart_preserves_active_revision(tmp_path: Path) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "restart.db"
    stack = EnterpriseAgentLifecycleProofStack.build(tmp_path, db_path=db_path)
    result = stack.run_happy_path()
    pointer = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    del stack
    reopened = EnterpriseAgentLifecycleProofStack.reopen(
        tmp_path / "reopen",
        db_path,
        config,
    )
    serving = reopened.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving.traffic_serving_revision_id == result.traffic_serving_revision_id
    assert serving.serving_pointer_revision == pointer.serving_pointer_revision


def test_enterprise_two_composition_stale_cas_rejects_second_writer(tmp_path: Path) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "enterprise-cas.db"
    stack_a = EnterpriseAgentLifecycleProofStack.build(tmp_path / "a", db_path=db_path)
    stack_a.install_from_catalog()
    stack_a.bind_enabled_agent()
    built = stack_a.build_revision()
    serving_before = stack_a.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    input_bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        runtime_revision_id=built.runtime_revision_id,
        manifest=stack_a.canonical.manifest,
        build_context=ApplicationBuildContext.for_manifest(stack_a.canonical.manifest),
        authority=stack_a.durable_runtime.registry_projection_authority,
    )
    activation_request = ActivateRuntimeRevisionRequest(
        mutation_id="mut-enterprise-cas-a",
        runtime_revision_id=built.runtime_revision_id,
        artifact_locator=built.artifact_locator,
        expected_artifact_digest=built.materialization_artifact_digest,
        expected_serving_pointer_revision=serving_before.serving_pointer_revision,
        expected_prior_traffic_revision_id=serving_before.traffic_serving_revision_id,
    )
    stack_a.launcher.deploy_and_activate(
        projection_input=input_bundle,
        activation_request=activation_request,
        principal=stack_a.canonical.governance.principal,
        admission_mutation_id=reference_admission_mutation_id(built.runtime_revision_id),
    )

    stack_b = EnterpriseAgentLifecycleProofStack.reopen(tmp_path / "b", db_path, config)
    stale_request = activation_request.model_copy(
        update={"mutation_id": "mut-enterprise-cas-b"},
    )
    with pytest.raises(RuntimeActivationConflict):
        stack_b.launcher.deploy_and_activate(
            projection_input=input_bundle,
            activation_request=stale_request,
            principal=stack_b.canonical.governance.principal,
            admission_mutation_id=reference_admission_mutation_id(
                built.runtime_revision_id,
            ),
        )


def test_enterprise_historical_projection_isolated_after_desired_state_change(
    tmp_path: Path,
) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "historical.db"
    stack = EnterpriseAgentLifecycleProofStack.build(tmp_path, db_path=db_path)
    result = stack.run_happy_path()
    build_context = ApplicationBuildContext.for_manifest(stack.canonical.manifest)
    historical = build_production_registry_projection_for_revision(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        runtime_revision_id=result.runtime_revision_id,
        manifest=stack.canonical.manifest,
        build_context=build_context,
        authority=stack.durable_runtime.registry_projection_authority,
    )
    binding = stack.admin.list_bindings(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    ).bindings[0]
    stack.admin.disable_binding(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        application_binding_id=config.application_binding_id,
        request=SetAgentEnablementRequest(
            mutation_id="mut-disable-after-serve",
            expected_revision=binding.binding_revision,
        ),
        principal=admin_test_principal(),
    )
    historical_after = build_production_registry_projection_for_revision(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        runtime_revision_id=result.runtime_revision_id,
        manifest=stack.canonical.manifest,
        build_context=build_context,
        authority=stack.durable_runtime.registry_projection_authority,
    )
    assert (
        historical.evidence.runtime_revision_id
        == historical_after.evidence.runtime_revision_id
    )
    assert (
        historical.evidence.effective_roster_revision_id
        == historical_after.evidence.effective_roster_revision_id
    )
    assert (
        historical.evidence.materialization_artifact_digest
        == historical_after.evidence.materialization_artifact_digest
    )
