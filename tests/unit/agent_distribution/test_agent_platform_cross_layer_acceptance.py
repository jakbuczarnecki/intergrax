# © Artur Czarnecki. All rights reserved.

"""Cross-layer Agent Platform identity chain acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent

from intergrax.agent_distribution.activation import FakeRuntimeServingProjectionCoordinator
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    AgentPlatformAdminBlockedError,
    RollbackRuntimeRevisionRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.errors import (
    MaterializationInputConflict,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.in_memory_stores import InMemoryRuntimeRevisionStore
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    InMemoryRuntimeRegistryProjectionStore,
    RegistryProjectionError,
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    InMemoryRuntimeAgentFactoryResolver,
)
from intergrax.applications._shared.wiring import (
    _index_manifest_bindings,
    factory_reference_for_roster_entry,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _APP,
    _ARTIFACT,
    _ENV,
    _bind_request,
    _build_request,
    _install_bind,
    _install_request,
    build_admin_stack,
    admin_test_principal,
)
from tests.unit.applications.test_registry_projection_ap10 import (
    _bundle_parts,
    _entry,
    _manifest,
    _revision,
    _roster,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PROOF_APP = "app_a"
_RESEARCHER_CONTRACT_ID = "researcher"


def _echo_factory(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> EchoAgent:
    return EchoAgent()


def _wire_real_projection_coordinator(stack: AdminStack) -> tuple[
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    InMemoryRuntimeRegistryProjectionStore,
]:
    input_store = InMemoryRegistryProjectionInputStore()
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=stack.service._revision_store,
        input_store=input_store,
        projection_store=projection_store,
    )
    stack.service._activation_service._projection_coordinator = coordinator
    return coordinator, input_store, projection_store


def _install_enable_build(stack: AdminStack, revision_id: str):
    stack.service.install_agent(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        request=_install_request(),
    )
    stack.service.bind_agent(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
    )
    stack.service.enable_binding(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    return stack.service.build_application_revision(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        request=_build_request(revision_id),
    )


def _exact_projection_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=_PROOF_APP,
        name="App A",
        agents=[
            AgentBinding.mount(
                EchoAgent,
                contract_id=_RESEARCHER_CONTRACT_ID,
                builder_key="researcher",
            )
        ],
    )


def _freeze_registry_projection_bundle(
    stack: AdminStack,
    runtime_revision_id: str,
) -> RegistryProjectionInputBundle:
    revision = stack.service._revision_store.get_revision(runtime_revision_id)
    assert revision is not None
    roster = stack.service._build_roster(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        manifest_release_id=revision.application_release_id,
    )
    assert roster.effective_roster_revision_id == revision.effective_roster_revision_id
    manifest = _exact_projection_manifest()
    build_context = ApplicationBuildContext.for_manifest(manifest)
    enabled = next(entry for entry in roster.entries if entry.effective_enablement)
    factory_reference = factory_reference_for_roster_entry(
        enabled,
        _index_manifest_bindings(manifest),
    )
    resolver = InMemoryRuntimeAgentFactoryResolver()
    resolver.register(
        package_digest=enabled.package_digest,
        factory_reference=factory_reference,
        factory=_echo_factory,
    )
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=build_context,
        factory_resolver=resolver,
        materialization_artifact_digest=revision.materialization_artifact_digest,
    )


def test_cross_layer_identity_chain_through_admin_serving_state() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )

    roster_view = stack.service.inspect_effective_roster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id="rel-1",
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-chain-1"),
    )
    revision = stack.service.inspect_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-chain-1",
    )

    assert revision.application_environment_id == _ENV
    assert revision.effective_roster_revision_id == built.effective_roster_revision_id
    assert revision.materialized_runtime_lock_id == built.materialized_runtime_lock_id
    assert revision.runtime_graph_digest == built.runtime_graph_digest
    assert revision.materialization_artifact_digest == built.materialization_artifact_digest
    assert revision.revision_state is RuntimeRevisionState.VALIDATED
    assert roster_view.effective_roster_revision_id == built.effective_roster_revision_id

    lock = stack.state.locks[built.materialized_runtime_lock_id or ""]
    assert lock.lock_digest == built.materialized_runtime_lock_digest

    activated = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-chain-1",
            runtime_revision_id="rev-chain-1",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
            expected_serving_pointer_revision=0,
        ),
    )
    serving = stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert activated.traffic_serving_revision_id == "rev-chain-1"
    assert serving.traffic_serving_revision_id == "rev-chain-1"
    assert serving.active_revision is not None
    assert serving.active_revision.runtime_revision_id == "rev-chain-1"
    assert serving.active_revision.effective_roster_revision_id == built.effective_roster_revision_id
    assert (
        serving.active_revision.materialized_runtime_lock_digest
        == built.materialized_runtime_lock_digest
    )


def test_cross_layer_build_blocked_without_explicit_dependency_resolver() -> None:
    stack = build_admin_stack()
    stack.service._lock_service = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_build_request("rev-no-resolver"),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_DEPENDENCY_RESOLVER"


def test_cross_layer_materialization_rejects_roster_application_mismatch(
    tmp_path: Path,
) -> None:
    from tests.unit.agent_distribution.test_agent_distribution_materialization import (
        _build_fixture,
    )

    materialization_input, _, _, _ = _build_fixture(tmp_path)
    foreign_roster = materialization_input.effective_roster.model_copy(
        update={"application_id": "foreign-app"},
    )
    tampered = materialization_input.model_copy(update={"effective_roster": foreign_roster})
    with pytest.raises(MaterializationInputConflict, match="application_id"):
        RuntimeMaterializationService._validate_input_consistency(tampered)


def test_cross_layer_registry_projection_rejects_app_b_revision_with_app_a_roster() -> None:
    roster_app_a = _roster((_entry("search"),))
    revision_app_b = _revision("rev-cross").model_copy(update={"application_id": "app_b"})
    bundle = _bundle_parts(revision_app_b, roster_app_a, _manifest(app_id="app_a"))
    with pytest.raises(RegistryProjectionError, match="application_id"):
        build_registry_projection(bundle)


def test_cross_layer_mark_validated_rejects_lock_digest_mutation() -> None:
    revision_service = RuntimeRevisionService(InMemoryRuntimeRevisionStore())
    candidate = RuntimeRevision(
        runtime_revision_id="rev-mut",
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-1",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="sha256:" + ("1" * 64),
        runtime_graph_digest="sha256:" + ("2" * 64),
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.CANDIDATE,
    )
    revision_service.persist_candidate_revision(candidate)
    mutated = candidate.model_copy(
        update={
            "revision_state": RuntimeRevisionState.VALIDATED,
            "materialization_artifact_digest": _ARTIFACT,
            "materialized_runtime_lock_digest": "sha256:" + ("9" * 64),
        }
    )
    with pytest.raises(RuntimeRevisionLifecycleError, match="materialized_runtime_lock_digest"):
        revision_service.mark_validated("rev-mut", validated_revision=mutated)


def test_real_registry_projection_composition_before_serving_cutover() -> None:
    stack = build_admin_stack()
    coordinator, input_store, projection_store = _wire_real_projection_coordinator(stack)
    assert isinstance(coordinator, ApplicationRegistryProjectionCoordinator)
    assert not isinstance(
        stack.service._activation_service._projection_coordinator,
        FakeRuntimeServingProjectionCoordinator,
    )

    built = _install_enable_build(stack, "rev-proof-1")
    revision = stack.service._revision_store.get_revision("rev-proof-1")
    assert revision is not None
    assert revision.revision_state is RuntimeRevisionState.VALIDATED
    bundle = _freeze_registry_projection_bundle(stack, "rev-proof-1")
    input_store.register(bundle)

    serving_before = stack.service.inspect_serving(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
    )
    assert serving_before.traffic_serving_revision_id is None
    activation = stack.service._activation_service
    activation.prepare_candidate(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-1",
        artifact_locator=built.artifact_locator or "test://artifact",
    )
    serving_after_prepare = stack.service.inspect_serving(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
    )
    assert serving_after_prepare.traffic_serving_revision_id is None
    activation.commit_activation(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-1",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
    )

    serving = stack.service.inspect_serving(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
    )
    projection = projection_store.get("rev-proof-1")
    assert projection is not None
    evidence = projection.evidence
    assert serving.traffic_serving_revision_id == "rev-proof-1"
    assert evidence.runtime_revision_id == serving.traffic_serving_revision_id
    assert evidence.application_id == _PROOF_APP
    assert evidence.application_environment_id == _ENV
    assert evidence.effective_roster_revision_id == revision.effective_roster_revision_id
    assert evidence.materialized_runtime_lock_id == revision.materialized_runtime_lock_id
    assert evidence.materialized_runtime_lock_digest == revision.materialized_runtime_lock_digest
    assert evidence.runtime_graph_digest == revision.runtime_graph_digest
    assert evidence.materialization_artifact_digest == revision.materialization_artifact_digest
    assert projection.agent_registry.has(_RESEARCHER_CONTRACT_ID)
    active = stack.service._revision_store.get_revision("rev-proof-1")
    assert active is not None
    assert active.revision_state is RuntimeRevisionState.ACTIVE


def test_missing_projection_inputs_block_serving_pointer_cutover() -> None:
    stack = build_admin_stack()
    _wire_real_projection_coordinator(stack)
    built = _install_enable_build(stack, "rev-proof-fail")
    serving_before = stack.service.inspect_serving(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
    )
    activation = stack.service._activation_service
    activation.prepare_candidate(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-fail",
        artifact_locator=built.artifact_locator or "test://artifact",
    )
    with pytest.raises(RegistryProjectionError, match="missing frozen projection inputs"):
        activation.commit_activation(
            application_id=_PROOF_APP,
            application_environment_id=_ENV,
            runtime_revision_id="rev-proof-fail",
            expected_prior_traffic_revision_id=None,
            expected_serving_pointer_revision=0,
            expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
        )
    serving_after = stack.service.inspect_serving(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
    )
    revision = stack.service._revision_store.get_revision("rev-proof-fail")
    assert revision is not None
    assert serving_after.traffic_serving_revision_id == serving_before.traffic_serving_revision_id
    assert serving_after.serving_pointer_revision == serving_before.serving_pointer_revision
    assert serving_after.traffic_serving_revision_id is None
    assert revision.revision_state is RuntimeRevisionState.VALIDATED
    assert serving_after.active_revision is None


def test_real_projection_rollback_reuses_frozen_registry_n() -> None:
    stack = build_admin_stack()
    coordinator, input_store, projection_store = _wire_real_projection_coordinator(stack)
    first = _install_enable_build(stack, "rev-proof-n")
    input_store.register(_freeze_registry_projection_bundle(stack, "rev-proof-n"))
    activation = stack.service._activation_service
    activation.prepare_candidate(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-n",
        artifact_locator=first.artifact_locator or "test://artifact",
    )
    activation.commit_activation(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-n",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=first.materialization_artifact_digest or _ARTIFACT,
    )
    projection_n = projection_store.get("rev-proof-n")
    assert projection_n is not None
    token_n = projection_n.evidence.readiness_token

    second = stack.service.build_application_revision(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-proof-n1"),
    )
    input_store.register(_freeze_registry_projection_bundle(stack, "rev-proof-n1"))
    activation.prepare_candidate(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-n1",
        artifact_locator=second.artifact_locator or "test://artifact",
    )
    activation.commit_activation(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-proof-n1",
        expected_prior_traffic_revision_id="rev-proof-n",
        expected_serving_pointer_revision=1,
        expected_artifact_digest=second.materialization_artifact_digest or _ARTIFACT,
    )
    stack.service.rollback_revision(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=RollbackRuntimeRevisionRequest(
            mutation_id="mut-rollback-proof",
            expected_current_traffic_revision_id="rev-proof-n1",
            expected_serving_pointer_revision=2,
            target_runtime_revision_id="rev-proof-n",
        ),
    )
    restored = projection_store.get("rev-proof-n")
    serving = stack.service.inspect_serving(
        application_id=_PROOF_APP,
        application_environment_id=_ENV,
    )
    assert restored is projection_n
    assert restored.evidence.readiness_token == token_n
    assert serving.traffic_serving_revision_id == "rev-proof-n"
    assert coordinator.get_projection("rev-proof-n") is projection_n
