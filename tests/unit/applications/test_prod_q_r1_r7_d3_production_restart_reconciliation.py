# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D3 — durable projection inputs and post-restart reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
    FakeInMemoryRuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.errors import RuntimeActivationError
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.application_environment_identity import (
    ApplicationEnvironmentIdentity,
)
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.applications._shared.durable_agent_platform_runtime import (
    build_durable_production_agent_platform_runtime,
)
from intergrax.applications._shared.durable_registry_projection_input_store import (
    DescriptorBackedRegistryProjectionInputStore,
)
from intergrax.applications._shared.production_lifecycle import (
    build_descriptor_backed_projection_input_store,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionServingReconciliationError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.production_process_serving_reconciliation import (
    RECONCILIATION_OUTCOME_READY,
    reconcile_production_serving_environment_on_process_open,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    build_reference_production_control_plane_governance,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
    wire_durable_reference_production_lifecycle_services,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _ENV,
    _activate_bundle,
    _canonical_bundle,
)
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _governance,
    _venv_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _durable_launcher(
    tmp_path: Path,
    db_path: Path,
) -> tuple[ReferenceProductionLifecycleLauncher, object, object]:
    durable = build_durable_production_agent_platform_runtime(db_path)
    composition = ProductionProcessComposition(
        agent_platform_runtime=durable.agent_platform_runtime,
        agent_capability_runtime=None,
    )
    services = wire_durable_reference_production_lifecycle_services(
        composition,
        durable_store_bundle=durable.distribution_store_bundle,
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    governance = build_reference_production_control_plane_governance(env)
    launcher = ReferenceProductionLifecycleLauncher(
        composition,
        services=services,
        mutation_authorization_boundary=governance.mutation_authorization_boundary,
        environment_tenant_resolver=governance.environment_tenant_resolver,
    )
    return launcher, durable, governance


def test_d3_descriptor_backed_projection_input_reconstructs_after_sqlite_restart(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "d3-projection.db"
    launcher, durable, governance = _durable_launcher(tmp_path, db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        stores=launcher.process_composition.agent_platform_runtime.stores,
    )
    revision_id = _activate_bundle(
        launcher,
        bundle,
        locator,
        principal=governance.principal,
    )
    del launcher
    fresh_store = build_descriptor_backed_projection_input_store(
        durable.distribution_store_bundle,
    )
    assert isinstance(fresh_store, DescriptorBackedRegistryProjectionInputStore)
    reconstructed = fresh_store.get(revision_id)
    assert reconstructed is not None
    assert reconstructed.runtime_revision.runtime_revision_id == revision_id


def test_d3_missing_descriptor_blocks_projection_input_reconstruction(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "d3-missing-desc.db"
    launcher, durable, governance = _durable_launcher(tmp_path, db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id="rev-missing-desc",
        stores=launcher.process_composition.agent_platform_runtime.stores,
    )
    revision_id = _activate_bundle(
        launcher,
        bundle,
        locator,
        principal=governance.principal,
    )
    bundle_store = durable.distribution_store_bundle
    fresh_store = build_descriptor_backed_projection_input_store(bundle_store)
    assert fresh_store.get(revision_id) is not None
    database = bundle_store.database
    connection_factory = getattr(database, "_connect")
    with connection_factory() as conn:
        conn.execute(
            "DELETE FROM projection_descriptors WHERE runtime_revision_id = ?",
            (revision_id,),
        )
        conn.commit()
    assert fresh_store.get(revision_id) is None


def _activation_harness() -> tuple[ActivationService, FakeInMemoryRuntimeDeploymentAdapter]:
    state = AgentDistributionStoreState()
    adapter = FakeInMemoryRuntimeDeploymentAdapter()
    revision_store = InMemoryRuntimeRevisionStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    activation_store = InMemoryApplicationEnvironmentActivationStore(state)
    materialization_store = InMemoryRuntimeMaterializationStore(state)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        activation_store=activation_store,
        deployment_adapter=adapter,
        projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        materialization_store=materialization_store,
    )
    return activation, adapter


def test_d3_reconcile_skipped_without_traffic() -> None:
    from tests.unit.agent_distribution.test_prod_q_r1_r7_d0_serving_unit_attestation import (
        _APP as D0_APP,
        _ENV as D0_ENV,
    )

    activation, _adapter = _activation_harness()
    transition = activation.reconcile_traffic_serving_physical_unit_after_process_restart(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    assert transition.value is None


def test_d3_reconcile_attests_serving_instance() -> None:
    from tests.unit.agent_distribution.test_prod_q_r1_r7_d0_serving_unit_attestation import (
        _APP as D0_APP,
        _D0Harness,
        _ENV as D0_ENV,
    )

    harness = _D0Harness()
    harness.seed_validated("rev-serving")
    ready = harness.prepare("rev-serving")
    scope = ApplicationEnvironmentIdentity(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    harness.state.serving_records[scope] = ApplicationEnvironmentServingRecord(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
        traffic_serving_revision_id="rev-serving",
        serving_pointer_revision=1,
    )
    serving_instance = ready.model_copy(
        update={"instance_state": DeploymentInstanceState.SERVING},
    )
    harness.deployment_store.persist_instance(serving_instance)
    transition = harness.activation.reconcile_traffic_serving_physical_unit_after_process_restart(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    assert transition.value is not None
    assert transition.value.physical_unit_live is True


def test_d3_stale_physical_unit_fails_reconciliation() -> None:
    from tests.unit.agent_distribution.test_prod_q_r1_r7_d0_serving_unit_attestation import (
        _APP as D0_APP,
        _D0Harness,
        _ENV as D0_ENV,
    )

    harness = _D0Harness()
    harness.seed_validated("rev-stale")
    ready = harness.prepare("rev-stale")
    scope = ApplicationEnvironmentIdentity(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    harness.state.serving_records[scope] = ApplicationEnvironmentServingRecord(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
        traffic_serving_revision_id="rev-stale",
        serving_pointer_revision=1,
    )
    harness.deployment_store.persist_instance(
        ready.model_copy(update={"instance_state": DeploymentInstanceState.SERVING}),
    )
    harness.replace_adapter(FakeInMemoryRuntimeDeploymentAdapter())
    with pytest.raises(RuntimeActivationError, match="attestation"):
        harness.activation.reconcile_traffic_serving_physical_unit_after_process_restart(
            application_id=D0_APP,
            application_environment_id=D0_ENV,
        )


def test_d3_full_reconcile_pipeline_after_sqlite_deploy(tmp_path: Path) -> None:
    db_path = tmp_path / "d3-pipeline.db"
    launcher, durable, governance = _durable_launcher(tmp_path, db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id="rev-pipeline",
        stores=launcher.process_composition.agent_platform_runtime.stores,
    )
    _activate_bundle(
        launcher,
        bundle,
        locator,
        principal=governance.principal,
    )
    pointer = (
        durable.distribution_store_bundle.serving_store.get_serving_record(_APP, _ENV)
    )
    assert pointer is not None
    result = reconcile_production_serving_environment_on_process_open(
        lifecycle_services=launcher.services,
        serving_store=durable.distribution_store_bundle.serving_store,
        registry_projection_rehydrator=durable.registry_projection_rehydrator,
        application_id=_APP,
        application_environment_id=_ENV,
        expected_serving_pointer_revision=pointer.serving_pointer_revision,
    )
    assert result.evidence.outcome == RECONCILIATION_OUTCOME_READY


def test_d3_open_fails_reconciliation_on_empty_adapter_after_serving(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ad-stale.db"
    launcher, durable, governance = _durable_launcher(tmp_path, db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id="rev-open-stale",
        stores=launcher.process_composition.agent_platform_runtime.stores,
    )
    _activate_bundle(
        launcher,
        bundle,
        locator,
        principal=governance.principal,
    )
    del launcher
    env = ApplicationEnvironmentProfile(profile_id=_ENV)
    artifact_roots = (tmp_path / "artifacts",)
    artifact_roots[0].mkdir(parents=True, exist_ok=True)
    fresh_adapter = _venv_adapter(tmp_path)
    with pytest.raises(ProductionServingReconciliationError, match="attestation"):
        open_production_process_lifecycle(
            agent_distribution_db=db_path,
            continuation_state_file=tmp_path / "continuation.json",
            deployment_adapter=fresh_adapter,
            governance=_governance(env),
            trusted_artifact_parent_roots=artifact_roots,
            missing_file_policy="empty",
        )


def test_d3_reconciliation_retry_is_idempotent() -> None:
    from tests.unit.agent_distribution.test_prod_q_r1_r7_d0_serving_unit_attestation import (
        _APP as D0_APP,
        _D0Harness,
        _ENV as D0_ENV,
    )

    harness = _D0Harness()
    harness.seed_validated("rev-retry")
    ready = harness.prepare("rev-retry")
    scope = ApplicationEnvironmentIdentity(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    harness.state.serving_records[scope] = ApplicationEnvironmentServingRecord(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
        traffic_serving_revision_id="rev-retry",
        serving_pointer_revision=1,
    )
    harness.deployment_store.persist_instance(
        ready.model_copy(update={"instance_state": DeploymentInstanceState.SERVING}),
    )
    first = harness.activation.reconcile_traffic_serving_physical_unit_after_process_restart(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    second = harness.activation.reconcile_traffic_serving_physical_unit_after_process_restart(
        application_id=D0_APP,
        application_environment_id=D0_ENV,
    )
    assert first.value is not None and second.value is not None
    assert first.value.serving_unit_ref == second.value.serving_unit_ref
