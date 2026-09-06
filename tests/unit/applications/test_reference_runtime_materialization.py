# © Artur Czarnecki. All rights reserved.

"""Reference runtime materialization authority proofs (NPSC-3B-R3)."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.errors import RuntimeMaterializationConflict
from intergrax.agent_distribution.runtime_materialization_record import RuntimeMaterializationRecord
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    build_reference_activation_request,
    build_reference_registry_projection_input_bundle,
    reference_admission_mutation_id,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleError,
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications._shared.reference_runtime_materialization import (
    prepare_reference_runtime_materialization,
)
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _research_projection(revision_id: str):
    settings = ResearchBackendSettings(use_nexus_loop=True)
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    projection_input = build_reference_registry_projection_input_bundle(
        manifest,
        env,
        builders=RESEARCH_AGENT_BUILDERS,
        runtime_revision_id=revision_id,
        enabled_contract_stems=frozenset({"research"}),
    )
    activation_request = build_reference_activation_request(projection_input)
    return projection_input, activation_request, env


def test_deploy_without_materialization_fails_closed() -> None:
    composition = create_reference_production_process_composition()
    projection_input, activation_request, env = _research_projection("rev-mat-missing")
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    with pytest.raises(
        ReferenceProductionLifecycleError,
        match="missing canonical materialization record",
    ):
        launcher.deploy_and_activate(
            projection_input,
            activation_request,
            principal=governance.principal,
            admission_mutation_id=reference_admission_mutation_id(
                projection_input.runtime_revision.runtime_revision_id
            ),
        )


def test_canonical_materialization_enables_activation() -> None:
    composition = create_reference_production_process_composition()
    projection_input, activation_request, env = _research_projection("rev-mat-ok")
    stores = composition.agent_platform_runtime.stores
    prepare_reference_runtime_materialization(
        stores,
        projection_input,
        artifact_locator=activation_request.artifact_locator,
    )
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    result = launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )
    assert result.runtime_revision_id == "rev-mat-ok"
    record = stores.materialization_store.get_by_revision("rev-mat-ok")
    assert record is not None


def test_materialization_record_scope_matches_revision() -> None:
    composition = create_reference_production_process_composition()
    projection_input, activation_request, _ = _research_projection("rev-mat-scope")
    revision = projection_input.runtime_revision
    record = prepare_reference_runtime_materialization(
        composition.agent_platform_runtime.stores,
        projection_input,
        artifact_locator=activation_request.artifact_locator,
    )
    assert record.runtime_revision_id == revision.runtime_revision_id
    assert record.application_id == revision.application_id
    assert record.application_environment_id == revision.application_environment_id
    assert record.materialization_topology == revision.materialization_topology
    assert record.artifact_locator == activation_request.artifact_locator
    assert (
        record.materialization_artifact_digest
        == projection_input.materialization_artifact_digest
    )
    assert record.materialized_runtime_lock_id == revision.materialized_runtime_lock_id
    assert (
        record.materialized_runtime_lock_digest
        == revision.materialized_runtime_lock_digest
    )


def test_conflicting_materialization_artifact_locator_fails_closed() -> None:
    composition = create_reference_production_process_composition()
    projection_input, activation_request, _ = _research_projection("rev-mat-conflict")
    stores = composition.agent_platform_runtime.stores
    prepare_reference_runtime_materialization(
        stores,
        projection_input,
        artifact_locator=activation_request.artifact_locator,
    )
    revision = projection_input.runtime_revision
    conflicting = RuntimeMaterializationRecord(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        artifact_locator="reference://process-local/venv-bundle/conflicting",
        materialization_artifact_digest=revision.materialization_artifact_digest or "",
        materialized_runtime_lock_id=revision.materialized_runtime_lock_id or "",
        materialized_runtime_lock_digest=revision.materialized_runtime_lock_digest or "",
    )
    with pytest.raises(RuntimeMaterializationConflict):
        stores.materialization_store.persist(conflicting)


def test_reference_launcher_does_not_create_materialization_record() -> None:
    composition = create_reference_production_process_composition()
    projection_input, activation_request, env = _research_projection("rev-mat-launcher")
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    stores = composition.agent_platform_runtime.stores
    with pytest.raises(ReferenceProductionLifecycleError):
        launcher.deploy_and_activate(
            projection_input,
            activation_request,
            principal=governance.principal,
            admission_mutation_id=reference_admission_mutation_id(
                projection_input.runtime_revision.runtime_revision_id
            ),
        )
    assert stores.materialization_store.get_by_revision("rev-mat-launcher") is None
    assert not hasattr(ReferenceProductionLifecycleLauncher, "materialize")
