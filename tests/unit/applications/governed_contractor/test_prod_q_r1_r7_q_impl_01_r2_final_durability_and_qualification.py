# © Artur Czarnecki. All rights reserved.

"""IMPL-01-R2 — consolidated durability & qualification gates (G-R2-02, G-R2-09)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from governed_contractor_application.host.sanctioned_external_work_operation_port import (
    GovernedSanctionedExternalWorkOperationPort,
)
from governed_contractor_application.host.agent_builders import (
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    InMemoryRuntimeAgentFactoryResolver,
)
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoff,
    ServingUnitIdentity,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkAcceptQuoteRequest,
    SanctionedExternalWorkMutationOperation,
    SanctionedExternalWorkOperationRequest,
)
from tests.unit.contracts.test_external_work import _acceptance
from tests.qualification.prod_q._prod_q_strict_external_work_provider_qualification_config import (
    with_qualified_strict_external_work_provider_pin,
)
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection
from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_q_gca_sqlite_durable_recovery import (
    _sqlite_orchestrator,
)
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _governance,
)
from tests.unit.contracts.test_governed_execution_result import _ger_v2, _proof
from tests.unit.runtime.architecture.test_gr10_r13_r3_production_host_reliability_adoption import (
    _continuation_store,
    _default_production_host_fixture,
    _seed_active_registry_projection,
)
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _isolate_production_process_lifecycle_slot() -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    active = ppl._active_production_process_lifecycle
    if active is not None and not active._process_closed:
        active.close_process_resources()
    yield
    active_after = ppl._active_production_process_lifecycle
    if active_after is not None and not active_after._process_closed:
        active_after.close_process_resources()


def _strict_factory_fixture_missing_gca_bundle(
    tmp_path: Path,
) -> tuple[object, ProductionProcessComposition, object, object]:
    settings, host_runtime = _default_production_host_fixture()
    settings = with_qualified_strict_external_work_provider_pin(settings)
    manifest = build_governed_contractor_manifest()
    settings = replace(
        settings,
        identity_source="context_only",
        include_task_control=False,
        include_interaction_routes=False,
        include_mcp=False,
        api_keys_map={
            "impl01-r2-qual-key": ApiKeyIdentity(
                tenant_id=manifest.app_id,
                user_id="impl01-r2-operator",
                scopes=("*",),
            ),
        },
    )
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=build_governed_contractor_agent_builders(host_runtime),
        revision_id="impl01-r2-missing-gca-bundle",
        settings=settings,
    )
    platform_persistence = build_reference_production_platform_persistence(
        db_path=tmp_path / "platform-kv.db",
    )
    handoff = ServingRuntimeHandoff(
        identity=ServingUnitIdentity(
            application_id=manifest.app_id,
            application_environment_id=env.profile_id,
            runtime_revision_id=projection.evidence.runtime_revision_id,
            serving_unit_ref="unit-impl01-r2",
            materialization_artifact_digest=projection.evidence.materialization_artifact_digest,
            materialization_topology=MaterializationTopology.VENV_BUNDLE,
        ),
        runtime_agent_factory_resolver=InMemoryRuntimeAgentFactoryResolver(),
    )
    composition = ProductionProcessComposition(
        agent_platform_runtime=build_production_agent_platform_runtime(
            platform_persistence=platform_persistence,
        ),
        provider_invocation_store=DurableTestProviderInvocationStore(),
        execution_continuation_state_store=_continuation_store(),
        governed_external_work_host_store_bundle=None,
        serving_runtime_handoff=handoff,
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    governance = _governance(env)
    return projection, composition, settings, host_runtime, governance


def test_g_r2_02_strict_factory_rejects_missing_gca_host_store_bundle_behaviorally(
    tmp_path: Path,
) -> None:
    projection, composition, settings, host_runtime, governance = (
        _strict_factory_fixture_missing_gca_bundle(tmp_path)
    )
    platform = composition.agent_platform_runtime.platform_persistence
    with patch(
        "governed_contractor_application.host.factory."
        "build_strict_external_work_host_session_resources",
    ) as build_strict:
        with pytest.raises(
            HarnessHostOrchestrationTopologyReliabilityCompositionError,
            match="governed_external_work_host_store_bundle",
        ):
            create_governed_contractor_backend_app(
                registry_projection=projection,
                process_composition=composition,
                production_governance=governance,
                settings=settings,
                host_runtime=host_runtime,
                document_store=platform.document_store,
                key_value_cache=platform.kv_store,
                trace_db_path=tmp_path / "trace.db",
            )
        build_strict.assert_not_called()


def test_g_r2_09_accept_quote_prior_create_snapshot_missing_through_sanctioned_port(
    tmp_path: Path,
) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = str(mint_execution_id())
    orch, fake, backend = _sqlite_orchestrator(tmp_path)
    ger = _ger_v2(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id="tenant-a",
        principal_id="principal-a",
        action=ACTION_CREATE_EXTERNAL_WORK,
        proof=_proof(
            task_id=str(task_id),
            run_id=str(run_id),
            tenant_id="tenant-a",
            principal_id="principal-a",
            action=ACTION_CREATE_EXTERNAL_WORK,
        ),
    )
    backend.bundle.execution_store.put_state(
        execution_id,
        GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED,
    )
    backend.bundle.execution_store.put_result(ger)
    port = GovernedSanctionedExternalWorkOperationPort(orch)
    acceptance = _acceptance()
    with patch.object(orch, "accept") as accept_mutator:
        result = port.execute(
            SanctionedExternalWorkOperationRequest(
                operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
                accept_quote=SanctionedExternalWorkAcceptQuoteRequest(
                    execution_id=execution_id,
                    principal_id="principal-a",
                    tenant_id="tenant-a",
                    acceptance=acceptance,
                    acceptance_idempotency_key="accept-idem-1",
                ),
            ),
        )
    assert result.admitted is False
    assert result.reason == "prior_create_snapshot_missing"
    assert str(result.execution_id) == execution_id
    assert result.provider_mutation_dispatched is False
    accept_mutator.assert_not_called()
    assert fake.create_calls == 0
    backend.close()
