# © Artur Czarnecki. All rights reserved.

"""Product observability dashboard real harness host composition (ONE-SPINE-2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    HostDiagnosticReadDependencies,
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared import harness_host_runtime as harness_host_runtime_module
from intergrax.applications._shared.product_observability_dashboard_wiring import (
    wire_harness_product_observability_dashboard,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
    resolve_reference_production_strict_host_environment,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from intergrax.runtime.diagnostics.deterministic_problem_grouping import STRATEGY_ID
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
    wire_problem_persistence,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    HarnessMeaningfulSideEffectAuthorizationWiring,
)
from tests.unit.applications.test_product_observability_dashboard_wiring import (
    _assess_retry_pair,
    _grouping_engine,
)
from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_occurrence_persistence_for_tests,
)


def _durable_continuation_store_for_tests() -> object:
    backing = ExecutionContinuationDurableBacking()
    backing_execution_continuation_state_store(backing)
    export = export_durable_continuation_state(backing)
    return execution_continuation_state_store_from_durable_export(export)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT_A = "governed_contractor.product"
_TENANT_B = "tenant-product-host-b"
_OBSERVED_AT = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


def _strict_governed_contractor_harness_kwargs(
    *,
    document_store: InMemoryDocumentStore,
    tmp_path: Path | None = None,
) -> dict[str, object]:
    if tmp_path is None:
        raise ValueError("tmp_path is required for strict governed contractor harness")
    kv_path = tmp_path / "strict_host_kv.db"
    platform = build_reference_production_platform_persistence(db_path=kv_path)
    return {
        "document_store": document_store,
        "key_value_cache": platform.kv_store,
        "execution_continuation_state_store": _durable_continuation_store_for_tests(),
    }


def _product_env() -> object:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    return manifest.environment or build_governed_contractor_environment_profile(settings)


def _seed_problems_via_lifecycle(
    persistence: object,
    *,
    occurrence_persistence: object,
    tenant_id: str,
    open_count: int,
    resolved_count: int,
) -> None:
    runtime_store = InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence, occurrence_persistence)
    grouping_engine = _grouping_engine()

    open_violations = [
        RuntimeEventType.RETRY_SCHEDULED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.PAUSE_REQUESTED,
    ]
    for index in range(open_count):
        grouping = grouping_engine.group(
            _assess_retry_pair(
                tenant_id=tenant_id,
                runtime_store=runtime_store,
                violating_event_type=open_violations[index % len(open_violations)],
            ),
            strategy_id=STRATEGY_ID,
        )
        lifecycle.reconcile(
            grouping,
            observed_at=_OBSERVED_AT + timedelta(minutes=index),
        )

    resolved_violations = [
        RuntimeEventType.PAUSE_REQUESTED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.RETRY_SCHEDULED,
    ]
    for index in range(resolved_count):
        grouping = grouping_engine.group(
            _assess_retry_pair(
                tenant_id=tenant_id,
                runtime_store=runtime_store,
                violating_event_type=resolved_violations[index % len(resolved_violations)],
            ),
            strategy_id=STRATEGY_ID,
        )
        result = lifecycle.reconcile(
            grouping,
            observed_at=_OBSERVED_AT + timedelta(hours=1, minutes=index),
        )
        problem = result.created[0] if result.created else result.updated[0]
        lifecycle.resolve(
            tenant_id=tenant_id,
            problem_id=problem.problem_id,
            resolved_at=_OBSERVED_AT + timedelta(hours=2, minutes=index),
        )


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    install_diagnostic_cursor_secret(monkeypatch)
    real_build_harness_host_runtime = harness_host_runtime_module.build_harness_host_runtime

    def _build_harness_host_runtime_with_test_governance(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("meaningful_side_effect_authorization", MagicMock())
        return real_build_harness_host_runtime(*args, **kwargs)

    monkeypatch.setattr(
        "intergrax.applications._shared.harness_host_runtime.build_harness_host_runtime",
        _build_harness_host_runtime_with_test_governance,
    )
    monkeypatch.setattr(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
        _build_harness_host_runtime_with_test_governance,
    )
    def _resolve_meaningful_side_effect_wiring(*args: object, **kwargs: object) -> HarnessMeaningfulSideEffectAuthorizationWiring:
        explicit = kwargs.get("explicit")
        if explicit is not None:
            return HarnessMeaningfulSideEffectAuthorizationWiring(authorization_port=explicit)
        return HarnessMeaningfulSideEffectAuthorizationWiring(
            authorization_port=None,
            owned_collaborative_work_persistence=None,
        )

    monkeypatch.setattr(
        "intergrax.applications._shared.harness_host_runtime.resolve_harness_host_meaningful_side_effect_authorization_wiring",
        _resolve_meaningful_side_effect_wiring,
    )
    monkeypatch.setattr(
        "governed_contractor_application.host.factory.resolve_governed_contractor_collaborative_work_integration_profile",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.runtime.nexus.nexus_loop.validate_durable_attempt_lifecycle_for_composition",
        lambda **_kwargs: None,
    )
    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_host_composition_dashboard_diagnostics_ready_with_tenant_scope(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    persistence = wire_problem_persistence(list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET, document_store=document_store)
    occurrence = document_store_occurrence_persistence_for_tests(document_store)
    _seed_problems_via_lifecycle(
        persistence,
        occurrence_persistence=occurrence,
        tenant_id=_TENANT_A,
        open_count=1,
        resolved_count=1,
    )
    _seed_problems_via_lifecycle(
        persistence,
        occurrence_persistence=occurrence,
        tenant_id=_TENANT_B,
        open_count=1,
        resolved_count=0,
    )

    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = resolve_reference_production_strict_host_environment(_product_env())
    runtime = harness_host_runtime_module.build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        registry_projection=build_governed_contractor_test_registry_projection(),
        tenant_id=manifest.app_id,
        **_strict_governed_contractor_harness_kwargs(
            document_store=document_store,
            tmp_path=tmp_path,
        ),
    )
    app = FastAPI()
    wire_harness_product_observability_dashboard(
        app,
        runtime=runtime,
        repo_root=_REPO_ROOT,
    )

    client = TestClient(app)
    response = client.get("/ops/dashboard/unified")
    assert response.status_code == 200
    payload = response.json()
    diagnostics = payload["dashboard"]["diagnostics"]
    assert diagnostics["ready"] is True
    assert diagnostics["problem_count"] == 2
    assert diagnostics["open_problem_count"] == 1
    auditability = payload["dashboard"]["health"]["auditability"]
    assert auditability["auditability_ready"] is True
    assert auditability["diagnostics_attached"] is True


def test_governed_contractor_factory_mounts_product_observability_dashboard(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    settings = GovernedContractorBackendSettings.from_env()
    trace_db_path = tmp_path / "trace.db"
    app = create_governed_contractor_backend_app(
        registry_projection=build_governed_contractor_test_registry_projection(),
        settings=settings,
        trace_db_path=trace_db_path,
        runtime_events_db_path=tmp_path / "runtime_events.db",
        checkpoints_db_path=tmp_path / "checkpoints.db",
        **_strict_governed_contractor_harness_kwargs(
            document_store=document_store,
            tmp_path=tmp_path,
        ),
    )
    paths = {route.path for route in app.routes}
    assert "/ops/dashboard/unified" in paths


def test_shared_problem_persistence_visible_after_lifecycle_reconcile(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = resolve_reference_production_strict_host_environment(_product_env())
    runtime = harness_host_runtime_module.build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        registry_projection=build_governed_contractor_test_registry_projection(),
        tenant_id=manifest.app_id,
        **_strict_governed_contractor_harness_kwargs(
            document_store=document_store,
            tmp_path=tmp_path,
        ),
    )
    deps = resolve_host_diagnostic_read_dependencies(runtime)
    _seed_problems_via_lifecycle(
        deps.problem_persistence,
        occurrence_persistence=deps.occurrence_persistence,
        tenant_id=_TENANT_A,
        open_count=1,
        resolved_count=0,
    )

    app = FastAPI()
    wiring = wire_harness_product_observability_dashboard(
        app,
        runtime=runtime,
        repo_root=_REPO_ROOT,
    )
    assert wiring.dashboard is not None
    assert wiring.dashboard.diagnostics.ready is True
    assert wiring.dashboard.diagnostics.problem_count == 1
    assert wiring.dashboard.diagnostics.open_problem_count == 1
    assert wiring.dashboard.health.auditability.auditability_ready is True
    assert wiring.dashboard.health.auditability.diagnostics_attached is True


def test_durable_problem_persistence_survives_adapter_restart(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    first = wire_problem_persistence(list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET, document_store=document_store)
    occurrence = document_store_occurrence_persistence_for_tests(document_store)
    _seed_problems_via_lifecycle(
        first,
        occurrence_persistence=occurrence,
        tenant_id=_TENANT_A,
        open_count=1,
        resolved_count=0,
    )
    if isinstance(first, DocumentStoreProblemPersistence):
        first.close()

    restarted = wire_problem_persistence(list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET, document_store=document_store)
    strict_env = resolve_reference_production_strict_host_environment(_product_env())
    manifest = build_governed_contractor_manifest()
    runtime = harness_host_runtime_module.build_harness_host_runtime(
        manifest.model_copy(update={"environment": strict_env}),
        strict_env,
        settings=GovernedContractorBackendSettings.from_env(),
        registry_projection=build_governed_contractor_test_registry_projection(),
        tenant_id=manifest.app_id,
        **_strict_governed_contractor_harness_kwargs(
            document_store=document_store,
            tmp_path=tmp_path,
        ),
    )
    deps = resolve_host_diagnostic_read_dependencies(runtime)
    from intergrax.applications._shared.diagnostic_composition import (
        DiagnosticComponentOwnership,
        DiagnosticPersistenceComposition,
    )

    service = build_diagnostic_read_service(
        HostDiagnosticReadDependencies(
            persistence=DiagnosticPersistenceComposition(
                problem_persistence=restarted,
                occurrence_persistence=deps.occurrence_persistence,
                causal_evidence_persistence=deps.causal_evidence_persistence,
                runtime_event_persistence=deps.runtime_event_persistence,
                problem_persistence_ownership=DiagnosticComponentOwnership.BORROWED,
                occurrence_persistence_ownership=(
                    deps.persistence.occurrence_persistence_ownership
                ),
                causal_evidence_persistence_ownership=(
                    deps.persistence.causal_evidence_persistence_ownership
                ),
            ),
            execution_lineage_reader=deps.execution_lineage_reader,
        ),
    )
    listed = service.list_problems(tenant_id=_TENANT_A)
    assert listed.total_count == 1
