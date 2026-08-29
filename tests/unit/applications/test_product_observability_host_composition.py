# © Artur Czarnecki. All rights reserved.

"""Product observability dashboard real harness host composition (ONE-SPINE-2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.product_observability_dashboard_wiring import (
    wire_harness_product_observability_dashboard,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.deterministic_problem_grouping import STRATEGY_ID
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
    wire_problem_persistence,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from tests.unit.applications.test_product_observability_dashboard_wiring import (
    _assess_retry_pair,
    _grouping_engine,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT_A = "governed_contractor.product"
_TENANT_B = "tenant-product-host-b"
_OBSERVED_AT = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


def _product_env() -> object:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    return manifest.environment or build_governed_contractor_environment_profile(settings)


def _seed_problems_via_lifecycle(
    persistence: object,
    *,
    tenant_id: str,
    open_count: int,
    resolved_count: int,
) -> None:
    runtime_store = InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence)
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


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

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
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    persistence = wire_problem_persistence(document_store=document_store)
    _seed_problems_via_lifecycle(
        persistence,
        tenant_id=_TENANT_A,
        open_count=1,
        resolved_count=1,
    )
    _seed_problems_via_lifecycle(
        persistence,
        tenant_id=_TENANT_B,
        open_count=1,
        resolved_count=0,
    )

    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = _product_env()
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        registry_projection=build_governed_contractor_test_registry_projection(),
        document_store=document_store,
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
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    settings = GovernedContractorBackendSettings.from_env()
    app = create_governed_contractor_backend_app(
        registry_projection=build_governed_contractor_test_registry_projection(),
        settings=settings,
        document_store=document_store,
    )
    paths = {route.path for route in app.routes}
    assert "/ops/dashboard/unified" in paths


def test_shared_problem_persistence_visible_after_lifecycle_reconcile(
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = _product_env()
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        registry_projection=build_governed_contractor_test_registry_projection(),
        document_store=document_store,
    )
    deps = resolve_host_diagnostic_read_dependencies(runtime)
    _seed_problems_via_lifecycle(
        deps.problem_persistence,
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
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    first = wire_problem_persistence(document_store=document_store)
    _seed_problems_via_lifecycle(
        first,
        tenant_id=_TENANT_A,
        open_count=1,
        resolved_count=0,
    )
    if isinstance(first, DocumentStoreProblemPersistence):
        first.close()

    restarted = wire_problem_persistence(document_store=document_store)
    runtime = build_harness_host_runtime(
        build_governed_contractor_manifest(),
        _product_env(),
        settings=GovernedContractorBackendSettings.from_env(),
        registry_projection=build_governed_contractor_test_registry_projection(),
        document_store=document_store,
    )
    deps = resolve_host_diagnostic_read_dependencies(runtime)
    service = build_diagnostic_read_service(
        HostDiagnosticReadDependencies(
            problem_persistence=restarted,
            runtime_event_persistence=deps.runtime_event_persistence,
            causal_evidence_persistence=deps.causal_evidence_persistence,
        ),
    )
    listed = service.list_problems(tenant_id=_TENANT_A)
    assert listed.total_count == 1
