# © Artur Czarnecki. All rights reserved.

"""HARDEN-1C — durable central Problem store survives full store/runtime restart."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_integration
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    grouping_provenance_from_problem_provenance,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_TENANT = "harden-1c-tenant"
_OTHER_TENANT = "harden-1c-other-tenant"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_1c"
_COLLECTION_PREFIX = "harden_1c_"


def _resolve_mongodb_uri() -> str:
    return os.environ.get("INTERGRAX_MONGODB_URI", _DEFAULT_URI).strip() or _DEFAULT_URI


def _probe_mongodb_backend(*, collection_name: str) -> None:
    try:
        bundle = create_mongodb_integration(
            uri=_resolve_mongodb_uri(),
            database=_DEFAULT_DATABASE,
            collection_name=collection_name,
        )
        store = bundle.document_store.as_document_store()
        store.close()
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        pytest.skip(
            "MongoDB backend unavailable for HARDEN-1C: "
            f"{type(exc).__name__}: {exc}. "
            "Start infra/docker/mongodb/docker-compose.yml or set INTERGRAX_MONGODB_URI.",
        )


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _product_manifest(*, integration_profile: IntegrationProfile) -> ApplicationManifest:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="harden.1c.product")
    env.application_profile = ApplicationProfile.PRODUCT
    env.integration_profile = integration_profile
    env.observability_profile = ObservabilityProfile(
        trace_sqlite_enabled=True,
        otel_enabled=False,
        metrics_plugins_enabled=True,
    )
    return ApplicationManifest.lab(
        app_id="harden_1c_product",
        name="HARDEN-1C Product Host",
        route_prefix="/v1/harden_1c",
        env_prefix="HARDEN_1C_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=env,
        integration_profile=integration_profile,
    )


def _build_product_runtime(
    tmp_path: Path,
    *,
    collection_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[HarnessHostRuntime, ConditionalDocumentStore]:
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", _resolve_mongodb_uri())
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", _DEFAULT_DATABASE)
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", collection_name)

    bootstrap_application_integration_catalog()
    integration_profile = IntegrationProfile(document_store="mongodb")
    manifest = _product_manifest(integration_profile=integration_profile)
    env = manifest.environment
    assert env is not None

    runtime = build_harness_host_runtime(
        manifest,
        env,
        registry=_echo_registry(),
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
    )
    wiring_context = runtime.env_wiring.build_context.tool_wiring_context
    assert wiring_context is not None
    assert wiring_context.document_store is not None
    store = assert_conditional_document_store(wiring_context.document_store)
    return runtime, store


def _shutdown_runtime_phase(store: ConditionalDocumentStore) -> None:
    store.close()


@pytest.fixture
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_harden_1c_durable_problem_survives_full_store_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _stub_llm: None,
) -> None:
    """
    HARDEN-1C process-equivalent proof:

    IntegrationProfile → DOCUMENT_STORE → ConditionalDocumentStore →
    DocumentStoreProblemPersistence (phase A write) → close store/runtime A →
    new IntegrationProfile resolution (phase B) → DiagnosticReadService read.
    """
    collection_name = f"{_COLLECTION_PREFIX}{uuid.uuid4().hex}"
    _probe_mongodb_backend(collection_name=collection_name)

    runtime_a, store_a = _build_product_runtime(
        tmp_path / "phase_a",
        collection_name=collection_name,
        monkeypatch=monkeypatch,
    )
    runtime_deps_a = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime_a.env_wiring,
        observability=runtime_a.observability,
    )
    assert runtime_deps_a is not None
    assert isinstance(runtime_deps_a.problem_persistence, DocumentStoreProblemPersistence)

    record = runtime_deps_a.problem_persistence.create(sample_problem(tenant_id=_TENANT))
    problem_id = record.problem_id
    expected_status = record.status
    expected_provenance = record.provenance
    expected_occurrences = record.occurrences
    expected_occurrence_count = record.occurrence_count

    store_a_id = id(store_a)
    runtime_a_id = id(runtime_a)
    _shutdown_runtime_phase(store_a)
    del runtime_a
    del store_a
    del runtime_deps_a

    runtime_b, store_b = _build_product_runtime(
        tmp_path / "phase_b",
        collection_name=collection_name,
        monkeypatch=monkeypatch,
    )
    assert id(store_b) != store_a_id
    assert id(runtime_b) != runtime_a_id

    read_deps_b = resolve_host_diagnostic_read_dependencies(runtime_b)
    assert isinstance(read_deps_b.problem_persistence, DocumentStoreProblemPersistence)
    read_service_b = build_diagnostic_read_service(read_deps_b)

    listed = read_service_b.list_problems(tenant_id=_TENANT)
    assert listed.total_count == 1
    summary = listed.problems[0]
    assert summary.problem_id == problem_id
    assert summary.tenant_id == _TENANT
    assert summary.status == expected_status
    assert summary.occurrence_count == expected_occurrence_count

    detail = read_service_b.get_problem(tenant_id=_TENANT, problem_id=problem_id)
    assert detail is not None
    assert detail.grouping_provenance == grouping_provenance_from_problem_provenance(
        expected_provenance,
    )
    assert detail.total_occurrence_count == len(expected_occurrences)
    assert detail.occurrences[0].observed_at == expected_occurrences[0].observed_at

    other_tenant = read_service_b.list_problems(tenant_id=_OTHER_TENANT)
    assert other_tenant.total_count == 0
    assert other_tenant.problems == ()

    _shutdown_runtime_phase(store_b)
