# © Artur Czarnecki. All rights reserved.

"""HARDEN-1B — durable ConditionalDocumentStore diagnostic composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = pytest.mark.unit

_TENANT = "harden-1b-tenant"


class _QualifiedConditionalDocumentStore(InMemoryDocumentStore):
    """Test double implementing ConditionalDocumentStore — not a vendor provider."""


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _product_manifest() -> ApplicationManifest:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="harden.1b.product")
    env.application_profile = ApplicationProfile.PRODUCT
    env.observability_profile = ObservabilityProfile(
        trace_sqlite_enabled=True,
        otel_enabled=False,
        metrics_plugins_enabled=True,
    )
    return ApplicationManifest.lab(
        app_id="harden_1b_product",
        name="HARDEN-1B Product Host",
        route_prefix="/v1/harden_1b",
        env_prefix="HARDEN_1B_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
        environment=env,
    )


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


def test_harden_1b_write_and_read_share_platform_conditional_document_store(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    """
    HARDEN-1B composition proof:
    ConditionalDocumentStore → environment wiring → ToolWiringContext →
    DocumentStoreProblemPersistence for both runtime write and read paths.
    """
    store = _QualifiedConditionalDocumentStore()
    manifest = _product_manifest()
    env = manifest.environment
    assert env is not None

    runtime = build_harness_host_runtime(
        manifest,
        env,
        registry=_echo_registry(),
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        document_store=store,
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
    )
    wiring_context = runtime.env_wiring.build_context.tool_wiring_context
    assert wiring_context is not None
    assert wiring_context.document_store is store
    assert_conditional_document_store(wiring_context.document_store)

    runtime_deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
    )
    assert runtime_deps is not None
    assert isinstance(runtime_deps.problem_persistence, DocumentStoreProblemPersistence)

    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    assert isinstance(read_deps.problem_persistence, DocumentStoreProblemPersistence)

    record = runtime_deps.problem_persistence.create(sample_problem(tenant_id=_TENANT))
    read_service = build_diagnostic_read_service(read_deps)
    listed = read_service.list_problems(tenant_id=_TENANT)
    assert listed.total_count == 1
    assert listed.problems[0].problem_id == record.problem_id
