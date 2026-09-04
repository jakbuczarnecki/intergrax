# © Artur Czarnecki. All rights reserved.

"""DIAG-FOUNDATION-5 — destructive proof that central diagnostics cannot be removed silently."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.auditability_health_wiring import (
    assert_host_auditability_health_valid,
    project_host_auditability_health_facts_from_runtime,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticAssemblyError,
    DiagnosticReadiness,
    DiagnosticWiring,
    resolve_central_diagnostics_required,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.health_dashboard_wiring import (
    resolve_health_dashboard_wiring_from_runtime,
)
from intergrax.applications._shared.observability_assembly_resolver import ObservabilityAssemblyError
from intergrax.applications._shared.scenario_runtime_baseline import ScenarioRuntimeBuildError
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DiagnosticPosture,
    DiagnosticProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
    diagnostic_subsystem_failure_observed_for_run,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge import (
    invoke_terminal_execution_diagnostics,
)
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.unit.applications.scenario_runtime_test_support import (
    MinimalProductionScenarioTestConfig,
    build_valid_minimal_production_scenario_fixture,
    echo_agent_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "df5-destructive"
_OBSERVED_AT = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)


def _echo_registry() -> AgentRegistry:
    return echo_agent_registry()


def _product_manifest(profile_id: str = "df5.product.host") -> ApplicationManifest:
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)
    environment.execution_mode = ExecutionMode.BALANCED
    environment.observability_profile = environment.observability_profile.model_copy(
        update={"otel_enabled": False},
    )
    return ApplicationManifest.lab(
        app_id="df5_product_host",
        name="DF5 Product Host",
        route_prefix="/v1/df5_product",
        env_prefix="DF5_PRODUCT_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=environment,
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


def test_df5_entrypoint_consistency_prerequisite_from_df4() -> None:
    from tests.unit.runtime.architecture.test_diag_foundation_4_entrypoint_consistency import (
        DF4_BEHAVIOR_TABLE,
        test_df4_behavior_table_covers_all_required_entrypoints,
    )

    test_df4_behavior_table_covers_all_required_entrypoints()
    assert {row.entrypoint for row in DF4_BEHAVIOR_TABLE} == {
        "standard_task",
        "scenario_task",
        "background_task",
        "child_execution",
        "hosted_application",
    }


def test_df5_case_a_product_without_document_store_fails(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest("df5.case.a")
    env = manifest.environment
    assert env is not None
    with pytest.raises(DiagnosticAssemblyError, match="document store"):
        build_harness_host_runtime(
            manifest,
            env,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            trace_db_path=tmp_path / "trace.db",
            runtime_events_db_path=tmp_path / "events.db",
            document_store=None,
        )


def test_df5_case_b_product_without_runtime_event_persistence_fails(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest("df5.case.b")
    env = manifest.environment
    assert env is not None
    env.observability_profile = env.observability_profile.model_copy(
        update={"trace_sqlite_enabled": False},
    )
    with pytest.raises(DiagnosticAssemblyError, match="RuntimeEvent persistence"):
        build_harness_host_runtime(
            manifest,
            env,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            document_store=InMemoryDocumentStore(),
            use_in_memory_trace=True,
            runtime_events_db_path=None,
        )


def test_df5_case_c_product_without_attached_diagnostics_fails_closed(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _product_manifest("df5.case.c")
    env = manifest.environment
    assert env is not None
    monkeypatch.setattr(
        "intergrax.applications._shared.diagnostic_runtime_wiring.try_build_terminal_execution_diagnostic_trigger",
        lambda **_kwargs: None,
    )
    with pytest.raises(DiagnosticAssemblyError, match="could not be attached"):
        build_harness_host_runtime(
            manifest,
            env,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            document_store=InMemoryDocumentStore(),
            trace_db_path=tmp_path / "trace.db",
            runtime_events_db_path=tmp_path / "events.db",
        )


def test_df5_case_d_production_scenario_without_diagnostics_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.diagnostic_runtime_wiring.try_build_terminal_execution_diagnostic_trigger",
        lambda **_kwargs: None,
    )
    with pytest.raises(ScenarioRuntimeBuildError, match="central diagnostics are required"):
        build_valid_minimal_production_scenario_fixture(
            tmp_path,
            MinimalProductionScenarioTestConfig(
                tenant_id=_TENANT,
                profile_id="df5.case.d",
                app_id="df5_prod_no_diag",
                document_store=InMemoryDocumentStore(),
            ),
        )


def test_df5_case_d_production_scenario_with_diagnostics_attaches(
    tmp_path: Path,
) -> None:
    composition = build_valid_minimal_production_scenario_fixture(
        tmp_path,
        MinimalProductionScenarioTestConfig(
            tenant_id=_TENANT,
            profile_id="df5.case.d.ok",
            app_id="df5_prod_ok",
            document_store=InMemoryDocumentStore(),
        ),
    )
    assert composition.diagnostic_wiring.required is True
    assert composition.diagnostic_wiring.readiness is DiagnosticReadiness.ATTACHED
    assert composition.has_terminal_diagnostic_trigger is True


def test_df5_case_e_product_not_required_profile_still_requires_diagnostics() -> None:
    product_env = ApplicationEnvironmentProfile.product_defaults(profile_id="df5.case.e")
    product_env.diagnostic_profile = DiagnosticProfile(posture=DiagnosticPosture.NOT_REQUIRED)
    assert resolve_central_diagnostics_required(product_env) is True


def test_df5_case_f_side_diagnostic_orchestrator_architecture_gate() -> None:
    from tests.unit.runtime.architecture.test_one_spine_diagnostic_orchestrator_gate import (
        test_production_code_cannot_invoke_diagnostic_orchestrator_directly,
    )

    test_production_code_cannot_invoke_diagnostic_orchestrator_directly()


def test_df5_case_g_side_generic_problem_store_architecture_gate() -> None:
    from tests.unit.runtime.architecture.test_one_spine_problem_store_gate import (
        test_production_code_cannot_construct_side_problem_stores,
    )

    test_production_code_cannot_construct_side_problem_stores()


def test_df5_health_gate_fails_on_real_runtime_when_diagnostics_unavailable(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _product_manifest("df5.health.gate")
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(
        manifest,
        env,
        registry=_echo_registry(),
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        document_store=InMemoryDocumentStore(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
    )
    object.__setattr__(
        runtime,
        "diagnostic_wiring",
        DiagnosticWiring(required=True, attached=False),
    )
    wiring = resolve_health_dashboard_wiring_from_runtime(
        runtime,
        diagnostic_read_side_ready=False,
    )
    assert wiring.contract is not None
    assert wiring.contract.auditability.auditability_ready is False
    facts = project_host_auditability_health_facts_from_runtime(
        runtime,
        diagnostic_read_side_ready=False,
    )
    with pytest.raises(ObservabilityAssemblyError, match="auditability is not ready"):
        assert_host_auditability_health_valid(facts, env)


def test_df5_observability_gate_fails_when_central_diagnostics_not_attached(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.maintenance import check_harness_observability_wiring as gate

    manifest = _product_manifest("df5.obs.gate")
    env = manifest.environment
    assert env is not None

    def _runtime_without_attached_diagnostics(*args: object, **kwargs: object) -> object:
        runtime = build_harness_host_runtime(*args, **kwargs)  # type: ignore[arg-type]
        object.__setattr__(
            runtime,
            "diagnostic_wiring",
            DiagnosticWiring(required=True, attached=False),
        )
        return runtime

    monkeypatch.setattr(gate, "build_harness_host_runtime", _runtime_without_attached_diagnostics)
    assert gate._audit_product_host_auditability() == 1


def test_df5_case_h_diagnostic_runtime_failure_emits_durable_evidence() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    identity = ExecutionIdentityBinding(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
            execution_identity=identity,
        )
    finally:
        reset_active_execution_identity(token)

    assert result is None
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT,
        run_id=run_id,
    )


def test_df5_case_i_evidence_persistence_failure_preserves_business_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    identity = ExecutionIdentityBinding(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )

    def _raise_evidence(*_args: object, **_kwargs: object) -> None:
        raise OSError("evidence journal unavailable")

    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge._persist_diagnostic_subsystem_failure",
        _raise_evidence,
    )

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
            execution_identity=identity,
        )
    finally:
        reset_active_execution_identity(token)

    assert result is None
    assert not diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT,
        run_id=run_id,
    )
