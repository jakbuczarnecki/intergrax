# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X2A — canonical host diagnostic composition ownership and one-resolution."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticCompositionOverrides,
    build_grouping_strategy_registry,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    build_diagnostic_scope_discovery_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
    resolve_host_terminal_execution_diagnostic_trigger,
)
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
    close_harness_host_runtime,
)
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.diagnostic_assembly_resolver import DiagnosticAssemblyError
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DiagnosticPosture,
    DiagnosticProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.diagnostics.problem_grouping import (
    DuplicateProblemGroupingStrategyError,
    ProblemGroupingEngine,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
    STRATEGY_ID as DETERMINISTIC_STRATEGY_ID,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingStrategy,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyVersion,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.unit.applications._shared.test_obs_diag_x2_diagnostic_composition_pluginability import (
    _CustomGroupingStrategy,
    _RecordingOccurrencePersistence,
    _RecordingProblemPersistence,
    _RecordingReconstructionReader,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ARCHITECTURE_MODULES = (
    _REPO_ROOT / "intergrax/applications/_shared/diagnostic_composition.py",
    _REPO_ROOT / "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
    _REPO_ROOT / "intergrax/applications/_shared/diagnostic_read_wiring.py",
    _REPO_ROOT / "intergrax/applications/_shared/harness_host_runtime.py",
)


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _required_diagnostics_manifest(profile_id: str = "obs.diag.x2a.host") -> ApplicationManifest:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.diagnostic_profile = DiagnosticProfile(posture=DiagnosticPosture.REQUIRED)
    return ApplicationManifest.lab(
        app_id="obs_diag_x2a_host",
        name="OBS-DIAG-X2A Host",
        route_prefix="/v1/obs_diag_x2a",
        env_prefix="OBS_DIAG_X2A_",
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


def _build_product_host(
    tmp_path: Path,
    *,
    document_store: InMemoryDocumentStore | None = None,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> object:
    manifest = _required_diagnostics_manifest()
    env = manifest.environment
    assert env is not None
    return build_harness_host_runtime(
        manifest,
        env,
        registry=_echo_registry(),
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        document_store=document_store or InMemoryDocumentStore(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
        diagnostic_composition_overrides=overrides,
    )


def test_x2a_canonical_factory_default_attached_and_shared(tmp_path: Path, _stub_llm: None) -> None:
    runtime = _build_product_host(tmp_path)
    assert runtime.diagnostic_wiring.attached is True
    assert runtime.host_diagnostic_dependencies is not None

    write_deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
        runtime=runtime,
    )
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    assert write_deps is not None
    assert write_deps.persistence.problem_persistence is read_deps.problem_persistence
    assert (
        write_deps.persistence.occurrence_persistence is read_deps.occurrence_persistence
    )


def test_x2a_resolve_diagnostic_persistence_composition_called_once(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intergrax.applications._shared.diagnostic_read_wiring as read_wiring_module

    counter = {"count": 0}
    original = read_wiring_module.resolve_diagnostic_persistence_composition

    def _counting(*args: object, **kwargs: object) -> object:
        counter["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        read_wiring_module,
        "resolve_diagnostic_persistence_composition",
        _counting,
    )
    runtime = _build_product_host(tmp_path)
    resolve_host_diagnostic_read_dependencies(runtime)
    resolve_host_diagnostic_runtime_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
        runtime=runtime,
    )
    assert counter["count"] == 1


def test_x2a_custom_problem_persistence_via_factory(tmp_path: Path, _stub_llm: None) -> None:
    custom_problem = _RecordingProblemPersistence()
    custom_occurrence = _RecordingOccurrencePersistence()
    custom_causal = InMemoryCausalEvidencePersistence()
    overrides = DiagnosticCompositionOverrides(
        problem_persistence=custom_problem,
        occurrence_persistence=custom_occurrence,
        causal_evidence_persistence=custom_causal,
    )
    runtime = _build_product_host(tmp_path, overrides=overrides)
    assert runtime.diagnostic_wiring.attached is True
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    assert read_deps.problem_persistence is custom_problem
    trigger = resolve_host_terminal_execution_diagnostic_trigger(runtime)
    assert trigger is not None


def test_x2a_custom_reconstruction_reader_via_factory(tmp_path: Path, _stub_llm: None) -> None:
    reader = _RecordingReconstructionReader()
    overrides = DiagnosticCompositionOverrides(execution_reconstruction_reader=reader)
    runtime = _build_product_host(tmp_path, overrides=overrides)
    read_service = build_diagnostic_read_service(
        resolve_host_diagnostic_read_dependencies(runtime),
        overrides=overrides,
    )
    trigger = resolve_host_terminal_execution_diagnostic_trigger(runtime)
    assert read_service._reconstructor is reader  # noqa: SLF001
    assert trigger._orchestrator._execution_reconstructor is reader  # noqa: SLF001
    assert not isinstance(reader, ExecutionReconstructor)


def test_x2a_custom_grouping_strategy_via_factory(tmp_path: Path, _stub_llm: None) -> None:
    custom = _CustomGroupingStrategy()
    overrides = DiagnosticCompositionOverrides(additional_grouping_strategies=(custom,))
    runtime = _build_product_host(tmp_path, overrides=overrides)
    trigger = resolve_host_terminal_execution_diagnostic_trigger(runtime)
    registry = trigger._orchestrator._grouping_engine._registry  # noqa: SLF001
    assert custom.strategy_id in registry.registered_strategy_ids()
    assert registry.resolve(custom.strategy_id) is custom
    assert type(trigger._orchestrator._grouping_engine) is ProblemGroupingEngine  # noqa: SLF001


def test_x2a_scope_discovery_same_persistence(tmp_path: Path, _stub_llm: None) -> None:
    runtime = _build_product_host(tmp_path)
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    discovery = build_diagnostic_scope_discovery_service(read_deps)
    assert discovery is not None
    problem_provider = discovery._registry._providers[0]  # noqa: SLF001
    assert problem_provider._problem_persistence is read_deps.problem_persistence  # noqa: SLF001


def test_x2a_borrowed_persistence_not_closed_on_host_shutdown(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    custom_problem = _RecordingProblemPersistence()
    custom_occurrence = _RecordingOccurrencePersistence()
    overrides = DiagnosticCompositionOverrides(
        problem_persistence=custom_problem,
        occurrence_persistence=custom_occurrence,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    runtime = _build_product_host(tmp_path, overrides=overrides)
    close_harness_host_runtime(runtime)
    assert custom_problem.closed is False
    assert custom_occurrence.closed is False


def test_x2a_host_shutdown_closes_host_owned_persistence(
    tmp_path: Path,
    _stub_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intergrax.applications._shared.harness_host_runtime as host_runtime_module

    closed = {"count": 0}

    def _record_close(persistence: object) -> None:
        del persistence
        closed["count"] += 1

    monkeypatch.setattr(
        host_runtime_module,
        "close_host_owned_diagnostic_persistence",
        _record_close,
    )
    runtime = _build_product_host(tmp_path)
    close_harness_host_runtime(runtime)
    assert closed["count"] == 1


def test_x2a_strict_host_missing_document_store_fail_closed(
    tmp_path: Path,
    _stub_llm: None,
) -> None:
    manifest = _required_diagnostics_manifest("obs.diag.x2a.strict")
    env = manifest.environment
    assert env is not None
    with pytest.raises(DiagnosticAssemblyError, match="document store"):
        build_harness_host_runtime(
            manifest,
            env,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            document_store=None,
            trace_db_path=tmp_path / "trace.db",
            runtime_events_db_path=tmp_path / "events.db",
        )


def test_x2a_partial_override_custom_problem_only(tmp_path: Path, _stub_llm: None) -> None:
    custom_problem = _RecordingProblemPersistence()
    overrides = DiagnosticCompositionOverrides(problem_persistence=custom_problem)
    runtime = _build_product_host(tmp_path, overrides=overrides)
    read_deps = resolve_host_diagnostic_read_dependencies(runtime)
    assert read_deps.problem_persistence is custom_problem
    assert read_deps.occurrence_persistence is not None
    assert read_deps.causal_evidence_persistence is not None


def test_x2a_duplicate_grouping_strategy_still_fail_closed() -> None:
    duplicate = DeterministicProblemGroupingStrategy()
    overrides = DiagnosticCompositionOverrides(
        additional_grouping_strategies=(duplicate,),
    )
    with pytest.raises(DuplicateProblemGroupingStrategyError):
        build_grouping_strategy_registry(overrides)


def test_x2a_application_build_context_has_no_runtime_diagnostics_fields() -> None:
    forbidden = {
        "DiagnosticCompositionOverrides",
        "DiagnosticPersistenceComposition",
        "ResolvedDiagnosticComposition",
        "ProblemPersistence",
        "DiagnosticOrchestrator",
    }
    source = inspect.getsource(ApplicationBuildContext)
    tree = ast.parse(source)
    field_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.add(node.target.id)
    assert forbidden.isdisjoint(field_names)


def _imported_module_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _contains_getattr_or_hasattr(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "hasattr", "setattr"}:
                hits.append(f"{path.name}:{node.lineno}:{node.func.id}")
    return hits


def test_x2a_architecture_gates_on_touched_modules() -> None:
    forbidden_vendors = frozenset(
        {"pymongo", "motor", "redis", "psycopg", "psycopg2", "opentelemetry", "sentry_sdk"}
    )
    violations: list[str] = []
    for path in _ARCHITECTURE_MODULES:
        for root in sorted(_imported_module_roots(path)):
            if root in forbidden_vendors:
                violations.append(f"{path.name} imports vendor root {root!r}")
        violations.extend(_contains_getattr_or_hasattr(path))
    assert not violations, "\n".join(violations)
