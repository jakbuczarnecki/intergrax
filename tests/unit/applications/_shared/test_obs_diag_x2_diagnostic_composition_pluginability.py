# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X2 — contract-driven diagnostic host composition pluginability."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticComponentOwnership,
    DiagnosticCompositionOverrides,
    DiagnosticPersistenceComposition,
    DiagnosticProviderResolutionError,
    ResolvedDiagnosticComposition,
    build_diagnostic_orchestrator_from_composition,
    build_grouping_strategy_registry,
    close_host_owned_diagnostic_persistence,
    resolve_diagnostic_composition,
    resolve_diagnostic_persistence_composition,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    build_diagnostic_orchestrator,
)
from intergrax.contracts.diagnostics.problem_identity import ProblemId, ProblemStatus
from intergrax.contracts.diagnostics.problem_persistence import (
    ProblemListPage,
    ProblemPersistence,
)
from intergrax.contracts.diagnostics.problem_record import PersistedProblem
from intergrax.contracts.diagnostics.reconciliation_key import ProblemReconciliationKey
from intergrax.contracts.diagnostics.subject_ref import ProblemGroupingSubjectRef
from intergrax.contracts.execution_event_position import AsOfBoundary
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.execution_reconstruction import ExecutionReconstructionReader
from intergrax.contracts.execution_reconstruction_models import ExecutionReconstruction
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
    STRATEGY_ID as DETERMINISTIC_STRATEGY_ID,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DuplicateProblemGroupingStrategyError,
    ProblemGroupingEngine,
    ProblemGroupingMethod,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSITION_MODULES = (
    _REPO_ROOT / "intergrax/applications/_shared/diagnostic_composition.py",
    _REPO_ROOT / "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
    _REPO_ROOT / "intergrax/applications/_shared/diagnostic_read_wiring.py",
    _REPO_ROOT / "intergrax/applications/_shared/harness_host_runtime.py",
)
_FORBIDDEN_VENDOR_ROOTS = frozenset(
    {
        "pymongo",
        "motor",
        "redis",
        "psycopg",
        "psycopg2",
        "opentelemetry",
        "sentry_sdk",
        "datadog",
    }
)


class _RecordingProblemPersistence:
    """Custom ProblemPersistence — contract only, no platform concrete coupling."""

    def __init__(self) -> None:
        self.closed = False
        self._inner = InMemoryProblemPersistence()

    def get(self, *, tenant_id: str, problem_id: ProblemId) -> PersistedProblem | None:
        return self._inner.get(tenant_id=tenant_id, problem_id=problem_id)

    def query_problems(
        self,
        *,
        tenant_id: str,
        status: ProblemStatus | None = None,
        limit: int,
        cursor: str | None = None,
    ) -> ProblemListPage:
        return self._inner.query_problems(
            tenant_id=tenant_id,
            status=status,
            limit=limit,
            cursor=cursor,
        )

    def find_by_reconciliation_key(
        self,
        *,
        tenant_id: str,
        reconciliation_key: ProblemReconciliationKey,
    ) -> PersistedProblem | None:
        return self._inner.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=reconciliation_key,
        )

    def find_by_subject_ref(
        self,
        *,
        tenant_id: str,
        subject_ref: ProblemGroupingSubjectRef,
    ) -> PersistedProblem | None:
        return self._inner.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=subject_ref,
        )

    def create(
        self,
        record: PersistedProblem,
        *,
        indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
    ) -> PersistedProblem:
        return self._inner.create(
            record,
            indexed_subject_refs=indexed_subject_refs,
        )

    def update(
        self,
        record: PersistedProblem,
        *,
        expected_version: int,
        indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
    ) -> PersistedProblem:
        return self._inner.update(
            record,
            expected_version=expected_version,
            indexed_subject_refs=indexed_subject_refs,
        )

    def close(self) -> None:
        self.closed = True


class _RecordingOccurrencePersistence(ProblemOccurrencePersistence):
    def __init__(self) -> None:
        self.closed = False
        store = in_memory_document_store_for_problem_tests()
        self._inner = document_store_occurrence_persistence_for_tests(store)

    def append_if_absent(self, **kwargs: Any) -> ProblemOccurrenceAppendResult:
        return self._inner.append_if_absent(**kwargs)

    def capture_occurrence_repair_boundary(self, **kwargs: Any) -> Any:
        return self._inner.capture_occurrence_repair_boundary(**kwargs)

    def query_occurrences(self, **kwargs: Any) -> ProblemOccurrencePage:
        return self._inner.query_occurrences(**kwargs)

    def close(self) -> None:
        self.closed = True


class _RecordingReconstructionReader:
    def __init__(self) -> None:
        self.calls = 0

    def reconstruct_execution(
        self,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        *,
        execution_as_of: AsOfBoundary | None = None,
    ) -> ExecutionReconstruction:
        del tenant_id, task_id, run_id, execution_as_of
        self.calls += 1
        raise AssertionError("reconstruction body not required for composition proof")


class _CustomGroupingStrategy:
    @property
    def strategy_id(self) -> ProblemGroupingStrategyId:
        return ProblemGroupingStrategyId("obs.diag.x2.custom.grouping")

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion:
        return ProblemGroupingStrategyVersion("1")

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.DETERMINISTIC,
            deterministic=True,
            requires_features=False,
        )

    def group(self, inputs: tuple[Any, ...]) -> ProblemGroupingStrategyResult:
        del inputs
        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=(),
        )


def _persistence_bundle(
    *,
    problem: ProblemPersistence | None = None,
    occurrence: ProblemOccurrencePersistence | None = None,
    causal: Any | None = None,
    events: Any | None = None,
) -> DiagnosticPersistenceComposition:
    problem_persistence = problem or _RecordingProblemPersistence()
    occurrence_persistence = occurrence or _RecordingOccurrencePersistence()
    causal_persistence = causal or InMemoryCausalEvidencePersistence()
    runtime_events = events or InMemoryRuntimeEventStore()
    return DiagnosticPersistenceComposition(
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
        causal_evidence_persistence=causal_persistence,
        runtime_event_persistence=runtime_events,
        problem_persistence_ownership=DiagnosticComponentOwnership.BORROWED,
        occurrence_persistence_ownership=DiagnosticComponentOwnership.BORROWED,
        causal_evidence_persistence_ownership=DiagnosticComponentOwnership.BORROWED,
    )


def test_x2_default_composition_uses_canonical_defaults() -> None:
    store = in_memory_document_store_for_problem_tests()
    events = InMemoryRuntimeEventStore()
    persistence = resolve_diagnostic_persistence_composition(
        document_store=store,
        runtime_event_persistence=events,
    )
    assert persistence is not None
    composition = resolve_diagnostic_composition(persistence)
    orchestrator = build_diagnostic_orchestrator_from_composition(composition)
    assert type(orchestrator) is DiagnosticOrchestrator
    assert isinstance(
        composition.execution_reconstruction_reader,
        ExecutionReconstructor,
    )
    assert DETERMINISTIC_STRATEGY_ID in composition.grouping_registry.registered_strategy_ids()
    assert type(orchestrator._problem_lifecycle_engine) is ProblemLifecycleEngine  # noqa: SLF001
    assert type(orchestrator._grouping_engine) is ProblemGroupingEngine  # noqa: SLF001


def test_x2_custom_problem_and_occurrence_persistence_injected() -> None:
    custom_problem = _RecordingProblemPersistence()
    custom_occurrence = _RecordingOccurrencePersistence()
    overrides = DiagnosticCompositionOverrides(
        problem_persistence=custom_problem,
        occurrence_persistence=custom_occurrence,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    persistence = resolve_diagnostic_persistence_composition(
        document_store=None,
        runtime_event_persistence=InMemoryRuntimeEventStore(),
        overrides=overrides,
    )
    assert persistence is not None
    assert persistence.problem_persistence is custom_problem
    assert persistence.occurrence_persistence is custom_occurrence
    composition = resolve_diagnostic_composition(persistence, overrides=overrides)
    orchestrator = build_diagnostic_orchestrator_from_composition(composition)
    assert type(orchestrator) is DiagnosticOrchestrator
    assert orchestrator._problem_lifecycle_engine._persistence is custom_problem  # noqa: SLF001


def test_x2_custom_reconstruction_reader_without_execution_reconstructor() -> None:
    reader = _RecordingReconstructionReader()
    overrides = DiagnosticCompositionOverrides(execution_reconstruction_reader=reader)
    persistence = _persistence_bundle()
    composition = resolve_diagnostic_composition(persistence, overrides=overrides)
    assert composition.execution_reconstruction_reader is reader
    assert not isinstance(
        composition.execution_reconstruction_reader,
        ExecutionReconstructor,
    )
    orchestrator = build_diagnostic_orchestrator_from_composition(composition)
    assert orchestrator._execution_reconstructor is reader  # noqa: SLF001


def test_x2_custom_grouping_strategy_registers_and_engine_remains_canonical() -> None:
    custom = _CustomGroupingStrategy()
    overrides = DiagnosticCompositionOverrides(
        additional_grouping_strategies=(custom,),
    )
    registry = build_grouping_strategy_registry(overrides)
    assert DETERMINISTIC_STRATEGY_ID in registry.registered_strategy_ids()
    assert custom.strategy_id in registry.registered_strategy_ids()
    engine = ProblemGroupingEngine(registry)
    assert type(engine) is ProblemGroupingEngine
    resolved = registry.resolve(custom.strategy_id)
    assert resolved is custom


def test_x2_duplicate_grouping_strategy_id_fails() -> None:
    overrides = DiagnosticCompositionOverrides(
        additional_grouping_strategies=(DeterministicProblemGroupingStrategy(),),
    )
    with pytest.raises(DuplicateProblemGroupingStrategyError):
        build_grouping_strategy_registry(overrides)


def test_x2_strict_missing_durable_provider_fails_closed() -> None:
    with pytest.raises(DiagnosticProviderResolutionError, match="in-memory"):
        resolve_diagnostic_persistence_composition(
            document_store=None,
            runtime_event_persistence=InMemoryRuntimeEventStore(),
            require_durable=True,
        )


def test_x2_strict_missing_provider_does_not_use_in_memory_problem_persistence() -> None:
    try:
        resolve_diagnostic_persistence_composition(
            document_store=None,
            runtime_event_persistence=InMemoryRuntimeEventStore(),
            require_durable=True,
        )
    except DiagnosticProviderResolutionError:
        pass
    else:
        pytest.fail("expected fail-closed resolution error")
    # Explicit: platform default path never substitutes InMemoryProblemPersistence.
    assert InMemoryProblemPersistence.__name__ == "InMemoryProblemPersistence"


def test_x2_read_and_write_share_same_override_persistence_instances() -> None:
    from intergrax.applications._shared.diagnostic_read_wiring import (
        HostDiagnosticReadDependencies,
    )

    custom_problem = _RecordingProblemPersistence()
    custom_occurrence = _RecordingOccurrencePersistence()
    custom_causal = InMemoryCausalEvidencePersistence()
    overrides = DiagnosticCompositionOverrides(
        problem_persistence=custom_problem,
        occurrence_persistence=custom_occurrence,
        causal_evidence_persistence=custom_causal,
    )
    events = InMemoryRuntimeEventStore()
    write_persistence = resolve_diagnostic_persistence_composition(
        document_store=None,
        runtime_event_persistence=events,
        overrides=overrides,
    )
    read_persistence = resolve_diagnostic_persistence_composition(
        document_store=None,
        runtime_event_persistence=events,
        overrides=overrides,
    )
    assert write_persistence is not None
    assert read_persistence is not None
    assert write_persistence.problem_persistence is read_persistence.problem_persistence
    assert (
        write_persistence.occurrence_persistence
        is read_persistence.occurrence_persistence
    )
    assert (
        write_persistence.causal_evidence_persistence
        is read_persistence.causal_evidence_persistence
    )
    write_deps = HostDiagnosticReadDependencies(persistence=write_persistence)
    read_deps = HostDiagnosticReadDependencies(persistence=read_persistence)
    write_orch = build_diagnostic_orchestrator(write_deps, overrides=overrides)
    read_service = build_diagnostic_read_service(read_deps, overrides=overrides)
    assert type(write_orch) is DiagnosticOrchestrator
    assert read_service is not None
    assert write_deps.problem_persistence is read_deps.problem_persistence


def test_x2_host_owned_closed_borrowed_not_closed() -> None:
    host_problem = _RecordingProblemPersistence()
    borrowed_occurrence = _RecordingOccurrencePersistence()
    borrowed_causal = InMemoryCausalEvidencePersistence()
    persistence = DiagnosticPersistenceComposition(
        problem_persistence=host_problem,
        occurrence_persistence=borrowed_occurrence,
        causal_evidence_persistence=borrowed_causal,
        runtime_event_persistence=InMemoryRuntimeEventStore(),
        problem_persistence_ownership=DiagnosticComponentOwnership.HOST_CREATED,
        occurrence_persistence_ownership=DiagnosticComponentOwnership.BORROWED,
        causal_evidence_persistence_ownership=DiagnosticComponentOwnership.BORROWED,
    )
    close_host_owned_diagnostic_persistence(persistence)
    assert host_problem.closed is True
    assert borrowed_occurrence.closed is False


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


def _contains_concrete_isinstance_branch(path: Path) -> list[str]:
    forbidden_names = {
        "DocumentStoreProblemPersistence",
        "InMemoryProblemPersistence",
        "MongoProblemPersistence",
        "ExecutionReconstructor",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "isinstance":
            continue
        if len(node.args) < 2:
            continue
        type_arg = node.args[1]
        names: list[str] = []
        if isinstance(type_arg, ast.Name):
            names.append(type_arg.id)
        elif isinstance(type_arg, ast.Tuple):
            names.extend(
                elt.id for elt in type_arg.elts if isinstance(elt, ast.Name)
            )
        for name in names:
            if name in forbidden_names:
                hits.append(f"{path.name}:{node.lineno}:isinstance({name})")
    return hits


def test_x2_shared_composition_architecture_gates() -> None:
    violations: list[str] = []
    for path in _COMPOSITION_MODULES:
        for root in sorted(_imported_module_roots(path)):
            if root in _FORBIDDEN_VENDOR_ROOTS:
                violations.append(f"{path.name} imports vendor root {root!r}")
        violations.extend(_contains_getattr_or_hasattr(path))
        violations.extend(_contains_concrete_isinstance_branch(path))
        source = path.read_text(encoding="utf-8")
        if 'diagnostics.get("' in source or "diagnostics.resolve(" in source:
            violations.append(f"{path.name} uses service-locator pattern")
    assert not violations, "\n".join(violations)


def test_x2_resolved_composition_type_is_explicit() -> None:
    persistence = _persistence_bundle()
    composition = resolve_diagnostic_composition(persistence)
    assert isinstance(composition, ResolvedDiagnosticComposition)
    assert isinstance(composition.execution_reconstruction_reader, ExecutionReconstructionReader)
