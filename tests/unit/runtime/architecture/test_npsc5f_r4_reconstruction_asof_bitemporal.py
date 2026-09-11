# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 — historical reconstruction / as-of / bitemporal qualification."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    BitemporalKnowledgeBasis,
    KnowledgeOrderingScope,
    KnowledgeRevisionId,
    KnowledgeRevisionWatermark,
    SystemTimeBasis,
    ValidTimeBasis,
    mint_knowledge_revision_id,
    mint_revision_acceptance_key,
)
from intergrax.contracts.execution_identity import RunId, mint_run_id, mint_task_id
from intergrax.contracts.historical_reconstruction import (
    ExecutionHistoricalReconstructionRequest,
    HistoricalEvidenceIntegrityError,
    HistoricalScopeMismatchError,
    KnowledgeBoundaryNotFinalizedError,
    revision_admissible_at_bitemporal_query,
)
from intergrax.runtime.events.execution_position import AsOfBoundary, ExecutionEventPosition
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import (
    PositionedJournalBoundaryNotFoundError,
    PositionedJournalPrefixTruncatedError,
    load_positioned_run_journal_through,
)
from intergrax.runtime.observability.export_boundary import ObservabilityExportEnvelope
from intergrax.runtime.observability.historical_reconstruction import HistoricalReconstructionService
from intergrax.runtime.observability.journal_export import JournalExportSnapshot
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.contracts.test_bitemporal_revision_ordering import (
    _InMemoryRevisionOrderingAuthority,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-r4"
_T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

_R4_SURFACES = (
    "intergrax/contracts/historical_reconstruction.py",
    "intergrax/runtime/observability/historical_reconstruction.py",
)

_FORBIDDEN_CONTROL_PLANE = frozenset(
    {
        "ExecutionRuntime",
        "ChildExecutionRunner",
        "AttemptLifecycleService",
        "LongRunningCoordinator",
        "FanOutPartialRecoveryService",
    },
)

_FORBIDDEN_REPLAY_NAMES = frozenset(
    {
        "HistoricalRuntime",
        "ReplayRuntime",
        "TemporalRuntime",
    },
)


def _scope() -> KnowledgeOrderingScope:
    return KnowledgeOrderingScope(tenant_id=_TENANT)


def _neutral_bitemporal_query() -> BitemporalKnowledgeBasis:
    return BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=30)),
    )


@dataclass(frozen=True, slots=True)
class _Revision:
    revision_id: KnowledgeRevisionId
    payload: str
    basis: BitemporalKnowledgeBasis


class _RevisionReader:
    def __init__(self, revisions: dict[KnowledgeRevisionId, _Revision]) -> None:
        self._revisions = revisions

    def load_revision(self, revision_id: KnowledgeRevisionId) -> _Revision:
        return self._revisions[revision_id]


def _reduce_payloads(state: tuple[str, ...], revision: _Revision) -> tuple[str, ...]:
    return state + (revision.payload,)


def _append_events(
    store: RuntimeEventPersistence,
    *,
    run_id: RunId,
    count: int,
    tenant_id: str = _TENANT,
    same_timestamp: datetime | None = None,
) -> None:
    task_id = mint_task_id()
    for _ in range(count):
        event = sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id)
        if same_timestamp is not None:
            event = event.model_copy(update={"timestamp": same_timestamp})
        store.append(event, tenant_id=tenant_id)


def _service(
    store: RuntimeEventPersistence,
    authority: _InMemoryRevisionOrderingAuthority | None = None,
) -> HistoricalReconstructionService:
    return HistoricalReconstructionService(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
        revision_ordering=authority or _InMemoryRevisionOrderingAuthority(),
    )


def _request(
    *,
    run_id: RunId,
    position: int,
    watermark: KnowledgeRevisionWatermark,
) -> ExecutionHistoricalReconstructionRequest:
    return ExecutionHistoricalReconstructionRequest(
        tenant_id=_TENANT,
        run_id=run_id,
        execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(position)),
        knowledge_watermark=watermark,
        bitemporal_query=_neutral_bitemporal_query(),
    )


def _accept_knowledge(
    authority: _InMemoryRevisionOrderingAuthority,
    count: int,
    *,
    revisions: dict[KnowledgeRevisionId, _Revision],
) -> KnowledgeRevisionWatermark:
    scope = _scope()
    for index in range(count):
        revision_id = mint_knowledge_revision_id()
        revisions[revision_id] = _Revision(
            revision_id=revision_id,
            payload=f"k{index}",
            basis=_neutral_bitemporal_query(),
        )
        authority.accept_revision(
            scope=scope,
            revision_id=revision_id,
            acceptance_key=mint_revision_acceptance_key(),
        )
    return authority.watermark(scope)


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda: InMemoryRuntimeEventStore()),
    ],
)
def test_r4_execution_boundary_inclusive(label: str, factory) -> None:
    store = factory()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=3)
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(2))
    prefix = load_positioned_run_journal_through(
        store,
        tenant_id=_TENANT,
        boundary=boundary,
    )
    assert [row.position.value for row in prefix] == [1, 2]
    with pytest.raises(PositionedJournalBoundaryNotFoundError):
        load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(99)),
        )


def test_r4_missing_execution_boundary_fail_closed(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "missing.db")
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1)
    authority = _InMemoryRevisionOrderingAuthority()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    watermark = _accept_knowledge(authority, 1, revisions=revisions)
    service = _service(store, authority)
    request = _request(run_id=run_id, position=5, watermark=watermark)
    with pytest.raises(HistoricalEvidenceIntegrityError):
        service.reconstruct(
            request,
            revision_reader=_RevisionReader(revisions),
            revision_bitemporal_basis=lambda revision: revision.basis,
            reducer=_reduce_payloads,
            initial_state=(),
        )
    store.close()


def test_r4_prefix_truncation_fail_closed() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=3)
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(2))
    with pytest.raises(PositionedJournalPrefixTruncatedError):
        load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=boundary,
            initial_limit=1,
            max_limit=1,
        )


def test_r4_same_timestamp_orders_by_position() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    ts = _T0
    _append_events(store, run_id=run_id, count=3, same_timestamp=ts)
    prefix = load_positioned_run_journal_through(
        store,
        tenant_id=_TENANT,
        boundary=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(3)),
    )
    positions = [row.position.value for row in prefix]
    assert positions == [1, 2, 3]
    assert len({row.event.timestamp for row in prefix}) == 1


def test_r4_k_finalized_only_and_k_too_new() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    watermark = _accept_knowledge(authority, 2, revisions=revisions)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1)
    service = _service(store, authority)
    ok = _request(run_id=run_id, position=1, watermark=watermark)
    result = service.reconstruct(
        ok,
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert result.knowledge_view.state == ("k0", "k1")

    too_new = KnowledgeRevisionWatermark(
        scope=_scope(),
        finalized_through_value=watermark.finalized_through_value + 1,
    )
    bad = _request(run_id=run_id, position=1, watermark=too_new)
    with pytest.raises(KnowledgeBoundaryNotFinalizedError):
        service.reconstruct(
            bad,
            revision_reader=_RevisionReader(revisions),
            revision_bitemporal_basis=lambda revision: revision.basis,
            reducer=_reduce_payloads,
            initial_state=(),
        )


def test_r4_late_knowledge_bitemporal() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    scope = _scope()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    early_id = mint_knowledge_revision_id()
    revisions[early_id] = _Revision(
        revision_id=early_id,
        payload="early",
        basis=BitemporalKnowledgeBasis(
            valid_time=ValidTimeBasis.instant(_T0),
            system_time=SystemTimeBasis.instant(_T0 + timedelta(days=1)),
        ),
    )
    authority.accept_revision(
        scope=scope,
        revision_id=early_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    late_id = mint_knowledge_revision_id()
    revisions[late_id] = _Revision(
        revision_id=late_id,
        payload="late",
        basis=BitemporalKnowledgeBasis(
            valid_time=ValidTimeBasis.instant(_T0),
            system_time=SystemTimeBasis.instant(_T0 + timedelta(days=10)),
        ),
    )
    authority.accept_revision(
        scope=scope,
        revision_id=late_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = authority.watermark(scope)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1)
    service = _service(store, authority)
    query_early = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=5)),
    )
    request_early = ExecutionHistoricalReconstructionRequest(
        tenant_id=_TENANT,
        run_id=run_id,
        execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(1)),
        knowledge_watermark=watermark,
        bitemporal_query=query_early,
    )
    early_view = service.reconstruct(
        request_early,
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert early_view.knowledge_view.state == ("early",)

    query_late = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=20)),
    )
    request_late = ExecutionHistoricalReconstructionRequest(
        tenant_id=_TENANT,
        run_id=run_id,
        execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(1)),
        knowledge_watermark=watermark,
        bitemporal_query=query_late,
    )
    late_view = service.reconstruct(
        request_late,
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=_reduce_payloads,
        initial_state=(),
    )
    assert late_view.knowledge_view.state == ("early", "late")


def test_r4_cross_tenant_scope_blocked() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=1, tenant_id="tenant-other")
    watermark = KnowledgeRevisionWatermark(
        scope=KnowledgeOrderingScope(tenant_id="tenant-other"),
        finalized_through_value=0,
    )
    with pytest.raises(HistoricalScopeMismatchError):
        ExecutionHistoricalReconstructionRequest(
            tenant_id=_TENANT,
            run_id=run_id,
            execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(1)),
            knowledge_watermark=watermark,
            bitemporal_query=_neutral_bitemporal_query(),
        )


def test_r4_determinism_and_clock_independence() -> None:
    authority = _InMemoryRevisionOrderingAuthority()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    watermark = _accept_knowledge(authority, 1, revisions=revisions)
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_events(store, run_id=run_id, count=2)
    service = _service(store, authority)
    request = _request(run_id=run_id, position=2, watermark=watermark)
    kwargs = {
        "revision_reader": _RevisionReader(revisions),
        "revision_bitemporal_basis": lambda revision: revision.basis,
        "reducer": _reduce_payloads,
        "initial_state": (),
    }
    first = service.reconstruct(request, **kwargs)
    second = service.reconstruct(request, **kwargs)
    assert first == second


def test_r4_no_export_envelope_as_source() -> None:
    source = Path(_REPO_ROOT / "intergrax/runtime/observability/historical_reconstruction.py")
    text = source.read_text(encoding="utf-8")
    assert "ObservabilityExportEnvelope" not in text
    assert "JournalExportSnapshot" not in text
    assert ObservabilityExportEnvelope.__name__ not in text
    assert JournalExportSnapshot.__name__ not in text


def test_r4_static_no_active_execution_surface() -> None:
    for relative in _R4_SURFACES:
        tree = ast.parse(Path(_REPO_ROOT / relative).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert names.isdisjoint(_FORBIDDEN_CONTROL_PLANE)
        assert names.isdisjoint(_FORBIDDEN_REPLAY_NAMES)


def test_r4_revision_admissible_independent_axes() -> None:
    revision = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=10)),
    )
    query_before = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=5)),
    )
    assert not revision_admissible_at_bitemporal_query(query=query_before, revision=revision)
    query_after = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=10)),
    )
    assert revision_admissible_at_bitemporal_query(query=query_after, revision=revision)


def test_r4_provider_parity_execution_prefix(tmp_path: Path) -> None:
    run_id = mint_run_id()
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(2))
    stores: list[tuple[str, RuntimeEventPersistence]] = [
        ("memory", InMemoryRuntimeEventStore()),
        ("sqlite", SQLiteRuntimeEventStore(db_path=tmp_path / "parity.db")),
        (
            "document",
            DocumentBackedRuntimeEventStore(InMemoryDocumentStore()),
        ),
    ]
    prefixes: list[tuple[int, ...]] = []
    for label, store in stores:
        _append_events(store, run_id=run_id, count=3)
        prefix = load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=boundary,
        )
        prefixes.append(tuple(row.position.value for row in prefix))
        if hasattr(store, "close"):
            store.close()
    assert prefixes[0] == prefixes[1] == prefixes[2] == (1, 2)


def _run_pytest(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *targets, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R4 gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py"],
    ),
    (
        "R3 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r3_final_governed_evidence_export.py"],
    ),
    (
        "R2 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r2_final_journal_completeness_ordering.py"],
    ),
    (
        "R1 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
            "tests/unit/runtime/observability/test_knowledge_reconstruction.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
)


@pytest.mark.gate
def test_r4_mandatory_regression_matrix() -> None:
    failures: list[str] = []
    for label, targets in _MANDATORY_SUITES:
        completed = _run_pytest(targets)
        if completed.returncode != 0:
            failures.append(f"{label}: {completed.stdout}\n{completed.stderr}")
    assert not failures, "\n".join(failures)
