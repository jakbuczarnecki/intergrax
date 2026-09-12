# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — complete Evidence Plane qualification, ownership, and freeze gate."""

from __future__ import annotations

import subprocess
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
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.historical_reconstruction import ExecutionHistoricalReconstructionRequest
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.execution_position import AsOfBoundary, ExecutionEventPosition
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import load_complete_run_journal
from intergrax.runtime.observability.historical_reconstruction import HistoricalReconstructionService
from intergrax.runtime.observability.journal_export import serialize_runtime_event
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.npsc5f_final_evidence_plane_drift import (
    NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA,
    NPSC_5E_FINAL_SHA,
    NPSC_5F_R1_FINAL_SHA,
    NPSC_5F_R2_FINAL_SHA,
    NPSC_5F_R3_FINAL_SHA,
    NPSC_5F_R4_FINAL_SHA,
    collect_breaking_evidence_plane_production_drift,
    git_head_sha,
)
from testing_support.npsc5f_final_evidence_plane_ownership import (
    EVIDENCE_PLANE_OWNERSHIP,
    FROZEN_EVIDENCE_PLANE_CONTRACTS,
    collect_forbidden_execution_control_calls,
)
from testing_support.npsc5f_final_regression_matrix import (
    MANDATORY_REGRESSION_SUITES,
    run_mandatory_regression_matrix,
)
from testing_support.npsc5f_r1_protected_drift import (
    R1_POST_R2_QUALIFIED_BASELINE_SHA,
    collect_r1_protected_production_drift,
)
from testing_support.npsc5f_r2_protected_drift import (
    R2_POST_QUALIFIED_BASELINE_SHA,
    collect_r2_protected_production_drift,
)
from testing_support.npsc5f_r3_protected_drift import collect_r3_protected_production_drift
from tests.unit.contracts.test_bitemporal_revision_ordering import _InMemoryRevisionOrderingAuthority
from tests.unit.runtime.architecture.test_npsc5f_r4_reconstruction_asof_bitemporal import (
    _Revision,
    _RevisionReader,
    _reduce_payloads,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-npsc5f-final"
_T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _neutral_bitemporal_query() -> BitemporalKnowledgeBasis:
    return BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0 + timedelta(days=30)),
    )


@pytest.mark.gate
def test_npsc5f_final_mandatory_regression_matrix_passes() -> None:
    """One ``uv run pytest`` for the full matrix (no per-suite subprocess fan-out)."""
    proc = run_mandatory_regression_matrix(_REPO_ROOT)
    assert proc.returncode == 0, (
        f"NPSC-5F Final mandatory regression matrix failed:\n{proc.stdout}\n{proc.stderr}"
    )


def test_npsc5f_final_frozen_predecessor_shas_recorded() -> None:
    assert NPSC_5E_FINAL_SHA == "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"
    assert NPSC_5F_R1_FINAL_SHA == "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"
    assert NPSC_5F_R2_FINAL_SHA == "76c92847f67da22d97943b55896a88c814d7e39d"
    assert NPSC_5F_R3_FINAL_SHA == "0346face3ef68d8f21504822a26f8f45f2384cf9"
    assert NPSC_5F_R4_FINAL_SHA == "37fb051c7f164d705f628760436b8ea10ee0289f"
    assert NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA == "3bec620ab56417a469487347f68045bf3dec6bd5"


def test_npsc5f_final_predecessor_drift_sentinels_empty() -> None:
    assert collect_r1_protected_production_drift(
        _REPO_ROOT,
        from_sha=R1_POST_R2_QUALIFIED_BASELINE_SHA,
        to_ref="origin/development",
    ) == []
    assert collect_r2_protected_production_drift(
        _REPO_ROOT,
        from_sha=R2_POST_QUALIFIED_BASELINE_SHA,
        to_ref="origin/development",
    ) == []
    assert collect_r3_protected_production_drift(_REPO_ROOT, to_ref="origin/development") == []
    assert collect_breaking_evidence_plane_production_drift(_REPO_ROOT) == []


def test_npsc5f_final_ownership_map_recorded() -> None:
    owners = dict(EVIDENCE_PLANE_OWNERSHIP)
    assert owners["Durable evidence"] == "RuntimeEventPersistence"
    assert owners["Historical reconstruction"] == "HistoricalReconstructionService"
    assert owners["Export boundary"] == "ObservabilityExportEnvelope"
    assert len(owners) == 10


def test_npsc5f_final_frozen_contract_matrix_recorded() -> None:
    matrix = dict(FROZEN_EVIDENCE_PLANE_CONTRACTS)
    assert matrix["Durable evidence"] == "FROZEN"
    assert matrix["Bitemporal semantics"] == "FROZEN"
    assert len(matrix) == 8


def test_npsc5f_final_evidence_plane_has_no_execution_control_calls() -> None:
    assert collect_forbidden_execution_control_calls(_REPO_ROOT) == []


def test_npsc5f_final_mandatory_suite_labels_recorded() -> None:
    labels = [label for label, _ in MANDATORY_REGRESSION_SUITES]
    for required in (
        "Recovery",
        "NPSC-5E Final",
        "NPSC-5D Final",
        "HITL R3",
        "Child execution",
        "Checkpoint",
        "Retry",
        "Cancellation",
        "Evidence",
        "R4 implementation gate",
        "DG_001",
        "TRACE-ASOF",
        "TRACE-BITEMP",
        "Execution reconstruction",
    ):
        assert required in labels


@pytest.mark.gate
def test_npsc5f_final_end_to_end_evidence_lifecycle_qualification() -> None:
    tenant_id = _TENANT
    run_id = mint_run_id()
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    event_id = mint_event_id()

    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)
    event = sample_runtime_event(
        tenant_id=tenant_id,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        event_id=event_id,
    )
    bus.record(event, tenant_id=tenant_id)

    positioned = store.get_by_event_id(tenant_id=tenant_id, event_id=event_id)
    assert positioned is not None
    persisted = positioned.event

    journal = load_complete_run_journal(store, tenant_id=tenant_id, run_id=str(run_id))
    assert len(journal) == 1
    assert journal[0].event_id == event_id

    exported = serialize_runtime_event(persisted)
    assert exported.event_id == str(event_id)

    authority = _InMemoryRevisionOrderingAuthority()
    revisions: dict[KnowledgeRevisionId, _Revision] = {}
    scope = KnowledgeOrderingScope(tenant_id=tenant_id)
    revision_id = mint_knowledge_revision_id()
    revisions[revision_id] = _Revision(
        revision_id=revision_id,
        payload="k0",
        basis=_neutral_bitemporal_query(),
    )
    authority.accept_revision(
        scope=scope,
        revision_id=revision_id,
        acceptance_key=mint_revision_acceptance_key(),
    )
    watermark = authority.watermark(scope)
    causal = InMemoryCausalEvidencePersistence()
    service = HistoricalReconstructionService(
        runtime_events=store,
        causal_evidence=causal,
        revision_ordering=authority,
    )
    events_before = len(store._accepted_by_event_id)
    causal_before = len(causal._accepted_by_evidence_id)
    bindings_before = len(authority._bindings)

    request = ExecutionHistoricalReconstructionRequest(
        tenant_id=tenant_id,
        run_id=run_id,
        execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(1)),
        knowledge_watermark=watermark,
        bitemporal_query=_neutral_bitemporal_query(),
    )
    view = service.reconstruct(
        request,
        revision_reader=_RevisionReader(revisions),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=_reduce_payloads,
        initial_state=(),
    )

    assert len(store._accepted_by_event_id) == events_before
    assert len(causal._accepted_by_evidence_id) == causal_before
    assert len(authority._bindings) == bindings_before

    audit_refs = view.execution_projection.source_events
    assert len(audit_refs) == 1
    assert audit_refs[0].event_id == event_id
    assert audit_refs[0].attempt_id == attempt_id

    positioned = view.execution_reconstruction.positioned_events
    assert len(positioned) == 1
    reconstructed = positioned[0].event

    identity_fields = (
        ("tenant_id", tenant_id, persisted.tenant_id, exported.tenant_id, reconstructed.tenant_id),
        ("run_id", run_id, persisted.run_id, RunId(exported.run_id), reconstructed.run_id),
        ("execution_id", execution_id, persisted.execution_id, ExecutionId(exported.execution_id), reconstructed.execution_id),
        ("attempt_id", attempt_id, persisted.attempt_id, AttemptId(exported.attempt_id), reconstructed.attempt_id),
        ("task_id", task_id, persisted.task_id, TaskId(exported.task_id), reconstructed.task_id),
        ("event_id", event_id, persisted.event_id, EventId(exported.event_id), reconstructed.event_id),
    )
    for label, expected, *values in identity_fields:
        for actual in values:
            assert actual == expected, f"{label} continuity failed: {actual!r} != {expected!r}"

    assert exported.tenant_id == tenant_id
    assert exported.attempt_id == str(attempt_id)


def test_npsc5f_final_head_matches_origin_development() -> None:
    head = git_head_sha(_REPO_ROOT)
    origin = subprocess.run(
        ["git", "rev-parse", "origin/development"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert origin.returncode == 0
    assert head == origin.stdout.strip()


@pytest.mark.gate
def test_npsc5f_final_qualification_gate() -> None:
    assert collect_breaking_evidence_plane_production_drift(_REPO_ROOT) == []
    assert collect_forbidden_execution_control_calls(_REPO_ROOT) == []
