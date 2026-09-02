# © Artur Czarnecki. All rights reserved.

"""DIAG-DURABILITY-D1 qualification orchestrator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intergrax.contracts.execution_identity import mint_attempt_id, mint_event_id, mint_run_id, mint_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
    wire_functional_evidence_persistence,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    assert_functional_evidence_conflicting_append_fails_closed,
    assert_functional_evidence_cross_domain_round_trip,
    assert_functional_evidence_persistence_conformance,
    assert_functional_evidence_tenant_run_isolation,
    collect_all_evidence,
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    C1_RAG_QUERY_ID,
    C1_RAG_RETRIEVE_OPERATION_ID,
    build_c1_rag_functional_diagnostic_specification,
)

_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_DURABILITY_D1_ARTIFACT_DIR",
        ".tmp/session/diag-functional-durability-d1",
    ),
)
_CURSOR_SECRET = b"diag-functional-durability-d1-secret-32b"
_BASE_TIME = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class QualificationReport:
    verdict: str
    gates: tuple[GateResult, ...]
    backend: str
    durable_append_read: str
    idempotent_duplicate: str
    conflicting_duplicate_rejected: str
    restart_recovery: str
    cross_domain_round_trip: str
    tenant_run_isolation: str
    partial_index_repair: str
    corruption_fail_closed: str
    concurrent_append: str
    backend_pluginability: str
    evidence_round_trip_fidelity: str
    identity_fidelity: str
    assessment_recovery_fidelity: str


class _SyntheticFunctionalEvidencePersistence(FunctionalEvidencePersistence):
    """Pluginability proof — minimal contract implementation."""

    def __init__(self) -> None:
        self._records: dict[str, PlatformFunctionalEvidence] = {}

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        existing = self._records.get(str(evidence.evidence_id))
        if existing is not None:
            if existing != evidence:
                raise FunctionalEvidencePersistenceConflictError("conflict")
            return existing
        self._records[str(evidence.evidence_id)] = evidence
        return evidence

    def query_evidence(self, request):
        from intergrax.runtime.diagnostics.functional_evidence_persistence import (
            FunctionalEvidenceQueryPage,
            FunctionalEvidenceQueryRequest,
            functional_evidence_query_order_key,
        )

        if not isinstance(request, FunctionalEvidenceQueryRequest):
            raise TypeError("invalid request")
        items = [
            record
            for record in self._records.values()
            if (
                record.scope.tenant_id == request.tenant_id
                and record.scope.task_id == request.task_id
                and record.scope.run_id == request.run_id
            )
        ]
        items.sort(key=functional_evidence_query_order_key)
        return FunctionalEvidenceQueryPage(
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            run_id=request.run_id,
            items=tuple(items[: request.page_size]),
            next_cursor=None,
        )


class _FailingPutIfAbsentDocumentStore(InMemoryDocumentStore):
    def __init__(self, *, fail_keys: frozenset[tuple[str, str]] = frozenset()) -> None:
        super().__init__()
        self._fail_keys = fail_keys
        self._failed_keys: set[tuple[str, str]] = set()

    def put_if_absent(self, document: DocumentRecord) -> bool:
        key = (document.partition_key, document.row_key)
        if key in self._fail_keys and key not in self._failed_keys:
            self._failed_keys.add(key)
            raise RuntimeError("simulated functional evidence index write failure")
        return super().put_if_absent(document)


def _durable_store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


def _durable_persistence(
    store: InMemoryDocumentStore,
) -> DocumentStoreFunctionalEvidencePersistence:
    return DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
    )


def _analyze(scope: PipelineEvidenceScope, persistence: FunctionalEvidencePersistence):
    spec = build_c1_rag_functional_diagnostic_specification(include_validation=False)
    return FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        specification=spec,
        validations=FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
        ),
    )


def _restart_recovery_evidence(scope: PipelineEvidenceScope) -> list[PlatformFunctionalEvidence]:
    return [
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component="d1.restart",
                operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
                recorded_at=_BASE_TIME,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name="retrieve",
                status=PipelineOperationStatus.SUCCEEDED,
            ),
        ),
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.CANDIDATE_RANK,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component="d1.restart",
                operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
                recorded_at=_BASE_TIME + timedelta(seconds=1),
            ),
            candidate=PipelineCandidateFact(
                query_id=C1_RAG_QUERY_ID,
                candidate_artifact_ref=ObservabilityArtifactReference(artifact_ref="chunk:test"),
                rank=1,
                selected=True,
            ),
        ),
    ]


def _gate_d1_a(store: InMemoryDocumentStore) -> GateResult:
    persistence = _durable_persistence(store)
    scope = sample_functional_evidence_scope(tenant_id="d1-a")
    evidence = sample_functional_evidence(scope=scope)
    persistence.append(evidence)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    passed = collected == (evidence,)
    return GateResult("D1-A", passed, "durable append/read")


def _gate_d1_b(store: InMemoryDocumentStore) -> GateResult:
    persistence = _durable_persistence(store)
    scope = sample_functional_evidence_scope(tenant_id="d1-b")
    evidence = sample_functional_evidence(scope=scope)
    persistence.append(evidence)
    duplicate = persistence.append(evidence)
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    passed = duplicate == evidence and collected == (evidence,)
    return GateResult("D1-B", passed, "idempotent duplicate")


def _gate_d1_c(store: InMemoryDocumentStore) -> GateResult:
    persistence = _durable_persistence(store)
    scope = sample_functional_evidence_scope(tenant_id="d1-c")
    evidence_id = mint_event_id()
    original = sample_functional_evidence(
        evidence_id=evidence_id,
        scope=scope,
        operation_name="original",
    )
    conflicting = sample_functional_evidence(
        evidence_id=evidence_id,
        scope=scope,
        operation_name="conflicting",
    )
    persistence.append(original)
    try:
        persistence.append(conflicting)
    except FunctionalEvidencePersistenceConflictError:
        return GateResult("D1-C", True, "conflicting duplicate rejected")
    return GateResult("D1-C", False, "conflict not rejected")


def _gate_d1_d(store: InMemoryDocumentStore) -> GateResult:
    scope = sample_functional_evidence_scope(tenant_id="d1-d")
    evidence = _restart_recovery_evidence(scope)
    writer = _durable_persistence(store)
    for item in evidence:
        writer.append(item)
    before = _analyze(scope, writer)

    reader = _durable_persistence(store)
    after = _analyze(scope, reader)
    collected = collect_all_evidence(
        reader,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    passed = (
        len(collected) == len(evidence)
        and before.first_proven_failure == after.first_proven_failure
        and before.check_results == after.check_results
    )
    return GateResult("D1-D", passed, "restart recovery")


def _gate_d1_e(store: InMemoryDocumentStore) -> GateResult:
    persistence = _durable_persistence(store)
    try:
        assert_functional_evidence_cross_domain_round_trip(
            persistence,
            label="d1-e",
        )
    except AssertionError as exc:
        return GateResult("D1-E", False, str(exc))
    return GateResult("D1-E", True, "cross-domain round-trip")


def _gate_d1_f(store: InMemoryDocumentStore) -> GateResult:
    persistence = _durable_persistence(store)
    try:
        assert_functional_evidence_tenant_run_isolation(persistence, label="d1-f")
    except AssertionError as exc:
        return GateResult("D1-F", False, str(exc))
    return GateResult("D1-F", True, "tenant/run isolation")


def _gate_d1_g(store: InMemoryDocumentStore) -> GateResult:
    scope = sample_functional_evidence_scope(tenant_id="d1-g")
    evidence = sample_functional_evidence(scope=scope)
    partition_key = f"intergrax.functional_evidence.v1:{scope.tenant_id}"
    exec_key = (
        partition_key,
        f"exec:{scope.task_id}:{scope.run_id}:{evidence.evidence_id}",
    )
    failing_store = _FailingPutIfAbsentDocumentStore(fail_keys=frozenset({exec_key}))
    writer = DocumentStoreFunctionalEvidencePersistence(
        failing_store,
        cursor_secret=_CURSOR_SECRET,
    )
    try:
        writer.append(evidence)
    except RuntimeError:
        pass
    repaired = DocumentStoreFunctionalEvidencePersistence(
        failing_store,
        cursor_secret=_CURSOR_SECRET,
    )
    result = repaired.append(evidence)
    collected = collect_all_evidence(
        repaired,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    passed = result == evidence and collected == (evidence,)
    return GateResult("D1-G", passed, "partial index repair")


def _gate_d1_h(store: InMemoryDocumentStore) -> GateResult:
    scope = sample_functional_evidence_scope(tenant_id="d1-h")
    evidence = sample_functional_evidence(scope=scope)
    partition_key = f"intergrax.functional_evidence.v1:{scope.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"exec:{scope.task_id}:{scope.run_id}:{evidence.evidence_id}",
            data={
                "schema_version": "intergrax.functional_evidence.index.v1",
                "evidence_id": str(evidence.evidence_id),
            },
        )
    )
    persistence = _durable_persistence(store)
    try:
        collect_all_evidence(
            persistence,
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    except FunctionalEvidencePersistenceIntegrityError:
        return GateResult("D1-H", True, "corruption fail-closed")
    return GateResult("D1-H", False, "corruption not rejected")


def _gate_d1_i(store: InMemoryDocumentStore) -> GateResult:
    from concurrent.futures import ThreadPoolExecutor
    import threading

    scope = sample_functional_evidence_scope(tenant_id="d1-i")
    evidence = sample_functional_evidence(scope=scope)
    barrier = threading.Barrier(2)
    results: list[PlatformFunctionalEvidence] = []

    def _append() -> None:
        persistence = _durable_persistence(store)
        barrier.wait(timeout=5)
        results.append(persistence.append(evidence))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_append), executor.submit(_append)]
        for future in futures:
            future.result(timeout=10)
    passed = len(results) == 2 and results[0] == evidence and results[1] == evidence
    return GateResult("D1-I", passed, "concurrent duplicate append")


def _gate_d1_j() -> GateResult:
    scope = PipelineEvidenceScope(
        tenant_id="d1-j",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    persistence = _SyntheticFunctionalEvidencePersistence()
    evidence = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="synthetic",
            operation_id="plugin-proof",
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name="plugin-proof",
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )
    persistence.append(evidence)
    analysis = _analyze(scope, persistence)
    passed = any(
        result.status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
        for result in analysis.check_results
    )
    return GateResult("D1-J", passed, "backend pluginability")


def run_qualification() -> QualificationReport:
    store = _durable_store()
    gates = (
        _gate_d1_a(store),
        _gate_d1_b(store),
        _gate_d1_c(store),
        _gate_d1_d(store),
        _gate_d1_e(store),
        _gate_d1_f(store),
        _gate_d1_g(store),
        _gate_d1_h(store),
        _gate_d1_i(store),
        _gate_d1_j(),
    )
    in_memory = InMemoryFunctionalEvidencePersistence(cursor_secret=_CURSOR_SECRET)
    assert_functional_evidence_persistence_conformance(in_memory, label="in-memory")
    assert_functional_evidence_conflicting_append_fails_closed(in_memory, label="in-memory")
    durable = _durable_persistence(_durable_store())
    assert_functional_evidence_persistence_conformance(durable, label="document-store")
    wire_functional_evidence_persistence(
        document_store=_durable_store(),
        cursor_secret=_CURSOR_SECRET,
    )
    passed = all(gate.passed for gate in gates)
    verdict = "PASS" if passed else "FAILED"
    return QualificationReport(
        verdict=verdict,
        gates=gates,
        backend="InMemoryDocumentStore (ConditionalDocumentStore conformance)",
        durable_append_read=_gate_status(gates, "D1-A"),
        idempotent_duplicate=_gate_status(gates, "D1-B"),
        conflicting_duplicate_rejected=_gate_status(gates, "D1-C"),
        restart_recovery=_gate_status(gates, "D1-D"),
        cross_domain_round_trip=_gate_status(gates, "D1-E"),
        tenant_run_isolation=_gate_status(gates, "D1-F"),
        partial_index_repair=_gate_status(gates, "D1-G"),
        corruption_fail_closed=_gate_status(gates, "D1-H"),
        concurrent_append=_gate_status(gates, "D1-I"),
        backend_pluginability=_gate_status(gates, "D1-J"),
        evidence_round_trip_fidelity="100%" if passed else "FAILED",
        identity_fidelity="100%" if passed else "FAILED",
        assessment_recovery_fidelity="100%" if passed else "FAILED",
    )


def _gate_status(gates: tuple[GateResult, ...], gate_id: str) -> str:
    for gate in gates:
        if gate.gate_id == gate_id:
            return "PASS" if gate.passed else "FAILED"
    return "FAILED"


def _write_artifact(report: QualificationReport) -> Path:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = _ARTIFACT_DIR / "qualification-report.json"
    payload = {
        "verdict": report.verdict,
        "backend": report.backend,
        "gates": [
            {"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail}
            for gate in report.gates
        ],
        "durable_append_read": report.durable_append_read,
        "idempotent_duplicate": report.idempotent_duplicate,
        "conflicting_duplicate_rejected": report.conflicting_duplicate_rejected,
        "restart_recovery": report.restart_recovery,
        "cross_domain_round_trip": report.cross_domain_round_trip,
        "tenant_run_isolation": report.tenant_run_isolation,
        "partial_index_repair": report.partial_index_repair,
        "corruption_fail_closed": report.corruption_fail_closed,
        "concurrent_append": report.concurrent_append,
        "backend_pluginability": report.backend_pluginability,
        "evidence_round_trip_fidelity": report.evidence_round_trip_fidelity,
        "identity_fidelity": report.identity_fidelity,
        "assessment_recovery_fidelity": report.assessment_recovery_fidelity,
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def main() -> int:
    report = run_qualification()
    artifact_path = _write_artifact(report)
    summary = {
        "verdict": report.verdict,
        "backend": report.backend,
        "artifact": str(artifact_path),
        "gates": {gate.gate_id: gate.passed for gate in report.gates},
    }
    print(json.dumps(summary, indent=2))
    if report.verdict == "PASS":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
