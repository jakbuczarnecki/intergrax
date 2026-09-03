# © Artur Czarnecki. All rights reserved.

"""DIAG-DURABILITY-D1 / D1-R1 qualification orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from tests.system.functional_diagnostics_durability.durability_orchestrator import (
  DurabilityRunOutcome,
  GateResult,
  run_durability_qualification,
)
from tests.system.functional_diagnostics_durability.mongodb_durable_backend import mongodb_available

_ARTIFACT_DIR = Path(
  os.environ.get(
    "DIAG_FUNCTIONAL_DURABILITY_D1_ARTIFACT_DIR",
    ".tmp/session/diag-functional-durability-d1",
  ),
)
_CURSOR_SECRET = b"diag-functional-durability-d1-secret-32b"
_BASE_TIME = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class ContractQualificationReport:
  verdict: str
  mode: str
  gates: tuple[GateResult, ...]
  backend: str
  note: str


@dataclass(frozen=True, slots=True)
class DurabilityQualificationReport:
  verdict: str
  mode: str
  durability: DurabilityRunOutcome
  contract_gates: tuple[GateResult, ...]


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


def _gate_contract_a(store: InMemoryDocumentStore) -> GateResult:
  persistence = _durable_persistence(store)
  scope = sample_functional_evidence_scope(tenant_id="d1-contract-a")
  evidence = sample_functional_evidence(scope=scope)
  persistence.append(evidence)
  collected = collect_all_evidence(
    persistence,
    tenant_id=scope.tenant_id,
    task_id=scope.task_id,
    run_id=scope.run_id,
  )
  passed = collected == (evidence,)
  return GateResult("D1-CONTRACT-A", passed, "durable append/read (in-memory store)")


def _gate_contract_b(store: InMemoryDocumentStore) -> GateResult:
  persistence = _durable_persistence(store)
  scope = sample_functional_evidence_scope(tenant_id="d1-contract-b")
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
  return GateResult("D1-CONTRACT-B", passed, "idempotent duplicate")


def _gate_contract_c(store: InMemoryDocumentStore) -> GateResult:
  persistence = _durable_persistence(store)
  scope = sample_functional_evidence_scope(tenant_id="d1-contract-c")
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
    return GateResult("D1-CONTRACT-C", True, "conflicting duplicate rejected")
  return GateResult("D1-CONTRACT-C", False, "conflict not rejected")


def _gate_contract_d(store: InMemoryDocumentStore) -> GateResult:
  scope = sample_functional_evidence_scope(tenant_id="d1-contract-d")
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
  return GateResult(
    "D1-CONTRACT-D",
    passed,
    "adapter replacement only — NOT process restart durability",
  )


def _gate_contract_e(store: InMemoryDocumentStore) -> GateResult:
  persistence = _durable_persistence(store)
  try:
    assert_functional_evidence_cross_domain_round_trip(
      persistence,
      label="d1-contract-e",
    )
  except AssertionError as exc:
    return GateResult("D1-CONTRACT-E", False, str(exc))
  return GateResult("D1-CONTRACT-E", True, "cross-domain round-trip")


def _gate_contract_f(store: InMemoryDocumentStore) -> GateResult:
  persistence = _durable_persistence(store)
  try:
    assert_functional_evidence_tenant_run_isolation(persistence, label="d1-contract-f")
  except AssertionError as exc:
    return GateResult("D1-CONTRACT-F", False, str(exc))
  return GateResult("D1-CONTRACT-F", True, "tenant/run isolation")


def _gate_contract_g(store: InMemoryDocumentStore) -> GateResult:
  scope = sample_functional_evidence_scope(tenant_id="d1-contract-g")
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
  return GateResult("D1-CONTRACT-G", passed, "partial index repair")


def _gate_contract_h(store: InMemoryDocumentStore) -> GateResult:
  scope = sample_functional_evidence_scope(tenant_id="d1-contract-h")
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
    return GateResult("D1-CONTRACT-H", True, "corruption fail-closed")
  return GateResult("D1-CONTRACT-H", False, "corruption not rejected")


def _gate_contract_i(store: InMemoryDocumentStore) -> GateResult:
  from concurrent.futures import ThreadPoolExecutor
  import threading

  scope = sample_functional_evidence_scope(tenant_id="d1-contract-i")
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
  return GateResult("D1-CONTRACT-I", passed, "concurrent duplicate append")


def _gate_contract_j() -> GateResult:
  scope = PipelineEvidenceScope(
    tenant_id="d1-contract-j",
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
  return GateResult("D1-CONTRACT-J", passed, "backend pluginability")


def run_contract_qualification() -> ContractQualificationReport:
  store = _durable_store()
  gates = (
    _gate_contract_a(store),
    _gate_contract_b(store),
    _gate_contract_c(store),
    _gate_contract_d(store),
    _gate_contract_e(store),
    _gate_contract_f(store),
    _gate_contract_g(store),
    _gate_contract_h(store),
    _gate_contract_i(store),
    _gate_contract_j(),
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
  return ContractQualificationReport(
    verdict=verdict,
    mode="contract-only",
    gates=gates,
    backend="InMemoryDocumentStore (ConditionalDocumentStore conformance)",
    note="NOT DURABILITY QUALIFICATION — adapter replacement only for D1-CONTRACT-D",
  )


def run_real_durability_qualification() -> DurabilityQualificationReport:
  if not mongodb_available():
    durability = DurabilityRunOutcome(
      verdict="BLOCKED",
      blocker="INTERGRAX_MONGODB_URI missing",
      gates=(),
      writer_pid=None,
      reader_pid=None,
      writer_exit_code=None,
      reader_exit_code=None,
      backend_provider="mongodb",
      backend_document_store_type="",
      database_name="",
      collection_name="",
      evidence_count_written=0,
      evidence_count_recovered=0,
      evidence_round_trip_fidelity="BLOCKED",
      identity_fidelity="BLOCKED",
      assessment_recovery_fidelity="BLOCKED",
      cross_process_idempotency="BLOCKED",
      cross_process_conflict="BLOCKED",
      tenant_isolation="BLOCKED",
      pagination_complete="BLOCKED",
      writer_reader_same_process=False,
      backend_in_memory=False,
      backend_mocked=False,
      raw_pymongo_bypass=False,
      production_provider_factory_used=True,
    )
    return DurabilityQualificationReport(
      verdict="BLOCKED",
      mode="real-durability",
      durability=durability,
      contract_gates=(),
    )

  contract = run_contract_qualification()
  durability = run_durability_qualification(work_dir=_ARTIFACT_DIR / "process-work")
  verdict = durability.verdict
  if contract.verdict != "PASS" and verdict == "PASS":
    verdict = "FAILED"
  return DurabilityQualificationReport(
    verdict=verdict,
    mode="real-durability",
    durability=durability,
    contract_gates=contract.gates,
  )


def _preserve_pre_r1_artifact() -> None:
  artifact_path = _ARTIFACT_DIR / "qualification-report.json"
  pre_r1_path = _ARTIFACT_DIR / "qualification-report-pre-r1.json"
  if artifact_path.exists() and not pre_r1_path.exists():
    shutil.copyfile(artifact_path, pre_r1_path)


def _write_contract_artifact(report: ContractQualificationReport) -> Path:
  _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  artifact_path = _ARTIFACT_DIR / "qualification-report-contract.json"
  payload = {
    "verdict": report.verdict,
    "mode": report.mode,
    "backend": report.backend,
    "note": report.note,
    "gates": [
      {"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail}
      for gate in report.gates
    ],
  }
  artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return artifact_path


def _write_durability_artifact(report: DurabilityQualificationReport) -> Path:
  _preserve_pre_r1_artifact()
  _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  artifact_path = _ARTIFACT_DIR / "qualification-report.json"
  durability = report.durability
  payload = {
    "verdict": report.verdict,
    "mode": report.mode,
    "blocker": durability.blocker,
    "backend_provider": durability.backend_provider,
    "backend_document_store_type": durability.backend_document_store_type,
    "database_name": durability.database_name,
    "collection_name": durability.collection_name,
    "writer_pid": durability.writer_pid,
    "reader_pid": durability.reader_pid,
    "writer_exit_code": durability.writer_exit_code,
    "reader_exit_code": durability.reader_exit_code,
    "evidence_count_written": durability.evidence_count_written,
    "evidence_count_recovered": durability.evidence_count_recovered,
    "evidence_round_trip_fidelity": durability.evidence_round_trip_fidelity,
    "identity_fidelity": durability.identity_fidelity,
    "assessment_recovery_fidelity": durability.assessment_recovery_fidelity,
    "cross_process_idempotency": durability.cross_process_idempotency,
    "cross_process_conflict": durability.cross_process_conflict,
    "tenant_isolation": durability.tenant_isolation,
    "pagination_complete": durability.pagination_complete,
    "writer_reader_same_process": durability.writer_reader_same_process,
    "backend_in_memory": durability.backend_in_memory,
    "backend_mocked": durability.backend_mocked,
    "raw_pymongo_bypass": durability.raw_pymongo_bypass,
    "production_provider_factory_used": durability.production_provider_factory_used,
    "gates": [
      {"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail}
      for gate in durability.gates
    ],
    "contract_gates": [
      {"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail}
      for gate in report.contract_gates
    ],
  }
  artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return artifact_path


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="DIAG-DURABILITY-D1 / D1-R1 qualification runner")
  parser.add_argument(
    "--contract-only",
    action="store_true",
    help="Run in-memory contract gates only (NOT durability qualification)",
  )
  parser.add_argument(
    "--conformance-only",
    action="store_true",
    help="Alias for --contract-only",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  contract_only = args.contract_only or args.conformance_only
  if contract_only:
    report = run_contract_qualification()
    artifact_path = _write_contract_artifact(report)
    summary = {
      "verdict": report.verdict,
      "mode": report.mode,
      "note": report.note,
      "artifact": str(artifact_path),
      "gates": {gate.gate_id: gate.passed for gate in report.gates},
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.verdict == "PASS" else 1

  report = run_real_durability_qualification()
  artifact_path = _write_durability_artifact(report)
  summary = {
    "verdict": report.verdict,
    "mode": report.mode,
    "blocker": report.durability.blocker,
    "artifact": str(artifact_path),
    "gates": {gate.gate_id: gate.passed for gate in report.durability.gates},
  }
  print(json.dumps(summary, indent=2))
  if report.verdict == "PASS":
    return 0
  if report.verdict == "BLOCKED":
    return 2
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
