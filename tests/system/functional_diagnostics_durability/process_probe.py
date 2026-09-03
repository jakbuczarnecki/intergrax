# © Artur Czarnecki. All rights reserved.

"""D1-R1 subprocess worker — real process-boundary functional evidence durability proof."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from intergrax.contracts.execution_identity import (
  AttemptId,
  mint_attempt_id,
  mint_run_id,
  mint_task_id,
  validate_attempt_id,
  validate_event_id,
  validate_run_id,
  validate_task_id,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
  DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_evidence import (
  PipelineEvidenceScope,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
  FunctionalEvidencePersistenceConflictError,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
  collect_all_evidence,
  sample_functional_evidence,
  sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
  build_c1_rag_functional_diagnostic_specification,
)
from tests.system.functional_diagnostics_durability.assessment_fingerprint import (
  DiagnosticAssessmentFingerprint,
)
from tests.system.functional_diagnostics_durability.mongodb_durable_backend import (
  MongoDurableBackendProbe,
)
from tests.system.functional_diagnostics_durability.process_ipc import (
  IPC_SCHEMA_VERSION,
  ConflictAppendResult,
  DurabilityProbePhase,
  DurabilityProbeResult,
  DurabilityReadResult,
  DurabilityWriteResult,
  ExecutionIdentity,
  IdempotentRetryResult,
  TenantIsolationResult,
  encode_ipc_payload,
)
from tests.system.functional_diagnostics_durability.qualification_evidence import (
  build_cross_domain_codec_evidence,
  build_pagination_evidence,
)

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_BLOCKED = 2
_DOCUMENT_PARTITION_PREFIX = "intergrax.functional_evidence.v1"
_CURSOR_SECRET_ENV = "DIAG_FUNCTIONAL_DURABILITY_D1_CURSOR_SECRET"
_DEFAULT_CURSOR_SECRET = b"diag-functional-durability-d1-secret-32b"
_PAGINATION_PAGE_SIZE = 2


def _emit(payload: dict[str, object]) -> None:
  sys.stdout.write(encode_ipc_payload(payload))
  sys.stdout.write("\n")
  sys.stdout.flush()


def _fail(message: str, *, code: int = _EXIT_ERROR) -> None:
  sys.stderr.write(message)
  if not message.endswith("\n"):
    sys.stderr.write("\n")
  sys.stderr.flush()
  raise SystemExit(code)


def _resolve_cursor_secret() -> bytes:
  raw = os.environ.get(_CURSOR_SECRET_ENV, "").strip()
  if raw:
    encoded = raw.encode("utf-8")
    if len(encoded) < 32:
      _fail("cursor secret too short", code=_EXIT_ERROR)
    return encoded
  return _DEFAULT_CURSOR_SECRET


def _build_probe(collection_name: str) -> MongoDurableBackendProbe:
  return MongoDurableBackendProbe(collection_name=collection_name)


def _build_persistence(
  store: ConditionalDocumentStore,
  *,
  query_page_limit: int | None = None,
) -> DocumentStoreFunctionalEvidencePersistence:
  kwargs: dict[str, bytes | int] = {"cursor_secret": _resolve_cursor_secret()}
  if query_page_limit is not None:
    kwargs["query_page_limit"] = query_page_limit
  return DocumentStoreFunctionalEvidencePersistence(store, **kwargs)


def _scope_from_identity(identity: ExecutionIdentity) -> PipelineEvidenceScope:
  attempt_id: AttemptId | None = None
  if identity.attempt_id is not None:
    attempt_id = validate_attempt_id(identity.attempt_id)
  return PipelineEvidenceScope(
    tenant_id=identity.tenant_id,
    task_id=validate_task_id(identity.task_id),
    run_id=validate_run_id(identity.run_id),
    attempt_id=attempt_id,
  )


def _analyze(scope: PipelineEvidenceScope, persistence: DocumentStoreFunctionalEvidencePersistence):
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


def _document_partition(tenant_id: str) -> str:
  return f"{_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"


def _purge_tenant_documents(store: ConditionalDocumentStore, tenant_id: str) -> None:
  partition_key = _document_partition(tenant_id)
  cursor: str | None = None
  while True:
    page = store.query(partition_key, limit=5000, cursor=cursor)
    for document in page.documents:
      store.delete(document.partition_key, document.row_key)
    if page.next_cursor is None:
      break
    cursor = page.next_cursor


def _identity_from_args(
  *,
  tenant_id: str,
  task_id: str | None,
  run_id: str | None,
  attempt_id: str | None,
) -> ExecutionIdentity:
  return ExecutionIdentity(
    tenant_id=tenant_id,
    task_id=task_id or str(mint_task_id()),
    run_id=run_id or str(mint_run_id()),
    attempt_id=attempt_id,
  )


def _run_probe(collection_name: str) -> None:
  probe = _build_probe(collection_name)
  try:
    store = probe.build_document_store()
    identity = probe.backend_identity()
    probe.close_document_store(store)
  except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
    _emit(
      DurabilityProbeResult(
        schema_version=IPC_SCHEMA_VERSION,
        pid=os.getpid(),
        phase=DurabilityProbePhase.PROBE.value,
        ok=False,
        detail=f"{type(exc).__name__}: MongoDB backend unavailable",
        exit_code=_EXIT_BLOCKED,
      ).to_json_dict(),
    )
    raise SystemExit(_EXIT_BLOCKED) from exc
  _emit(
    DurabilityProbeResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.PROBE.value,
      ok=True,
      detail=(
        f"provider={identity.provider_id};"
        f"database={identity.database_name};"
        f"collection={identity.collection_name}"
      ),
      exit_code=_EXIT_OK,
    ).to_json_dict(),
  )


def _run_write_main(
  *,
  collection_name: str,
  tenant_id: str,
  task_id: str | None,
  run_id: str | None,
  attempt_id: str | None,
) -> None:
  identity = _identity_from_args(
    tenant_id=tenant_id,
    task_id=task_id,
    run_id=run_id,
    attempt_id=attempt_id,
  )
  scope = _scope_from_identity(identity)
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store)
  evidence_items = build_cross_domain_codec_evidence(scope)
  for evidence in evidence_items:
    persistence.append(evidence)
  analysis = _analyze(scope, persistence)
  fingerprint = DiagnosticAssessmentFingerprint.from_analysis(analysis)
  evidence_ids = tuple(str(item.evidence_id) for item in evidence_items)
  store_type = type(store).__name__
  probe.close_document_store(store)
  _emit(
    DurabilityWriteResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.WRITE_MAIN.value,
      identity=identity,
      evidence_ids=evidence_ids,
      evidence_count=len(evidence_ids),
      assessment_fingerprint=fingerprint,
      store_type=store_type,
      exit_code=_EXIT_OK,
    ).to_json_dict(),
  )


def _run_read_main(
  *,
  collection_name: str,
  identity_file: Path,
) -> None:
  raw = json.loads(identity_file.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    _fail("identity file invalid")
  identity_payload = raw.get("identity")
  if not isinstance(identity_payload, dict):
    _fail("identity payload invalid")
  identity = ExecutionIdentity(
    tenant_id=str(identity_payload["tenant_id"]),
    task_id=str(identity_payload["task_id"]),
    run_id=str(identity_payload["run_id"]),
    attempt_id=(
      str(identity_payload["attempt_id"])
      if identity_payload.get("attempt_id") is not None
      else None
    ),
  )
  scope = _scope_from_identity(identity)
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store)
  collected = collect_all_evidence(
    persistence,
    tenant_id=scope.tenant_id,
    task_id=scope.task_id,
    run_id=scope.run_id,
    page_size=_PAGINATION_PAGE_SIZE,
  )
  analysis = _analyze(scope, persistence)
  fingerprint = DiagnosticAssessmentFingerprint.from_analysis(analysis)
  evidence_ids = tuple(str(item.evidence_id) for item in collected)
  store_type = type(store).__name__
  probe.close_document_store(store)
  _emit(
    DurabilityReadResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.READ_MAIN.value,
      identity=identity,
      evidence_ids=evidence_ids,
      evidence_count=len(evidence_ids),
      assessment_fingerprint=fingerprint,
      store_type=store_type,
      exit_code=_EXIT_OK,
    ).to_json_dict(),
  )


def _run_idempotent_retry(
  *,
  collection_name: str,
  identity_file: Path,
  evidence_id: str,
) -> None:
  raw = json.loads(identity_file.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    _fail("identity file invalid")
  identity_payload = raw.get("identity")
  if not isinstance(identity_payload, dict):
    _fail("identity payload invalid")
  identity = ExecutionIdentity(
    tenant_id=str(identity_payload["tenant_id"]),
    task_id=str(identity_payload["task_id"]),
    run_id=str(identity_payload["run_id"]),
    attempt_id=(
      str(identity_payload["attempt_id"])
      if identity_payload.get("attempt_id") is not None
      else None
    ),
  )
  scope = _scope_from_identity(identity)
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store)
  collected = collect_all_evidence(
    persistence,
    tenant_id=scope.tenant_id,
    task_id=scope.task_id,
    run_id=scope.run_id,
  )
  target = next((item for item in collected if str(item.evidence_id) == evidence_id), None)
  if target is None:
    _fail(f"evidence {evidence_id!r} not found in durable store")
  retry_result = persistence.append(target)
  recount = collect_all_evidence(
    persistence,
    tenant_id=scope.tenant_id,
    task_id=scope.task_id,
    run_id=scope.run_id,
  )
  idempotent = retry_result == target and len(recount) == len(collected)
  probe.close_document_store(store)
  _emit(
    IdempotentRetryResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.IDEMPOTENT_RETRY.value,
      ok=idempotent,
      idempotent=idempotent,
      evidence_count=len(recount),
      exit_code=_EXIT_OK if idempotent else _EXIT_ERROR,
    ).to_json_dict(),
  )


def _run_conflict_append(
  *,
  collection_name: str,
  identity_file: Path,
  evidence_id: str,
) -> None:
  raw = json.loads(identity_file.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    _fail("identity file invalid")
  identity_payload = raw.get("identity")
  if not isinstance(identity_payload, dict):
    _fail("identity payload invalid")
  identity = ExecutionIdentity(
    tenant_id=str(identity_payload["tenant_id"]),
    task_id=str(identity_payload["task_id"]),
    run_id=str(identity_payload["run_id"]),
    attempt_id=(
      str(identity_payload["attempt_id"])
      if identity_payload.get("attempt_id") is not None
      else None
    ),
  )
  scope = _scope_from_identity(identity)
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store)
  conflicting = sample_functional_evidence(
    evidence_id=validate_event_id(evidence_id),
    scope=scope,
    operation_name="conflicting-payload",
  )
  conflict_detected = False
  try:
    persistence.append(conflicting)
  except FunctionalEvidencePersistenceConflictError:
    conflict_detected = True
  probe.close_document_store(store)
  _emit(
    ConflictAppendResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.CONFLICT_APPEND.value,
      ok=conflict_detected,
      conflict_detected=conflict_detected,
      exit_code=_EXIT_OK if conflict_detected else _EXIT_ERROR,
    ).to_json_dict(),
  )


def _run_tenant_write(
  *,
  collection_name: str,
  tenant_a: str,
  tenant_b: str,
  identity_file: Path,
) -> None:
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store)
  scope_a = sample_functional_evidence_scope(tenant_id=tenant_a)
  scope_b = sample_functional_evidence_scope(tenant_id=tenant_b)
  persistence.append(sample_functional_evidence(scope=scope_a, operation_name="tenant-a"))
  persistence.append(sample_functional_evidence(scope=scope_b, operation_name="tenant-b"))
  probe.close_document_store(store)
  identity_a = ExecutionIdentity(
    tenant_id=scope_a.tenant_id,
    task_id=str(scope_a.task_id),
    run_id=str(scope_a.run_id),
    attempt_id=str(scope_a.attempt_id) if scope_a.attempt_id is not None else None,
  )
  identity_b = ExecutionIdentity(
    tenant_id=scope_b.tenant_id,
    task_id=str(scope_b.task_id),
    run_id=str(scope_b.run_id),
    attempt_id=str(scope_b.attempt_id) if scope_b.attempt_id is not None else None,
  )
  identity_file.write_text(
    json.dumps(
      {
        "schema_version": IPC_SCHEMA_VERSION,
        "tenant_a": identity_a.to_json_dict(),
        "tenant_b": identity_b.to_json_dict(),
      },
      indent=2,
    ),
    encoding="utf-8",
  )
  _emit(
    DurabilityProbeResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.TENANT_WRITE.value,
      ok=True,
      detail=f"tenant_a={tenant_a};tenant_b={tenant_b}",
      exit_code=_EXIT_OK,
    ).to_json_dict(),
  )


def _run_tenant_read(
  *,
  collection_name: str,
  requested_tenant: str,
  other_tenant: str,
  identity_file: Path,
) -> None:
  raw = json.loads(identity_file.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    _fail("identity file invalid")
  tenant_a_payload = raw.get("tenant_a")
  tenant_b_payload = raw.get("tenant_b")
  if not isinstance(tenant_a_payload, dict) or not isinstance(tenant_b_payload, dict):
    _fail("tenant identity payload invalid")
  identity_a = ExecutionIdentity(
    tenant_id=str(tenant_a_payload["tenant_id"]),
    task_id=str(tenant_a_payload["task_id"]),
    run_id=str(tenant_a_payload["run_id"]),
    attempt_id=(
      str(tenant_a_payload["attempt_id"])
      if tenant_a_payload.get("attempt_id") is not None
      else None
    ),
  )
  identity_b = ExecutionIdentity(
    tenant_id=str(tenant_b_payload["tenant_id"]),
    task_id=str(tenant_b_payload["task_id"]),
    run_id=str(tenant_b_payload["run_id"]),
    attempt_id=(
      str(tenant_b_payload["attempt_id"])
      if tenant_b_payload.get("attempt_id") is not None
      else None
    ),
  )
  if requested_tenant == identity_a.tenant_id:
    requested_identity = identity_a
    other_identity = identity_b
  elif requested_tenant == identity_b.tenant_id:
    requested_identity = identity_b
    other_identity = identity_a
  else:
    _fail("requested tenant not found in identity file")
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store)
  requested_scope = _scope_from_identity(requested_identity)
  other_scope = _scope_from_identity(other_identity)
  requested_count = len(
    collect_all_evidence(
      persistence,
      tenant_id=requested_scope.tenant_id,
      task_id=requested_scope.task_id,
      run_id=requested_scope.run_id,
    ),
  )
  leak_scope = PipelineEvidenceScope(
    tenant_id=requested_scope.tenant_id,
    task_id=other_scope.task_id,
    run_id=other_scope.run_id,
    attempt_id=other_scope.attempt_id,
  )
  leak_count = len(
    collect_all_evidence(
      persistence,
      tenant_id=leak_scope.tenant_id,
      task_id=leak_scope.task_id,
      run_id=leak_scope.run_id,
    ),
  )
  other_direct_count = len(
    collect_all_evidence(
      persistence,
      tenant_id=other_scope.tenant_id,
      task_id=other_scope.task_id,
      run_id=other_scope.run_id,
    ),
  )
  ok = requested_count == 1 and leak_count == 0 and other_direct_count == 1
  probe.close_document_store(store)
  _emit(
    TenantIsolationResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.TENANT_READ.value,
      ok=ok,
      requested_tenant_count=requested_count,
      other_tenant_count=leak_count,
      exit_code=_EXIT_OK if ok else _EXIT_ERROR,
    ).to_json_dict(),
  )


def _run_pagination_write(
  *,
  collection_name: str,
  tenant_id: str,
) -> None:
  identity = _identity_from_args(tenant_id=tenant_id, task_id=None, run_id=None, attempt_id=None)
  scope = _scope_from_identity(identity)
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store, query_page_limit=_PAGINATION_PAGE_SIZE)
  evidence_items = build_pagination_evidence(scope)
  for evidence in evidence_items:
    persistence.append(evidence)
  probe.close_document_store(store)
  identity_path = Path(os.environ["DIAG_D1_R1_IDENTITY_FILE"])
  identity_path.write_text(
    json.dumps(
      {
        "schema_version": IPC_SCHEMA_VERSION,
        "identity": identity.to_json_dict(),
        "expected_count": len(evidence_items),
      },
      indent=2,
    ),
    encoding="utf-8",
  )
  _emit(
    DurabilityProbeResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.PAGINATION_WRITE.value,
      ok=True,
      detail=f"evidence_count={len(evidence_items)}",
      exit_code=_EXIT_OK,
    ).to_json_dict(),
  )


def _run_pagination_read(*, collection_name: str, identity_file: Path) -> None:
  raw = json.loads(identity_file.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    _fail("identity file invalid")
  identity_payload = raw.get("identity")
  expected_count = raw.get("expected_count")
  if not isinstance(identity_payload, dict):
    _fail("identity payload invalid")
  if not isinstance(expected_count, int) or isinstance(expected_count, bool):
    _fail("expected_count invalid")
  identity = ExecutionIdentity(
    tenant_id=str(identity_payload["tenant_id"]),
    task_id=str(identity_payload["task_id"]),
    run_id=str(identity_payload["run_id"]),
    attempt_id=None,
  )
  scope = _scope_from_identity(identity)
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  persistence = _build_persistence(store, query_page_limit=_PAGINATION_PAGE_SIZE)
  collected = collect_all_evidence(
    persistence,
    tenant_id=scope.tenant_id,
    task_id=scope.task_id,
    run_id=scope.run_id,
    page_size=_PAGINATION_PAGE_SIZE,
  )
  ok = len(collected) == expected_count
  probe.close_document_store(store)
  _emit(
    DurabilityProbeResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.PAGINATION_READ.value,
      ok=ok,
      detail=f"collected={len(collected)};expected={expected_count}",
      exit_code=_EXIT_OK if ok else _EXIT_ERROR,
    ).to_json_dict(),
  )


def _run_cleanup(*, collection_name: str, tenant_ids: tuple[str, ...]) -> None:
  probe = _build_probe(collection_name)
  store = probe.build_document_store()
  for tenant_id in tenant_ids:
    _purge_tenant_documents(store, tenant_id)
  probe.close_document_store(store)
  _emit(
    DurabilityProbeResult(
      schema_version=IPC_SCHEMA_VERSION,
      pid=os.getpid(),
      phase=DurabilityProbePhase.CLEANUP.value,
      ok=True,
      detail=f"purged={len(tenant_ids)}",
      exit_code=_EXIT_OK,
    ).to_json_dict(),
  )


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="D1-R1 functional evidence durability process probe")
  subparsers = parser.add_subparsers(dest="command", required=True)

  probe = subparsers.add_parser("probe")
  probe.add_argument("--collection-name", required=True)

  write_main = subparsers.add_parser("write-main")
  write_main.add_argument("--collection-name", required=True)
  write_main.add_argument("--tenant-id", required=True)
  write_main.add_argument("--task-id")
  write_main.add_argument("--run-id")
  write_main.add_argument("--attempt-id")

  read_main = subparsers.add_parser("read-main")
  read_main.add_argument("--collection-name", required=True)
  read_main.add_argument("--identity-file", type=Path, required=True)

  idempotent = subparsers.add_parser("idempotent-retry")
  idempotent.add_argument("--collection-name", required=True)
  idempotent.add_argument("--identity-file", type=Path, required=True)
  idempotent.add_argument("--evidence-id", required=True)

  conflict = subparsers.add_parser("conflict-append")
  conflict.add_argument("--collection-name", required=True)
  conflict.add_argument("--identity-file", type=Path, required=True)
  conflict.add_argument("--evidence-id", required=True)

  tenant_write = subparsers.add_parser("tenant-write")
  tenant_write.add_argument("--collection-name", required=True)
  tenant_write.add_argument("--tenant-a", required=True)
  tenant_write.add_argument("--tenant-b", required=True)
  tenant_write.add_argument("--identity-file", type=Path, required=True)

  tenant_read = subparsers.add_parser("tenant-read")
  tenant_read.add_argument("--collection-name", required=True)
  tenant_read.add_argument("--requested-tenant", required=True)
  tenant_read.add_argument("--other-tenant", required=True)
  tenant_read.add_argument("--identity-file", type=Path, required=True)

  pagination_write = subparsers.add_parser("pagination-write")
  pagination_write.add_argument("--collection-name", required=True)
  pagination_write.add_argument("--tenant-id", required=True)

  pagination_read = subparsers.add_parser("pagination-read")
  pagination_read.add_argument("--collection-name", required=True)
  pagination_read.add_argument("--identity-file", type=Path, required=True)

  cleanup = subparsers.add_parser("cleanup")
  cleanup.add_argument("--collection-name", required=True)
  cleanup.add_argument("--tenant-id", action="append", required=True)

  return parser


def main(argv: list[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  if args.command == "probe":
    _run_probe(args.collection_name)
    return
  if args.command == "write-main":
    _run_write_main(
      collection_name=args.collection_name,
      tenant_id=args.tenant_id,
      task_id=args.task_id,
      run_id=args.run_id,
      attempt_id=args.attempt_id,
    )
    return
  if args.command == "read-main":
    _run_read_main(collection_name=args.collection_name, identity_file=args.identity_file)
    return
  if args.command == "idempotent-retry":
    _run_idempotent_retry(
      collection_name=args.collection_name,
      identity_file=args.identity_file,
      evidence_id=args.evidence_id,
    )
    return
  if args.command == "conflict-append":
    _run_conflict_append(
      collection_name=args.collection_name,
      identity_file=args.identity_file,
      evidence_id=args.evidence_id,
    )
    return
  if args.command == "tenant-write":
    _run_tenant_write(
      collection_name=args.collection_name,
      tenant_a=args.tenant_a,
      tenant_b=args.tenant_b,
      identity_file=args.identity_file,
    )
    return
  if args.command == "tenant-read":
    _run_tenant_read(
      collection_name=args.collection_name,
      requested_tenant=args.requested_tenant,
      other_tenant=args.other_tenant,
      identity_file=args.identity_file,
    )
    return
  if args.command == "pagination-write":
    _run_pagination_write(collection_name=args.collection_name, tenant_id=args.tenant_id)
    return
  if args.command == "pagination-read":
    _run_pagination_read(collection_name=args.collection_name, identity_file=args.identity_file)
    return
  if args.command == "cleanup":
    _run_cleanup(collection_name=args.collection_name, tenant_ids=tuple(args.tenant_id))
    return
  _fail(f"unsupported command: {args.command!r}")


if __name__ == "__main__":
  main()
