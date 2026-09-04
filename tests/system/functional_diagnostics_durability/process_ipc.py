# © Artur Czarnecki. All rights reserved.

"""Typed IPC contracts for D1-R1 process probe subprocess communication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from tests.system.functional_diagnostics_durability.assessment_fingerprint import (
  DiagnosticAssessmentFingerprint,
)

IPC_SCHEMA_VERSION = 1


class DurabilityProbePhase(StrEnum):
  PROBE = "probe"
  WRITE_MAIN = "write-main"
  READ_MAIN = "read-main"
  IDEMPOTENT_RETRY = "idempotent-retry"
  CONFLICT_APPEND = "conflict-append"
  TENANT_WRITE = "tenant-write"
  TENANT_READ = "tenant-read"
  PAGINATION_WRITE = "pagination-write"
  PAGINATION_READ = "pagination-read"
  CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
  tenant_id: str
  task_id: str
  run_id: str
  attempt_id: str | None

  def to_json_dict(self) -> dict[str, str | None]:
    return {
      "tenant_id": self.tenant_id,
      "task_id": self.task_id,
      "run_id": self.run_id,
      "attempt_id": self.attempt_id,
    }


@dataclass(frozen=True, slots=True)
class DurabilityWriteResult:
  schema_version: int
  pid: int
  phase: str
  identity: ExecutionIdentity
  evidence_ids: tuple[str, ...]
  evidence_count: int
  assessment_fingerprint: DiagnosticAssessmentFingerprint
  store_type: str
  exit_code: int

  def to_json_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.schema_version,
      "pid": self.pid,
      "phase": self.phase,
      "identity": self.identity.to_json_dict(),
      "evidence_ids": self.evidence_ids,
      "evidence_count": self.evidence_count,
      "assessment_fingerprint": self.assessment_fingerprint.to_json_dict(),
      "store_type": self.store_type,
      "exit_code": self.exit_code,
    }


@dataclass(frozen=True, slots=True)
class DurabilityReadResult:
  schema_version: int
  pid: int
  phase: str
  identity: ExecutionIdentity
  evidence_ids: tuple[str, ...]
  evidence_count: int
  assessment_fingerprint: DiagnosticAssessmentFingerprint
  store_type: str
  exit_code: int

  def to_json_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.schema_version,
      "pid": self.pid,
      "phase": self.phase,
      "identity": self.identity.to_json_dict(),
      "evidence_ids": self.evidence_ids,
      "evidence_count": self.evidence_count,
      "assessment_fingerprint": self.assessment_fingerprint.to_json_dict(),
      "store_type": self.store_type,
      "exit_code": self.exit_code,
    }


@dataclass(frozen=True, slots=True)
class DurabilityProbeResult:
  schema_version: int
  pid: int
  phase: str
  ok: bool
  detail: str
  exit_code: int

  def to_json_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.schema_version,
      "pid": self.pid,
      "phase": self.phase,
      "ok": self.ok,
      "detail": self.detail,
      "exit_code": self.exit_code,
    }


@dataclass(frozen=True, slots=True)
class IdempotentRetryResult:
  schema_version: int
  pid: int
  phase: str
  ok: bool
  idempotent: bool
  evidence_count: int
  exit_code: int

  def to_json_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.schema_version,
      "pid": self.pid,
      "phase": self.phase,
      "ok": self.ok,
      "idempotent": self.idempotent,
      "evidence_count": self.evidence_count,
      "exit_code": self.exit_code,
    }


@dataclass(frozen=True, slots=True)
class ConflictAppendResult:
  schema_version: int
  pid: int
  phase: str
  ok: bool
  conflict_detected: bool
  exit_code: int

  def to_json_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.schema_version,
      "pid": self.pid,
      "phase": self.phase,
      "ok": self.ok,
      "conflict_detected": self.conflict_detected,
      "exit_code": self.exit_code,
    }


@dataclass(frozen=True, slots=True)
class TenantIsolationResult:
  schema_version: int
  pid: int
  phase: str
  ok: bool
  requested_tenant_count: int
  other_tenant_count: int
  exit_code: int

  def to_json_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.schema_version,
      "pid": self.pid,
      "phase": self.phase,
      "ok": self.ok,
      "requested_tenant_count": self.requested_tenant_count,
      "other_tenant_count": self.other_tenant_count,
      "exit_code": self.exit_code,
    }


def encode_ipc_payload(payload: dict[str, object]) -> str:
  return json.dumps(payload, sort_keys=True)


def _require_int(value: object, field: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool):
    raise ValueError(f"{field}_invalid")
  return value


def _require_str(value: object, field: str) -> str:
  if not isinstance(value, str):
    raise ValueError(f"{field}_invalid")
  return value


def _require_bool(value: object, field: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"{field}_invalid")
  return value


def _parse_identity(payload: object) -> ExecutionIdentity:
  if not isinstance(payload, dict):
    raise ValueError("execution_identity_invalid")
  attempt_id = payload.get("attempt_id")
  if attempt_id is not None and not isinstance(attempt_id, str):
    raise ValueError("execution_identity_attempt_id_invalid")
  return ExecutionIdentity(
    tenant_id=_require_str(payload.get("tenant_id"), "tenant_id"),
    task_id=_require_str(payload.get("task_id"), "task_id"),
    run_id=_require_str(payload.get("run_id"), "run_id"),
    attempt_id=attempt_id,
  )


def _parse_schema_version(payload: dict[str, object]) -> int:
  version = _require_int(payload.get("schema_version"), "schema_version")
  if version != IPC_SCHEMA_VERSION:
    raise ValueError("ipc_schema_version_mismatch")
  return version


def parse_write_result(stdout: str) -> DurabilityWriteResult:
  raw = json.loads(stdout)
  if not isinstance(raw, dict):
    raise ValueError("write_result_invalid")
  schema_version = _parse_schema_version(raw)
  evidence_ids_raw = raw.get("evidence_ids")
  if not isinstance(evidence_ids_raw, list):
    raise ValueError("evidence_ids_invalid")
  evidence_ids = tuple(_require_str(item, "evidence_id") for item in evidence_ids_raw)
  return DurabilityWriteResult(
    schema_version=schema_version,
    pid=_require_int(raw.get("pid"), "pid"),
    phase=_require_str(raw.get("phase"), "phase"),
    identity=_parse_identity(raw.get("identity")),
    evidence_ids=evidence_ids,
    evidence_count=_require_int(raw.get("evidence_count"), "evidence_count"),
    assessment_fingerprint=DiagnosticAssessmentFingerprint.from_json_mapping(
      raw.get("assessment_fingerprint", {}),
    ),
    store_type=_require_str(raw.get("store_type"), "store_type"),
    exit_code=_require_int(raw.get("exit_code"), "exit_code"),
  )


def parse_read_result(stdout: str) -> DurabilityReadResult:
  raw = json.loads(stdout)
  if not isinstance(raw, dict):
    raise ValueError("read_result_invalid")
  schema_version = _parse_schema_version(raw)
  evidence_ids_raw = raw.get("evidence_ids")
  if not isinstance(evidence_ids_raw, list):
    raise ValueError("evidence_ids_invalid")
  evidence_ids = tuple(_require_str(item, "evidence_id") for item in evidence_ids_raw)
  return DurabilityReadResult(
    schema_version=schema_version,
    pid=_require_int(raw.get("pid"), "pid"),
    phase=_require_str(raw.get("phase"), "phase"),
    identity=_parse_identity(raw.get("identity")),
    evidence_ids=evidence_ids,
    evidence_count=_require_int(raw.get("evidence_count"), "evidence_count"),
    assessment_fingerprint=DiagnosticAssessmentFingerprint.from_json_mapping(
      raw.get("assessment_fingerprint", {}),
    ),
    store_type=_require_str(raw.get("store_type"), "store_type"),
    exit_code=_require_int(raw.get("exit_code"), "exit_code"),
  )


def parse_probe_result(stdout: str) -> DurabilityProbeResult:
  raw = json.loads(stdout)
  if not isinstance(raw, dict):
    raise ValueError("probe_result_invalid")
  schema_version = _parse_schema_version(raw)
  return DurabilityProbeResult(
    schema_version=schema_version,
    pid=_require_int(raw.get("pid"), "pid"),
    phase=_require_str(raw.get("phase"), "phase"),
    ok=_require_bool(raw.get("ok"), "ok"),
    detail=_require_str(raw.get("detail"), "detail"),
    exit_code=_require_int(raw.get("exit_code"), "exit_code"),
  )


def parse_idempotent_retry_result(stdout: str) -> IdempotentRetryResult:
  raw = json.loads(stdout)
  if not isinstance(raw, dict):
    raise ValueError("idempotent_retry_result_invalid")
  schema_version = _parse_schema_version(raw)
  return IdempotentRetryResult(
    schema_version=schema_version,
    pid=_require_int(raw.get("pid"), "pid"),
    phase=_require_str(raw.get("phase"), "phase"),
    ok=_require_bool(raw.get("ok"), "ok"),
    idempotent=_require_bool(raw.get("idempotent"), "idempotent"),
    evidence_count=_require_int(raw.get("evidence_count"), "evidence_count"),
    exit_code=_require_int(raw.get("exit_code"), "exit_code"),
  )


def parse_conflict_append_result(stdout: str) -> ConflictAppendResult:
  raw = json.loads(stdout)
  if not isinstance(raw, dict):
    raise ValueError("conflict_append_result_invalid")
  schema_version = _parse_schema_version(raw)
  return ConflictAppendResult(
    schema_version=schema_version,
    pid=_require_int(raw.get("pid"), "pid"),
    phase=_require_str(raw.get("phase"), "phase"),
    ok=_require_bool(raw.get("ok"), "ok"),
    conflict_detected=_require_bool(raw.get("conflict_detected"), "conflict_detected"),
    exit_code=_require_int(raw.get("exit_code"), "exit_code"),
  )


def parse_tenant_isolation_result(stdout: str) -> TenantIsolationResult:
  raw = json.loads(stdout)
  if not isinstance(raw, dict):
    raise ValueError("tenant_isolation_result_invalid")
  schema_version = _parse_schema_version(raw)
  return TenantIsolationResult(
    schema_version=schema_version,
    pid=_require_int(raw.get("pid"), "pid"),
    phase=_require_str(raw.get("phase"), "phase"),
    ok=_require_bool(raw.get("ok"), "ok"),
    requested_tenant_count=_require_int(raw.get("requested_tenant_count"), "requested_tenant_count"),
    other_tenant_count=_require_int(raw.get("other_tenant_count"), "other_tenant_count"),
    exit_code=_require_int(raw.get("exit_code"), "exit_code"),
  )


__all__ = [
  "ConflictAppendResult",
  "DurabilityProbePhase",
  "DurabilityProbeResult",
  "DurabilityReadResult",
  "DurabilityWriteResult",
  "ExecutionIdentity",
  "IPC_SCHEMA_VERSION",
  "IdempotentRetryResult",
  "TenantIsolationResult",
  "encode_ipc_payload",
  "parse_conflict_append_result",
  "parse_idempotent_retry_result",
  "parse_probe_result",
  "parse_read_result",
  "parse_tenant_isolation_result",
  "parse_write_result",
]
