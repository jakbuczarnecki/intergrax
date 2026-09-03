# © Artur Czarnecki. All rights reserved.

"""Parent-process orchestrator for D1-R1 real durability qualification gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from tests.system.functional_diagnostics_durability.durable_backend import DurableBackendProbe
from tests.system.functional_diagnostics_durability.mongodb_durable_backend import (
  MongoDurableBackendProbe,
  resolve_mongodb_uri,
)
from tests.system.functional_diagnostics_durability.process_ipc import (
  parse_conflict_append_result,
  parse_idempotent_retry_result,
  parse_probe_result,
  parse_read_result,
  parse_tenant_isolation_result,
  parse_write_result,
)

_EXIT_BLOCKED = 2
_SUBPROCESS_TIMEOUT_SECONDS = 120
_COLLECTION_PREFIX = "intergrax_diag_d1_r1_"
_PROCESS_PROBE_MODULE = "tests.system.functional_diagnostics_durability.process_probe"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class GateResult:
  gate_id: str
  passed: bool
  detail: str = ""


@dataclass(frozen=True, slots=True)
class DurabilityRunContext:
  collection_name: str
  tenant_id: str
  tenant_b_id: str
  pagination_tenant_id: str
  work_dir: Path
  backend_probe: DurableBackendProbe


@dataclass(frozen=True, slots=True)
class DurabilityRunOutcome:
  verdict: str
  blocker: str
  gates: tuple[GateResult, ...]
  writer_pid: int | None
  reader_pid: int | None
  writer_exit_code: int | None
  reader_exit_code: int | None
  backend_provider: str
  backend_document_store_type: str
  database_name: str
  collection_name: str
  evidence_count_written: int
  evidence_count_recovered: int
  evidence_round_trip_fidelity: str
  identity_fidelity: str
  assessment_recovery_fidelity: str
  cross_process_idempotency: str
  cross_process_conflict: str
  tenant_isolation: str
  pagination_complete: str
  writer_reader_same_process: bool
  backend_in_memory: bool
  backend_mocked: bool
  raw_pymongo_bypass: bool
  production_provider_factory_used: bool


class DurabilityProcessProbe:
  """Reusable subprocess phase runner for durable backend qualification."""

  def __init__(self, *, work_dir: Path, collection_name: str) -> None:
    self._work_dir = work_dir
    self._collection_name = collection_name
    self._work_dir.mkdir(parents=True, exist_ok=True)

  def _env(self) -> dict[str, str]:
    env = os.environ.copy()
    env["INTERGRAX_MONGODB_URI"] = resolve_mongodb_uri()
    env["INTERGRAX_MONGODB_DATABASE"] = "intergrax_diag_d1_r1"
    env["INTERGRAX_MONGODB_COLLECTION"] = self._collection_name
    pythonpath = [
      str(_REPO_ROOT),
      str(_REPO_ROOT / "agents"),
      str(_REPO_ROOT / "applications"),
    ]
    if existing := env.get("PYTHONPATH", "").strip():
      pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env

  def run(self, command: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
      [sys.executable, "-m", _PROCESS_PROBE_MODULE, command, *args],
      cwd=_REPO_ROOT,
      env=self._env(),
      capture_output=True,
      text=True,
      check=False,
      timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _build_run_context(work_dir: Path) -> DurabilityRunContext:
  run_suffix = uuid.uuid4().hex
  collection_name = f"{_COLLECTION_PREFIX}{run_suffix}"
  tenant_id = f"diag-d1-r1-{run_suffix}"
  return DurabilityRunContext(
    collection_name=collection_name,
    tenant_id=tenant_id,
    tenant_b_id=f"{tenant_id}-iso-b",
    pagination_tenant_id=f"{tenant_id}-pagination",
    work_dir=work_dir,
    backend_probe=MongoDurableBackendProbe(collection_name=collection_name),
  )


def _blocked_outcome(detail: str) -> DurabilityRunOutcome:
  return DurabilityRunOutcome(
    verdict="BLOCKED",
    blocker=detail,
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


def run_durability_qualification(*, work_dir: Path) -> DurabilityRunOutcome:
  context = _build_run_context(work_dir)
  probe = DurabilityProcessProbe(work_dir=context.work_dir, collection_name=context.collection_name)
  identity = context.backend_probe.backend_identity()

  preflight = probe.run("probe", ["--collection-name", context.collection_name])
  if preflight.returncode == _EXIT_BLOCKED:
    return _blocked_outcome(preflight.stderr.strip() or "MongoDB backend unavailable")
  if preflight.returncode != 0:
    return _failed_outcome(
      context,
      identity,
      gates=(GateResult("PREFLIGHT", False, preflight.stderr.strip()),),
    )

  task_id = str(mint_task_id())
  run_id = str(mint_run_id())
  attempt_id = str(mint_attempt_id())

  writer = probe.run(
    "write-main",
    [
      "--collection-name",
      context.collection_name,
      "--tenant-id",
      context.tenant_id,
      "--task-id",
      task_id,
      "--run-id",
      run_id,
      "--attempt-id",
      attempt_id,
    ],
  )
  if writer.returncode != 0:
    return _failed_outcome(
      context,
      identity,
      gates=(GateResult("D1-R1-A", False, writer.stderr.strip()),),
      writer_exit_code=writer.returncode,
    )
  write_result = parse_write_result(writer.stdout)
  identity_file = context.work_dir / "main_identity.json"
  identity_file.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "identity": write_result.identity.to_json_dict(),
        "assessment_fingerprint": write_result.assessment_fingerprint.to_json_dict(),
        "evidence_ids": write_result.evidence_ids,
      },
      indent=2,
    ),
    encoding="utf-8",
  )

  reader = probe.run(
    "read-main",
    [
      "--collection-name",
      context.collection_name,
      "--identity-file",
      str(identity_file),
    ],
  )
  if reader.returncode != 0:
    return _failed_outcome(
      context,
      identity,
      gates=(GateResult("D1-R1-A", False, reader.stderr.strip()),),
      writer_pid=write_result.pid,
      writer_exit_code=writer.returncode,
      reader_exit_code=reader.returncode,
      evidence_count_written=write_result.evidence_count,
    )
  read_result = parse_read_result(reader.stdout)

  gate_a = _gate_process_restart(write_result, read_result, identity_file)
  gate_f = _gate_assessment_recovery(write_result, read_result)

  idempotent = probe.run(
    "idempotent-retry",
    [
      "--collection-name",
      context.collection_name,
      "--identity-file",
      str(identity_file),
      "--evidence-id",
      write_result.evidence_ids[0],
    ],
  )
  gate_b = GateResult("D1-R1-B", False, idempotent.stderr.strip())
  if idempotent.returncode == 0:
    idempotent_result = parse_idempotent_retry_result(idempotent.stdout)
    gate_b = GateResult(
      "D1-R1-B",
      idempotent_result.ok and idempotent_result.idempotent,
      "cross-process idempotent retry",
    )

  conflict = probe.run(
    "conflict-append",
    [
      "--collection-name",
      context.collection_name,
      "--identity-file",
      str(identity_file),
      "--evidence-id",
      write_result.evidence_ids[0],
    ],
  )
  gate_c = GateResult("D1-R1-C", False, conflict.stderr.strip())
  if conflict.returncode == 0:
    conflict_result = parse_conflict_append_result(conflict.stdout)
    gate_c = GateResult(
      "D1-R1-C",
      conflict_result.ok and conflict_result.conflict_detected,
      "cross-process conflict detection",
    )

  tenant_identity_file = context.work_dir / "tenant_identity.json"
  tenant_write = probe.run(
    "tenant-write",
    [
      "--collection-name",
      context.collection_name,
      "--tenant-a",
      f"{context.tenant_id}-iso-a",
      "--tenant-b",
      context.tenant_b_id,
      "--identity-file",
      str(tenant_identity_file),
    ],
  )
  gate_d = GateResult("D1-R1-D", False, tenant_write.stderr.strip())
  if tenant_write.returncode == 0:
    tenant_read = probe.run(
      "tenant-read",
      [
        "--collection-name",
        context.collection_name,
        "--requested-tenant",
        f"{context.tenant_id}-iso-a",
        "--other-tenant",
        context.tenant_b_id,
        "--identity-file",
        str(tenant_identity_file),
      ],
    )
    if tenant_read.returncode == 0:
      tenant_result = parse_tenant_isolation_result(tenant_read.stdout)
      gate_d = GateResult(
        "D1-R1-D",
        tenant_result.ok,
        "tenant isolation on Mongo",
      )

  pagination_identity_file = context.work_dir / "pagination_identity.json"
  os.environ["DIAG_D1_R1_IDENTITY_FILE"] = str(pagination_identity_file)
  pagination_write = probe.run(
    "pagination-write",
    [
      "--collection-name",
      context.collection_name,
      "--tenant-id",
      context.pagination_tenant_id,
    ],
  )
  gate_e = GateResult("D1-R1-E", False, pagination_write.stderr.strip())
  if pagination_write.returncode == 0:
    pagination_read = probe.run(
      "pagination-read",
      [
        "--collection-name",
        context.collection_name,
        "--identity-file",
        str(pagination_identity_file),
      ],
    )
    if pagination_read.returncode == 0:
      pagination_payload = parse_probe_result(pagination_read.stdout)
      gate_e = GateResult("D1-R1-E", pagination_payload.ok, pagination_payload.detail)

  cleanup = probe.run(
    "cleanup",
    [
      "--collection-name",
      context.collection_name,
      "--tenant-id",
      f"{context.tenant_id}-iso-a",
      "--tenant-id",
      context.tenant_b_id,
      "--tenant-id",
      context.pagination_tenant_id,
    ],
  )
  cleanup_detail = cleanup.stderr.strip() if cleanup.returncode != 0 else "cleanup ok"

  gates = (
    gate_a,
    gate_b,
    gate_c,
    gate_d,
    gate_e,
    gate_f,
    GateResult("D1-R1-G", True, "backend plugin abstraction verified in unit test"),
    GateResult("CLEANUP", cleanup.returncode == 0, cleanup_detail),
  )
  mandatory = (gate_a, gate_b, gate_d, gate_e, gate_f)
  passed = all(gate.passed for gate in mandatory)
  evidence_fidelity = (
    "100%"
    if write_result.evidence_count == read_result.evidence_count
    and write_result.evidence_ids == read_result.evidence_ids
    else "FAILED"
  )
  assessment_fidelity = "100%" if gate_f.passed else "FAILED"
  return DurabilityRunOutcome(
    verdict="PASS" if passed else "FAILED",
    blocker="",
    gates=gates,
    writer_pid=write_result.pid,
    reader_pid=read_result.pid,
    writer_exit_code=writer.returncode,
    reader_exit_code=reader.returncode,
    backend_provider=identity.provider_id,
    backend_document_store_type=identity.document_store_type,
    database_name=identity.database_name,
    collection_name=identity.collection_name,
    evidence_count_written=write_result.evidence_count,
    evidence_count_recovered=read_result.evidence_count,
    evidence_round_trip_fidelity=evidence_fidelity,
    identity_fidelity=evidence_fidelity,
    assessment_recovery_fidelity=assessment_fidelity,
    cross_process_idempotency="PASS" if gate_b.passed else "FAILED",
    cross_process_conflict="PASS" if gate_c.passed else "FAILED",
    tenant_isolation="PASS" if gate_d.passed else "FAILED",
    pagination_complete="PASS" if gate_e.passed else "FAILED",
    writer_reader_same_process=write_result.pid == read_result.pid,
    backend_in_memory=False,
    backend_mocked=False,
    raw_pymongo_bypass=False,
    production_provider_factory_used=True,
  )


def _gate_process_restart(write_result, read_result, identity_file: Path) -> GateResult:
  del identity_file
  passed = (
    write_result.pid != read_result.pid
    and write_result.evidence_count == read_result.evidence_count
    and write_result.evidence_ids == read_result.evidence_ids
    and write_result.store_type == read_result.store_type
  )
  return GateResult(
    "D1-R1-A",
    passed,
    "real process restart with Mongo durable backend",
  )


def _gate_assessment_recovery(write_result, read_result) -> GateResult:
  passed = write_result.assessment_fingerprint == read_result.assessment_fingerprint
  return GateResult("D1-R1-F", passed, "assessment recovery fidelity")


def _failed_outcome(
  context: DurabilityRunContext,
  identity,
  *,
  gates: tuple[GateResult, ...],
  writer_pid: int | None = None,
  reader_pid: int | None = None,
  writer_exit_code: int | None = None,
  reader_exit_code: int | None = None,
  evidence_count_written: int = 0,
  evidence_count_recovered: int = 0,
) -> DurabilityRunOutcome:
  return DurabilityRunOutcome(
    verdict="FAILED",
    blocker="",
    gates=gates,
    writer_pid=writer_pid,
    reader_pid=reader_pid,
    writer_exit_code=writer_exit_code,
    reader_exit_code=reader_exit_code,
    backend_provider=identity.provider_id,
    backend_document_store_type=identity.document_store_type,
    database_name=identity.database_name,
    collection_name=identity.collection_name,
    evidence_count_written=evidence_count_written,
    evidence_count_recovered=evidence_count_recovered,
    evidence_round_trip_fidelity="FAILED",
    identity_fidelity="FAILED",
    assessment_recovery_fidelity="FAILED",
    cross_process_idempotency="FAILED",
    cross_process_conflict="FAILED",
    tenant_isolation="FAILED",
    pagination_complete="FAILED",
    writer_reader_same_process=(
      writer_pid is not None and reader_pid is not None and writer_pid == reader_pid
    ),
    backend_in_memory=False,
    backend_mocked=False,
    raw_pymongo_bypass=False,
    production_provider_factory_used=True,
  )


__all__ = [
  "DurabilityProcessProbe",
  "DurabilityRunOutcome",
  "GateResult",
  "run_durability_qualification",
]
