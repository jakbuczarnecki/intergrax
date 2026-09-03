# © Artur Czarnecki. All rights reserved.

"""S1 scale subprocess worker — multi-process writer/reader qualification."""

from __future__ import annotations

import hashlib
import argparse
import binascii
import os
import sys

from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceError,
    FunctionalEvidenceQueryRequest,
)
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceScope,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
    sample_functional_evidence,
)
from tests.system.functional_diagnostics_scale.metrics import MonotonicTimer
from tests.system.functional_diagnostics_scale.mongodb_backend import (
    MongoFunctionalDiagnosticsScaleProbe,
)
from tests.system.functional_diagnostics_scale.process_ipc import (
    IPC_SCHEMA_VERSION,
    ScaleWorkerPhase,
    ScaleWorkerResult,
    encode_ipc_payload,
)
from tests.system.functional_diagnostics_scale.profile import resolve_scale_profile
from tests.system.functional_diagnostics_scale.workload import (
    FunctionalEvidenceWorkloadGenerator,
)

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_BLOCKED = 2


def _isolated_scope_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}{digest}"


def _emit(result: ScaleWorkerResult) -> None:
    sys.stdout.write(encode_ipc_payload(result.to_json_dict()))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, *, code: int = _EXIT_ERROR) -> None:
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")
    sys.stderr.flush()
    raise SystemExit(code)


def _resolve_cursor_secret(hex_value: str) -> bytes:
    if not hex_value:
        _fail("cursor secret missing")
    try:
        secret = binascii.unhexlify(hex_value)
    except binascii.Error as exc:
        raise SystemExit(_EXIT_ERROR) from exc
    if len(secret) < 32:
        _fail("cursor secret too short")
    return secret


def _build_persistence(
    collection_name: str,
    *,
    cursor_secret: bytes,
    query_page_limit: int,
) -> DocumentStoreFunctionalEvidencePersistence:
    probe = MongoFunctionalDiagnosticsScaleProbe(collection_name=collection_name)
    probe.prepare()
    store = probe.build_document_store()
    return DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=cursor_secret,
        query_page_limit=query_page_limit,
    )


def _worker_write(args: argparse.Namespace) -> int:
    profile = resolve_scale_profile(args.profile_name)
    generator = FunctionalEvidenceWorkloadGenerator(profile)
    persistence = _build_persistence(
        args.collection_name,
        cursor_secret=_resolve_cursor_secret(args.cursor_secret_hex),
        query_page_limit=args.query_page_limit,
    )
    written = 0
    append_latencies: list[float] = []
    errors = 0
    for index, identity in enumerate(generator.execution_identities()):
        if index % args.worker_count != args.worker_index:
            continue
        for evidence in generator.evidence_for_execution(identity):
            timer = MonotonicTimer()
            try:
                persistence.append(evidence)
            except (FunctionalEvidencePersistenceError, ValueError, TypeError):
                errors += 1
                continue
            append_latencies.append(timer.elapsed_ms())
            written += 1
    _emit(
        ScaleWorkerResult(
            schema_version=IPC_SCHEMA_VERSION,
            pid=os.getpid(),
            phase=ScaleWorkerPhase.WRITE.value,
            worker_index=args.worker_index,
            written_count=written,
            read_count=0,
            append_latency_ms=tuple(append_latencies),
            read_latency_ms=(),
            conflicts=0,
            errors=errors,
            exit_code=_EXIT_OK,
            detail="write-complete",
        ),
    )
    return _EXIT_OK


def _worker_read(args: argparse.Namespace) -> int:
    profile = resolve_scale_profile(args.profile_name)
    generator = FunctionalEvidenceWorkloadGenerator(profile)
    persistence = _build_persistence(
        args.collection_name,
        cursor_secret=_resolve_cursor_secret(args.cursor_secret_hex),
        query_page_limit=args.query_page_limit,
    )
    read_latencies: list[float] = []
    read_count = 0
    errors = 0
    for index, identity in enumerate(generator.execution_identities()):
        if index % args.worker_count != args.worker_index:
            continue
        timer = MonotonicTimer()
        try:
            collected = collect_all_evidence(
                persistence,
                tenant_id=identity.tenant_id,
                task_id=identity.task_id,
                run_id=identity.run_id,
                page_size=args.page_size,
            )
        except (FunctionalEvidencePersistenceError, ValueError, TypeError):
            errors += 1
            continue
        read_latencies.append(timer.elapsed_ms())
        read_count += len(collected)
    _emit(
        ScaleWorkerResult(
            schema_version=IPC_SCHEMA_VERSION,
            pid=os.getpid(),
            phase=ScaleWorkerPhase.READ.value,
            worker_index=args.worker_index,
            written_count=0,
            read_count=read_count,
            append_latency_ms=(),
            read_latency_ms=tuple(read_latencies),
            conflicts=0,
            errors=errors,
            exit_code=_EXIT_OK,
            detail="read-complete",
        ),
    )
    return _EXIT_OK


def _worker_idempotent(args: argparse.Namespace) -> int:
    profile = resolve_scale_profile(args.profile_name)
    generator = FunctionalEvidenceWorkloadGenerator(profile)
    persistence = _build_persistence(
        args.collection_name,
        cursor_secret=_resolve_cursor_secret(args.cursor_secret_hex),
        query_page_limit=args.query_page_limit,
    )
    identity = generator.execution_identities()[-1]
    evidence = generator.evidence_for_execution(identity)[0]
    target_id = evidence.evidence_id
    conflicts = 0
    errors = 0
    for _ in range(20):
        try:
            persistence.append(evidence)
        except FunctionalEvidencePersistenceConflictError:
            conflicts += 1
        except (FunctionalEvidencePersistenceError, ValueError, TypeError):
            errors += 1
    collected = collect_all_evidence(
        persistence,
        tenant_id=identity.tenant_id,
        task_id=identity.task_id,
        run_id=identity.run_id,
        page_size=args.page_size,
    )
    matching = [item for item in collected if item.evidence_id == target_id]
    idempotent_ok = len(matching) == 1
    detail = "idempotent-count-ok" if idempotent_ok else "idempotent-count-fail"
    _emit(
        ScaleWorkerResult(
            schema_version=IPC_SCHEMA_VERSION,
            pid=os.getpid(),
            phase=ScaleWorkerPhase.IDEMPOTENT.value,
            worker_index=args.worker_index,
            written_count=1 if idempotent_ok else 0,
            read_count=len(matching),
            append_latency_ms=(),
            read_latency_ms=(),
            conflicts=conflicts,
            errors=errors,
            exit_code=_EXIT_OK if idempotent_ok else _EXIT_ERROR,
            detail=detail,
        ),
    )
    return _EXIT_OK if idempotent_ok else _EXIT_ERROR


def _worker_conflict(args: argparse.Namespace) -> int:
    profile = resolve_scale_profile(args.profile_name)
    generator = FunctionalEvidenceWorkloadGenerator(profile)
    persistence = _build_persistence(
        args.collection_name,
        cursor_secret=_resolve_cursor_secret(args.cursor_secret_hex),
        query_page_limit=args.query_page_limit,
    )
    conflict_scope = PipelineEvidenceScope(
        tenant_id=f"s1-conflict-{args.collection_name}",
        task_id=validate_task_id(_isolated_scope_id("task_", args.collection_name, str(args.worker_index), "task")),
        run_id=validate_run_id(_isolated_scope_id("run_", args.collection_name, str(args.worker_index), "run")),
        attempt_id=validate_attempt_id(
            _isolated_scope_id("attempt_", args.collection_name, str(args.worker_index), "attempt"),
        ),
    )
    original = sample_functional_evidence(
        scope=conflict_scope,
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        operation_name="scale-conflict-original",
    )
    conflicting = sample_functional_evidence(
        evidence_id=original.evidence_id,
        scope=original.scope,
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        operation_name="scale-conflict-conflicting",
    )
    conflicts = 0
    errors = 0
    persistence.append(original)
    for _ in range(10):
        try:
            persistence.append(conflicting)
        except FunctionalEvidencePersistenceConflictError:
            conflicts += 1
        except (FunctionalEvidencePersistenceError, ValueError, TypeError):
            errors += 1
    detail = "conflict-ok" if conflicts > 0 else "conflict-missing"
    _emit(
        ScaleWorkerResult(
            schema_version=IPC_SCHEMA_VERSION,
            pid=os.getpid(),
            phase=ScaleWorkerPhase.CONFLICT.value,
            worker_index=args.worker_index,
            written_count=1,
            read_count=0,
            append_latency_ms=(),
            read_latency_ms=(),
            conflicts=conflicts,
            errors=errors,
            exit_code=_EXIT_OK if conflicts > 0 else _EXIT_ERROR,
            detail=detail,
        ),
    )
    return _EXIT_OK if conflicts > 0 else _EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S1 scale worker subprocess")
    parser.add_argument("phase", choices=[item.value for item in ScaleWorkerPhase])
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--cursor-secret-hex", required=True)
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--query-page-limit", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-name", required=True)
    args = parser.parse_args(argv)
    if args.phase == ScaleWorkerPhase.WRITE.value:
        return _worker_write(args)
    if args.phase == ScaleWorkerPhase.READ.value:
        return _worker_read(args)
    if args.phase == ScaleWorkerPhase.IDEMPOTENT.value:
        return _worker_idempotent(args)
    if args.phase == ScaleWorkerPhase.CONFLICT.value:
        return _worker_conflict(args)
    _fail(f"unsupported phase: {args.phase}")
    return _EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
