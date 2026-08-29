# © Artur Czarnecki. All rights reserved.

"""HARDEN-2A subprocess worker — real cross-process Problem persistence concurrency proof."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.registry.factory import resolve as resolve_integration
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingMethod
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemOccurrence,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceConflictError

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_SKIP = 2

_DOCUMENT_PARTITION_PREFIX = "intergrax.diagnostic_problem.v1"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_2a"
_START_POLL_SECONDS = 0.05
_START_TIMEOUT_SECONDS = 30.0


def _json_default(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if is_dataclass(value):
        return {key: _json_default(item) for key, item in asdict(value).items()}
    raise TypeError(f"unsupported JSON value type: {type(value)!r}")


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=_json_default))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, *, code: int = _EXIT_ERROR) -> None:
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")
    sys.stderr.flush()
    raise SystemExit(code)


def _resolve_platform_document_store() -> ConditionalDocumentStore:
    bootstrap_application_integration_catalog()
    integration_profile = IntegrationProfile(document_store="mongodb")
    store = resolve_integration(
        IntegrationCategory.DOCUMENT_STORE,
        profile=integration_profile,
    )
    return assert_conditional_document_store(store)


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


def _wait_for_start(start_path: Path) -> None:
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while not start_path.exists():
        if time.monotonic() >= deadline:
            _fail(f"start signal not observed: {start_path}")
        time.sleep(_START_POLL_SECONDS)


def _append_occurrence(
    base: Problem,
    *,
    subject_ref,
    observed_at: datetime,
) -> Problem:
    new_occurrence = ProblemOccurrence(
        subject_ref=subject_ref,
        observed_at=observed_at,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
    )
    merged_occurrences = base.occurrences + (new_occurrence,)
    merged_subject_refs = base.current_subject_refs
    if subject_ref not in merged_subject_refs:
        merged_subject_refs = base.current_subject_refs + (subject_ref,)
    return Problem(
        problem_id=base.problem_id,
        tenant_id=base.tenant_id,
        status=ProblemStatus.OPEN,
        first_seen_at=min(base.first_seen_at, observed_at),
        last_seen_at=max(base.last_seen_at, observed_at),
        occurrence_count=base.occurrence_count + 1,
        current_subject_refs=merged_subject_refs,
        occurrences=merged_occurrences,
        provenance=base.provenance,
        record_version=base.record_version + 1,
    )


def _run_probe() -> None:
    try:
        store = _resolve_platform_document_store()
        store.close()
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        _fail(
            "MongoDB backend unavailable for HARDEN-2A: "
            f"{type(exc).__name__}: {exc}. "
            "Start infra/docker/mongodb/docker-compose.yml or set INTERGRAX_MONGODB_URI.",
            code=_EXIT_SKIP,
        )
    _emit({"ok": True, "pid": os.getpid(), "phase": "probe"})


def _run_seed_baseline(*, tenant_id: str, other_tenant_id: str) -> None:
    store = _resolve_platform_document_store()
    persistence = DocumentStoreProblemPersistence(store)
    _purge_tenant_documents(store, tenant_id)
    _purge_tenant_documents(store, other_tenant_id)
    baseline = sample_problem(tenant_id=tenant_id)
    created = persistence.create(baseline)
    payload = {
        "ok": True,
        "pid": os.getpid(),
        "phase": "seed",
        "tenant_id": tenant_id,
        "other_tenant_id": other_tenant_id,
        "problem_id": str(created.problem_id),
        "baseline_occurrence_count": created.occurrence_count,
        "baseline_record_version": created.record_version,
        "baseline_subject_refs": created.current_subject_refs,
        "reconciliation_key": created.provenance.reconciliation_key,
    }
    persistence.close()
    store.close()
    _emit(payload)


def _run_concurrent_update(
    *,
    tenant_id: str,
    problem_id: str,
    worker_label: str,
    start_path: Path,
    update_path: Path,
    read_snapshot_path: Path,
    done_path: Path,
) -> None:
    _wait_for_start(start_path)
    store = _resolve_platform_document_store()
    persistence = DocumentStoreProblemPersistence(store)
    validated_id = ProblemId(problem_id)
    baseline = persistence.get(tenant_id=tenant_id, problem_id=validated_id)
    if baseline is None:
        _fail(f"baseline problem missing for update worker {worker_label!r}")

    read_snapshot_path.write_text(
        json.dumps(
            {
                "worker": worker_label,
                "record_version": baseline.record_version,
                "occurrence_count": baseline.occurrence_count,
            },
            default=_json_default,
        ),
        encoding="utf-8",
    )

    _wait_for_start(update_path)

    observed_at = datetime(2026, 8, 29, 12, worker_label.count("a") + 1, tzinfo=UTC)
    new_subject = _sample_subject_ref(tenant_id=tenant_id)
    candidate = _append_occurrence(
        baseline,
        subject_ref=new_subject,
        observed_at=observed_at,
    )

    outcome: dict[str, Any]
    try:
        updated = persistence.update(
            candidate,
            expected_version=baseline.record_version,
        )
        outcome = {
            "status": "updated",
            "occurrence_count": updated.occurrence_count,
            "record_version": updated.record_version,
            "new_subject_ref": new_subject,
        }
    except ProblemPersistenceConflictError as exc:
        outcome = {
            "status": "conflict",
            "error": str(exc),
            "new_subject_ref": new_subject,
        }

    persistence.close()
    store.close()
    done_path.write_text(
        json.dumps(
            {
                "ok": True,
                "pid": os.getpid(),
                "phase": "update",
                "worker": worker_label,
                "problem_id": problem_id,
                "tenant_id": tenant_id,
                **outcome,
            },
            default=_json_default,
        ),
        encoding="utf-8",
    )
    _emit({"ok": True, "pid": os.getpid(), "phase": "update", "worker": worker_label})


def _run_read_final(*, tenant_id: str, problem_id: str, other_tenant_id: str) -> None:
    store = _resolve_platform_document_store()
    persistence = DocumentStoreProblemPersistence(store)
    validated_id = ProblemId(problem_id)
    final = persistence.get(tenant_id=tenant_id, problem_id=validated_id)
    if final is None:
        _fail(f"final problem missing for tenant {tenant_id!r}")

    by_key = persistence.find_by_reconciliation_key(
        tenant_id=tenant_id,
        reconciliation_key=final.provenance.reconciliation_key,
    )
    listed = persistence.list_for_tenant(tenant_id)
    other_listed = persistence.list_for_tenant(other_tenant_id)

    payload = {
        "ok": True,
        "pid": os.getpid(),
        "phase": "read",
        "tenant_id": tenant_id,
        "problem_id": problem_id,
        "occurrence_count": final.occurrence_count,
        "record_version": final.record_version,
        "occurrence_subject_refs": [item.subject_ref for item in final.occurrences],
        "listed_problem_ids": [str(item.problem_id) for item in listed],
        "reconciliation_lookup_problem_id": (
            None if by_key is None else str(by_key.problem_id)
        ),
        "other_tenant_problem_count": len(other_listed),
    }
    persistence.close()
    store.close()
    _emit(payload)


def _run_cleanup(*, tenant_ids: tuple[str, ...]) -> None:
    store = _resolve_platform_document_store()
    for tenant_id in tenant_ids:
        _purge_tenant_documents(store, tenant_id)
    store.close()
    _emit({"ok": True, "pid": os.getpid(), "phase": "cleanup"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HARDEN-2A process concurrency proof worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="verify MongoDB via platform document-store resolution")

    seed = subparsers.add_parser("seed-baseline", help="purge tenants and seed baseline Problem")
    seed.add_argument("--tenant-id", required=True)
    seed.add_argument("--other-tenant-id", required=True)

    update = subparsers.add_parser(
        "concurrent-update",
        help="wait for start signal and append one occurrence via CAS update",
    )
    update.add_argument("--tenant-id", required=True)
    update.add_argument("--problem-id", required=True)
    update.add_argument("--worker-label", required=True)
    update.add_argument("--start-path", type=Path, required=True)
    update.add_argument("--update-path", type=Path, required=True)
    update.add_argument("--read-snapshot-path", type=Path, required=True)
    update.add_argument("--done-path", type=Path, required=True)

    read_final = subparsers.add_parser("read-final", help="fresh read-side verification")
    read_final.add_argument("--tenant-id", required=True)
    read_final.add_argument("--problem-id", required=True)
    read_final.add_argument("--other-tenant-id", required=True)

    cleanup = subparsers.add_parser("cleanup", help="purge proof tenant documents via store API")
    cleanup.add_argument("--tenant-id", action="append", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "probe":
        _run_probe()
        return
    if args.command == "seed-baseline":
        _run_seed_baseline(
            tenant_id=args.tenant_id,
            other_tenant_id=args.other_tenant_id,
        )
        return
    if args.command == "concurrent-update":
        _run_concurrent_update(
            tenant_id=args.tenant_id,
            problem_id=args.problem_id,
            worker_label=args.worker_label,
            start_path=args.start_path,
            update_path=args.update_path,
            read_snapshot_path=args.read_snapshot_path,
            done_path=args.done_path,
        )
        return
    if args.command == "read-final":
        _run_read_final(
            tenant_id=args.tenant_id,
            problem_id=args.problem_id,
            other_tenant_id=args.other_tenant_id,
        )
        return
    if args.command == "cleanup":
        _run_cleanup(tenant_ids=tuple(args.tenant_id))
        return
    _fail(f"unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
