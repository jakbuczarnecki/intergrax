# © Artur Czarnecki. All rights reserved.

"""Cross-process PostgreSQL CAS proof helpers for Collaborative Work MP-2."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.collaborative_work.persistence import open_postgresql_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    UpdateWorkItemCommand,
    WorkItemRevisionConflict,
    WorkItemScopeKey,
)
from intergrax.contracts.collaborative_work import WorkItemState
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

_WORKER_JOIN_TIMEOUT_SECONDS = 30.0
_BARRIER_TIMEOUT_SECONDS = 10.0
_RESULT_TIMEOUT_SECONDS = 30.0


class CrossProcessCasProofFailure(Exception):
    """Cross-process CAS proof rejected by semantic invariants."""


@dataclass(frozen=True, slots=True)
class CrossProcessCasProofResult:
    """Observed cross-process CAS race outcome."""

    worker_pids: tuple[int, ...]
    successes: int
    conflicts: int
    final_revision: int


def _cross_process_work_item_cas_worker(
    dsn: str,
    schema_name: str,
    tenant_id: str,
    workspace_id: str,
    work_item_id: str,
    expected_revision: int,
    updated_at_iso: str,
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue[tuple[str, ...]],
) -> None:
    bundle = None
    try:
        config = PostgreSQLIntegrationConfig(dsn=dsn)
        bundle = open_postgresql_collaborative_work_repositories(
            config=config,
            schema_name=schema_name,
        )
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        bundle.work_item.update(
            UpdateWorkItemCommand(
                scope=WorkItemScopeKey(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    work_item_id=work_item_id,
                ),
                expected_revision=expected_revision,
                state=WorkItemState.ACTIVE,
                updated_at=datetime.fromisoformat(updated_at_iso),
            ),
        )
        result_queue.put(("updated", str(os.getpid())))
    except WorkItemRevisionConflict:
        result_queue.put(("conflict", str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))
    finally:
        if bundle is not None:
            bundle.close()


def run_postgresql_work_item_cross_process_cas_proof(
    *,
    config: PostgreSQLIntegrationConfig,
    schema_name: str,
    tenant_id: str,
    workspace_id: str,
    work_item_id: str,
    expected_revision: int,
    updated_at: datetime,
) -> CrossProcessCasProofResult:
    """
    Prove PostgreSQL CAS across two distinct OS processes.

    Each worker independently materializes repositories, reads the same revision
    baseline, and races on ``update(expected_revision=N)``.
    """
    dsn = config.connection_string()
    if not dsn:
        raise CrossProcessCasProofFailure("PostgreSQL connection string is required")

    bundle_a = open_postgresql_collaborative_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        read_a = bundle_a.work_item.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
        )
        read_b = bundle_b.work_item.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
        )
        if read_a is None or read_b is None:
            raise CrossProcessCasProofFailure("cross-process pre-read missing work item")
        if not (
            read_a.revision
            == read_b.revision
            == expected_revision
        ):
            raise CrossProcessCasProofFailure("cross-process revision baseline mismatch")
    finally:
        bundle_a.close()
        bundle_b.close()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2, timeout=_BARRIER_TIMEOUT_SECONDS)
    result_queue: mp.Queue[tuple[str, ...]] = ctx.Queue()
    worker_args = (
        dsn,
        schema_name,
        tenant_id,
        workspace_id,
        work_item_id,
        expected_revision,
        updated_at.astimezone(UTC).isoformat(),
        barrier,
        result_queue,
    )
    processes = [
        ctx.Process(target=_cross_process_work_item_cas_worker, args=worker_args),
        ctx.Process(target=_cross_process_work_item_cas_worker, args=worker_args),
    ]
    outcomes: list[tuple[str, ...]] = []
    try:
        for process in processes:
            process.start()
        for _ in processes:
            outcomes.append(result_queue.get(timeout=_RESULT_TIMEOUT_SECONDS))
        for process in processes:
            process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
            if process.exitcode != 0:
                raise CrossProcessCasProofFailure(
                    f"cross-process worker exited with code {process.exitcode}",
                )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()

    if len(outcomes) != 2:
        raise CrossProcessCasProofFailure("expected exactly two worker outcomes")

    worker_pids: list[int] = []
    successes = 0
    conflicts = 0
    for outcome in outcomes:
        status = outcome[0]
        if status == "updated":
            successes += 1
            worker_pids.append(int(outcome[1]))
            continue
        if status == "conflict":
            conflicts += 1
            worker_pids.append(int(outcome[1]))
            continue
        raise CrossProcessCasProofFailure(
            f"unexpected worker outcome: {outcome}",
        )

    if successes != 1 or conflicts != 1:
        raise CrossProcessCasProofFailure(
            f"expected one success and one conflict, got successes={successes} conflicts={conflicts}",
        )
    if len(set(worker_pids)) != 2:
        raise CrossProcessCasProofFailure("expected distinct worker PIDs")

    verify_bundle = open_postgresql_collaborative_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        final = verify_bundle.work_item.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
        )
        if final is None:
            raise CrossProcessCasProofFailure("final work item missing after cross-process race")
        if final.revision != expected_revision + 1:
            raise CrossProcessCasProofFailure("cross-process final revision mismatch")
        final_revision = final.revision
    finally:
        verify_bundle.close()

    return CrossProcessCasProofResult(
        worker_pids=tuple(worker_pids),
        successes=successes,
        conflicts=conflicts,
        final_revision=final_revision,
    )
