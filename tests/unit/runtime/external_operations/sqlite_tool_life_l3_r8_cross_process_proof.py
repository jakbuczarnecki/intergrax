# © Artur Czarnecki. All rights reserved.

"""Local cross-process adversarial proofs for SQLite TOOL-LIFE-X L3-R8 (single-host)."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
    QualificationWorkerExecutionRevocationSupervision,
)
from tests.unit.runtime.external_operations.tool_life_x_test_commit_authority_loss import (
    issue_sqlite_commit_authority_loss_decision_for_bound_operation,
)

_BARRIER_TIMEOUT_SECONDS = 15.0
_RESULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class CrossProcessRaceResult:
    worker_pids: tuple[int, ...]
    exit_codes: tuple[int, ...]
    outcomes: tuple[str, ...]


def _configure_qualification_composition(bundle: object) -> None:
    from intergrax.contracts.certified_owner_loss_evidence import (
        CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
        CompositionCertifiedOwnerLossIssuerBinding,
    )
    from intergrax.runtime.external_operations.certified_owner_loss_composition import (
        CertifiedOwnerLossComposition,
        _replace_certified_owner_loss_composition,
    )
    from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
        WorkerExecutionOwnershipRevocationAuthorizer,
        WorkerExecutionOwnershipRevocationRecorder,
    )

    from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
        qualification_worker_execution_revocation_supervision,
    )

    supervision = qualification_worker_execution_revocation_supervision()
    authorizer = WorkerExecutionOwnershipRevocationAuthorizer(
        authoritative_state_store=bundle.external_operation_state,
        supervision=supervision,
    )
    recorder = WorkerExecutionOwnershipRevocationRecorder(
        authoritative_state_store=bundle.external_operation_state,
        revocation_port=bundle.worker_execution_ownership_revocation,
        authorizer=authorizer,
    )
    _replace_certified_owner_loss_composition(
        CertifiedOwnerLossComposition(
            issuer_binding=CompositionCertifiedOwnerLossIssuerBinding(
                allowed_issuer_ids=frozenset(
                    {CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID},
                ),
            ),
            worker_execution_ownership_revocation=bundle.worker_execution_ownership_revocation,
            worker_execution_revocation_supervision=supervision,
            worker_execution_revocation_authorizer=authorizer,
            worker_execution_revocation_recorder=recorder,
        ),
    )


def _worker_revoke_after_terminal(
    db_path: str,
    tenant_id: str,
    operation_id: str,
    barrier: mp.synchronize.Barrier,
    terminal_done: mp.synchronize.Event,
    result_queue: mp.Queue[tuple[str, ...]],
) -> None:
    from intergrax.contracts.worker_execution_ownership_revocation import (
        WorkerExecutionOwnershipRevocationUnauthorizedError,
    )
    from intergrax.runtime.external_operations.certified_owner_loss_composition import (
        require_certified_owner_loss_composition,
    )
    from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
        SqliteToolLifeDurableStoresBundle,
    )

    try:
        bundle = SqliteToolLifeDurableStoresBundle(db_path=Path(db_path))
        _configure_qualification_composition(bundle)
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        if not terminal_done.wait(timeout=_BARRIER_TIMEOUT_SECONDS):
            raise TimeoutError("terminal commit did not complete before revocation")
        recorder = require_certified_owner_loss_composition().worker_execution_revocation_recorder
        if recorder is None:
            raise RuntimeError("revocation recorder not configured in child process")
        issue_sqlite_commit_authority_loss_decision_for_bound_operation(
            bundle,
            tenant_id=tenant_id,
            operation_id=operation_id,
        )
        recorder.record_revocation_for_bound_operation(
            tenant_id=tenant_id,
            operation_id=operation_id,
        )
        result_queue.put(("revoked", str(os.getpid())))
    except WorkerExecutionOwnershipRevocationUnauthorizedError:
        result_queue.put(("unauthorized", str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))


def _worker_revoke_then_barrier(
    db_path: str,
    tenant_id: str,
    operation_id: str,
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue[tuple[str, ...]],
) -> None:
    from intergrax.runtime.external_operations.certified_owner_loss_composition import (
        require_certified_owner_loss_composition,
    )
    from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
        SqliteToolLifeDurableStoresBundle,
    )

    try:
        bundle = SqliteToolLifeDurableStoresBundle(db_path=Path(db_path))
        _configure_qualification_composition(bundle)
        recorder = require_certified_owner_loss_composition().worker_execution_revocation_recorder
        if recorder is None:
            raise RuntimeError("revocation recorder not configured in child process")
        issue_sqlite_commit_authority_loss_decision_for_bound_operation(
            bundle,
            tenant_id=tenant_id,
            operation_id=operation_id,
        )
        recorder.record_revocation_for_bound_operation(
            tenant_id=tenant_id,
            operation_id=operation_id,
        )
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        result_queue.put(("revoked", str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))


def _worker_terminal_then_barrier(
    db_path: str,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue[tuple[str, ...]],
    terminal_done: mp.synchronize.Event | None = None,
) -> None:
    from intergrax.contracts.certified_owner_loss_evidence import (
        CertifiedOwnerLossEvidenceRejectedError,
    )
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationPhysicalState,
    )
    from intergrax.runtime.external_operations.external_operation_ownership import (
        ExternalOperationFencedCommitRejectedError,
    )
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateUpdate,
        StaleExternalOperationStateError,
    )
    from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
        SqliteToolLifeDurableStoresBundle,
    )

    try:
        bundle = SqliteToolLifeDurableStoresBundle(db_path=Path(db_path))
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        bundle.external_operation_state.compare_and_set(
            tenant_id,
            operation_id,
            expected_revision=expected_revision,
            update=ExternalOperationStateUpdate(
                physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            ),
        )
        if terminal_done is not None:
            terminal_done.set()
        result_queue.put(("terminal", str(os.getpid())))
    except (
        StaleExternalOperationStateError,
        CertifiedOwnerLossEvidenceRejectedError,
        ExternalOperationFencedCommitRejectedError,
    ):
        result_queue.put(("fenced", str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))


def _collect_process_results(
    processes: list[mp.Process],
    result_queue: mp.Queue[tuple[str, ...]],
) -> CrossProcessRaceResult:
    for process in processes:
        process.join(timeout=_RESULT_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    outcomes: list[str] = []
    pids: list[int] = []
    deadline = time.monotonic() + _RESULT_TIMEOUT_SECONDS
    while time.monotonic() < deadline and len(outcomes) < len(processes):
        try:
            item = result_queue.get(timeout=0.25)
        except Empty:
            if all(not process.is_alive() for process in processes):
                break
            continue
        outcomes.append(item[0])
        pids.append(int(item[-1]))
    exit_codes = tuple(
        process.exitcode if process.exitcode is not None else -1 for process in processes
    )
    return CrossProcessRaceResult(
        worker_pids=tuple(pids),
        exit_codes=exit_codes,
        outcomes=tuple(outcomes),
    )


def run_revocation_wins_before_terminal_race(
    *,
    db_path: Path,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
) -> CrossProcessRaceResult:
    """Process A revokes (deterministic); process B attempts stale terminal CAS after barrier."""
    barrier = mp.Barrier(2)
    result_queue: mp.Queue[tuple[str, ...]] = mp.Queue()
    processes = [
        mp.Process(
            target=_worker_revoke_then_barrier,
            args=(str(db_path), tenant_id, operation_id, barrier, result_queue),
        ),
        mp.Process(
            target=_worker_terminal_then_barrier,
            args=(
                str(db_path),
                tenant_id,
                operation_id,
                expected_revision,
                barrier,
                result_queue,
            ),
        ),
    ]
    for process in processes:
        process.start()
    return _collect_process_results(processes, result_queue)


def run_terminal_wins_before_revocation_race(
    *,
    db_path: Path,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
) -> CrossProcessRaceResult:
    """Process A commits terminal first; process B revocation must fail (not RUNNING)."""
    barrier = mp.Barrier(2)
    terminal_done = mp.Event()
    result_queue: mp.Queue[tuple[str, ...]] = mp.Queue()
    processes = [
        mp.Process(
            target=_worker_terminal_then_barrier,
            args=(
                str(db_path),
                tenant_id,
                operation_id,
                expected_revision,
                barrier,
                result_queue,
                terminal_done,
            ),
        ),
        mp.Process(
            target=_worker_revoke_after_terminal,
            args=(
                str(db_path),
                tenant_id,
                operation_id,
                barrier,
                terminal_done,
                result_queue,
            ),
        ),
    ]
    for process in processes:
        process.start()
    return _collect_process_results(processes, result_queue)
