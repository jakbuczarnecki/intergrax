# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Local cross-process adversarial proofs for SQLite TOOL-LIFE-X L3-R8 (single-host)."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

_BARRIER_TIMEOUT_SECONDS = 15.0
_RESULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class CrossProcessRaceResult:
    worker_pids: tuple[int, ...]
    outcomes: tuple[str, ...]


def _worker_revoke_after_terminal(
    db_path: str,
    tenant_id: str,
    operation_id: str,
    barrier: mp.synchronize.Barrier,
    terminal_done: mp.synchronize.Event,
    result_queue: mp.Queue[tuple[str, ...]],
) -> None:
    from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
        SqliteToolLifeDurableStoresBundle,
    )
    from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
        _sanctioned_revocation_write,
    )

    try:
        bundle = SqliteToolLifeDurableStoresBundle(db_path=Path(db_path))
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        if not terminal_done.wait(timeout=_BARRIER_TIMEOUT_SECONDS):
            raise TimeoutError("terminal commit did not complete before revocation")
        current = bundle.external_operation_state.load(tenant_id, operation_id)
        if current is None or current.worker_instance_id is None:
            raise RuntimeError("bound worker execution record missing")
        with _sanctioned_revocation_write():
            bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=current.worker_instance_id,
                ownership_epoch=current.worker_ownership_epoch or 0,
            )
        result_queue.put(("revoked", str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))


def _worker_revoke_then_barrier(
    db_path: str,
    tenant_id: str,
    operation_id: str,
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue[tuple[str, ...]],
) -> None:
    from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
        SqliteToolLifeDurableStoresBundle,
    )
    from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
        _sanctioned_revocation_write,
    )

    try:
        bundle = SqliteToolLifeDurableStoresBundle(db_path=Path(db_path))
        current = bundle.external_operation_state.load(tenant_id, operation_id)
        if current is None or current.worker_instance_id is None:
            raise RuntimeError("bound worker execution record missing")
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        with _sanctioned_revocation_write():
            bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=current.worker_instance_id,
                ownership_epoch=current.worker_ownership_epoch or 0,
            )
        result_queue.put(("revoked", str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))


def _worker_terminal_then_barrier(
    db_path: str,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    owner_token: str,
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue[tuple[str, ...]],
    terminal_done: mp.synchronize.Event | None = None,
) -> None:
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationPhysicalState,
    )
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateUpdate,
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
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("fenced", type(exc).__name__, str(exc), str(os.getpid())))


def run_revocation_wins_before_terminal_race(
    *,
    db_path: Path,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    owner_token: str,
) -> CrossProcessRaceResult:
    """Process A revokes; process B attempts stale terminal CAS after barrier."""
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
                owner_token,
                barrier,
                result_queue,
            ),
        ),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=_RESULT_TIMEOUT_SECONDS)
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
    return CrossProcessRaceResult(worker_pids=tuple(pids), outcomes=tuple(outcomes))


def run_terminal_wins_before_revocation_race(
    *,
    db_path: Path,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    owner_token: str,
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
                owner_token,
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
    for process in processes:
        process.join(timeout=_RESULT_TIMEOUT_SECONDS)
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
    return CrossProcessRaceResult(worker_pids=tuple(pids), outcomes=tuple(outcomes))


__all__ = [
    "CrossProcessRaceResult",
    "run_revocation_wins_before_terminal_race",
    "run_terminal_wins_before_revocation_race",
]
