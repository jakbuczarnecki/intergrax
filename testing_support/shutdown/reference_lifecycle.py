# © Artur Czarnecki. All rights reserved.

"""Reference Execution Engine shutdown lifecycle for EE-B4-B (not production authority)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityExceededError,
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPermit,
    ExecutionCapacityPolicy,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_reliability import (
    EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER,
    ExecutionRuntimeShutdownPhase,
)
from intergrax.runtime.execution.local_execution_capacity_admission import (
    LocalExecutionCapacityAdmission,
)
from testing_support.shutdown.models import (
    ReferenceRootAdmissionDecision,
    ReferenceShutdownFailureKind,
    ReferenceShutdownTerminalOutcome,
    _ShutdownState,
)
from testing_support.shutdown.ports import (
    InMemoryFinalStateStore,
    RecordingMandatoryEvidenceFlush,
    RecordingObservabilityExporter,
)


@dataclass(frozen=True, slots=True)
class ReferenceRootExecutionHandle:
    execution_id: ExecutionId
    permit: ExecutionCapacityPermit
    worker_task: asyncio.Task[None]


@dataclass
class ReferenceExecutionShutdownLifecycle:
    """Composes capacity, worker drain, evidence flush, and termination for certification."""

    capacity_policy: ExecutionCapacityPolicy = field(
        default_factory=lambda: ExecutionCapacityPolicy(
            max_concurrent_root_executions=4,
            overload_mode=ExecutionCapacityOverloadMode.REJECT,
        )
    )
    drain_timeout_seconds: float = 30.0
    mandatory_evidence: RecordingMandatoryEvidenceFlush = field(
        default_factory=RecordingMandatoryEvidenceFlush
    )
    final_state: InMemoryFinalStateStore = field(
        default_factory=InMemoryFinalStateStore
    )
    observability_exporter: RecordingObservabilityExporter = field(
        default_factory=RecordingObservabilityExporter
    )
    stop_boundary_event: asyncio.Event = field(default_factory=asyncio.Event)
    _drain_idle: asyncio.Event = field(default_factory=asyncio.Event)

    _admission: LocalExecutionCapacityAdmission = field(init=False)
    _state: _ShutdownState = field(default_factory=_ShutdownState)
    _shutdown_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _active_roots: dict[str, ReferenceRootExecutionHandle] = field(default_factory=dict)
    _managed_workers: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _held_permits: int = 0
    _admitted_parent_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._admission = LocalExecutionCapacityAdmission(self.capacity_policy)
        self._drain_idle.set()

    @property
    def shutdown_phase(self) -> ExecutionRuntimeShutdownPhase | None:
        return self._state.phase

    @property
    def accepting_new_root_work(self) -> bool:
        return (
            not self._state.terminated
            and not self._state.stop_boundary_reached
            and self._state.phase is None
        )

    @property
    def held_root_permits(self) -> int:
        return self._held_permits

    @property
    def managed_worker_count(self) -> int:
        return sum(1 for task in self._managed_workers.values() if not task.done())

    @property
    def managed_task_count(self) -> int:
        return len(self._active_roots)

    def operational_shutdown_phase(self) -> ExecutionRuntimeShutdownPhase | None:
        return self._state.phase

    async def try_admit_root(
        self,
        *,
        tenant_id: str = "cert",
        execution_id: ExecutionId | None = None,
    ) -> tuple[ReferenceRootAdmissionDecision, ExecutionCapacityPermit | None]:
        if self._state.terminated:
            return ReferenceRootAdmissionDecision.REJECTED_TERMINATED, None
        if self._state.stop_boundary_reached:
            self._state.new_roots_after_stop += 1
            return ReferenceRootAdmissionDecision.REJECTED_SHUTDOWN, None
        exec_id = execution_id if execution_id is not None else mint_execution_id()
        request = ExecutionCapacityAdmissionRequest(
            tenant_id=tenant_id,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=exec_id,
        )
        try:
            permit = await self._admission.acquire(request)
        except ExecutionCapacityExceededError:
            return ReferenceRootAdmissionDecision.REJECTED_CAPACITY, None
        self._held_permits += 1
        self._admitted_parent_ids.add(exec_id)
        return ReferenceRootAdmissionDecision.ADMITTED, permit

    def try_admit_child_continuation(
        self,
        parent_execution_id: ExecutionId,
    ) -> ReferenceRootAdmissionDecision:
        if self._state.terminated:
            return ReferenceRootAdmissionDecision.REJECTED_TERMINATED
        if parent_execution_id not in self._admitted_parent_ids:
            return ReferenceRootAdmissionDecision.REJECTED_SHUTDOWN
        if parent_execution_id not in self._active_roots:
            return ReferenceRootAdmissionDecision.REJECTED_SHUTDOWN
        return ReferenceRootAdmissionDecision.ADMITTED

    async def run_root_work(
        self,
        *,
        execution_id: ExecutionId,
        permit: ExecutionCapacityPermit,
        work_factory: Callable[[], Awaitable[None]],
    ) -> ReferenceRootExecutionHandle:
        async def _worker() -> None:
            try:
                await work_factory()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._state.worker_fault_during_drain = True
                raise
            finally:
                await self._release_permit(permit)
                self._active_roots.pop(execution_id, None)
                self._admitted_parent_ids.discard(execution_id)
                if not self._active_roots:
                    self._drain_idle.set()

        task = asyncio.create_task(_worker(), name=f"ee-b4-b-worker-{execution_id}")
        self._managed_workers[execution_id] = task
        handle = ReferenceRootExecutionHandle(
            execution_id=execution_id,
            permit=permit,
            worker_task=task,
        )
        self._active_roots[execution_id] = handle
        self._drain_idle.clear()
        return handle

    async def release_root_permit(self, permit: ExecutionCapacityPermit) -> None:
        await self._release_permit(permit)

    async def _release_permit(self, permit: ExecutionCapacityPermit) -> None:
        if self._held_permits <= 0:
            self._state.double_release_attempts += 1
            await permit.release()
            return
        await permit.release()
        self._held_permits -= 1

    async def shutdown(
        self,
        *,
        cancel_stuck_on_drain_timeout: bool = True,
    ) -> ReferenceShutdownTerminalOutcome:
        async with self._shutdown_lock:
            if self._state.outcome is not None:
                return self._state.outcome
            return await self._run_shutdown_once(
                cancel_stuck_on_drain_timeout=cancel_stuck_on_drain_timeout
            )

    async def _run_shutdown_once(
        self,
        *,
        cancel_stuck_on_drain_timeout: bool,
    ) -> ReferenceShutdownTerminalOutcome:
        primary: ReferenceShutdownFailureKind | None = None
        secondary: list[ReferenceShutdownFailureKind] = []

        for phase in EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER:
            self._state.phase = phase
            self._state.phase_trail.append(phase)
            if phase is ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK:
                self._state.stop_boundary_reached = True
                self.stop_boundary_event.set()
            elif phase is ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS:
                worker_fault = await self._drain_active_executions(
                    cancel_stuck_on_drain_timeout=cancel_stuck_on_drain_timeout
                )
                if self._state.worker_fault_during_drain and primary is None:
                    primary = ReferenceShutdownFailureKind.WORKER
                elif worker_fault is not None and primary is None:
                    primary = worker_fault
            elif phase is ExecutionRuntimeShutdownPhase.FLUSH_REQUIRED_EVIDENCE:
                try:
                    await self.mandatory_evidence.flush_required_evidence()
                except Exception:
                    if primary is None:
                        primary = ReferenceShutdownFailureKind.MANDATORY_EVIDENCE
                    else:
                        secondary.append(
                            ReferenceShutdownFailureKind.MANDATORY_EVIDENCE
                        )
            elif phase is ExecutionRuntimeShutdownPhase.PERSIST_FINAL_STATE:
                try:
                    await self.final_state.persist_final_state(
                        {
                            "phase": phase.value,
                            "active_roots": str(len(self._active_roots)),
                        }
                    )
                except Exception:
                    if primary is None:
                        primary = ReferenceShutdownFailureKind.FINAL_STATE
                    else:
                        secondary.append(ReferenceShutdownFailureKind.FINAL_STATE)
            elif phase is ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS:
                await self._terminate_workers()
                try:
                    await self.observability_exporter.close_export()
                except Exception:
                    secondary.append(ReferenceShutdownFailureKind.OBSERVABILITY_EXPORT)

        self._state.terminated = True
        self._state.phase = ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS
        clean = (
            primary is None
            and self.mandatory_evidence.flushed
            and self.final_state.persisted
            and self._held_permits == 0
            and self.managed_worker_count == 0
        )
        outcome = ReferenceShutdownTerminalOutcome(
            clean_success=clean,
            primary_failure_kind=primary,
            secondary_failure_kinds=tuple(secondary),
            phase_trail=tuple(self._state.phase_trail),
            held_root_permits=self._held_permits,
            managed_worker_count=self.managed_worker_count,
            managed_task_count=self.managed_task_count,
            double_release_attempts=self._state.double_release_attempts,
            new_roots_after_stop=self._state.new_roots_after_stop,
        )
        self._state.outcome = outcome
        return outcome

    async def _drain_active_executions(
        self,
        *,
        cancel_stuck_on_drain_timeout: bool,
    ) -> ReferenceShutdownFailureKind | None:
        if not self._active_roots:
            return None

        worker_fault: ReferenceShutdownFailureKind | None = None
        try:
            await asyncio.wait_for(
                self._drain_idle.wait(),
                timeout=self.drain_timeout_seconds,
            )
        except TimeoutError:
            if cancel_stuck_on_drain_timeout:
                for handle in list(self._active_roots.values()):
                    handle.worker_task.cancel()
                for handle in list(self._active_roots.values()):
                    try:
                        await handle.worker_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        worker_fault = ReferenceShutdownFailureKind.WORKER
            else:
                return ReferenceShutdownFailureKind.DRAIN_TIMEOUT
        return worker_fault

    async def _terminate_workers(self) -> None:
        for task in list(self._managed_workers.values()):
            if not task.done():
                task.cancel()
        for task in list(self._managed_workers.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._managed_workers.clear()
        self._active_roots.clear()
