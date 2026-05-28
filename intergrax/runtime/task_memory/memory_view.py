# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Policy-scoped MemoryView gateway over TaskMemory (§42.35, Phase I.2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.task_memory.coordinator import TaskMemoryCoordinator
from intergrax.runtime.task_memory.limits import TaskMemoryLimits
from intergrax.runtime.task_memory.models import TaskMemoryRecord
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.contracts.memory_write_policy import MemoryWritePolicy
from intergrax.runtime.task_memory.policy import MemoryAccessPolicy, memory_access_policy_from_metadata


class MemoryViewError(RuntimeError):
    """Base error for MemoryView operations."""


class MemoryViewAccessDenied(MemoryViewError):
    """Raised when policy blocks a namespace or write."""


class PolicyScopedMemoryView:
    """
    UAEP-bound memory gateway — agents use ``ctx.memory_view``, never the store.

    Emits ``MEMORY_READ`` / ``MEMORY_WRITE`` runtime events on every operation.
    """

    def __init__(
        self,
        exec_ctx: RuntimeExecutionContext,
        store: TaskMemoryPersistence,
        *,
        tenant_id: str,
        task_id: str,
        access_policy: Optional[MemoryAccessPolicy] = None,
        limits: Optional[TaskMemoryLimits] = None,
    ) -> None:
        self._exec_ctx = exec_ctx
        self._store = store
        self._tenant_id = tenant_id
        self._task_id = task_id
        self._access_policy = access_policy or MemoryAccessPolicy()
        self._limits = limits or TaskMemoryLimits()

    async def read(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        self._guard_namespace(namespace, write=False)
        record = TaskMemoryCoordinator.read(
            self._store,
            tenant_id=self._tenant_id,
            task_id=self._task_id,
            namespace=namespace,
            key=key,
        )
        await self._emit(
            RuntimeEventType.MEMORY_READ,
            namespace=namespace,
            key=key,
            found=record is not None,
            record_id=record.record_id if record is not None else None,
        )
        return dict(record.value) if record is not None else None

    async def write(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
        *,
        policy: MemoryWritePolicy = MemoryWritePolicy.REPLACE,
    ) -> None:
        self._guard_namespace(namespace, write=True)
        resolved_value = dict(value)
        if policy == MemoryWritePolicy.MERGE:
            existing = TaskMemoryCoordinator.read(
                self._store,
                tenant_id=self._tenant_id,
                task_id=self._task_id,
                namespace=namespace,
                key=key,
            )
            if existing is not None:
                resolved_value = {**existing.value, **resolved_value}

        record = TaskMemoryCoordinator.write(
            self._store,
            tenant_id=self._tenant_id,
            task_id=self._task_id,
            namespace=namespace,
            key=key,
            value=resolved_value,
            provenance={
                "agent_id": self._exec_ctx.agent_id,
                "run_id": self._exec_ctx.run_id,
                "node_id": self._exec_ctx.node_id,
                "write_policy": policy.value,
            },
            limits=self._limits,
        )
        await self._emit(
            RuntimeEventType.MEMORY_WRITE,
            namespace=namespace,
            key=key,
            found=True,
            record_id=record.record_id,
            extra={"write_policy": policy.value},
        )

    async def list(self, namespace: str, prefix: str = "") -> List[TaskMemoryRecord]:
        self._guard_namespace(namespace, write=False)
        records = TaskMemoryCoordinator.list_namespace(
            self._store,
            tenant_id=self._tenant_id,
            task_id=self._task_id,
            namespace=namespace,
            prefix=prefix,
            limit=self._access_policy.list_limit,
        )
        await self._emit(
            RuntimeEventType.MEMORY_READ,
            namespace=namespace,
            key=prefix or "*",
            found=bool(records),
            extra={"operation": "list", "count": len(records), "prefix": prefix},
        )
        return records

    def _guard_namespace(self, namespace: str, *, write: bool) -> None:
        ns = (namespace or "").strip()
        if not ns:
            raise MemoryViewError("memory namespace must not be empty")
        if write and self._access_policy.read_only:
            raise MemoryViewAccessDenied("memory view is read-only")
        denied = self._access_policy.write_denied_namespaces
        if write and denied is not None and ns in denied:
            raise MemoryViewAccessDenied(f"namespace write denied: {ns}")
        allowed = self._access_policy.allowed_namespaces
        if allowed is not None and ns not in allowed:
            raise MemoryViewAccessDenied(f"namespace not allowed: {ns}")

    async def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        namespace: str,
        key: str,
        found: bool,
        record_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        emitter = self._exec_ctx.event_emitter
        if emitter is None:
            return
        payload: Dict[str, Any] = {
            "namespace": namespace.strip(),
            "key": key.strip(),
            "found": found,
        }
        if record_id is not None:
            payload["record_id"] = record_id
        if extra:
            payload.update(extra)
        await emitter.emit(
            RuntimeEvent(
                tenant_id=self._tenant_id,
                task_id=self._exec_ctx.task_id,
                run_id=self._exec_ctx.run_id,
                node_id=self._exec_ctx.node_id,
                agent_id=self._exec_ctx.agent_id,
                event_type=event_type,
                phase=self._exec_ctx.phase,
                payload=payload,
                correlation_id=self._exec_ctx.correlation_id or self._exec_ctx.task_id,
            )
        )
