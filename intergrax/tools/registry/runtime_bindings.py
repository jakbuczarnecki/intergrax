# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime-bound tool dependency protocols (avoid Tier-0 ↔ UAEP import cycles)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from intergrax.contracts.memory_write_policy import MemoryWritePolicy


@runtime_checkable
class RunTraceReaderBinding(Protocol):
    """Structural binding for persisted run trace reads (``RunTraceReader``)."""

    def read_run(self, run_id: str, tenant_id: str) -> Any: ...

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> List[Any]: ...


@runtime_checkable
class OnlineEvaluationRegistryBinding(Protocol):
    """Append-only harness evaluation registry (V-EVAL / W-OPS.11)."""

    def append(self, observation: Any) -> None: ...

    def list_observations(self) -> List[Any]: ...


@runtime_checkable
class TaskMemoryViewBinding(Protocol):
    """Structural binding for policy-scoped task memory (``PolicyScopedMemoryView``)."""

    async def read(self, namespace: str, key: str) -> Optional[Dict[str, Any]]: ...

    async def write(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
        *,
        policy: MemoryWritePolicy = MemoryWritePolicy.REPLACE,
    ) -> None: ...

    async def list(self, namespace: str, prefix: str = "") -> List[Any]: ...
