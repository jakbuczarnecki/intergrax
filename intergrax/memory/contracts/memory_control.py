# © Artur Czarnecki. All rights reserved.

"""Canonical Memory Control Plane contract (MEM-ENT-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleOutcome,
    MemoryReconciliationOutcome,
)
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordTrust,
)
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.contracts.memory_recall import (
    MemoryRankingScore,
    MemoryRecallReasonCode,
    MemorySupersessionIntent,
)

__all__ = [
    "EpisodicMemoryCapability",
    "MemoryControlAccessDenied",
    "MemoryControlBackendError",
    "MemoryControlForgetRequest",
    "MemoryControlForgetResult",
    "MemoryControlNotFound",
    "MemoryControlPartialLifecycleError",
    "MemoryControlPlane",
    "MemoryControlPlaneScope",
    "MemoryControlRecallItem",
    "MemoryControlRecallRequest",
    "MemoryControlRecallResult",
    "MemoryControlReconcileRequest",
    "MemoryControlReconcileResult",
    "MemoryControlRememberRequest",
    "MemoryControlRememberResult",
    "MemoryControlScopeRef",
    "MemoryControlUnsupportedScope",
    "TaskMemoryCapability",
    "UserMemoryForgetCapabilityResult",
    "UserMemoryRecallCapabilityResult",
    "UserMemoryRememberCapabilityResult",
    "UserProfileMemoryCapability",
    "MemoryControlSupersessionApplyResult",
    "user_memory_scope",
]


class MemoryControlPlaneScope(str, Enum):
    """Logical memory surface routed by the control plane."""

    TASK = "task"
    USER = "user"
    SESSION = "session"


class MemoryControlAccessDenied(PermissionError):
    """Canonical identity does not authorize the requested memory scope."""


class MemoryControlUnsupportedScope(LookupError):
    """No capability is configured for the requested scope."""


class MemoryControlNotFound(LookupError):
    """Requested memory entity does not exist or is not recallable."""


class MemoryControlBackendError(RuntimeError):
    """Underlying memory capability failed."""


class MemoryControlPartialLifecycleError(RuntimeError):
    """Primary mutation applied but projection lifecycle did not complete."""

    def __init__(
        self,
        lifecycle: MemoryLifecycleOutcome,
        *,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            f"memory control partial lifecycle: operation={lifecycle.operation.value}, "
            f"user_id={lifecycle.user_id}"
        )
        self.lifecycle = lifecycle
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class UserMemoryRememberCapabilityResult:
    entry: UserProfileMemoryEntry
    lifecycle: MemoryLifecycleOutcome


@dataclass(frozen=True, slots=True)
class UserMemoryForgetCapabilityResult:
    entry_id: str
    lifecycle: MemoryLifecycleOutcome


@dataclass(frozen=True, slots=True)
class UserMemoryRecallCapabilityResult:
    entries: tuple[UserProfileMemoryEntry, ...]
    scores: tuple[float | None, ...]
    used_semantic: bool
    reason: str


@dataclass(frozen=True, slots=True)
class MemoryControlScopeRef:
    """Typed scope bound to canonical tenant; user/session fields are validated against identity."""

    kind: MemoryControlPlaneScope
    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None
    task_namespace: str | None = None
    task_key: str | None = None


def user_memory_scope(identity: RequestIdentity) -> MemoryControlScopeRef:
    """Build USER scope from canonical identity (no caller-supplied user authority)."""
    user_id = (identity.user_id or "").strip()
    if not user_id:
        raise MemoryControlAccessDenied("user memory scope requires canonical user_id")
    return MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=identity.tenant_id,
        user_id=user_id,
    )


@dataclass(frozen=True, slots=True)
class MemoryControlRememberRequest:
    content: str = ""
    entry: UserProfileMemoryEntry | None = None
    kind: MemoryKind = MemoryKind.OTHER
    title: str | None = None
    task_value_json: str | None = None
    provenance: MemoryProvenance | None = None
    trust: MemoryRecordTrust | None = None
    governance: MemoryRecordGovernance | None = None


@dataclass(frozen=True, slots=True)
class MemoryControlRecallRequest:
    query: str = ""
    top_k: int = 6
    score_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryControlForgetRequest:
    entry_id: str = ""
    task_namespace: str | None = None
    task_key: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryControlReconcileRequest:
    """USER-scope reconciliation uses canonical user from scope ref."""


@dataclass(frozen=True, slots=True)
class MemoryControlRecallItem:
    entry_id: str
    content: str
    kind: MemoryKind
    score: float | None = None
    score_breakdown: MemoryRankingScore | None = None
    reason_codes: tuple[MemoryRecallReasonCode, ...] = ()
    conflict_unresolved: bool = False


@dataclass(frozen=True, slots=True)
class MemoryControlSupersessionApplyResult:
    scope: MemoryControlPlaneScope
    superseded_memory_id: str
    superseding_memory_id: str
    lifecycle: MemoryLifecycleOutcome


@dataclass(frozen=True, slots=True)
class MemoryControlRememberResult:
    scope: MemoryControlPlaneScope
    entry_id: str | None = None
    lifecycle: MemoryLifecycleOutcome | None = None
    task_written: bool = False


@dataclass(frozen=True, slots=True)
class MemoryControlRecallResult:
    scope: MemoryControlPlaneScope
    items: tuple[MemoryControlRecallItem, ...]
    used_semantic: bool = False
    reason: str = "ok"


@dataclass(frozen=True, slots=True)
class MemoryControlForgetResult:
    scope: MemoryControlPlaneScope
    entry_id: str | None = None
    lifecycle: MemoryLifecycleOutcome | None = None
    task_deleted: bool = False


@dataclass(frozen=True, slots=True)
class MemoryControlReconcileResult:
    scope: MemoryControlPlaneScope
    reconciliation: MemoryReconciliationOutcome | None = None


@runtime_checkable
class UserProfileMemoryCapability(Protocol):
    """Narrow user LTM surface for control-plane routing."""

    async def add_memory_entry(
        self,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> UserMemoryRememberCapabilityResult: ...

    async def remove_memory_entry(
        self,
        user_id: str,
        entry_id: str,
    ) -> UserMemoryForgetCapabilityResult: ...

    async def list_active_memory_entries(
        self,
        user_id: str,
    ) -> tuple[UserProfileMemoryEntry, ...]: ...

    def is_longterm_rag_enabled(self) -> bool: ...

    async def search_longterm_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> UserMemoryRecallCapabilityResult: ...

    async def reconcile_memory_projections(
        self,
        user_id: str,
    ) -> MemoryReconciliationOutcome: ...

    async def apply_memory_supersession(
        self,
        user_id: str,
        intent: MemorySupersessionIntent,
    ) -> MemoryControlSupersessionApplyResult: ...


@runtime_checkable
class TaskMemoryCapability(Protocol):
    async def write(
        self,
        namespace: str,
        key: str,
        value: dict[str, object],
    ) -> None: ...

    async def read(self, namespace: str, key: str) -> dict[str, object] | None: ...

    async def delete(self, namespace: str, key: str) -> bool: ...


@runtime_checkable
class EpisodicMemoryCapability(Protocol):
    """Placeholder episodic/session recall capability (MEM-ENT-3 minimal)."""

    async def recall_session_turns(
        self,
        *,
        tenant_id: str,
        session_id: str,
        query: str,
        top_k: int,
    ) -> tuple[MemoryControlRecallItem, ...]: ...


@runtime_checkable
class MemoryControlPlane(Protocol):
    """Single high-level memory boundary for platform callers."""

    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult: ...

    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult: ...

    async def forget(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlForgetRequest,
    ) -> MemoryControlForgetResult: ...

    async def apply_memory_supersession(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        intent: MemorySupersessionIntent,
    ) -> MemoryControlSupersessionApplyResult: ...

    async def reconcile(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlReconcileRequest,
    ) -> MemoryControlReconcileResult: ...
