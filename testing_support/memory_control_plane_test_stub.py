# © Artur Czarnecki. All rights reserved.

"""Protocol-complete MemoryControlPlane stubs for unit tests (runtime_checkable)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlForgetRequest,
    MemoryControlForgetResult,
    MemoryControlPlaneScope,
    MemoryControlReconcileRequest,
    MemoryControlReconcileResult,
    MemoryControlRecallRequest,
    MemoryControlRecallResult,
    MemoryControlRememberRequest,
    MemoryControlRememberResult,
    MemoryControlScopeRef,
    MemoryControlSupersessionApplyResult,
    MemorySupersessionIntent,
)


class MemoryControlPlaneTestStub:
    """Minimal plane surface so ``isinstance(..., MemoryControlPlane)`` succeeds in tests."""

    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        raise NotImplementedError

    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        raise NotImplementedError

    async def forget(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlForgetRequest,
    ) -> MemoryControlForgetResult:
        return MemoryControlForgetResult(scope=MemoryControlPlaneScope.USER)

    async def apply_memory_supersession(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        intent: MemorySupersessionIntent,
    ) -> MemoryControlSupersessionApplyResult:
        raise NotImplementedError

    async def reconcile(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlReconcileRequest,
    ) -> MemoryControlReconcileResult:
        return MemoryControlReconcileResult(scope=MemoryControlPlaneScope.USER)
