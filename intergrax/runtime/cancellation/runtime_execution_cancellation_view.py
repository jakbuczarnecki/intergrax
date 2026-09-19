# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime adapter: cooperative cancellation for ``RuntimeExecutionContext``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext


class RuntimeExecutionContextCancellationView:
    """Read-only cancellation view over request + execution metadata."""

    __slots__ = ("_ctx",)

    def __init__(self, ctx: RuntimeExecutionContext) -> None:
        self._ctx = ctx

    def is_cancelled(self) -> bool:
        from intergrax.runtime.cancellation.coordinator import CancellationCoordinator

        request = self._ctx.request
        if request is not None:
            metadata: dict[str, Any] = dict(request.metadata)
            if CancellationCoordinator.is_requested(metadata):
                return True
        return CancellationCoordinator.is_requested(dict(self._ctx.metadata))

    def cancellation_reason(self) -> str | None:
        from intergrax.runtime.cancellation.coordinator import (
            CANCELLATION_REASON_KEY,
            CancellationCoordinator,
        )

        request = self._ctx.request
        if request is not None:
            metadata = dict(request.metadata)
            if CancellationCoordinator.is_requested(metadata):
                reason = metadata.get(CANCELLATION_REASON_KEY)
                return reason if isinstance(reason, str) and reason else None
        reason = self._ctx.metadata.get(CANCELLATION_REASON_KEY)
        return reason if isinstance(reason, str) and reason else None


def attach_runtime_execution_cancellation_view(
    exec_ctx: RuntimeExecutionContext,
) -> None:
    exec_ctx.cancellation = RuntimeExecutionContextCancellationView(exec_ctx)
