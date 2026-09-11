# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool executor termination boundary (W4-D) — thread pool, no native cancel."""

from __future__ import annotations

from concurrent.futures import Future

from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationResult,
)


TOOL_EXTERNAL_OPERATION_CAPABILITIES = ExternalOperationCapabilities(
    supports_native_cancel=False,
    supports_stream_abort=False,
    supports_remote_termination=False,
)


class ToolExecutorTerminationPort:
    """Best-effort Future.cancel before worker start; otherwise not supported."""

    __slots__ = ("_futures",)

    def __init__(self) -> None:
        self._futures: dict[str, Future[object]] = {}

    def bind_future(self, operation_id: str, future: Future[object]) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        self._futures[operation_id] = future

    def unbind_future(self, operation_id: str) -> None:
        self._futures.pop(operation_id, None)

    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        future = self._futures.pop(identity.operation_id, None)
        if future is None:
            return TerminationResult.not_supported()
        cancelled = future.cancel()
        if cancelled:
            return TerminationResult.physical_stop_confirmed()
        return TerminationResult.not_supported()
