# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution Engine qualified capability dispatch with request-id idempotency (UCA-6C-R)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchPort,
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)


@dataclass(frozen=True, slots=True)
class _DispatchLedgerEntry:
    result: QualifiedCapabilityExecutionDispatchResult


class QualifiedCapabilityExecutionDispatchService(
    QualifiedCapabilityExecutionDispatchPort
):
    """Process-local exactly-once dispatch ledger keyed by tenant + execution_request_id."""

    def __init__(
        self,
        *,
        handler_registry: QualifiedCapabilityExecutionBindingHandlerRegistry,
    ) -> None:
        self._handlers = handler_registry
        self._ledger: dict[tuple[str, str], _DispatchLedgerEntry] = {}
        self._lock = threading.RLock()
        self.dispatch_side_effects = 0

    def dispatch(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        ledger_key = (request.tenant_id, request.execution_request_id)
        with self._lock:
            existing = self._ledger.get(ledger_key)
            if existing is not None:
                return existing.result

            handler = self._handlers.resolve(
                request.execution_target.binding_provider_id
            )
            if handler is None:
                result = QualifiedCapabilityExecutionDispatchResult(
                    disposition=QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE,
                    reason_detail="execution_handler_unavailable",
                )
                self._ledger[ledger_key] = _DispatchLedgerEntry(result=result)
                return result

            self.dispatch_side_effects += 1
            result = handler.dispatch_once(request)
            if (
                result.disposition
                is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
                and result.execution_request_id != request.execution_request_id
            ):
                result = QualifiedCapabilityExecutionDispatchResult(
                    disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                    execution_request_id=request.execution_request_id,
                    reason_detail="execution_request_id_integrity_mismatch",
                )

            if (
                result.disposition
                is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
                and result.execution_request_id is None
            ):
                result = QualifiedCapabilityExecutionDispatchResult(
                    disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                    execution_request_id=request.execution_request_id,
                    reason_detail="execution_request_id_missing",
                )

            self._ledger[ledger_key] = _DispatchLedgerEntry(result=result)
            return result


__all__ = ["QualifiedCapabilityExecutionDispatchService"]
