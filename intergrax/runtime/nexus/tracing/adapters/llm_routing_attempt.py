# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Failover routing attempt diagnostics (M-LLM-X.4.4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

from intergrax.llm_adapters.registry.failover_adapter import (
    FailoverLLMAdapter,
    LLMRoutingAttemptRecord,
)
from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceLevel,
)

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@dataclass(frozen=True)
class LLMRoutingAttemptDiagV1(DiagnosticPayload):
    """PII-safe record of a retriable LLM profile failover attempt."""

    profile_id: str
    provider: str
    model: str
    error: str
    profile_index: int

    def redact(self) -> LLMRoutingAttemptDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.engine.core_llm.routing_attempt"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
            "profile_index": self.profile_index,
        }


def routing_attempt_to_diag(record: LLMRoutingAttemptRecord) -> LLMRoutingAttemptDiagV1:
    return LLMRoutingAttemptDiagV1(
        profile_id=record.profile_id,
        provider=record.provider,
        model=record.model,
        error=record.error,
        profile_index=record.profile_index,
    )


TraceEmitFn = Callable[..., None]


def attach_failover_routing_trace_observer(
    adapter: LLMAdapter,
    trace_event: TraceEmitFn,
) -> None:
    """
    Wire Tier-1 trace emission for ``FailoverLLMAdapter`` routing attempts.

    Idempotent when called multiple times on the same adapter instance.
    """
    if not isinstance(adapter, FailoverLLMAdapter):
        return
    if adapter.routing_attempt_observer is not None:
        return

    def _on_attempt(record: LLMRoutingAttemptRecord) -> None:
        payload = routing_attempt_to_diag(record)
        trace_event(
            component=TraceComponent.ENGINE,
            step="llm_routing_attempt",
            message="LLM profile failover attempt recorded.",
            level=TraceLevel.WARNING,
            payload=payload,
        )

    adapter.routing_attempt_observer = _on_attempt
