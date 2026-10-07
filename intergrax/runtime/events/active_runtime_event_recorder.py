# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active RuntimeEvent recorder port for adapter-layer factual evidence (TRACE-X-P4)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort

_active_runtime_event_recorder: ContextVar[RuntimeEventRecorderPort | None] = ContextVar(
    "active_runtime_event_recorder",
    default=None,
)
_active_runtime_event_tenant_id: ContextVar[str] = ContextVar(
    "active_runtime_event_tenant_id",
    default="",
)


@dataclass(frozen=True, slots=True)
class ActiveRuntimeEventRecorderBinding:
    recorder_token: Token[RuntimeEventRecorderPort | None]
    tenant_token: Token[str]


def bind_active_runtime_event_recorder(
    recorder: RuntimeEventRecorderPort | None,
    *,
    tenant_id: str | None = None,
) -> ActiveRuntimeEventRecorderBinding:
    tenant_token = _active_runtime_event_tenant_id.set((tenant_id or "").strip())
    recorder_token = _active_runtime_event_recorder.set(recorder)
    return ActiveRuntimeEventRecorderBinding(
        recorder_token=recorder_token,
        tenant_token=tenant_token,
    )


def reset_active_runtime_event_recorder(binding: ActiveRuntimeEventRecorderBinding) -> None:
    _active_runtime_event_recorder.reset(binding.recorder_token)
    _active_runtime_event_tenant_id.reset(binding.tenant_token)


def peek_active_runtime_event_recorder() -> RuntimeEventRecorderPort | None:
    return _active_runtime_event_recorder.get()


def peek_active_runtime_event_tenant_id() -> str:
    return _active_runtime_event_tenant_id.get()


__all__ = [
    "ActiveRuntimeEventRecorderBinding",
    "bind_active_runtime_event_recorder",
    "peek_active_runtime_event_recorder",
    "peek_active_runtime_event_tenant_id",
    "reset_active_runtime_event_recorder",
]
