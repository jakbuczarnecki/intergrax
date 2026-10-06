# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active RuntimeEvent recorder port for adapter-layer factual evidence (TRACE-X-P4)."""

from __future__ import annotations

from contextvars import ContextVar, Token

from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort

_active_runtime_event_recorder: ContextVar[RuntimeEventRecorderPort | None] = ContextVar(
    "active_runtime_event_recorder",
    default=None,
)
_active_runtime_event_tenant_id: ContextVar[str] = ContextVar(
    "active_runtime_event_tenant_id",
    default="",
)


def bind_active_runtime_event_recorder(
    recorder: RuntimeEventRecorderPort | None,
    *,
    tenant_id: str | None = None,
) -> Token:
    tenant_token = _active_runtime_event_tenant_id.set((tenant_id or "").strip())
    recorder_token = _active_runtime_event_recorder.set(recorder)
    return recorder_token


def reset_active_runtime_event_recorder(token: Token) -> None:
    _active_runtime_event_recorder.reset(token)


def peek_active_runtime_event_recorder() -> RuntimeEventRecorderPort | None:
    return _active_runtime_event_recorder.get()


def peek_active_runtime_event_tenant_id() -> str:
    return _active_runtime_event_tenant_id.get()


__all__ = [
    "bind_active_runtime_event_recorder",
    "peek_active_runtime_event_recorder",
    "peek_active_runtime_event_tenant_id",
    "reset_active_runtime_event_recorder",
]
