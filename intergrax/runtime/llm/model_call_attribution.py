# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Request-scoped model-call attribution for runtime LLM_CALL evidence (TRACE-X-P4)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

_model_call_scope: ContextVar[ModelCallExecutionScope] = ContextVar(
    "intergrax_model_call_execution_scope",
    default=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
)
_pending_model_input_hash: ContextVar[str] = ContextVar(
    "intergrax_pending_model_input_messages_hash",
    default="",
)
_attribution_node_id: ContextVar[str] = ContextVar("intergrax_model_call_node_id", default="")
_attribution_agent_id: ContextVar[str] = ContextVar("intergrax_model_call_agent_id", default="")
_attribution_step_id: ContextVar[str] = ContextVar("intergrax_model_call_step_id", default="")
_attribution_label: ContextVar[str] = ContextVar("intergrax_model_call_label", default="")
_pending_context_assembly_event_id: ContextVar[str] = ContextVar(
    "intergrax_pending_context_assembly_event_id",
    default="",
)


def get_model_call_execution_scope() -> ModelCallExecutionScope:
    return _model_call_scope.get()


def peek_pending_model_input_messages_hash() -> str:
    return _pending_model_input_hash.get()


def set_pending_model_input_messages_hash(messages_hash: str) -> None:
    _pending_model_input_hash.set((messages_hash or "").strip())


def bind_model_input_messages(messages: Sequence[ChatMessage]) -> None:
    set_pending_model_input_messages_hash(compute_model_facing_messages_hash(messages))


def clear_pending_model_input_messages_hash() -> None:
    _pending_model_input_hash.set("")


def bind_pending_context_assembly_event_id(event_id: EventId | str) -> None:
    resolved = validate_event_id(event_id)
    _pending_context_assembly_event_id.set(str(resolved))


def peek_pending_context_assembly_event_id() -> str:
    return _pending_context_assembly_event_id.get()


def clear_pending_context_assembly_event_id() -> None:
    _pending_context_assembly_event_id.set("")


@dataclass(frozen=True, slots=True)
class ModelCallAttributionOverlay:
    execution_scope: ModelCallExecutionScope | None = None
    node_id: str | None = None
    agent_id: str | None = None
    step_id: str | None = None
    label: str | None = None


@contextmanager
def model_call_attribution_scope(
    overlay: ModelCallAttributionOverlay | None = None,
    *,
    execution_scope: ModelCallExecutionScope | None = None,
    messages: Sequence[ChatMessage] | None = None,
) -> Iterator[None]:
    previous_scope = get_model_call_execution_scope()
    previous_hash = peek_pending_model_input_messages_hash()
    previous_node = _attribution_node_id.get()
    previous_agent = _attribution_agent_id.get()
    previous_step = _attribution_step_id.get()
    previous_label = _attribution_label.get()

    resolved_scope = execution_scope
    if overlay is not None and overlay.execution_scope is not None:
        resolved_scope = overlay.execution_scope
    if resolved_scope is not None:
        _model_call_scope.set(resolved_scope)

    if messages is not None:
        bind_model_input_messages(messages)

    if overlay is not None:
        if overlay.node_id is not None:
            _attribution_node_id.set(overlay.node_id)
        if overlay.agent_id is not None:
            _attribution_agent_id.set(overlay.agent_id)
        if overlay.step_id is not None:
            _attribution_step_id.set(overlay.step_id)
        if overlay.label is not None:
            _attribution_label.set(overlay.label)

    try:
        yield
    finally:
        _model_call_scope.set(previous_scope)
        set_pending_model_input_messages_hash(previous_hash)
        _attribution_node_id.set(previous_node)
        _attribution_agent_id.set(previous_agent)
        _attribution_step_id.set(previous_step)
        _attribution_label.set(previous_label)


def peek_model_call_attribution_ids() -> tuple[str, str, str, str]:
    return (
        _attribution_node_id.get(),
        _attribution_agent_id.get(),
        _attribution_step_id.get(),
        _attribution_label.get(),
    )


__all__ = [
    "ModelCallAttributionOverlay",
    "bind_model_input_messages",
    "bind_pending_context_assembly_event_id",
    "clear_pending_context_assembly_event_id",
    "clear_pending_model_input_messages_hash",
    "get_model_call_execution_scope",
    "model_call_attribution_scope",
    "peek_model_call_attribution_ids",
    "peek_pending_context_assembly_event_id",
    "peek_pending_model_input_messages_hash",
    "set_pending_model_input_messages_hash",
]
