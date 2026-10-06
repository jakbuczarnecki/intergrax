# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4 qualification support (model ↔ context attribution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

TRACE_X_P4_START_HEAD: Final[str] = "8cbefef4c2a66fd2a195d18fb3482e8b65c9b141"


@dataclass(frozen=True, slots=True)
class ModelContextAttributionJoin:
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str
    model_input_messages_hash: str
    tenant_id: str | None = None
    execution_scope: str = ""


def _payload_data(event: Any) -> dict[str, Any]:
    raw = event.payload if isinstance(event.payload, dict) else {}
    data = raw.get("data")
    if isinstance(data, dict):
        return {**raw, **data}
    return raw


def attribution_from_runtime_event(event: Any) -> ModelContextAttributionJoin | None:
    merged = _payload_data(event)
    model_hash = str(merged.get("model_input_messages_hash") or "")
    if not model_hash:
        return None
    return ModelContextAttributionJoin(
        task_id=str(event.task_id),
        run_id=str(event.run_id),
        attempt_id=str(event.attempt_id),
        execution_id=str(event.execution_id),
        model_input_messages_hash=model_hash,
        tenant_id=event.tenant_id,
        execution_scope=str(merged.get("execution_scope") or ""),
    )


def attributions_joinable(left: ModelContextAttributionJoin, right: ModelContextAttributionJoin) -> bool:
    return (
        left.task_id == right.task_id
        and left.run_id == right.run_id
        and left.attempt_id == right.attempt_id
        and left.execution_id == right.execution_id
        and left.model_input_messages_hash == right.model_input_messages_hash
    )
