# © Artur Czarnecki. All rights reserved.

"""Streaming task intake contract (IDEAL-3.3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.task_envelope import TaskEnvelope


class TaskEnvelopeChunk(BaseModel):
    """Single chunk in a streaming intake sequence."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    content: str
    is_final: bool = False


class StreamingTaskIntake(Protocol):
    """Protocol for adapters that assemble a TaskEnvelope from streamed chunks."""

    async def chunks(self) -> AsyncIterator[TaskEnvelopeChunk]:
        ...

    async def finalize(self) -> TaskEnvelope:
        ...


def assemble_envelope_from_chunks(
    chunks: list[TaskEnvelopeChunk],
    *,
    tenant_id: str,
    user_id: str,
) -> TaskEnvelope:
    """Merge ordered chunks into a canonical TaskEnvelope."""
    ordered = sorted(chunks, key=lambda chunk: chunk.sequence)
    message = "".join(chunk.content for chunk in ordered)
    if not ordered or not ordered[-1].is_final:
        raise ValueError("streaming intake requires a final chunk")
    return TaskEnvelope(tenant_id=tenant_id, user_id=user_id, message=message)
