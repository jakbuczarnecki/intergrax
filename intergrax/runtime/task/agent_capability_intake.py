# © Artur Czarnecki. All rights reserved.

"""Bridge runtime task routing context to public TaskEnvelope for Agent.can_handle()."""

from __future__ import annotations

from intergrax.contracts.task_envelope import (
    CAPABILITY_ROUTING_METADATA_KEY,
    INTENT_ROUTING_METADATA_KEY,
    TaskEnvelope,
)
from intergrax.runtime.task.task import Task, TaskContext


def task_envelope_for_agent_capability_match(
    task: Task,
) -> TaskEnvelope:
    envelope = task.to_envelope()
    meta = dict(envelope.metadata)
    if task.context.capability:
        meta[CAPABILITY_ROUTING_METADATA_KEY] = task.context.capability
    if task.context.intent:
        meta[INTENT_ROUTING_METADATA_KEY] = task.context.intent
    if task.context.metadata:
        meta.update(task.context.metadata)
    return envelope.model_copy(update={"metadata": meta})


def task_envelope_from_task_context(context: TaskContext) -> TaskEnvelope:
    meta = dict(context.metadata)
    if context.capability:
        meta[CAPABILITY_ROUTING_METADATA_KEY] = context.capability
    if context.intent:
        meta[INTENT_ROUTING_METADATA_KEY] = context.intent
    return TaskEnvelope(tenant_id="routing", user_id="routing", metadata=meta)
