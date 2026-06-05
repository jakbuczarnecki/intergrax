# © Artur Czarnecki. All rights reserved.

"""Typed task security context keys for Nexus lifecycle hooks."""

from __future__ import annotations

from intergrax.runtime.task.task import Task

RESOURCE_TENANT_ID_METADATA_KEY = "resource_tenant_id"


def resource_tenant_id_for_task(task: Task) -> str:
    """Resolve the resource tenant bound to this task intake check."""
    value = task.context.metadata.get(RESOURCE_TENANT_ID_METADATA_KEY)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return task.tenant_id
