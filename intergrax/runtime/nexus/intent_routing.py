# © Artur Czarnecki. All rights reserved.

"""Apply Tier-3 intent routes to tasks before classification (ORCH-CONFIG.1)."""

from __future__ import annotations

from intergrax.contracts.intent_route import IntentRoute
from intergrax.runtime.task.task import Task, TaskContext


def apply_intent_routes(task: Task, routes: list[IntentRoute]) -> Task:
    """
    Set ``task.context.capability`` from the first matching route when capability is unset.

    Routes are evaluated in declaration order; first match wins.
    Mutates ``task`` in place so Nexus lifecycle transitions stay on the same object.
    """
    existing = (task.context.capability or "").strip()
    if existing:
        return task

    message = (task.message or "").strip()
    if not message or not routes:
        return task

    for route in routes:
        if not route.keywords:
            continue
        haystack = message.lower() if route.case_insensitive else message
        for keyword in route.keywords:
            needle = keyword.lower() if route.case_insensitive else keyword
            if needle and needle in haystack:
                task.context = TaskContext(
                    capability=route.capability,
                    intent=task.context.intent,
                    metadata=dict(task.context.metadata),
                )
                return task
    return task
