# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.task.task import Task


class TaskClassifier:
    """
    Minimal task classifier (§10.2).

    Enriches task metadata; state transitions are owned by TaskLifecycle.
    """

    def classify(self, task: Task) -> Task:
        if task.agent_id:
            task.metadata.setdefault("classification", "single_agent_explicit")

        capability = task.context.capability
        if capability:
            task.metadata["requested_capability"] = capability
            task.metadata.setdefault("classification", "capability_routed")

        if "classification" not in task.metadata:
            task.metadata["classification"] = "single_agent_default"

        return task
