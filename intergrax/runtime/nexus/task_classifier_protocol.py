# © Artur Czarnecki. All rights reserved.

"""Nexus task classifier protocol (Phase ORCH-1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.task.task import Task


class NexusTaskClassifierProtocol(Protocol):
    """Enriches task classification metadata without owning lifecycle transitions."""

    def classify(self, task: Task) -> Task:
        """Return the task with classification fields populated."""
