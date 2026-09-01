# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task lifecycle state enum (shared without task runtime imports)."""

from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    CREATED = "created"
    CLASSIFIED = "classified"
    PLANNED = "planned"
    WAITING_FOR_RESOURCES = "waiting_for_resources"
    WAITING_FOR_HUMAN = "waiting_for_human"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    NEEDS_MORE_INFORMATION = "needs_more_information"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
