# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical worker lifecycle state contract (AW-1A)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

CANONICAL_WORKER_LIFECYCLE_STATES: Final = (
    "PROVISIONING",
    "ACTIVE",
    "IDLE",
    "WORKING",
    "WAITING_EXTERNAL",
    "WAITING_FOR_HUMAN",
    "RECOVERING",
    "DEGRADED",
    "PAUSED",
    "QUARANTINED",
    "STOPPED",
)


class WorkerLifecycleState(StrEnum):
    """Semantic worker lifecycle — distinct from Task/Run/Execution lifecycle."""

    PROVISIONING = "PROVISIONING"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    WORKING = "WORKING"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    RECOVERING = "RECOVERING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    QUARANTINED = "QUARANTINED"
    STOPPED = "STOPPED"
