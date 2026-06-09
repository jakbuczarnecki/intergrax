# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""User-facing autonomy steering (architecture REL §35, UAEP §42.10.2)."""

from __future__ import annotations

from enum import Enum


class AutonomyLevel(str, Enum):
    """How much the harness may act without explicit human confirmation."""

    MANUAL = "manual"
    ASK = "ask"
    AUTONOMOUS = "autonomous"
