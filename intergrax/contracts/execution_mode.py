# © Artur Czarnecki. All rights reserved.

"""Neutral execution posture for runtime and agent merge (architecture §3.4)."""

from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    """Execution posture shared by application hosts and runtime merge."""

    STRICT = "strict"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"
