# © Artur Czarnecki. All rights reserved.

"""Memory scope modes for agent session environment (architecture §30.9)."""

from __future__ import annotations

from enum import Enum


class MemoryScope(str, Enum):
    USER = "user"
    ORG = "org"
    TASK = "task"
    CUSTOM = "custom"
