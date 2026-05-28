# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent-facing memory write policy (architecture §42.35)."""

from __future__ import annotations

from enum import Enum


class MemoryWritePolicy(str, Enum):
    REPLACE = "replace"
    MERGE = "merge"
