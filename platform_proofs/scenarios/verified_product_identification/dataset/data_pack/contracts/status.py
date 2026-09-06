"""Data pack lifecycle status."""

from __future__ import annotations

from enum import StrEnum


class DataPackStatus(StrEnum):
    BUILDING = "BUILDING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
