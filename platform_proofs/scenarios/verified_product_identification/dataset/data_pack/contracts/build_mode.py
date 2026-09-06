"""Data pack build policy modes."""

from __future__ import annotations

from enum import StrEnum


class DataPackBuildMode(StrEnum):
    """Controls strictness of model identity and compatibility gates."""

    PROOF = "PROOF"
    CANONICAL = "CANONICAL"
