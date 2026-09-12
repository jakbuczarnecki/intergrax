# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from enum import StrEnum


class ModelAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


__all__ = ["ModelAvailability"]
