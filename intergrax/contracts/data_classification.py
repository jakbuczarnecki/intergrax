# © Artur Czarnecki. All rights reserved.

"""Data classification for security governance (FAUDIT-SEC.1)."""

from __future__ import annotations

from enum import Enum


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    def allows_export(self) -> bool:
        return self in {DataClassification.PUBLIC, DataClassification.INTERNAL}

    def requires_encryption(self) -> bool:
        return self in {DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED}
