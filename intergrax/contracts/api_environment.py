# © Artur Czarnecki. All rights reserved.

"""Host deployment environment classification (Tier-3 application settings)."""

from __future__ import annotations

from enum import Enum


class ApiEnvironment(str, Enum):
    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"


__all__ = ["ApiEnvironment"]
