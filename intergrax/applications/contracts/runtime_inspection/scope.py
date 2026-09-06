# © Artur Czarnecki. All rights reserved.

"""Core runtime inspection scopes (P1.4)."""

from __future__ import annotations

from enum import StrEnum


class InspectionScope(StrEnum):
    """Bounded inspection domains for the canonical read model."""

    PROFILE = "profile"
    CAPABILITY = "capability"
    EXECUTION = "execution"
