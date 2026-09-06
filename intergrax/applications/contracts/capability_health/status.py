# © Artur Czarnecki. All rights reserved.

"""Canonical effective capability health status (P1.5)."""

from __future__ import annotations

from enum import StrEnum


class CapabilityHealthStatus(StrEnum):
    """Cross-domain operational readiness projection — not semantic authority."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
