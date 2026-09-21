# © Artur Czarnecki. All rights reserved.

"""Declarative policy enforcement posture — public contract."""

from __future__ import annotations

from enum import StrEnum


class PolicyEnforcementMode(StrEnum):
    """Declarative policy enforcement posture for a host bundle."""

    AUDIT_ONLY = "audit_only"
    ENFORCE = "enforce"
