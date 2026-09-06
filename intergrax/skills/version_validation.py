# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Exact skill version identity validation (P1.10)."""

from __future__ import annotations

_FORBIDDEN_VERSION_SENTINELS = frozenset({"unknown", "latest", "current", "0"})


def validate_skill_version(value: object, *, label: str = "version") -> str:
    """Reject ambiguous or empty skill version labels."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if normalized.lower() in _FORBIDDEN_VERSION_SENTINELS:
        raise ValueError(f"{label} must be explicit, got {normalized!r}")
    return normalized
