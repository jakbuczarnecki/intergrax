# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal validation helpers for Capability Catalog Stage-1 contracts."""

from __future__ import annotations


def require_non_empty_text(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if value != normalized:
        raise ValueError(f"{label} must not have surrounding whitespace")
    return normalized


def normalize_optional_text(value: object | None, *, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{label} must be str or None, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be omitted or non-empty")
    if value != normalized:
        raise ValueError(f"{label} must not have surrounding whitespace")
    return normalized
