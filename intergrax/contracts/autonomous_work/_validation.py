# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal validation helpers for Autonomous Work contracts (AW-1A)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypeVar

_T = TypeVar("_T")


def require_non_empty_text(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def require_opaque_ref(value: object, *, label: str) -> str:
    return require_non_empty_text(value, label=label)


def require_aware_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{label} must be datetime, got {type(value).__name__}")
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def require_non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{label} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def freeze_tuple(value: tuple[_T, ...] | list[_T], *, label: str) -> tuple[_T, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be tuple or list, got {type(value).__name__}")
    return tuple(value)
