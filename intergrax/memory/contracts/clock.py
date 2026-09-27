# © Artur Czarnecki. All rights reserved.

"""Clock helpers for Memory contract defaults (no globals/settings coupling)."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = ["utc_now"]
