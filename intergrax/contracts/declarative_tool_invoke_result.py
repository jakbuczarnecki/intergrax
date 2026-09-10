# © Artur Czarnecki. All rights reserved.

"""Typed declarative tool invoke outcome (shared ACP / execution-bound U2 contract)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DeclarativeToolInvokeResult:
    status: Literal["success", "failed", "denied"]
    output: dict[str, Any] | None = None
    external_ref: str | None = None
    error: str | None = None
    duration_ms: int = 0
