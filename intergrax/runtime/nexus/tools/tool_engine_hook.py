# © Artur Czarnecki. All rights reserved.

"""Adaptive tool engine hook contract (TOOL-ENG-10)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolEngineHook:
    """When enabled, per-run tool selection / invocation mode may be adapted."""

    enabled: bool
    engine_id: str
