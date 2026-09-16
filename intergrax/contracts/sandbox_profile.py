# © Artur Czarnecki. All rights reserved.

"""Sandbox session configuration surface for execution environment resolution (P1.8)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SandboxProfile(BaseModel):
    """Sandbox session manager configuration (Phase H-APP.3.5)."""

    model_config = ConfigDict(extra="forbid")

    root: Path | None = None
    enable_exec_tool: bool = False
