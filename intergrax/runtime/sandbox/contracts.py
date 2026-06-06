# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox execution contracts shared by local and hosted sessions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.sandbox.models import SandboxExecutionResult


@runtime_checkable
class SandboxExecCapable(Protocol):
    """Minimal surface required by the ``sandbox.exec`` catalog tool."""

    session_id: str

    def execute(self, operation: str, payload: dict | None = None) -> SandboxExecutionResult:
        """Run an allowlisted sandbox operation."""
