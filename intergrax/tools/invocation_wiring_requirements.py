# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Invocation wiring requirement metadata (no registry imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolInvocationWiringRequirements:
    """Provider-declared invocation-scoped dependencies (default-empty ABI)."""

    workspace: bool = False
    memory_view: bool = False
    trace_reader: bool = False
    run_budget: bool = False
    sandbox_session: bool = False
