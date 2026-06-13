# © Artur Czarnecki. All rights reserved.

"""Selective capture policy for execution boundary export."""

from __future__ import annotations

from enum import Enum

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionResult


class AttestationCaptureMode(str, Enum):
    OFF = "off"
    SIDE_EFFECTS_ONLY = "side_effects_only"
    ALLOWLIST = "allowlist"


def should_emit_boundary_event(
    *,
    contract: ToolContract,
    result: ToolExecutionResult,
    capture_mode: AttestationCaptureMode,
    allowlist: frozenset[str],
) -> bool:
    """Return True when the harness should emit an unsigned boundary event."""
    _ = result
    if capture_mode == AttestationCaptureMode.OFF:
        return False
    if capture_mode == AttestationCaptureMode.ALLOWLIST:
        return contract.tool_id in allowlist
    return contract.side_effects
