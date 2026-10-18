# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-owned domain signal payload families (OBS-DIAG-EC2)."""

from __future__ import annotations

from enum import Enum

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.middleware_hook_point import HookPoint
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.hooks.hook_context import HookAction


class PlatformHookViolationKind(str, Enum):
    """How the hook runtime guard classified the violation."""

    TIMEOUT = "timeout"
    ERROR = "error"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"


class PlatformHookViolationPayloadV1(RuntimeEventPayload):
    """
    Typed body for ``platform.hook.hook_blocked`` / ``hook_error`` / ``hook_timeout``.

    Shared semantics across the hook violation family (APP-CON-5).
    """

    schema_id = "platform.hook.violation.v1"

    hook_name: str = ""
    point: HookPoint = HookPoint.BEFORE_TASK_INTAKE
    action: HookAction = HookAction.BLOCK
    reason: str = ""
    violation_kind: PlatformHookViolationKind = PlatformHookViolationKind.BLOCK
    source_execution_phase: ExecutionPhase = ExecutionPhase.STEP_EXECUTION


class DiagnosticSubsystemFailurePayloadV1(RuntimeEventPayload):
    """Emitted when terminal execution diagnostics fail to persist (DIAG-FOUNDATION-3)."""

    schema_id = "platform.diagnostic.subsystem_failure.v1"

    error_type: str = ""
    source: str = ""


__all__ = [
    "DiagnosticSubsystemFailurePayloadV1",
    "PlatformHookViolationKind",
    "PlatformHookViolationPayloadV1",
]
