# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class ToolInvocationStartDiagV1(DiagnosticPayload):
    tool_id: str
    step_id: str
    side_effects: bool
    input_payload: Dict[str, Any]

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.tools.invocation.start"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "step_id": self.step_id,
            "side_effects": self.side_effects,
            "input_payload": dict(self.input_payload),
        }


@dataclass(frozen=True)
class ToolInvocationEndDiagV1(DiagnosticPayload):
    tool_id: str
    step_id: str
    success: bool
    output_preview: Optional[str]

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.tools.invocation.end"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "step_id": self.step_id,
            "success": self.success,
            "output_preview": self.output_preview,
        }


@dataclass(frozen=True)
class ToolInvocationErrorDiagV1(DiagnosticPayload):
    tool_id: str
    step_id: str
    error_code: RuntimeErrorCode
    error_message: str

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.tools.invocation.error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "step_id": self.step_id,
            "error_code": self.error_code.value,
            "error_message": self.error_message,
        }
