# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True)
class ToolInvocationStartDiagV1(DiagnosticPayload):
    tool_id: str
    step_id: str
    side_effects: bool
    input_payload: Dict[str, Any]    

    def redact(self) -> ToolInvocationStartDiagV1:
        """
        This diagnostic payload may contain arbitrary tool input data
        which can include user content, RAG data, memory content
        or organization data.
        In production, tool input payload must not be persisted.
        We preserve structural metadata only.
        """
        return ToolInvocationStartDiagV1(
            tool_id=self.tool_id,
            step_id=self.step_id,
            side_effects=self.side_effects,
            input_payload={"_redacted": True},
        )

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
    duration_ms: Optional[int]

    def redact(self) -> ToolInvocationEndDiagV1:
        """
        This diagnostic payload may contain tool output preview
        which can include user data, RAG data or external API content.
        In production, tool output preview must not be persisted.
        """
        return ToolInvocationEndDiagV1(
            tool_id=self.tool_id,
            step_id=self.step_id,
            success=self.success,
            output_preview=DEFAULT_REDACTED_TEXT if self.output_preview is not None else None,
            duration_ms=self.duration_ms,
        )

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.tools.invocation.end"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "step_id": self.step_id,
            "success": self.success,
            "output_preview": self.output_preview,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class ToolInvocationErrorDiagV1(DiagnosticPayload):
    tool_id: str
    step_id: str
    error_code: RuntimeErrorCode
    error_message: str

    def redact(self) -> ToolInvocationErrorDiagV1:
        """
        This diagnostic payload may contain raw tool error messages
        which can include user content, tool inputs/outputs or external API data.
        In production, error_message must not be persisted.
        """
        return ToolInvocationErrorDiagV1(
            tool_id=self.tool_id,
            step_id=self.step_id,
            error_code=self.error_code,
            error_message=DEFAULT_REDACTED_TEXT,
        )

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
