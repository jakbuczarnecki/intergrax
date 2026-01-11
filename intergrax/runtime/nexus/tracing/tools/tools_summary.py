# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class ToolsSummaryDiagV1(DiagnosticPayload):
    tools_mode: str
    used_tools: bool

    tool_calls_count: int
    tool_names: List[str]

    warning: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.tools.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tools_mode": self.tools_mode,
            "used_tools": self.used_tools,
            "tool_calls_count": self.tool_calls_count,
            "tool_names": list(self.tool_names),
            "warning": self.warning,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
