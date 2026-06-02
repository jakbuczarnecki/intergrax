# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class CoreLLMUsedToolsAgentAnswerDiagV1(DiagnosticPayload):
    used_tools_answer: bool
    has_tool_planner_answer: bool

    def redact(self) -> CoreLLMUsedToolsAgentAnswerDiagV1:
        """
        This diagnostic payload contains only boolean metadata
        about tool agent answer usage.
        It does not include user content or identifying information
        and is considered PII-safe.
        """
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.engine.core_llm.used_tool_planner_answer"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used_tools_answer": self.used_tools_answer,
            "has_tool_planner_answer": self.has_tool_planner_answer,
        }
