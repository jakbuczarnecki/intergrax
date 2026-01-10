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
    has_tools_agent_answer: bool

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.engine.core_llm.used_tools_agent_answer"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used_tools_answer": self.used_tools_answer,
            "has_tools_agent_answer": self.has_tools_agent_answer,
        }
