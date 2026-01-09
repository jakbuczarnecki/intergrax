# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class CoreLLMAdapterFailedDiagV1(DiagnosticPayload):
    error_type: str
    error_message: str
    has_tools_agent_answer: bool

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.engine.core_llm.adapter_failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "has_tools_agent_answer": self.has_tools_agent_answer,
        }
