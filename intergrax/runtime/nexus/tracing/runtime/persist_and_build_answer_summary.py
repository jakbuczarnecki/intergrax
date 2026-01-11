# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PersistAndBuildAnswerSummaryDiagV1(DiagnosticPayload):
    session_id: str
    strategy: str
    used_rag: bool
    used_websearch: bool
    used_tools: bool

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.engine.persist_and_build_answer.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "strategy": self.strategy,
            "used_rag": self.used_rag,
            "used_websearch": self.used_websearch,
            "used_tools": self.used_tools,
        }
