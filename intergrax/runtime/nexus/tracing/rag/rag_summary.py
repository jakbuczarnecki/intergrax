# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RagSummaryDiagV1(DiagnosticPayload):
    rag_enabled: bool
    used_rag: bool
    chunks_count: int
    context_messages_count: int
    warning: Optional[str]

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.rag.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rag_enabled": self.rag_enabled,
            "used_rag": self.used_rag,
            "chunks_count": self.chunks_count,
            "context_messages_count": self.context_messages_count,
            "warning": self.warning,
        }