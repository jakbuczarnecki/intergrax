# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class HistorySummaryDiagV1(DiagnosticPayload):
    base_history_length: int
    history_length: int
    history_includes_current_user: bool
    history_tokens: Optional[int]

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.history.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_history_length": self.base_history_length,
            "history_length": self.history_length,
            "history_includes_current_user": self.history_includes_current_user,
            "history_tokens": self.history_tokens,
        }
