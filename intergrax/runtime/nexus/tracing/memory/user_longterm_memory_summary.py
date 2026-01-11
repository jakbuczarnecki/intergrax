# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class UserLongtermMemorySummaryDiagV1(DiagnosticPayload):
    enabled: bool
    used_user_longterm_memory: bool
    reason: Optional[str]

    hits_count: int
    top_k: int

    context_blocks_count: int
    context_preview_chars: int
    context_preview: str  # limited

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.user_longterm_memory.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used_user_longterm_memory": self.used_user_longterm_memory,
            "reason": self.reason,
            "hits_count": self.hits_count,
            "top_k": self.top_k,
            "context_blocks_count": self.context_blocks_count,
            "context_preview_chars": self.context_preview_chars,
            "context_preview": self.context_preview,
        }
