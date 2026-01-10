# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class AttachmentsContextSummaryDiagV1(DiagnosticPayload):
    configured: bool
    has_session: bool

    used_attachments_context: bool
    hits_count: int

    top_k: int

    reason: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.attachments.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "has_session": self.has_session,
            "used_attachments_context": self.used_attachments_context,
            "hits_count": self.hits_count,
            "top_k": self.top_k,
            "reason": self.reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
