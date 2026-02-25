# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


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

    def redact(self) -> AttachmentsContextSummaryDiagV1:
        """
        This diagnostic payload may contain dynamic reason or error messages
        which can include user content or attachment-derived data.
        In production, raw reason and error_message must not be persisted.
        """
        return AttachmentsContextSummaryDiagV1(
            configured=self.configured,
            has_session=self.has_session,
            used_attachments_context=self.used_attachments_context,
            hits_count=self.hits_count,
            top_k=self.top_k,
            reason=DEFAULT_REDACTED_TEXT if self.reason is not None else None,
            error_type=self.error_type,
            error_message=DEFAULT_REDACTED_TEXT if self.error_message is not None else None,
        )

    @classmethod
    def schema_id(cls) -> str:
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
