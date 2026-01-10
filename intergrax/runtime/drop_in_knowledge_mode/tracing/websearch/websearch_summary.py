# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class WebsearchSummaryDiagV1(DiagnosticPayload):
    enabled: bool
    configured: bool

    used_websearch: bool
    results_count: int

    context_blocks_count: int

    no_evidence: bool

    error_type: Optional[str]
    error_message: Optional[str]

    context_preview_chars: int
    context_preview: str  # limited

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.websearch.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "used_websearch": self.used_websearch,
            "results_count": self.results_count,
            "context_blocks_count": self.context_blocks_count,
            "no_evidence": self.no_evidence,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "context_preview_chars": self.context_preview_chars,
            "context_preview": self.context_preview,
        }
