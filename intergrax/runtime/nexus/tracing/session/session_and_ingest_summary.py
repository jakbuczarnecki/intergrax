# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class SessionAndIngestSummaryDiagV1(DiagnosticPayload):
    session_id: str
    user_id: str
    tenant_id: Optional[str]

    attachments_count: int
    ingestion_results_count: int

    # Keep this small and stable (optional)
    ingestion_preview: List[Dict[str, Any]]

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.session_and_ingest.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "attachments_count": self.attachments_count,
            "ingestion_results_count": self.ingestion_results_count,
            "ingestion_preview": list(self.ingestion_preview),
        }
