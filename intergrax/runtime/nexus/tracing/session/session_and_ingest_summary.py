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
    ingestion_preview: List[IngestionPreviewItemV1]

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.session_and_ingest.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "attachments_count": self.attachments_count,
            "ingestion_results_count": self.ingestion_results_count,
            "ingestion_preview": [p.to_dict() for p in self.ingestion_preview],
        }

@dataclass(frozen=True)
class IngestionPreviewItemV1:
    attachment_id: str
    attachment_type: str
    num_chunks: int
    vector_ids_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "attachment_type": self.attachment_type,
            "num_chunks": int(self.num_chunks),
            "vector_ids_count": int(self.vector_ids_count),
        }
