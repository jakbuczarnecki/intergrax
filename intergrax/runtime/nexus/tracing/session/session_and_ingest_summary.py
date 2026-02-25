# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True)
class SessionAndIngestSummaryDiagV1(DiagnosticPayload):
    session_id: str
    user_id: str
    tenant_id: Optional[str]

    attachments_count: int
    ingestion_results_count: int

    def redact(self) -> SessionAndIngestSummaryDiagV1:
        """
        This diagnostic payload contains user- and tenant-level identifiers
        and attachment identifiers which must not be persisted in raw form
        in production traces.
        We preserve ingestion metrics but remove identifying fields.
        """
        redacted_preview = [
            IngestionPreviewItemV1(
                attachment_id=DEFAULT_REDACTED_TEXT,
                attachment_type=p.attachment_type,
                num_chunks=p.num_chunks,
                vector_ids_count=p.vector_ids_count,
            )
            for p in self.ingestion_preview
        ]

        return SessionAndIngestSummaryDiagV1(
            session_id=self.session_id,
            user_id=DEFAULT_REDACTED_TEXT,
            tenant_id=DEFAULT_REDACTED_TEXT if self.tenant_id is not None else None,
            attachments_count=self.attachments_count,
            ingestion_results_count=self.ingestion_results_count,
            ingestion_preview=redacted_preview,
        )

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
