# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.ingest_contracts import (
    RagScheduleIngestJobInput,
    RagScheduleIngestJobOutput,
)
from intergrax.tools.providers.rag.ingest_job_service import perform_rag_schedule_ingest_job


class RagScheduleIngestJobHandler(
    ServiceToolHandler[RagScheduleIngestJobInput, RagScheduleIngestJobOutput]
):
    _service = perform_rag_schedule_ingest_job
