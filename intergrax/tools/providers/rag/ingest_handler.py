# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput, RagIngestOutput
from intergrax.tools.providers.rag.ingest_service import perform_rag_ingest


class RagIngestHandler(ServiceToolHandler[RagIngestInput, RagIngestOutput]):
    _service = perform_rag_ingest
