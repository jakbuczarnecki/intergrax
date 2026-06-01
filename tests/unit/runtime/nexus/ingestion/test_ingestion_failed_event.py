# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.llm.messages import AttachmentRef
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.ingestion.attachments import AttachmentResolver
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService


class _FailingResolver(AttachmentResolver):
    async def resolve_to_path(self, attachment: AttachmentRef) -> Path:
        raise FileNotFoundError(f"missing:{attachment.id}")


@pytest.mark.asyncio
@pytest.mark.gate
async def test_ingestion_failure_emits_runtime_event() -> None:
    bus = RuntimeEventBus()
    service = AttachmentIngestionService(
        resolver=_FailingResolver(),
        embedding_manager=MagicMock(),
        vectorstore_manager=MagicMock(),
        loader=MagicMock(),
        splitter=MagicMock(),
        event_bus=bus,
    )
    attachment = AttachmentRef(id="att-1", type="file", uri="file:///missing.pdf")

    results = await service.ingest_attachments_for_session(
        [attachment],
        session_id="sess-1",
        user_id="user-1",
        tenant_id="t1",
        run_id="run-1",
    )

    assert len(results) == 1
    assert results[0].num_chunks == 0
    assert results[0].metadata.get("reason") == "ingestion_failed"
    failed = [e for e in bus.history if e.event_type == RuntimeEventType.INGESTION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["attachment_id"] == "att-1"
