# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
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
    run_id = mint_run_id()
    session_id = mint_task_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        results = await service.ingest_attachments_for_session(
            [attachment],
            session_id=session_id,
            user_id="user-1",
            tenant_id="t1",
            run_id=run_id,
        )
    finally:
        reset_active_execution_identity(token)

    assert len(results) == 1
    assert results[0].num_chunks == 0
    assert results[0].metadata.get("reason") == "ingestion_failed"
    failed = [e for e in bus.history if e.event_type == RuntimeEventType.INGESTION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["attachment_id"] == "att-1"
    assert failed[0].run_id == run_id
    assert failed[0].task_id == session_id
