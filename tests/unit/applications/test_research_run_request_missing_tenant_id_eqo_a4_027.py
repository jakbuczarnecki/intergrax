# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from pydantic import ValidationError

from research_application.serving.fastapi_router import ResearchRunService
from research_application.serving.schemas import ResearchRunRequestV1

pytestmark = [pytest.mark.unit]


def test_research_run_request_v1_rejects_missing_tenant_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ResearchRunRequestV1.model_validate({"message": "summarize", "user_id": "u1"})
    assert any(error.get("loc") == ("tenant_id",) for error in exc_info.value.errors())


def test_research_run_request_v1_rejects_whitespace_only_tenant_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ResearchRunRequestV1.model_validate(
            {"tenant_id": "   ", "user_id": "u1", "message": "summarize"},
        )
    assert any(error.get("loc") == ("tenant_id",) for error in exc_info.value.errors())


@pytest.mark.asyncio
async def test_research_run_service_rejects_whitespace_tenant_id_without_principal() -> None:
    host_execution = AsyncMock(spec=HostTaskExecutionPort)
    service = ResearchRunService(host_execution=host_execution)
    body = ResearchRunRequestV1.model_construct(
        tenant_id="   ",
        user_id="u1",
        message="summarize",
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.run_pipeline(body, authenticated_principal=None)
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "tenant_id_required"
    host_execution.execute.assert_not_called()
