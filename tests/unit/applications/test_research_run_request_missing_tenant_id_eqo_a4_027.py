# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_application.serving.schemas import ResearchRunRequestV1

pytestmark = [pytest.mark.unit]


def test_research_run_request_v1_rejects_missing_tenant_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ResearchRunRequestV1.model_validate({"message": "summarize", "user_id": "u1"})
    assert any(error.get("loc") == ("tenant_id",) for error in exc_info.value.errors())
