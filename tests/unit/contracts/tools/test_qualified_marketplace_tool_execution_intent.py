# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntent,
    SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1,
)

pytestmark = pytest.mark.unit


def _intent() -> QualifiedMarketplaceToolExecutionIntent:
    return QualifiedMarketplaceToolExecutionIntent(
        execution_request_id="exec-req-1",
        binding_operation_id="bind-1",
        resume_operation_id="resume-1",
        tenant_id="tenant-a",
        task_id="task_00000000000000000000000000000001",
        worker_need_id="worker-need-1",
        qualified_subject_reference="qualified-capability-subject:q:domain_handoff_reference:h",
        handoff_id="handoff-1",
        selected_operation="invoke",
    )


def test_valid_construction() -> None:
    intent = _intent()
    assert intent.schema_version == SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1


def test_empty_identity_rejected() -> None:
    with pytest.raises(ValidationError):
        QualifiedMarketplaceToolExecutionIntent(
            execution_request_id="",
            binding_operation_id="bind-1",
            resume_operation_id="resume-1",
            tenant_id="tenant-a",
            task_id="task_00000000000000000000000000000001",
            worker_need_id="worker-need-1",
            qualified_subject_reference="qualified-capability-subject:q:domain_handoff_reference:h",
            handoff_id="handoff-1",
            selected_operation="invoke",
        )


def test_frozen_model() -> None:
    intent = _intent()
    with pytest.raises(ValidationError):
        intent.execution_request_id = "other"


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        QualifiedMarketplaceToolExecutionIntent(
            **_intent().model_dump(),
            business_payload={"x": 1},
        )


def test_semantic_schema_fixed() -> None:
    assert _intent().schema_version == "qualified_marketplace_tool_execution_intent.v1"


def test_no_business_payload_field() -> None:
    assert "business_payload" not in QualifiedMarketplaceToolExecutionIntent.model_fields
