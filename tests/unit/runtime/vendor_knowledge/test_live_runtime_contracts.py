from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live import (
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    LiveCapabilityExecutionContextV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityHandlerV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
    ValidatedLiveCapabilityCallV1,
)

_NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    query: str


class _ForeignOutcome(StrEnum):
    COMPLETED = "completed"


class _RuntimeOnlyHandler:
    provider_id = "runtime_provider"
    integration_kind = IntegrationCategory.ISSUE_TRACKER
    source_kind = "issues"
    capability_id = "vendor.runtime_provider.issues.read"
    contract_version = "1"
    request_schema_ref = "schema://runtime/request/v1"
    result_schema_ref = "schema://runtime/result/v1"
    expected_request_model = _Request

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        item = LiveCapabilityResultItemV1(
            remote_item_id="item-1",
            safe_display_name="Runtime item",
            content="runtime content",
            content_hash=sha256(b"runtime content").hexdigest(),
            retrieved_at=context.started_at,
        )
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
            items=(item,),
            item_count=1,
            byte_count=len(item.content.encode()),
            started_at=context.started_at,
            completed_at=context.started_at,
        )


def _call() -> ValidatedLiveCapabilityCallV1:
    return ValidatedLiveCapabilityCallV1(
        call_id="call-1",
        capability_id=_RuntimeOnlyHandler.capability_id,
        contract_version="1",
        connection_ref="connection-1",
        live_access_binding_id="binding-1",
        remote_resource_id="resource-1",
        validated_request=_Request(query="read"),
        effective_budget=EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=1_000,
            max_result_items=10,
            max_result_bytes=1_000,
        ),
        audience_context_ref="personal",
        provider_id=_RuntimeOnlyHandler.provider_id,
        integration_kind=_RuntimeOnlyHandler.integration_kind,
        source_kind=_RuntimeOnlyHandler.source_kind,
    )


@pytest.mark.asyncio
async def test_runtime_only_handler_constructs_canonical_result() -> None:
    context = LiveCapabilityExecutionContextV1(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=100.0,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )

    result = await _RuntimeOnlyHandler().execute(
        integration=object(),
        call=_call(),
        context=context,
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.items[0].content == "runtime content"


def test_outcome_enum_remains_strict_and_canonical() -> None:
    assert LiveExecutionOutcomeV1.COMPLETED.value == "completed"
    with pytest.raises(ValidationError):
        LiveCapabilityExecutionResultV1(
            call_id="call-1",
            normalized_outcome=_ForeignOutcome.COMPLETED,
            item_count=0,
            byte_count=0,
            started_at=_NOW,
            completed_at=_NOW,
        )
    with pytest.raises(ValidationError):
        LiveCapabilityExecutionResultV1(
            call_id="call-1",
            normalized_outcome="completed",
            item_count=0,
            byte_count=0,
            started_at=_NOW,
            completed_at=_NOW,
        )


def test_application_execution_exports_exact_runtime_contracts() -> None:
    application_execution = import_module(
        "local_workspace_application.workspaces.hybrid_ask_execution"
    )

    assert (
        application_execution.LiveCapabilityExecutionContextV1
        is LiveCapabilityExecutionContextV1
    )
    assert (
        application_execution.LiveCapabilityExecutionResultV1
        is LiveCapabilityExecutionResultV1
    )
    assert application_execution.LiveCapabilityHandlerV1 is LiveCapabilityHandlerV1
    assert application_execution.LiveCapabilityResultItemV1 is LiveCapabilityResultItemV1
    assert application_execution.LiveExecutionOutcomeV1 is LiveExecutionOutcomeV1


def test_live_runtime_production_has_no_lkw_application_import() -> None:
    repository = Path(__file__).parents[4]
    live_root = repository / "intergrax" / "runtime" / "vendor_knowledge" / "live"
    forbidden = ("local_workspace_application", "applications.local_workspace_application")

    for source_file in live_root.glob("*.py"):
        source = source_file.read_text(encoding="utf-8")
        assert not any(term in source for term in forbidden), source_file
