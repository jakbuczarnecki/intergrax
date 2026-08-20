# © Artur Czarnecki. All rights reserved.

"""Change Approval live capability handler."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.change_approval.integration import ChangeApprovalIntegration
from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
    CHANGE_APPROVAL_SOURCE_KIND,
    ChangeApprovalNotFoundError,
    ChangeApprovalReadError,
    ChangeApprovalSnapshotV1,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.live import (
    LiveCapabilityExecutionContextV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityHandlerV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
    ValidatedLiveCapabilityCallV1,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    PointInTimeEvidenceTemporalV1,
)
from intergrax.runtime.vendor_knowledge.live.contracts import content_sha256
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)

CHANGE_APPROVAL_READ_CAPABILITY_ID: Final[str] = (
    f"vendor.{CHANGE_APPROVAL_PROVIDER_ID}.{CHANGE_APPROVAL_SOURCE_KIND}.read"
)
CHANGE_APPROVAL_READ_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/change_approval/change/read/request/v1"
)
CHANGE_APPROVAL_READ_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/change_approval/change/read/result/v1"
)

_MAX_RESULT_BYTES = 65_536
_MAX_CONTENT_BYTES_PER_ITEM = 16_384


class ChangeApprovalReadLiveRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    change_id: str = Field(..., min_length=1, max_length=64)

    @field_validator("change_id")
    @classmethod
    def _validate_change_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("change_id_invalid")
        return cleaned


def build_change_approval_read_descriptor() -> LiveCapabilityDescriptorV1:
    return LiveCapabilityDescriptorV1(
        capability_id=CHANGE_APPROVAL_READ_CAPABILITY_ID,
        provider_id=CHANGE_APPROVAL_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=CHANGE_APPROVAL_SOURCE_KIND,
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=False,
        request_schema_ref=CHANGE_APPROVAL_READ_REQUEST_SCHEMA_REF,
        result_schema_ref=CHANGE_APPROVAL_READ_RESULT_SCHEMA_REF,
        max_result_items=1,
        max_result_bytes=_MAX_RESULT_BYTES,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=1,
        max_provider_page_size=1,
        max_content_bytes_per_item=_MAX_CONTENT_BYTES_PER_ITEM,
        available=True,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalized_content(snapshot: ChangeApprovalSnapshotV1) -> str:
    payload = {
        "change_id": snapshot.change_id,
        "approval_state": snapshot.approval_state.value,
        "approved": snapshot.approved,
        "updated_at": snapshot.updated_at.isoformat(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class ChangeApprovalReadLiveHandlerV1(LiveCapabilityHandlerV1):
    provider_id = CHANGE_APPROVAL_PROVIDER_ID
    integration_kind = IntegrationCategory.ISSUE_TRACKER
    source_kind = CHANGE_APPROVAL_SOURCE_KIND
    capability_id = CHANGE_APPROVAL_READ_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = CHANGE_APPROVAL_READ_REQUEST_SCHEMA_REF
    result_schema_ref = CHANGE_APPROVAL_READ_RESULT_SCHEMA_REF
    expected_request_model = ChangeApprovalReadLiveRequestV1

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        try:
            request = self._validated_request(call)
            if not isinstance(integration, ChangeApprovalIntegration):
                raise self._vendor_error(
                    VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
                )
            snapshot = await integration.read_change_approval(
                change_id=request.change_id,
            )
            if type(snapshot) is not ChangeApprovalSnapshotV1:
                raise self._vendor_error(
                    VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
                )
            content = _normalized_content(snapshot)
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > min(
                call.effective_budget.max_content_bytes_per_item,
                _MAX_CONTENT_BYTES_PER_ITEM,
            ):
                raise self._vendor_error(
                    VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
                )
            retrieved_at = _utc_now()
            item = LiveCapabilityResultItemV1(
                remote_item_id=f"change:{snapshot.change_id}:approval",
                safe_display_name=f"Change approval {snapshot.change_id}",
                content=content,
                content_hash=content_sha256(content),
                retrieved_at=retrieved_at,
                remote_updated_at=snapshot.updated_at,
                temporal=PointInTimeEvidenceTemporalV1(
                    effective_at=snapshot.updated_at,
                ),
            )
            return LiveCapabilityExecutionResultV1(
                call_id=call.call_id,
                normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
                items=(item,),
                item_count=1,
                byte_count=content_bytes,
                started_at=context.started_at,
                completed_at=retrieved_at,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._failure(
                call=call,
                context=context,
                error_code=self._map_exception(exc),
            )

    def _validated_request(
        self,
        call: ValidatedLiveCapabilityCallV1,
    ) -> ChangeApprovalReadLiveRequestV1:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise self._vendor_error(VendorKnowledgeErrorCode.CONFIGURATION_ERROR)
        return call.validated_request

    def _vendor_error(self, code: VendorKnowledgeErrorCode) -> VendorKnowledgeError:
        return VendorKnowledgeError(code)

    def _failure(
        self,
        *,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
        error_code: str,
    ) -> LiveCapabilityExecutionResultV1:
        completed_at = _utc_now()
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=LiveExecutionOutcomeV1.FAILED,
            items=(),
            item_count=0,
            byte_count=0,
            started_at=context.started_at,
            completed_at=completed_at,
            error_code=error_code,
        )

    @staticmethod
    def _map_vendor_error(code: VendorKnowledgeErrorCode) -> str:
        return {
            VendorKnowledgeErrorCode.INVALID_SCOPE: "live_resource_scope_invalid",
            VendorKnowledgeErrorCode.CONFIGURATION_ERROR: "live_request_invalid",
            VendorKnowledgeErrorCode.AUTHENTICATION_FAILED: "live_provider_unauthorized",
            VendorKnowledgeErrorCode.AUTHORIZATION_DENIED: "live_provider_forbidden",
            VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND: "live_provider_not_found",
            VendorKnowledgeErrorCode.REMOTE_ITEM_REVOKED: "live_provider_not_found",
            VendorKnowledgeErrorCode.RATE_LIMITED: "live_provider_throttled",
            VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE: (
                "live_provider_temporarily_unavailable"
            ),
            VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE: (
                "live_provider_contract_violation"
            ),
        }.get(code, "live_execution_failed")

    def _map_exception(self, exc: BaseException) -> str:
        if isinstance(exc, VendorKnowledgeError):
            return self._map_vendor_error(exc.code)
        if isinstance(exc, ChangeApprovalNotFoundError):
            return "live_provider_not_found"
        if isinstance(exc, ChangeApprovalReadError):
            return "live_provider_contract_violation"
        if isinstance(exc, IntegrationConfigurationError):
            return "live_request_invalid"
        if isinstance(exc, IntegrationDependencyError):
            return "live_provider_temporarily_unavailable"
        if isinstance(exc, (ValidationError, TypeError, AttributeError, ValueError)):
            return "live_provider_contract_violation"
        return "live_execution_failed"
