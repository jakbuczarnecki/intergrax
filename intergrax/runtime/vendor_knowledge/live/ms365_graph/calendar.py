"""Microsoft Graph Calendar live list capability."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    MSGRAPH_CALENDAR_SCOPE_TYPE,
    MsGraphCalendarKnowledgeAdapter,
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
from intergrax.runtime.vendor_knowledge.live.contracts import (
    content_sha256,
    safe_locator_or_none,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)

MSGRAPH_CALENDAR_LIST_CAPABILITY_ID: Final[str] = (
    f"vendor.{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}.calendar.list"
)
MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/"
    f"{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}/calendar/list/request/v1"
)
MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/"
    f"{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}/calendar/list/result/v1"
)

_MAX_RESULT_ITEMS = 200
_MAX_RESULT_BYTES = 131_072
_MAX_PROVIDER_PAGE_SIZE = 200
_MAX_CONTENT_BYTES_PER_ITEM = 4_096
_EVENT_ITEM_TYPE = "msgraph_calendar_event"
_REMOVAL_SEMANTICS = "removed_from_synchronized_calendar_window_view"
_METADATA_KEYS = frozenset(
    {
        "event_state",
        "event_type",
        "start_at",
        "end_at",
        "original_start_at",
        "last_modified_at",
        "series_master_id",
        "i_cal_uid",
        "is_all_day",
        "is_cancelled",
        "is_draft",
        "has_attachments",
        "is_online_meeting",
        "removal_semantics",
    }
)


class MsGraphCalendarListLiveRequestV1(BaseModel):
    """Strict bounded input for one binding-selected Calendar page."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    page_size: int = Field(default=25, ge=1, le=_MAX_PROVIDER_PAGE_SIZE)


def build_msgraph_calendar_list_descriptor() -> LiveCapabilityDescriptorV1:
    """Build the single supported Microsoft Graph Calendar descriptor."""

    return LiveCapabilityDescriptorV1(
        capability_id=MSGRAPH_CALENDAR_LIST_CAPABILITY_ID,
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=True,
        supported_resource_types=(MSGRAPH_CALENDAR_SCOPE_TYPE,),
        request_schema_ref=MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF,
        result_schema_ref=MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF,
        max_result_items=_MAX_RESULT_ITEMS,
        max_result_bytes=_MAX_RESULT_BYTES,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=_MAX_RESULT_ITEMS,
        max_provider_page_size=_MAX_PROVIDER_PAGE_SIZE,
        max_content_bytes_per_item=_MAX_CONTENT_BYTES_PER_ITEM,
        available=True,
    )


class MsGraphCalendarListLiveHandlerV1(LiveCapabilityHandlerV1):
    """Stateless metadata-only mapping over one binding-selected Calendar scope."""

    provider_id = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    integration_kind = IntegrationCategory.COLLABORATION_SUITE
    source_kind = MSGRAPH_CALENDAR_SOURCE_KIND
    capability_id = MSGRAPH_CALENDAR_LIST_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF
    result_schema_ref = MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF
    expected_request_model = MsGraphCalendarListLiveRequestV1

    def __init__(
        self,
        adapter: MsGraphCalendarKnowledgeAdapter | None = None,
    ) -> None:
        self._adapter = adapter or MsGraphCalendarKnowledgeAdapter()

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        try:
            self._validate_call(call)
            request = self._validate_request(call)
            if not isinstance(integration, Ms365GraphCollaborationSuiteIntegration):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message=(
                        "Microsoft Graph Calendar live capability requires "
                        "the resolved collaboration-suite integration"
                    ),
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                )
            source = KnowledgeSourceRef(
                tenant_id=context.tenant_id,
                provider_id=self.provider_id,
                integration_kind=self.integration_kind,
                source_kind=self.source_kind,
                connection_ref=call.connection_ref,
                scope=KnowledgeSourceScope(
                    remote_scope_id=call.remote_resource_id,
                    remote_scope_type=MSGRAPH_CALENDAR_SCOPE_TYPE,
                    safe_display_name="Microsoft Graph Calendar",
                    parameters={},
                ),
            )
            effective_limit = min(
                request.page_size,
                call.effective_budget.max_result_items,
                call.effective_budget.max_upstream_items,
                call.effective_budget.max_provider_page_size,
                _MAX_PROVIDER_PAGE_SIZE,
            )
            page = await self._adapter.read_page(
                integration=integration,
                source=source,
                cursor=None,
                limit=effective_limit,
            )
            if len(page.changes) > effective_limit:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Microsoft Graph Calendar live page exceeds its bound",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                )
            retrieved_at = _utc_now()
            items = self._map_changes(page.changes, retrieved_at=retrieved_at)
            self._validate_result_bytes(
                items,
                max_item_bytes=min(
                    call.effective_budget.max_content_bytes_per_item,
                    _MAX_CONTENT_BYTES_PER_ITEM,
                ),
                max_result_bytes=min(
                    call.effective_budget.max_result_bytes,
                    _MAX_RESULT_BYTES,
                ),
            )
            outcome = (
                LiveExecutionOutcomeV1.TRUNCATED
                if page.has_more
                else LiveExecutionOutcomeV1.COMPLETED
            )
            return self._result(
                call=call,
                context=context,
                items=items,
                outcome=outcome,
                truncated=page.has_more,
            )
        except asyncio.CancelledError:
            raise
        except VendorKnowledgeError as exc:
            return self._failure(
                call=call,
                context=context,
                error_code=self._map_vendor_error(exc.code),
            )
        except IntegrationConfigurationError:
            return self._failure(
                call=call,
                context=context,
                error_code="live_request_invalid",
            )
        except IntegrationDependencyError:
            return self._failure(
                call=call,
                context=context,
                error_code="live_provider_temporarily_unavailable",
            )
        except (ValidationError, TypeError, AttributeError):
            return self._failure(
                call=call,
                context=context,
                error_code="live_provider_contract_violation",
            )
        except ValueError:
            return self._failure(
                call=call,
                context=context,
                error_code="live_resource_scope_invalid",
            )
        except Exception:
            return self._failure(
                call=call,
                context=context,
                error_code="live_execution_failed",
            )

    def _validate_call(self, call: ValidatedLiveCapabilityCallV1) -> None:
        try:
            call.assert_identity()
        except (TypeError, ValueError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar live capability identity is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if (
            call.provider_id != self.provider_id
            or call.integration_kind is not self.integration_kind
            or call.source_kind != self.source_kind
            or call.capability_id != self.capability_id
            or call.contract_version != self.contract_version
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar live capability scope is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if call.remote_resource_id is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar live resource scope is missing",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )

    def _validate_request(
        self,
        call: ValidatedLiveCapabilityCallV1,
    ) -> MsGraphCalendarListLiveRequestV1:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Calendar live request is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return call.validated_request

    def _map_changes(
        self,
        changes: tuple[KnowledgeChange, ...],
        *,
        retrieved_at: datetime,
    ) -> tuple[LiveCapabilityResultItemV1, ...]:
        seen_remote_ids: set[str] = set()
        items: list[LiveCapabilityResultItemV1] = []
        for change in changes:
            if change.remote_id in seen_remote_ids:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Microsoft Graph Calendar live page contains duplicate item IDs",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                )
            seen_remote_ids.add(change.remote_id)
            items.append(self._map_change(change, retrieved_at=retrieved_at))
        return tuple(items)

    def _map_change(
        self,
        change: KnowledgeChange,
        *,
        retrieved_at: datetime,
    ) -> LiveCapabilityResultItemV1:
        descriptor = change.descriptor
        if descriptor is None:
            if change.kind is not KnowledgeChangeKind.DELETED:
                raise self._invalid_provider_response()
            safe_display_name = "Removed Calendar event"
            normalized: dict[str, object] = {
                "change_kind": change.kind.value,
                "content_available": False,
                "item_type": "deleted",
                "event_state": "removed",
                "removal_semantics": _REMOVAL_SEMANTICS,
            }
            remote_updated_at = None
            safe_locator = None
        else:
            if change.kind is not KnowledgeChangeKind.UPSERT:
                raise self._invalid_provider_response()
            self._validate_active_descriptor(change, descriptor)
            metadata = descriptor.metadata
            normalized = {
                "change_kind": change.kind.value,
                "content_available": descriptor.content_available,
                "content_mode": descriptor.content_mode.value,
                "item_type": descriptor.item_type,
                "event_state": metadata["event_state"],
                "event_type": metadata["event_type"],
                "start_at": metadata["start_at"],
                "end_at": metadata["end_at"],
                "original_start_at": metadata["original_start_at"],
                "last_modified_at": metadata["last_modified_at"],
                "series_master_id": metadata["series_master_id"],
                "i_cal_uid": metadata["i_cal_uid"],
                "is_all_day": metadata["is_all_day"],
                "is_cancelled": metadata["is_cancelled"],
                "is_draft": metadata["is_draft"],
                "has_attachments": metadata["has_attachments"],
                "is_online_meeting": metadata["is_online_meeting"],
                "removal_semantics": metadata["removal_semantics"],
                "revision_version": descriptor.revision.version,
            }
            safe_display_name = descriptor.title
            remote_updated_at = descriptor.revision.updated_at
            safe_locator = safe_locator_or_none(descriptor.provenance.web_url)
        content = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return LiveCapabilityResultItemV1(
            remote_item_id=change.remote_id,
            safe_display_name=safe_display_name,
            content=content,
            content_hash=content_sha256(content),
            retrieved_at=retrieved_at,
            remote_updated_at=remote_updated_at,
            safe_locator=safe_locator,
        )

    def _validate_active_descriptor(
        self,
        change: KnowledgeChange,
        descriptor: object,
    ) -> None:
        if (
            descriptor.item_type != _EVENT_ITEM_TYPE
            or descriptor.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD
            or descriptor.content_available is not True
            or descriptor.provenance.provider_id != self.provider_id
            or descriptor.provenance.source_kind != self.source_kind
            or descriptor.provenance.remote_id != change.remote_id
            or set(descriptor.metadata) != _METADATA_KEYS
            or descriptor.metadata["event_state"] != "active"
            or descriptor.metadata["removal_semantics"] != _REMOVAL_SEMANTICS
        ):
            raise self._invalid_provider_response()
        metadata = descriptor.metadata
        for key in (
            "is_all_day",
            "is_cancelled",
            "is_draft",
            "has_attachments",
            "is_online_meeting",
        ):
            if type(metadata[key]) is not bool:
                raise self._invalid_provider_response()
        if not isinstance(metadata["event_type"], str) or not metadata["event_type"]:
            raise self._invalid_provider_response()
        for key in ("start_at", "end_at", "last_modified_at"):
            if not isinstance(metadata[key], str) or not metadata[key]:
                raise self._invalid_provider_response()
            try:
                self._validate_timestamp(metadata[key])
            except ValueError:
                raise self._invalid_provider_response() from None
        original_start_at = metadata["original_start_at"]
        if original_start_at is not None:
            if not isinstance(original_start_at, str) or not original_start_at:
                raise self._invalid_provider_response()
            try:
                self._validate_timestamp(original_start_at)
            except ValueError:
                raise self._invalid_provider_response() from None
        for key in ("series_master_id", "i_cal_uid"):
            if metadata[key] is not None and not isinstance(metadata[key], str):
                raise self._invalid_provider_response()
        if (
            not isinstance(descriptor.revision.version, str)
            or not descriptor.revision.version
            or descriptor.revision.updated_at is None
            or descriptor.revision.updated_at.tzinfo is None
            or descriptor.revision.updated_at.utcoffset() is None
        ):
            raise self._invalid_provider_response()

    @staticmethod
    def _validate_timestamp(value: str) -> None:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("metadata timestamp is invalid") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("metadata timestamp must be timezone-aware")

    def _validate_result_bytes(
        self,
        items: tuple[LiveCapabilityResultItemV1, ...],
        *,
        max_item_bytes: int,
        max_result_bytes: int,
    ) -> None:
        total = 0
        for item in items:
            size = len(item.content.encode("utf-8"))
            if size > max_item_bytes:
                raise self._invalid_provider_response()
            total += size
        if total > max_result_bytes:
            raise self._invalid_provider_response()

    def _result(
        self,
        *,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
        items: tuple[LiveCapabilityResultItemV1, ...],
        outcome: LiveExecutionOutcomeV1,
        truncated: bool,
    ) -> LiveCapabilityExecutionResultV1:
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=outcome,
            items=items,
            item_count=len(items),
            byte_count=sum(len(item.content.encode("utf-8")) for item in items),
            started_at=context.started_at,
            completed_at=_utc_now(),
            truncated=truncated,
            provider_id=call.provider_id,
            integration_kind=call.integration_kind,
            source_kind=call.source_kind,
            capability_id=call.capability_id,
            contract_version=call.contract_version,
            live_access_binding_id=call.live_access_binding_id,
            connection_ref=call.connection_ref,
            remote_resource_id=call.remote_resource_id,
        )

    def _failure(
        self,
        *,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
        error_code: str,
    ) -> LiveCapabilityExecutionResultV1:
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=LiveExecutionOutcomeV1.FAILED,
            item_count=0,
            byte_count=0,
            started_at=context.started_at,
            completed_at=_utc_now(),
            error_code=error_code,
            provider_id=call.provider_id,
            integration_kind=call.integration_kind,
            source_kind=call.source_kind,
            capability_id=call.capability_id,
            contract_version=call.contract_version,
            live_access_binding_id=call.live_access_binding_id,
            connection_ref=call.connection_ref,
            remote_resource_id=call.remote_resource_id,
        )

    def _invalid_provider_response(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message="Microsoft Graph Calendar live result is invalid",
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    @staticmethod
    def _map_vendor_error(code: VendorKnowledgeErrorCode) -> str:
        mapping = {
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
        }
        return mapping.get(code, "live_execution_failed")


def _utc_now() -> datetime:
    return datetime.now(UTC)
