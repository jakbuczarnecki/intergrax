"""Microsoft Graph Teams Chat live list capability."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    MsGraphTeamsChatKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.utils import attribute_access
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

MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID: Final[str] = (
    f"vendor.{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}.teams_chat.list"
)
MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/"
    f"{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}/teams_chat/list/request/v1"
)
MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/"
    f"{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}/teams_chat/list/result/v1"
)

_MAX_RESULT_ITEMS = 50
_MAX_RESULT_BYTES = 131_072
_MAX_PROVIDER_PAGE_SIZE = 50
_MAX_CONTENT_BYTES_PER_ITEM = 4_096
_MESSAGE_ITEM_TYPE = "msgraph_teams_chat_message"
_METADATA_KEYS = frozenset(
    {
        "message_state",
        "message_type",
        "importance",
        "body_kind",
        "has_attachments",
        "created_at",
        "last_modified_at",
        "last_edited_at",
        "event_detail_type",
        "locale",
        "attachment_inventory_in_content",
        "attachment_binary_content_included",
        "hosted_content_included",
        "reference_urls_included",
    }
)


class MsGraphTeamsChatListLiveRequestV1(BaseModel):
    """Strict bounded input for one binding-fixed Teams Chat page."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    page_size: int = Field(default=25, ge=1, le=_MAX_PROVIDER_PAGE_SIZE)


def build_msgraph_teams_chat_list_descriptor() -> LiveCapabilityDescriptorV1:
    """Build the single supported Microsoft Graph Teams Chat descriptor."""

    return LiveCapabilityDescriptorV1(
        capability_id=MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID,
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=True,
        supported_resource_types=(MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,),
        request_schema_ref=MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF,
        result_schema_ref=MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF,
        max_result_items=_MAX_RESULT_ITEMS,
        max_result_bytes=_MAX_RESULT_BYTES,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=_MAX_RESULT_ITEMS,
        max_provider_page_size=_MAX_PROVIDER_PAGE_SIZE,
        max_content_bytes_per_item=_MAX_CONTENT_BYTES_PER_ITEM,
        available=True,
    )


class MsGraphTeamsChatListLiveHandlerV1(LiveCapabilityHandlerV1):
    """Stateless metadata-only mapping over one fixed-window Chat scope."""

    provider_id = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    integration_kind = IntegrationCategory.COLLABORATION_SUITE
    source_kind = MSGRAPH_TEAMS_CHAT_SOURCE_KIND
    capability_id = MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF
    result_schema_ref = MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF
    expected_request_model = MsGraphTeamsChatListLiveRequestV1

    def __init__(
        self,
        adapter: MsGraphTeamsChatKnowledgeAdapter | None = None,
    ) -> None:
        self._adapter = adapter or MsGraphTeamsChatKnowledgeAdapter()

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
                        "Microsoft Graph Teams Chat live capability requires "
                        "the resolved collaboration-suite integration"
                    ),
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                )
            resolved_scope = attribute_access.optional(call, "resolved_resource_scope", None)
            scope_token = (
                attribute_access.optional(resolved_scope, "scope_token", None)
                if resolved_scope is not None
                else None
            )
            # The resolved token is authoritative. The remote-resource fallback
            # remains for legacy validated calls whose binding scope predates the
            # resolved-resource envelope; identity validation above still pins it
            # to the tenant/binding request before the provider adapter sees it.
            source = KnowledgeSourceRef(
                tenant_id=context.tenant_id,
                provider_id=self.provider_id,
                integration_kind=self.integration_kind,
                source_kind=self.source_kind,
                connection_ref=call.connection_ref,
                scope=KnowledgeSourceScope(
                    remote_scope_id=scope_token or call.remote_resource_id,
                    remote_scope_type=MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
                    safe_display_name="Microsoft Graph Teams Chat",
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
                    safe_message="Microsoft Graph Teams Chat live page exceeds its bound",
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
                safe_message="Microsoft Graph Teams Chat live capability identity is invalid",
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
                safe_message="Microsoft Graph Teams Chat live capability scope is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if call.remote_resource_id is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat live resource scope is missing",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )

    def _validate_request(
        self,
        call: ValidatedLiveCapabilityCallV1,
    ) -> MsGraphTeamsChatListLiveRequestV1:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Chat live request is invalid",
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
                    safe_message="Microsoft Graph Teams Chat live page contains duplicate item IDs",
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
            safe_display_name = "Deleted Teams Chat message"
            normalized: dict[str, object] = {
                "change_kind": change.kind.value,
                "content_available": False,
                "item_type": "deleted",
                "message_state": "removed",
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
                "message_state": metadata["message_state"],
                "message_type": metadata["message_type"],
                "importance": metadata["importance"],
                "body_kind": metadata["body_kind"],
                "has_attachments": metadata["has_attachments"],
                "created_at": metadata["created_at"],
                "last_modified_at": metadata["last_modified_at"],
                "last_edited_at": metadata["last_edited_at"],
                "event_detail_type": metadata["event_detail_type"],
                "locale": metadata["locale"],
                "attachment_inventory_in_content": metadata[
                    "attachment_inventory_in_content"
                ],
                "attachment_binary_content_included": metadata[
                    "attachment_binary_content_included"
                ],
                "hosted_content_included": metadata["hosted_content_included"],
                "reference_urls_included": metadata["reference_urls_included"],
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
            descriptor.item_type != _MESSAGE_ITEM_TYPE
            or descriptor.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD
            or descriptor.content_available is not True
            or descriptor.provenance.provider_id != self.provider_id
            or descriptor.provenance.source_kind != self.source_kind
            or descriptor.provenance.remote_id != change.remote_id
            or set(descriptor.metadata) != _METADATA_KEYS
            or descriptor.metadata["message_state"] != "active"
        ):
            raise self._invalid_provider_response()
        metadata = descriptor.metadata
        for key in (
            "has_attachments",
            "attachment_inventory_in_content",
            "attachment_binary_content_included",
            "hosted_content_included",
            "reference_urls_included",
        ):
            if type(metadata[key]) is not bool:
                raise self._invalid_provider_response()
        if (
            metadata["attachment_inventory_in_content"] is not True
            or metadata["attachment_binary_content_included"] is not False
            or metadata["hosted_content_included"] is not False
            or metadata["reference_urls_included"] is not False
        ):
            raise self._invalid_provider_response()
        for key in ("message_type", "importance", "body_kind", "created_at", "last_modified_at"):
            if not isinstance(metadata[key], str) or not metadata[key]:
                raise self._invalid_provider_response()
        for key in ("created_at", "last_modified_at"):
            try:
                self._validate_timestamp(metadata[key])
            except ValueError:
                raise self._invalid_provider_response() from None
        last_edited_at = metadata["last_edited_at"]
        if last_edited_at is not None:
            if not isinstance(last_edited_at, str) or not last_edited_at:
                raise self._invalid_provider_response()
            try:
                self._validate_timestamp(last_edited_at)
            except ValueError:
                raise self._invalid_provider_response() from None
        for key in ("event_detail_type", "locale"):
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
            safe_message="Microsoft Graph Teams Chat live result is invalid",
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
