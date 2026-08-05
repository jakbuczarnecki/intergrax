"""Microsoft Graph Mail live list capability."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_SCOPE_TYPE,
    MsGraphMailKnowledgeAdapter,
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
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)

MSGRAPH_MAIL_LIST_CAPABILITY_ID: Final[str] = (
    f"vendor.{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}.mail.list"
)
MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/"
    f"{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}/mail/list/request/v1"
)
MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/"
    f"{MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID}/mail/list/result/v1"
)

_MAX_RESULT_ITEMS = 100
_MAX_RESULT_BYTES = 131_072
_MAX_PROVIDER_PAGE_SIZE = 100
_MAX_CONTENT_BYTES_PER_ITEM = 4_096
_REMOVAL_SEMANTICS = "removed_from_synchronized_folder_view"


class MsGraphMailListLiveRequestV1(BaseModel):
    """Strict bounded input for one Mail mailbox-folder metadata page."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    page_size: int = Field(default=25, ge=1, le=_MAX_PROVIDER_PAGE_SIZE)


def build_msgraph_mail_list_descriptor() -> LiveCapabilityDescriptorV1:
    """Build the single supported Microsoft Graph Mail live descriptor."""

    return LiveCapabilityDescriptorV1(
        capability_id=MSGRAPH_MAIL_LIST_CAPABILITY_ID,
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=True,
        supported_resource_types=(MSGRAPH_MAIL_SCOPE_TYPE,),
        request_schema_ref=MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF,
        result_schema_ref=MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF,
        max_result_items=_MAX_RESULT_ITEMS,
        max_result_bytes=_MAX_RESULT_BYTES,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=_MAX_RESULT_ITEMS,
        max_provider_page_size=_MAX_PROVIDER_PAGE_SIZE,
        max_content_bytes_per_item=_MAX_CONTENT_BYTES_PER_ITEM,
        available=True,
    )


class MsGraphMailListLiveHandlerV1(LiveCapabilityHandlerV1):
    """Stateless live mapping over the existing public Mail adapter boundary."""

    provider_id = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    integration_kind = IntegrationCategory.COLLABORATION_SUITE
    source_kind = MSGRAPH_MAIL_SOURCE_KIND
    capability_id = MSGRAPH_MAIL_LIST_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF
    result_schema_ref = MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF
    expected_request_model = MsGraphMailListLiveRequestV1

    def __init__(self, adapter: MsGraphMailKnowledgeAdapter | None = None) -> None:
        self._adapter = adapter or MsGraphMailKnowledgeAdapter()

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
                        "Microsoft Graph Mail live capability requires "
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
                    remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE,
                    safe_display_name="Microsoft Graph Mail folder",
                    parameters={},
                ),
            )
            effective_limit = min(
                request.page_size,
                call.effective_budget.max_result_items,
                call.effective_budget.max_upstream_items,
                call.effective_budget.max_provider_page_size,
            )
            page = await self._adapter.read_page(
                integration=integration,
                source=source,
                cursor=None,
                limit=effective_limit,
            )
            retrieved_at = _utc_now()
            items = self._map_changes(page.changes, retrieved_at=retrieved_at)
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
        except (ValidationError, TypeError):
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
        except Exception:  # noqa: BLE001
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
                safe_message="Microsoft Graph Mail live capability identity is invalid",
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
                safe_message="Microsoft Graph Mail live capability scope is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if call.remote_resource_id is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail live resource scope is missing",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )

    def _validate_request(
        self,
        call: ValidatedLiveCapabilityCallV1,
    ) -> MsGraphMailListLiveRequestV1:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Mail live request is invalid",
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
                    safe_message="Microsoft Graph Mail live page contains duplicate item IDs",
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
            safe_display_name = "Deleted Mail message"
            normalized = {
                "change_kind": change.kind.value,
                "content_available": False,
                "item_type": "deleted",
                "message_state": "removed",
                "removal_semantics": _REMOVAL_SEMANTICS,
            }
            remote_updated_at = None
            safe_locator = None
        else:
            metadata = descriptor.metadata
            normalized = {
                "change_kind": change.kind.value,
                "content_available": descriptor.content_available,
                "content_mode": descriptor.content_mode.value,
                "item_type": descriptor.item_type,
                "message_state": metadata["message_state"],
                "is_read": metadata["is_read"],
                "is_draft": metadata["is_draft"],
                "has_attachments": metadata["has_attachments"],
                "importance": metadata["importance"],
                "attachment_inventory_included": metadata[
                    "attachment_inventory_included"
                ],
                "attachment_content_included": metadata["attachment_content_included"],
                "revision_version": descriptor.revision.version,
            }
            for key in (
                "created_at",
                "received_at",
                "sent_at",
                "last_modified_at",
            ):
                if key in metadata:
                    normalized[key] = metadata[key]
            safe_display_name = descriptor.title
            remote_updated_at = descriptor.revision.updated_at
            safe_locator = safe_locator_or_none(descriptor.provenance.web_url)
        content = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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
