"""Slack conversation live capabilities."""

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
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    DEFAULT_MESSAGE_MAX_CHARS,
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationContentTooLarge,
    SlackConversationExactMessageResult,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessageChanged,
    SlackConversationMessageNotFound,
    SlackConversationMessagePage,
    SlackConversationReadConfigurationError,
    SlackConversationReadError,
    SlackConversationSourceWindow,
    compute_slack_conversation_message_revision,
    validate_slack_conversation_message,
    validate_slack_timestamp,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    validate_safe_text,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    SlackConversationKnowledgeAdapter,
    decode_slack_conversation_scope_id,
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
    KnowledgePage,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)

SLACK_CONVERSATION_LIST_CAPABILITY_ID: Final[str] = (
    f"vendor.{SLACK_CONVERSATION_CHANNEL_PROVIDER_ID}."
    f"{SLACK_CONVERSATION_SOURCE_KIND}.list"
)
SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/slack/slack_conversation/list/request/v1"
)
SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/slack/slack_conversation/list/result/v1"
)

SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID: Final[str] = (
    f"vendor.{SLACK_CONVERSATION_CHANNEL_PROVIDER_ID}."
    f"{SLACK_CONVERSATION_SOURCE_KIND}.thread.read"
)
SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/slack/slack_conversation/"
    "thread.read/request/v1"
)
SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/slack/slack_conversation/"
    "thread.read/result/v1"
)

SLACK_CONVERSATION_READ_CAPABILITY_ID: Final[str] = (
    f"vendor.{SLACK_CONVERSATION_CHANNEL_PROVIDER_ID}."
    f"{SLACK_CONVERSATION_SOURCE_KIND}.read"
)
SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/slack/slack_conversation/read/request/v1"
)
SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF: Final[str] = (
    "schema://vendor-knowledge/live/slack/slack_conversation/read/result/v1"
)

_MAX_LIST_ITEMS = 1
_MAX_THREAD_ITEMS = 15
_MAX_RESULT_BYTES = 131_072
_MAX_CONTENT_BYTES_PER_ITEM = 16_384
_METADATA_MAX_CHARS = 4_096
_EXACT_DEFAULT_MAX_CHARS = 16_384
_MESSAGE_ITEM_TYPE = "slack_conversation_message"
_METADATA_KEYS = frozenset(
    {
        "subtype",
        "has_files",
        "reply_count",
        "created_at",
        "edited_at",
        "thread_root_ts",
        "attachment_inventory_in_content",
    }
)
_AUTH_ERRORS = frozenset(
    {
        "invalid_auth",
        "token_revoked",
        "not_authed",
        "account_inactive",
        "token_expired",
        "not_allowed_token_type",
    }
)
_FORBIDDEN_ERRORS = frozenset(
    {
        "missing_scope",
        "no_permission",
        "not_in_channel",
        "access_denied",
        "restricted_action",
        "team_access_not_granted",
        "accesslimited",
        "enterprise_is_restricted",
        "ekm_access_denied",
        "org_login_required",
        "two_factor_setup_required",
    }
)
_NOT_FOUND_ERRORS = frozenset({"channel_not_found", "thread_not_found", "message_not_found"})


class SlackConversationListLiveRequestV1(BaseModel):
    """Strict immutable zero-field request for one root message."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SlackConversationThreadReadLiveRequestV1(BaseModel):
    """Strict bounded request for one thread reply page."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    root_message_ts: str
    page_size: int = Field(default=_MAX_THREAD_ITEMS, ge=1, le=_MAX_THREAD_ITEMS)

    @field_validator("root_message_ts")
    @classmethod
    def _valid_root_message_ts(cls, value: str) -> str:
        return validate_slack_timestamp(value)


class SlackConversationReadLiveRequestV1(BaseModel):
    """Strict bounded request for one exact Slack message."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    message_ts: str
    root_thread_ts: str | None = None
    expected_revision: str | None = None
    max_chars: int = Field(
        default=_EXACT_DEFAULT_MAX_CHARS,
        ge=1,
        le=DEFAULT_MESSAGE_MAX_CHARS,
    )

    @field_validator("message_ts", "root_thread_ts")
    @classmethod
    def _valid_timestamp(cls, value: str | None) -> str | None:
        return None if value is None else validate_slack_timestamp(value)

    @field_validator("expected_revision")
    @classmethod
    def _valid_expected_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value:
            raise ValueError("expected_revision_invalid")
        return validate_safe_text(value, max_length=4096)


def _descriptor(
    *,
    capability_id: str,
    request_schema_ref: str,
    result_schema_ref: str,
    source_type: str = SLACK_CONVERSATION_SCOPE_TYPE,
    max_items: int,
    max_upstream_items: int,
    max_page_size: int,
    max_content_bytes: int = _MAX_CONTENT_BYTES_PER_ITEM,
) -> LiveCapabilityDescriptorV1:
    return LiveCapabilityDescriptorV1(
        capability_id=capability_id,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=True,
        supported_resource_types=(source_type,),
        request_schema_ref=request_schema_ref,
        result_schema_ref=result_schema_ref,
        max_result_items=max_items,
        max_result_bytes=_MAX_RESULT_BYTES,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=max_upstream_items,
        max_provider_page_size=max_page_size,
        max_content_bytes_per_item=max_content_bytes,
        available=True,
    )


def build_slack_conversation_list_descriptor() -> LiveCapabilityDescriptorV1:
    return _descriptor(
        capability_id=SLACK_CONVERSATION_LIST_CAPABILITY_ID,
        request_schema_ref=SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF,
        result_schema_ref=SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF,
        max_items=1,
        max_upstream_items=1,
        max_page_size=1,
    )


def build_slack_conversation_thread_read_descriptor() -> LiveCapabilityDescriptorV1:
    return _descriptor(
        capability_id=SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
        request_schema_ref=SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF,
        result_schema_ref=SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF,
        max_items=15,
        max_upstream_items=15,
        max_page_size=15,
    )


def build_slack_conversation_read_descriptor() -> LiveCapabilityDescriptorV1:
    return _descriptor(
        capability_id=SLACK_CONVERSATION_READ_CAPABILITY_ID,
        request_schema_ref=SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF,
        result_schema_ref=SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF,
        max_items=1,
        max_upstream_items=1,
        max_page_size=1,
    )


class _SlackLiveHandlerSupport:
    def _validate_call_and_scope(
        self,
        call: ValidatedLiveCapabilityCallV1,
    ) -> tuple[str, SlackConversationKind, SlackConversationSourceWindow]:
        try:
            call.assert_identity()
        except (TypeError, ValueError):
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_SCOPE) from None
        if (
            call.provider_id != self.provider_id
            or call.integration_kind is not self.integration_kind
            or call.source_kind != self.source_kind
            or call.capability_id != self.capability_id
            or call.contract_version != self.contract_version
            or call.remote_resource_id is None
        ):
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_SCOPE)
        try:
            return decode_slack_conversation_scope_id(call.remote_resource_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_SCOPE) from None

    def _source(
        self,
        *,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> KnowledgeSourceRef:
        if call.remote_resource_id is None:
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_SCOPE)
        return KnowledgeSourceRef(
            tenant_id=context.tenant_id,
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            source_kind=self.source_kind,
            connection_ref=call.connection_ref,
            scope=KnowledgeSourceScope(
                remote_scope_id=call.remote_resource_id,
                remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
                safe_display_name="Slack conversation",
                parameters={},
            ),
        )

    def _vendor_error(self, code: VendorKnowledgeErrorCode) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=code,
            safe_message="Slack live capability validation failed",
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
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
        self._validate_result_bytes(items, call=call)
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

    def _validate_result_bytes(
        self,
        items: tuple[LiveCapabilityResultItemV1, ...],
        *,
        call: ValidatedLiveCapabilityCallV1,
    ) -> None:
        max_item_bytes = min(
            call.effective_budget.max_content_bytes_per_item,
            _MAX_CONTENT_BYTES_PER_ITEM,
        )
        max_result_bytes = min(call.effective_budget.max_result_bytes, _MAX_RESULT_BYTES)
        total = 0
        for item in items:
            size = len(item.content.encode("utf-8"))
            if size > max_item_bytes:
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            total += size
        if total > max_result_bytes:
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)

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

    @staticmethod
    def _map_read_error(error: SlackConversationReadError) -> str:
        if error.slack_error in _AUTH_ERRORS:
            return "live_provider_unauthorized"
        if error.slack_error in _FORBIDDEN_ERRORS:
            return "live_provider_forbidden"
        if error.slack_error in _NOT_FOUND_ERRORS:
            return "live_provider_not_found"
        if error.slack_error == "ratelimited":
            return "live_provider_throttled"
        if error.slack_error == "malformed_response":
            return "live_provider_contract_violation"
        return "live_provider_temporarily_unavailable"

    def _map_exception(self, exc: BaseException) -> str:
        if isinstance(exc, VendorKnowledgeError):
            return self._map_vendor_error(exc.code)
        if isinstance(exc, SlackConversationMessageNotFound):
            return "live_provider_not_found"
        if isinstance(exc, SlackConversationMessageChanged):
            return "live_provider_temporarily_unavailable"
        if isinstance(exc, SlackConversationReadError):
            return self._map_read_error(exc)
        if isinstance(exc, SlackConversationReadConfigurationError):
            return "live_request_invalid"
        if isinstance(exc, SlackConversationContentTooLarge):
            return "live_provider_contract_violation"
        if isinstance(exc, IntegrationConfigurationError):
            return "live_request_invalid"
        if isinstance(exc, IntegrationDependencyError):
            return "live_provider_temporarily_unavailable"
        if isinstance(exc, (ValidationError, TypeError, AttributeError)):
            return "live_provider_contract_violation"
        if isinstance(exc, ValueError):
            return "live_provider_contract_violation"
        return "live_execution_failed"


class SlackConversationListLiveHandlerV1(
    _SlackLiveHandlerSupport,
    LiveCapabilityHandlerV1,
):
    provider_id = SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
    integration_kind = IntegrationCategory.CONVERSATION_CHANNEL
    source_kind = SLACK_CONVERSATION_SOURCE_KIND
    capability_id = SLACK_CONVERSATION_LIST_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF
    result_schema_ref = SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF
    expected_request_model = SlackConversationListLiveRequestV1

    def __init__(
        self,
        adapter: SlackConversationKnowledgeAdapter | None = None,
    ) -> None:
        self._adapter = adapter or SlackConversationKnowledgeAdapter()

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        try:
            self._validate_call_and_scope(call)
            self._validate_request(call)
            if not isinstance(integration, SlackConversationChannelIntegration):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            page = await self._adapter.read_page(
                integration=integration,
                source=self._source(call=call, context=context),
                cursor=None,
                limit=1,
            )
            if not isinstance(page, KnowledgePage):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            if len(page.changes) > 1:
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            retrieved_at = _utc_now()
            items = tuple(
                self._map_root_change(change, retrieved_at=retrieved_at)
                for change in page.changes
            )
            return self._result(
                call=call,
                context=context,
                items=items,
                outcome=(
                    LiveExecutionOutcomeV1.TRUNCATED
                    if page.has_more
                    else LiveExecutionOutcomeV1.COMPLETED
                ),
                truncated=page.has_more,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._failure(
                call=call,
                context=context,
                error_code=self._map_exception(exc),
            )

    def _validate_request(self, call: ValidatedLiveCapabilityCallV1) -> None:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise self._vendor_error(VendorKnowledgeErrorCode.CONFIGURATION_ERROR)

    def _map_root_change(
        self,
        change: KnowledgeChange,
        *,
        retrieved_at: datetime,
    ) -> LiveCapabilityResultItemV1:
        if change.kind is not KnowledgeChangeKind.UPSERT or change.descriptor is None:
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
        metadata = _validated_descriptor_metadata(
            change,
            require_root=True,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
        )
        content = _json_content(
            {
                "change_kind": change.kind.value,
                "item_type": change.descriptor.item_type,
                **metadata,
                "content_available": change.descriptor.content_available,
                "content_mode": change.descriptor.content_mode.value,
                "revision_version": change.descriptor.revision.version,
            }
        )
        return _item(
            remote_item_id=change.remote_id,
            safe_display_name=change.descriptor.title,
            content=content,
            retrieved_at=retrieved_at,
            remote_updated_at=change.descriptor.revision.updated_at,
            safe_locator=change.descriptor.provenance.web_url,
        )


class SlackConversationThreadReadLiveHandlerV1(
    _SlackLiveHandlerSupport,
    LiveCapabilityHandlerV1,
):
    provider_id = SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
    integration_kind = IntegrationCategory.CONVERSATION_CHANNEL
    source_kind = SLACK_CONVERSATION_SOURCE_KIND
    capability_id = SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF
    result_schema_ref = SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF
    expected_request_model = SlackConversationThreadReadLiveRequestV1

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        try:
            conversation_id, conversation_kind, window = self._validate_call_and_scope(call)
            request = self._validated_request(call)
            if not isinstance(integration, SlackConversationChannelIntegration):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            if not _timestamp_in_window(request.root_message_ts, window):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_SCOPE)
            effective_limit = min(
                request.page_size,
                call.effective_budget.max_result_items,
                call.effective_budget.max_upstream_items,
                call.effective_budget.max_provider_page_size,
                _MAX_THREAD_ITEMS,
            )
            page = await integration.read_thread_replies_page(
                conversation_id=conversation_id,
                conversation_kind=conversation_kind,
                root_message_ts=request.root_message_ts,
                window=window,
                cursor=None,
                limit=effective_limit,
                max_chars_per_message=_METADATA_MAX_CHARS,
            )
            if type(page) is not SlackConversationMessagePage:
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            if len(page.items) > effective_limit:
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            retrieved_at = _utc_now()
            items = tuple(
                self._map_reply(
                    message,
                    root_message_ts=request.root_message_ts,
                    retrieved_at=retrieved_at,
                )
                for message in page.items
            )
            return self._result(
                call=call,
                context=context,
                items=items,
                outcome=(
                    LiveExecutionOutcomeV1.TRUNCATED
                    if page.next_cursor is not None
                    else LiveExecutionOutcomeV1.COMPLETED
                ),
                truncated=page.next_cursor is not None,
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
    ) -> SlackConversationThreadReadLiveRequestV1:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise self._vendor_error(VendorKnowledgeErrorCode.CONFIGURATION_ERROR)
        return call.validated_request

    def _map_reply(
        self,
        message: SlackConversationMessage,
        *,
        root_message_ts: str,
        retrieved_at: datetime,
    ) -> LiveCapabilityResultItemV1:
        validated = validate_slack_conversation_message(message)
        if (
            validated.root_thread_ts != root_message_ts
            or validated.message_ts == root_message_ts
        ):
            raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
        content = _message_content(validated)
        return _item(
            remote_item_id=validated.message_ts,
            safe_display_name=_message_title(validated.text),
            content=content,
            retrieved_at=retrieved_at,
            remote_updated_at=validated.edited_at or validated.created_at,
        )


class SlackConversationReadLiveHandlerV1(
    _SlackLiveHandlerSupport,
    LiveCapabilityHandlerV1,
):
    provider_id = SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
    integration_kind = IntegrationCategory.CONVERSATION_CHANNEL
    source_kind = SLACK_CONVERSATION_SOURCE_KIND
    capability_id = SLACK_CONVERSATION_READ_CAPABILITY_ID
    contract_version = "1"
    request_schema_ref = SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF
    result_schema_ref = SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF
    expected_request_model = SlackConversationReadLiveRequestV1

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        try:
            conversation_id, conversation_kind, window = self._validate_call_and_scope(call)
            request = self._validated_request(call)
            if not isinstance(integration, SlackConversationChannelIntegration):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            if not _timestamp_in_window(request.message_ts, window) or (
                request.root_thread_ts is not None
                and not _timestamp_in_window(request.root_thread_ts, window)
            ):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_SCOPE)
            result = await integration.read_exact_message(
                conversation_id=conversation_id,
                conversation_kind=conversation_kind,
                message_ts=request.message_ts,
                root_thread_ts=request.root_thread_ts,
                window=window,
                expected_revision=request.expected_revision,
                max_chars_per_message=min(request.max_chars, DEFAULT_MESSAGE_MAX_CHARS),
            )
            if type(result) is not SlackConversationExactMessageResult:
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            if not result.found or result.message is None:
                raise self._vendor_error(VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND)
            message = validate_slack_conversation_message(result.message)
            if (
                message.conversation_id != conversation_id
                or message.message_ts != request.message_ts
                or message.root_thread_ts != request.root_thread_ts
            ):
                raise self._vendor_error(VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE)
            if (
                request.expected_revision is not None
                and compute_slack_conversation_message_revision(message)
                != request.expected_revision
            ):
                raise SlackConversationMessageChanged()
            content = _message_content(
                message,
                include_text=_content_fits(
                    message,
                    call=call,
                    include_text=True,
                ),
            )
            item = _item(
                remote_item_id=message.message_ts,
                safe_display_name=_message_title(message.text),
                content=content,
                retrieved_at=_utc_now(),
                remote_updated_at=message.edited_at or message.created_at,
            )
            return self._result(
                call=call,
                context=context,
                items=(item,),
                outcome=LiveExecutionOutcomeV1.COMPLETED,
                truncated=False,
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
    ) -> SlackConversationReadLiveRequestV1:
        if not isinstance(call.validated_request, self.expected_request_model):
            raise self._vendor_error(VendorKnowledgeErrorCode.CONFIGURATION_ERROR)
        return call.validated_request


def _validated_descriptor_metadata(
    change: KnowledgeChange,
    *,
    require_root: bool,
    provider_id: str,
    source_kind: str,
) -> dict[str, object]:
    descriptor = change.descriptor
    if descriptor is None:
        raise ValueError("descriptor_missing")
    if (
        descriptor.item_type != _MESSAGE_ITEM_TYPE
        or descriptor.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD
        or descriptor.content_available is not True
        or descriptor.provenance.provider_id != provider_id
        or descriptor.provenance.source_kind != source_kind
        or descriptor.provenance.remote_id != change.remote_id
        or set(descriptor.metadata) != _METADATA_KEYS
    ):
        raise ValueError("descriptor_invalid")
    metadata = descriptor.metadata
    if (
        (metadata["subtype"] is not None and not isinstance(metadata["subtype"], str))
        or type(metadata["has_files"]) is not bool
        or (
            metadata["reply_count"] is not None
            and (type(metadata["reply_count"]) is not int or metadata["reply_count"] < 0)
        )
        or not isinstance(metadata["created_at"], str)
        or (metadata["edited_at"] is not None and not isinstance(metadata["edited_at"], str))
        or (metadata["thread_root_ts"] is not None and not isinstance(metadata["thread_root_ts"], str))
        or metadata["attachment_inventory_in_content"] is not True
    ):
        raise ValueError("descriptor_metadata_invalid")
    if metadata["thread_root_ts"] is not None:
        validate_slack_timestamp(metadata["thread_root_ts"])
    if require_root and metadata["thread_root_ts"] is not None:
        raise ValueError("root_message_required")
    _validate_aware_iso(metadata["created_at"])
    if metadata["edited_at"] is not None:
        _validate_aware_iso(metadata["edited_at"])
    return dict(metadata)


def _message_content(
    message: SlackConversationMessage,
    *,
    include_text: bool = False,
) -> str:
    normalized: dict[str, object] = {
        "change_kind": KnowledgeChangeKind.UPSERT.value,
        "item_type": _MESSAGE_ITEM_TYPE,
        "subtype": message.subtype,
        "has_files": bool(message.files),
        "reply_count": message.reply_count,
        "created_at": message.created_at.isoformat(),
        "edited_at": message.edited_at.isoformat() if message.edited_at else None,
        "thread_root_ts": message.root_thread_ts,
        "attachment_inventory_in_content": True,
        "content_available": True,
        "content_mode": KnowledgeContentMode.STRUCTURED_RECORD.value,
        "revision_version": compute_slack_conversation_message_revision(message),
    }
    if include_text:
        normalized["message_ts"] = message.message_ts
        normalized["text"] = message.text
    else:
        normalized["content_deferred"] = True
    return _json_content(normalized)


def _content_fits(
    message: SlackConversationMessage,
    *,
    call: ValidatedLiveCapabilityCallV1,
    include_text: bool,
) -> bool:
    content = _message_content(message, include_text=include_text)
    return len(content.encode("utf-8")) <= min(
        call.effective_budget.max_content_bytes_per_item,
        _MAX_CONTENT_BYTES_PER_ITEM,
    )


def _json_content(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _item(
    *,
    remote_item_id: str,
    safe_display_name: str,
    content: str,
    retrieved_at: datetime,
    remote_updated_at: datetime | None,
    safe_locator: str | None = None,
) -> LiveCapabilityResultItemV1:
    return LiveCapabilityResultItemV1(
        remote_item_id=remote_item_id,
        safe_display_name=safe_display_name,
        content=content,
        content_hash=content_sha256(content),
        retrieved_at=retrieved_at,
        remote_updated_at=remote_updated_at,
        safe_locator=safe_locator_or_none(safe_locator),
    )


def _message_title(text: str) -> str:
    stripped = text.strip()
    return stripped[:120] + ("..." if len(stripped) > 120 else "") or "Slack message"


def _validate_aware_iso(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("metadata_timestamp_not_aware")


def _timestamp_in_window(
    value: str,
    window: SlackConversationSourceWindow,
) -> bool:
    from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.timestamp import (
        slack_timestamp_in_window,
    )

    return slack_timestamp_in_window(
        value=value,
        oldest=window.oldest,
        latest=window.latest,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)
