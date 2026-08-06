from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationExactMessageResult,
    SlackConversationFileReference,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessageChanged,
    SlackConversationMessagePage,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    parse_slack_ts,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    encode_slack_conversation_scope_id,
)
from intergrax.runtime.vendor_knowledge.live import (
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    LiveCapabilityExecutionContextV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
    ValidatedLiveCapabilityCallV1,
)
from intergrax.runtime.vendor_knowledge.live.contracts import (
    LiveCapabilityExecutionResultV1,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph import (
    build_msgraph_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    publish_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.slack import (
    SLACK_CONVERSATION_LIST_CAPABILITY_ID,
    SLACK_CONVERSATION_READ_CAPABILITY_ID,
    SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
    SlackConversationListLiveHandlerV1,
    SlackConversationListLiveRequestV1,
    SlackConversationReadLiveHandlerV1,
    SlackConversationReadLiveRequestV1,
    SlackConversationThreadReadLiveHandlerV1,
    SlackConversationThreadReadLiveRequestV1,
    build_slack_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.slack.conversation import (
    SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF,
    SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF,
    SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF,
    SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF,
    SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF,
    SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_CONVERSATION_ID = "C01234567"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_REPLY_TS = "1704153601.000001"
_OTHER_REPLY_TS = "1704153602.000001"
_OTHER_ROOT_TS = "1704153599.000001"
_NOW = datetime(2026, 8, 6, 4, 0, tzinfo=UTC)


def _scope_id() -> str:
    return encode_slack_conversation_scope_id(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        oldest=_OLDEST,
        latest=_LATEST,
    )


def _message(
    *,
    message_ts: str,
    text: str,
    root_thread_ts: str | None = None,
    reply_count: int | None = None,
    files: tuple[SlackConversationFileReference, ...] = (),
) -> SlackConversationMessage:
    return SlackConversationMessage(
        conversation_id=_CONVERSATION_ID,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id="U111",
        text=text,
        subtype=None,
        created_at=parse_slack_ts(message_ts) or _NOW,
        edited_at=None,
        reply_count=reply_count,
        files=files,
        provider_metadata={},
    )


class _FakeSlackBackend:
    def __init__(self) -> None:
        self.history_calls: list[dict[str, Any]] = []
        self.thread_calls: list[dict[str, Any]] = []
        self.exact_calls: list[dict[str, Any]] = []
        self.history_page = SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(_message(message_ts=_ROOT_TS, text="root", reply_count=2),),
            next_cursor="history-page-2",
        )
        self.thread_page = SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(
                _message(
                    message_ts=_REPLY_TS,
                    text="reply",
                    root_thread_ts=_ROOT_TS,
                ),
            ),
            next_cursor="reply-page-2",
        )
        self.exact_result: SlackConversationExactMessageResult | None = None
        self.raise_changed = False

    async def read_conversation_history_page(self, **kwargs: Any):
        self.history_calls.append(kwargs)
        return self.history_page

    async def read_recent_conversation_messages_page(self, **kwargs: Any):
        self.history_calls.append(kwargs)
        return self.history_page.model_copy(
            update={
                "oldest": kwargs["window"].oldest,
                "latest": kwargs["window"].latest,
            }
        )

    async def read_thread_replies_page(self, **kwargs: Any):
        self.thread_calls.append(kwargs)
        return self.thread_page

    async def read_exact_message(self, **kwargs: Any):
        self.exact_calls.append(kwargs)
        if self.raise_changed:
            raise SlackConversationMessageChanged()
        return self.exact_result or SlackConversationExactMessageResult(
            found=True,
            message=_message(message_ts=kwargs["message_ts"], text="exact"),
        )

    async def list_accessible_conversations_page(self, **kwargs: Any):
        raise NotImplementedError

    async def read_file_info(self, **kwargs: Any):
        raise NotImplementedError

    async def start(self, handler) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, message):
        raise NotImplementedError

    def health(self):
        return None


def _integration(backend: _FakeSlackBackend) -> SlackConversationChannelIntegration:
    return SlackConversationChannelIntegration.from_backend(
        backend,
        enabled=True,
        config=SlackConversationChannelIntegrationConfig(
            enabled=True,
            app_token="xapp-test-token-value",
            bot_token="xoxb-test-token-value",
        ),
    )


def _call(
    capability_id: str,
    request: object,
    *,
    budget_items: int = 15,
) -> ValidatedLiveCapabilityCallV1:
    return ValidatedLiveCapabilityCallV1(
        call_id="call-slack-1",
        capability_id=capability_id,
        contract_version="1",
        connection_ref="connection-1",
        live_access_binding_id="binding-1",
        remote_resource_id=_scope_id(),
        validated_request=request,
        effective_budget=EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=10_000,
            max_result_items=budget_items,
            max_result_bytes=131_072,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=budget_items,
            max_provider_page_size=budget_items,
            max_content_bytes_per_item=16_384,
        ),
        audience_context_ref=None,
        provider_id="slack",
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind="slack_conversation",
    )


def _context() -> LiveCapabilityExecutionContextV1:
    return LiveCapabilityExecutionContextV1(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=100,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )


async def test_slack_requests_registration_and_combined_publication() -> None:
    with pytest.raises(ValidationError):
        SlackConversationListLiveRequestV1(page_size=0)
    with pytest.raises(ValidationError):
        SlackConversationListLiveRequestV1(page_size=16)
    assert SlackConversationListLiveRequestV1().page_size == 1
    assert SlackConversationListLiveRequestV1(page_size=15).page_size == 15
    inverted_window = SlackConversationListLiveRequestV1(
        oldest=_LATEST, latest=_OLDEST
    )
    assert inverted_window.oldest == _LATEST
    assert inverted_window.latest == _OLDEST
    with pytest.raises(ValidationError):
        SlackConversationListLiveRequestV1.model_validate(
            {"conversation_id": _CONVERSATION_ID}
        )
    with pytest.raises(ValidationError):
        SlackConversationThreadReadLiveRequestV1(
            root_message_ts=_ROOT_TS,
            page_size=16,
        )
    with pytest.raises(ValidationError):
        SlackConversationReadLiveRequestV1.model_validate(
            {"message_ts": _ROOT_TS, "max_chars": 2_000_001}
        )
    with pytest.raises(ValidationError):
        SlackConversationReadLiveRequestV1.model_validate(
            {"message_ts": _ROOT_TS, "root_thread_ts": "bad", "max_chars": 10}
        )

    bundles = build_slack_live_registration_bundles()
    assert tuple(bundle.descriptor.capability_id for bundle in bundles) == (
        SLACK_CONVERSATION_LIST_CAPABILITY_ID,
        SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
        SLACK_CONVERSATION_READ_CAPABILITY_ID,
    )
    published = publish_live_registration_bundles(bundles)
    assert (
        published.schemas.resolve_request(
            SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF, "1"
        )
        is SlackConversationListLiveRequestV1
    )
    assert (
        published.schemas.resolve_request(
            SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF, "1"
        )
        is SlackConversationThreadReadLiveRequestV1
    )
    assert (
        published.schemas.resolve_request(
            SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF, "1"
        )
        is SlackConversationReadLiveRequestV1
    )
    assert published.schemas.resolve_result(
        SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF, "1"
    )
    assert published.schemas.resolve_result(
        SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF, "1"
    )
    assert published.schemas.resolve_result(
        SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF, "1"
    )
    with pytest.raises(ValueError, match="duplicate_live_capability_identity"):
        publish_live_registration_bundles(bundles + bundles)
    combined = publish_live_registration_bundles(
        (*build_msgraph_live_registration_bundles(), *bundles)
    )
    assert len(combined.descriptors) == 8
    assert {
        "vendor.ms365_graph.drive.list",
        "vendor.ms365_graph.mail.list",
        "vendor.ms365_graph.teams_channel.list",
        "vendor.ms365_graph.teams_chat.list",
        "vendor.ms365_graph.calendar.list",
    }.issubset({key[2] for key in combined.descriptors})


async def test_slack_list_window_validation_is_bounded_before_provider_call() -> None:
    backend = _FakeSlackBackend()
    narrowed = await SlackConversationListLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_LIST_CAPABILITY_ID,
            SlackConversationListLiveRequestV1(
                oldest=_ROOT_TS,
                latest=_REPLY_TS,
            ),
        ),
        context=_context(),
    )
    assert narrowed.error_code is None
    assert backend.history_calls[0]["window"].oldest == _ROOT_TS
    assert backend.history_calls[0]["window"].latest == _REPLY_TS

    backend.history_calls.clear()
    outside_binding = await SlackConversationListLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_LIST_CAPABILITY_ID,
            SlackConversationListLiveRequestV1(
                oldest="1700000000.000001",
                latest="1800000000.000001",
            ),
        ),
        context=_context(),
    )
    assert outside_binding.error_code is None
    assert backend.history_calls[0]["window"].oldest == _OLDEST
    assert backend.history_calls[0]["window"].latest == _LATEST

    backend.history_calls.clear()
    invalid = await SlackConversationListLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_LIST_CAPABILITY_ID,
            SlackConversationListLiveRequestV1(oldest=_LATEST, latest=_OLDEST),
        ),
        context=_context(),
    )
    assert invalid.error_code == "live_request_invalid"
    assert backend.history_calls == []


async def test_slack_list_is_one_call_bounded_text_and_truncated() -> None:
    backend = _FakeSlackBackend()
    result = await SlackConversationListLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_LIST_CAPABILITY_ID, SlackConversationListLiveRequestV1()
        ),
        context=_context(),
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert result.item_count == 1
    assert len(backend.history_calls) == 1
    assert backend.history_calls[0]["cursor"] is None
    assert backend.history_calls[0]["limit"] == 1
    assert '"text":"root"' in result.items[0].content
    assert '"thread_root_ts":null' in result.items[0].content
    assert "history-page-2" not in result.items[0].content


async def test_slack_list_rejects_active_reply() -> None:
    backend = _FakeSlackBackend()
    backend.history_page = backend.history_page.model_copy(
        update={
            "items": (
                _message(
                    message_ts=_REPLY_TS,
                    text="reply",
                    root_thread_ts=_ROOT_TS,
                ),
            ),
            "next_cursor": None,
        }
    )
    result = await SlackConversationListLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_LIST_CAPABILITY_ID, SlackConversationListLiveRequestV1()
        ),
        context=_context(),
    )
    assert result.error_code == "live_provider_contract_violation"


async def test_slack_thread_reads_one_provider_page_and_rejects_wrong_thread() -> None:
    backend = _FakeSlackBackend()
    result = await SlackConversationThreadReadLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
            SlackConversationThreadReadLiveRequestV1(
                root_message_ts=_ROOT_TS,
                page_size=15,
            ),
        ),
        context=_context(),
    )
    assert result.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert result.item_count == 1
    assert len(backend.thread_calls) == 1
    assert backend.thread_calls[0]["conversation_id"] == _CONVERSATION_ID
    assert backend.thread_calls[0]["cursor"] is None
    assert backend.thread_calls[0]["limit"] == 15
    assert '"text":"reply"' in result.items[0].content

    backend.thread_page = backend.thread_page.model_copy(
        update={
            "items": (
                _message(
                    message_ts=_OTHER_REPLY_TS,
                    text="wrong",
                    root_thread_ts=_OTHER_ROOT_TS,
                ),
            ),
            "next_cursor": None,
        }
    )
    rejected = await SlackConversationThreadReadLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
            SlackConversationThreadReadLiveRequestV1(root_message_ts=_ROOT_TS),
        ),
        context=_context(),
    )
    assert rejected.error_code == "live_provider_contract_violation"


async def test_slack_exact_read_maps_found_not_found_changed_and_bounds_content() -> (
    None
):
    backend = _FakeSlackBackend()
    backend.exact_result = SlackConversationExactMessageResult(
        found=True,
        message=_message(
            message_ts=_ROOT_TS,
            text="exact body",
            files=(
                SlackConversationFileReference(
                    file_id="F123",
                    safe_file_name="safe.txt",
                ),
            ),
        ),
    )
    result = await SlackConversationReadLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_READ_CAPABILITY_ID,
            SlackConversationReadLiveRequestV1(
                message_ts=_ROOT_TS,
                max_chars=100,
            ),
            budget_items=1,
        ),
        context=_context(),
    )
    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.item_count == 1
    assert '"text":"exact body"' in result.items[0].content
    assert "private" not in result.items[0].content
    assert len(backend.exact_calls) == 1
    assert backend.exact_calls[0]["max_chars_per_message"] == 100

    backend.exact_result = SlackConversationExactMessageResult(found=False)
    not_found = await SlackConversationReadLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_READ_CAPABILITY_ID,
            SlackConversationReadLiveRequestV1(message_ts=_ROOT_TS),
            budget_items=1,
        ),
        context=_context(),
    )
    assert not_found.error_code == "live_provider_not_found"

    backend.raise_changed = True
    changed = await SlackConversationReadLiveHandlerV1().execute(
        integration=_integration(backend),
        call=_call(
            SLACK_CONVERSATION_READ_CAPABILITY_ID,
            SlackConversationReadLiveRequestV1(
                message_ts=_ROOT_TS,
                expected_revision="a" * 64,
            ),
            budget_items=1,
        ),
        context=_context(),
    )
    assert changed.error_code == "live_provider_temporarily_unavailable"


async def test_slack_shared_executor_resolves_once_and_returns_receipt_only() -> None:
    from local_workspace_application.workspaces.hybrid_ask_execution import (
        LiveCapabilityExecutorV1,
    )
    from local_workspace_application.workspaces.hybrid_ask_policy import (
        ExecutableLiveCallV1,
        ResolvedLiveResourceScopeV1,
    )

    backend = _FakeSlackBackend()
    integration = _integration(backend)

    class _Resolver:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def resolve(self, **kwargs: object) -> object:
            self.calls.append(integration)
            return integration

    resolver = _Resolver()
    validated = _call(
        SLACK_CONVERSATION_LIST_CAPABILITY_ID,
        SlackConversationListLiveRequestV1(),
        budget_items=1,
    )
    executable = ExecutableLiveCallV1(
        **validated.model_dump(exclude={"validated_request"}),
        validated_request=validated.validated_request,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(
            remote_resource_id=_scope_id(),
            scope_token=None,
        ),
    )
    executor = LiveCapabilityExecutorV1(
        published_registration=publish_live_registration_bundles(
            build_slack_live_registration_bundles()
        ),
        integration_resolver=resolver,
        id_factory=lambda: "receipt-1",
    )
    result = await executor.execute(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        call=executable,
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.RECEIPT_ONLY,
    )

    assert isinstance(result, LiveCapabilityExecutionResultV1)
    assert result.error_code is None
    assert result.receipt is not None
    assert result.receipt.receipt_id == "receipt-1"
    assert result.items
    assert len(resolver.calls) == 1
    assert resolver.calls[0] is integration
    assert len(backend.history_calls) == 1
    assert result.receipt.error_code is None
