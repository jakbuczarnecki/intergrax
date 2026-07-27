# © Artur Czarnecki. All rights reserved.

"""Slack attachment → managed-file intake workflow tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationAttachmentContent,
    ConversationAttachmentFetchError,
    ConversationAttachmentReference,
    ConversationDeliveryReceipt,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from local_workspace_application.slack_companion.authorization import SlackCompanionAuthConfig
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
    build_slack_dedupe_key,
)
from local_workspace_application.slack_companion.models import (
    SlackAskClientError,
    SlackAskHttpResponse,
    SlackDedupeStatus,
    SlackManagedFileBatchItem,
    SlackManagedFileBatchResponse,
)
from local_workspace_application.slack_companion.rendering import (
    NO_WORKSPACE_AVAILABLE_TEXT,
    SELECTED_WORKSPACE_UNAVAILABLE_TEXT,
    render_attachment_intake_failed,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
    SlackWorkspaceSelection,
    slack_selection_actor_key,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    slack_attachment_intake_idempotency_key,
)

pytestmark = pytest.mark.unit


class FakeAskClient:
    def __init__(self) -> None:
        self.ask_calls = 0
        self.upload_calls: list[dict[str, Any]] = []
        self.upload_response = SlackManagedFileBatchResponse(
            batch_id="batch-1",
            workspace_id="ws-active",
            status="accepted",
            accepted_count=1,
            failed_count=0,
            items=[
                SlackManagedFileBatchItem(
                    position=0,
                    file_name="contract.pdf",
                    status="accepted",
                    input_id="in-1",
                    source_id="src-1",
                    operation_id="op-1",
                    operation_status="queued",
                )
            ],
        )
        self.upload_error: SlackAskClientError | None = None

    async def ask(self, **kwargs: Any) -> SlackAskHttpResponse:
        del kwargs
        self.ask_calls += 1
        return SlackAskHttpResponse(
            run_id="ask-1",
            workspace_id="ws-active",
            status="completed",
            question="q",
            answer="a",
        )

    async def upload_managed_files(self, **kwargs: Any) -> SlackManagedFileBatchResponse:
        self.upload_calls.append(kwargs)
        if self.upload_error is not None:
            raise self.upload_error
        return self.upload_response

    async def list_workspaces(self, **kwargs: Any) -> list[Any]:
        del kwargs
        return []


class FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_at: int | None = None
        self.fail_kind = "attachment_download_failed"
        self.bodies: dict[str, bytes] = {}

    async def fetch_attachment(
        self,
        attachment: ConversationAttachmentReference,
        *,
        max_bytes: int,
    ) -> ConversationAttachmentContent:
        del max_bytes
        self.calls.append(attachment.attachment_id)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise ConversationAttachmentFetchError(kind=self.fail_kind)
        return ConversationAttachmentContent(
            attachment_id=attachment.attachment_id,
            file_name=attachment.file_name or f"{attachment.attachment_id}.bin",
            content_type=attachment.content_type or "application/octet-stream",
            body=self.bodies.get(attachment.attachment_id, b"bytes"),
        )


class RecordingSender:
    def __init__(self) -> None:
        self.sent: list[OutboundConversationMessage] = []

    async def __call__(
        self, message: OutboundConversationMessage
    ) -> ConversationDeliveryReceipt:
        self.sent.append(message)
        return ConversationDeliveryReceipt(
            message_id=f"msg-{len(self.sent)}",
            address=message.address,
            delivered_at=datetime.now(timezone.utc),
        )


class SelectiveFailingSender(RecordingSender):
    """Fail on selected outbound texts while recording successful deliveries."""

    def __init__(
        self,
        *,
        fail_on_receiving: bool = False,
        fail_on_acceptance: bool = False,
        fail_on_intake_failed: bool = False,
        fail_exception: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.fail_on_receiving = fail_on_receiving
        self.fail_on_acceptance = fail_on_acceptance
        self.fail_on_intake_failed = fail_on_intake_failed
        self.fail_exception = fail_exception or RuntimeError(
            "slack delivery secret xoxb-leaked https://files.slack.com"
        )

    async def __call__(
        self, message: OutboundConversationMessage
    ) -> ConversationDeliveryReceipt:
        text = message.text
        if self.fail_on_receiving and text.startswith("Receiving "):
            raise self.fail_exception
        if self.fail_on_acceptance and (
            "Accepted " in text or "None of the attached files were accepted" in text
        ):
            raise self.fail_exception
        if self.fail_on_intake_failed and text == render_attachment_intake_failed():
            raise self.fail_exception
        return await super().__call__(message)


def _auth() -> SlackCompanionAuthConfig:
    return SlackCompanionAuthConfig(
        approved_team_id="T_OK",
        approved_user_id="U_OK",
        tenant_id="tenant-a",
        active_workspace_id="ws-active",
    )


def _event(
    *,
    event_id: str = "Ev-att-1",
    text: str | None = None,
    attachments: tuple[ConversationAttachmentReference, ...] = (),
) -> InboundConversationEvent:
    if not attachments:
        attachments = (
            ConversationAttachmentReference(
                attachment_id="F1",
                file_name="contract.pdf",
                content_type="application/pdf",
                size_bytes=10,
            ),
        )
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id="T_OK",
            conversation_id="Dchannel",
            thread_id="1711111.000200",
        ),
        actor=ConversationActor(actor_id="U_OK", is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text=text,
        attachments=attachments,
    )


def _workflow(
    *,
    ask: FakeAskClient | None = None,
    fetcher: FakeFetcher | None = None,
    sender: RecordingSender | None = None,
    selections: InMemorySlackWorkspaceSelectionStore | None = None,
    attachment_max_batch_files: int = 20,
    attachment_max_bytes: int = 1024,
    use_fetcher: bool = True,
) -> tuple[SlackAskWorkflow, FakeAskClient, FakeFetcher | None, RecordingSender, SlackEventDedupeRepository]:
    ask_client = ask or FakeAskClient()
    send = sender or RecordingSender()
    store = selections or InMemorySlackWorkspaceSelectionStore()
    dedupe = SlackEventDedupeRepository(InMemoryDocumentStore())
    resolved_fetcher: FakeFetcher | None
    if use_fetcher:
        resolved_fetcher = fetcher or FakeFetcher()
    else:
        resolved_fetcher = None
    workflow = SlackAskWorkflow(
        auth_config=_auth(),
        dedupe=dedupe,
        ask_client=ask_client,  # type: ignore[arg-type]
        send=send,
        selection_store=store,
        attachment_fetcher=resolved_fetcher,
        attachment_max_bytes=attachment_max_bytes,
        attachment_max_batch_files=attachment_max_batch_files,
    )
    return workflow, ask_client, resolved_fetcher, send, dedupe


@pytest.mark.asyncio
async def test_single_attachment_happy_path() -> None:
    workflow, ask, fetcher, sender, dedupe = _workflow()
    assert fetcher is not None
    event = _event()
    await workflow.handle(event)

    assert fetcher.calls == ["F1"]
    assert len(ask.upload_calls) == 1
    upload = ask.upload_calls[0]
    assert upload["tenant_id"] == "tenant-a"
    assert upload["workspace_id"] == "ws-active"
    assert upload["idempotency_key"] == slack_attachment_intake_idempotency_key(
        team_id="T_OK",
        event_id="Ev-att-1",
    )
    assert len(upload["attachments"]) == 1
    assert ask.ask_calls == 0
    assert len(sender.sent) == 2
    assert "Receiving 1 attached file" in sender.sent[0].text
    assert "Accepted 1 file for processing" in sender.sent[1].text
    assert "Processing continues asynchronously" in sender.sent[1].text
    assert "in-1" not in sender.sent[1].text
    assert "op-1" not in sender.sent[1].text
    assert sender.sent[0].address.thread_id == "1711111.000200"
    assert sender.sent[1].address.thread_id == "1711111.000200"
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_multiple_attachments_one_upload() -> None:
    workflow, ask, fetcher, _sender, _dedupe = _workflow()
    assert fetcher is not None
    attachments = tuple(
        ConversationAttachmentReference(attachment_id=f"F{i}", file_name=f"{i}.pdf")
        for i in range(1, 4)
    )
    await workflow.handle(_event(attachments=attachments))
    assert fetcher.calls == ["F1", "F2", "F3"]
    assert len(ask.upload_calls) == 1
    assert [a.attachment_id for a in ask.upload_calls[0]["attachments"]] == [
        "F1",
        "F2",
        "F3",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", ["workspaces", "What is leave policy?"])
async def test_attachment_plus_caption_does_not_run_command_or_ask(caption: str) -> None:
    workflow, ask, fetcher, sender, _dedupe = _workflow()
    assert fetcher is not None
    await workflow.handle(_event(text=caption))
    assert fetcher.calls == ["F1"]
    assert len(ask.upload_calls) == 1
    assert ask.ask_calls == 0
    joined = "\n".join(m.text for m in sender.sent)
    assert "Available workspaces" not in joined
    assert "Employees" not in joined


@pytest.mark.asyncio
async def test_dedupe_before_fetch_and_single_upload() -> None:
    fetcher = FakeFetcher()
    ask = FakeAskClient()
    sender = RecordingSender()
    workflow, _, _, _, _dedupe = _workflow(ask=ask, fetcher=fetcher, sender=sender)
    event = _event()
    await workflow.handle(event)
    await workflow.handle(event)
    assert fetcher.calls == ["F1"]
    assert len(ask.upload_calls) == 1
    assert sum(1 for m in sender.sent if "Accepted" in m.text) == 1


@pytest.mark.asyncio
async def test_too_many_files() -> None:
    attachments = tuple(
        ConversationAttachmentReference(attachment_id=f"F{i}") for i in range(3)
    )
    workflow, ask, fetcher, sender, dedupe = _workflow(attachment_max_batch_files=2)
    assert fetcher is not None
    await workflow.handle(_event(attachments=attachments))
    assert fetcher.calls == []
    assert ask.upload_calls == []
    assert "Too many files" in sender.sent[0].text
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_no_active_workspace() -> None:
    auth = SlackCompanionAuthConfig(
        approved_team_id="T_OK",
        approved_user_id="U_OK",
        tenant_id="tenant-a",
        active_workspace_id="ws-active",
    )
    ask = FakeAskClient()
    fetcher = FakeFetcher()
    sender = RecordingSender()
    selections = InMemorySlackWorkspaceSelectionStore()
    actor_key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selections.suppress_configured(actor_key)
    dedupe = SlackEventDedupeRepository(InMemoryDocumentStore())
    workflow = SlackAskWorkflow(
        auth_config=auth,
        dedupe=dedupe,
        ask_client=ask,  # type: ignore[arg-type]
        send=sender,
        selection_store=selections,
        attachment_fetcher=fetcher,
        attachment_max_bytes=1024,
        attachment_max_batch_files=20,
    )
    await workflow.handle(_event())
    assert fetcher.calls == []
    assert ask.upload_calls == []
    assert sender.sent[0].text == NO_WORKSPACE_AVAILABLE_TEXT


@pytest.mark.asyncio
async def test_missing_fetcher_keeps_text_ask() -> None:
    ask = FakeAskClient()
    sender = RecordingSender()
    workflow, _, _, _, _ = _workflow(ask=ask, sender=sender, use_fetcher=False)
    await workflow.handle(_event())
    assert ask.upload_calls == []
    assert "not available" in sender.sent[0].text.casefold()

    # text ask still works
    text_event = InboundConversationEvent(
        event_id="Ev-ask",
        address=ConversationAddress(
            installation_id="T_OK",
            conversation_id="Dchannel",
            thread_id="1711111.000200",
        ),
        actor=ConversationActor(actor_id="U_OK", is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text="What is leave policy?",
    )
    await workflow.handle(text_event)
    assert ask.ask_calls == 1


@pytest.mark.asyncio
async def test_download_failure_zero_upload() -> None:
    fetcher = FakeFetcher()
    fetcher.fail_at = 2
    fetcher.fail_kind = "attachment_download_failed"
    ask = FakeAskClient()
    sender = RecordingSender()
    attachments = tuple(
        ConversationAttachmentReference(attachment_id=f"F{i}") for i in range(1, 4)
    )
    workflow, _, _, _, dedupe = _workflow(ask=ask, fetcher=fetcher, sender=sender)
    await workflow.handle(_event(attachments=attachments))
    assert fetcher.calls == ["F1", "F2"]
    assert ask.upload_calls == []
    joined = "\n".join(m.text for m in sender.sent)
    assert "could not be received from Slack" in joined
    assert "xoxb" not in joined
    assert "files.slack.com" not in joined
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.FAILED


@pytest.mark.asyncio
async def test_file_too_large_specific_message() -> None:
    fetcher = FakeFetcher()
    fetcher.fail_at = 1
    fetcher.fail_kind = "attachment_too_large"
    sender = RecordingSender()
    workflow, ask, _, _, _ = _workflow(fetcher=fetcher, sender=sender)
    await workflow.handle(_event())
    assert ask.upload_calls == []
    assert "too large" in sender.sent[-1].text.casefold()


@pytest.mark.asyncio
async def test_partial_acceptance_safe_summary() -> None:
    ask = FakeAskClient()
    ask.upload_response = SlackManagedFileBatchResponse(
        batch_id="batch-partial",
        workspace_id="ws-active",
        status="partial",
        accepted_count=2,
        failed_count=1,
        items=[
            SlackManagedFileBatchItem(
                position=0,
                file_name="contract.pdf",
                status="accepted",
                input_id="in-1",
                source_id="src-1",
                operation_id="op-1",
            ),
            SlackManagedFileBatchItem(
                position=1,
                file_name="notes.txt",
                status="accepted",
                input_id="in-2",
            ),
            SlackManagedFileBatchItem(
                position=2,
                file_name="invalid-file",
                status="failed",
                error_code="managed_file_name_unsafe",
            ),
        ],
    )
    sender = RecordingSender()
    workflow, _, _, _, _ = _workflow(ask=ask, sender=sender)
    await workflow.handle(_event())
    text = sender.sent[-1].text
    assert "Accepted 2 of 3" in text
    assert "contract.pdf" in text
    assert "invalid-file" in text
    assert "Processing continues asynchronously" in text
    assert "batch-partial" not in text
    assert "in-1" not in text
    assert "op-1" not in text
    assert "managed_file_name_unsafe" not in text


@pytest.mark.asyncio
async def test_all_failed_acceptance() -> None:
    ask = FakeAskClient()
    ask.upload_response = SlackManagedFileBatchResponse(
        batch_id="batch-fail",
        workspace_id="ws-active",
        status="failed",
        accepted_count=0,
        failed_count=1,
        items=[
            SlackManagedFileBatchItem(
                position=0,
                file_name="bad.bin",
                status="failed",
                error_code="managed_file_empty",
            )
        ],
    )
    sender = RecordingSender()
    workflow, _, _, _, dedupe = _workflow(ask=ask, sender=sender)
    await workflow.handle(_event())
    text = sender.sent[-1].text
    assert "None of the attached files were accepted" in text
    assert "Processing continues asynchronously" not in text
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_successful_intake_summary_delivery_failure_keeps_completed() -> None:
    sender = SelectiveFailingSender(fail_on_acceptance=True)
    workflow, ask, fetcher, _, dedupe = _workflow(sender=sender)
    assert fetcher is not None
    await workflow.handle(_event())
    assert fetcher.calls == ["F1"]
    assert len(ask.upload_calls) == 1
    assert ask.ask_calls == 0
    assert len(sender.sent) == 1
    assert "Receiving 1 attached file" in sender.sent[0].text
    assert render_attachment_intake_failed() not in {m.text for m in sender.sent}
    joined = "\n".join(m.text for m in sender.sent)
    assert "xoxb" not in joined
    assert "files.slack.com" not in joined
    assert "secret" not in joined
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_ack_delivery_failure_does_not_block_intake() -> None:
    sender = SelectiveFailingSender(fail_on_receiving=True)
    workflow, ask, fetcher, _, dedupe = _workflow(sender=sender)
    assert fetcher is not None
    await workflow.handle(_event())
    assert fetcher.calls == ["F1"]
    assert len(ask.upload_calls) == 1
    assert len(sender.sent) == 1
    assert "Accepted 1 file for processing" in sender.sent[0].text
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_upload_failure_marks_dedupe_failed() -> None:
    ask = FakeAskClient()
    ask.upload_error = SlackAskClientError(kind="http_503")
    sender = RecordingSender()
    workflow, _, fetcher, _, dedupe = _workflow(ask=ask, sender=sender)
    assert fetcher is not None
    await workflow.handle(_event())
    assert len(ask.upload_calls) == 1
    assert sender.sent[-1].text == render_attachment_intake_failed()
    assert not any("Accepted" in m.text for m in sender.sent)
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.FAILED


@pytest.mark.asyncio
async def test_upload_failure_plus_error_delivery_failure_keeps_failed() -> None:
    ask = FakeAskClient()
    ask.upload_error = SlackAskClientError(kind="http_503")
    sender = SelectiveFailingSender(fail_on_intake_failed=True)
    workflow, _, fetcher, _, dedupe = _workflow(ask=ask, sender=sender)
    assert fetcher is not None
    await workflow.handle(_event())
    assert len(ask.upload_calls) == 1
    assert render_attachment_intake_failed() not in {m.text for m in sender.sent}
    joined = "\n".join(m.text for m in sender.sent)
    assert "xoxb" not in joined
    assert "http_503" not in joined
    assert "secret" not in joined
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-att-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.FAILED


@pytest.mark.asyncio
async def test_selected_workspace_404_clears_selection() -> None:
    ask = FakeAskClient()
    ask.upload_error = SlackAskClientError(kind="http_404")
    selections = InMemorySlackWorkspaceSelectionStore()
    actor_key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selections.set(
        actor_key,
        SlackWorkspaceSelection(
            workspace_id="ws-selected",
            workspace_name="Selected",
        ),
    )
    sender = RecordingSender()
    workflow, _, _, _, _ = _workflow(ask=ask, sender=sender, selections=selections)
    await workflow.handle(_event())
    assert selections.get(actor_key) is None
    assert sender.sent[-1].text == SELECTED_WORKSPACE_UNAVAILABLE_TEXT
    assert "http_404" not in sender.sent[-1].text


def test_idempotency_key_stable_and_opaque() -> None:
    a = slack_attachment_intake_idempotency_key(team_id="T_OK", event_id="Ev1")
    b = slack_attachment_intake_idempotency_key(team_id="T_OK", event_id="Ev1")
    c = slack_attachment_intake_idempotency_key(team_id="T_OK", event_id="Ev2")
    d = slack_attachment_intake_idempotency_key(team_id="T_OTHER", event_id="Ev1")
    assert a == b
    assert a != c
    assert a != d
    assert a.startswith("slack-attachment:v1:")
    assert "Ev1" not in a
    assert "T_OK" not in a
