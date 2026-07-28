# © Artur Czarnecki. All rights reserved.

"""Tests for the conversational interaction planner."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult

from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningAttachment,
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    LocalFileReferenceExtractedObject,
    MessageTextEvidenceSpan,
    WebUrlExtractedObject,
    WorkspaceActivatePlannedAction,
    WorkspaceAskPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    ConversationPlanningError,
    ConversationPlanningErrorCode,
    PlanRequestValidationError,
    validate_plan_against_request,
)
from local_workspace_application.conversation.interaction_prompt import (
    build_planning_messages,
    system_prompt_contains_required_rules,
)


def _magazyn_target() -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn")


def _span(message: str, substring: str) -> MessageTextEvidenceSpan:
    start = message.index(substring)
    return MessageTextEvidenceSpan(
        source="message_text",
        start=start,
        end=start + len(substring),
        text=substring,
    )


def _web_object(object_id: str, message: str, url: str) -> WebUrlExtractedObject:
    return WebUrlExtractedObject(
        object_id=object_id,
        object_type="web_url",
        value=url,
        evidence=_span(message, url),
    )


def _local_object(object_id: str, message: str, path: str) -> LocalFileReferenceExtractedObject:
    return LocalFileReferenceExtractedObject(
        object_id=object_id,
        object_type="local_file_reference",
        reference_kind="file",
        value=path,
        evidence=_span(message, path),
    )


MIXED_ROUTING_MESSAGE = (
    "ten adres https://cenniki.pl wrzuć do workspace numer 1, "
    r"a pliki C:\cenniki\hurt.xlsx i C:\cenniki\detal.xlsx "
    "dodaj do workspace numer 2"
)


def _mixed_routing_plan() -> ConversationInteractionPlan:
    message = MIXED_ROUTING_MESSAGE
    return ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            _web_object("url-1", message, "https://cenniki.pl"),
            _local_object("local-1", message, r"C:\cenniki\hurt.xlsx"),
            _local_object("local-2", message, r"C:\cenniki\detal.xlsx"),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                source_object_ids=("url-1",),
            ),
            KnowledgeAddSourcesPlannedAction(
                action_id="a2",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                source_object_ids=("local-1", "local-2"),
            ),
        ),
    )


def _mixed_routing_request() -> ConversationPlanningRequest:
    return ConversationPlanningRequest(
        message_text=MIXED_ROUTING_MESSAGE,
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="ws-1", name="finanse", is_active=True),
            ConversationPlanningWorkspace(workspace_id="ws-2", name="magazyn", is_active=False),
        ),
        active_workspace_id="ws-1",
    )


class RecordingPlannerAdapter(LLMAdapter):
    provider = "recording"
    model = "recording"

    def __init__(
        self,
        *,
        structured_outputs: Sequence[Any] | None = None,
        supports_structured: bool = True,
        provider_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._structured_outputs = list(structured_outputs or [])
        self._supports_structured = supports_structured
        self._provider_error = provider_error
        self.recorded_messages: list[list[ChatMessage]] = []
        self.call_count = 0

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def supports_structured_output(self) -> bool:
        return self._supports_structured

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ):
        return build_adapter_response(content="ok")

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        self.call_count += 1
        self.recorded_messages.append(list(messages))
        if self._provider_error is not None:
            raise self._provider_error
        if not self._structured_outputs:
            raise RuntimeError("no structured output configured")
        item = self._structured_outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, output_model):
            parsed = item
        elif isinstance(item, dict):
            parsed = output_model.model_validate(item)
        else:
            raise TypeError("unsupported structured output fixture")
        return LLMStructuredResult(parsed=parsed, response=build_adapter_response(content=""))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_routing_plan() -> None:
    adapter = RecordingPlannerAdapter(structured_outputs=[_mixed_routing_plan()])
    planner = ConversationInteractionPlanner(adapter)
    plan = await planner.plan(_mixed_routing_request())
    assert len(plan.objects) == 3
    assert len(plan.actions) == 2
    add_sources = [a for a in plan.actions if a.action_type == "knowledge.add_sources"]
    assert len(add_sources) == 2
    url_action = next(a for a in add_sources if a.workspace.value == "1")
    local_action = next(a for a in add_sources if a.workspace.value == "2")
    assert url_action.source_object_ids == ("url-1",)
    assert set(local_action.source_object_ids) == {"local-1", "local-2"}
    assert all(action.action_type != "workspace.activate" for action in plan.actions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_workspace_activation() -> None:
    activate_plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        actions=(
            WorkspaceActivatePlannedAction(
                action_id="act-1",
                action_type="workspace.activate",
                workspace=_magazyn_target(),
                evidence_quotes=("przełącz mnie na workspace magazyn",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[activate_plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(
        message_text="przełącz mnie na workspace magazyn",
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="ws-2", name="magazyn", is_active=False),
        ),
    )
    plan = await planner.plan(request)
    assert plan.actions[0].action_type == "workspace.activate"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_attachments_and_text_single_plan() -> None:
    message = (
        "załącz pliki i https://docs.example.com/page "
        r"oraz c:\data\report.docx do workspace magazyn"
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            _web_object("url-1", message, "https://docs.example.com/page"),
            _local_object("local-1", message, r"c:\data\report.docx"),
        ),
        actions=(
            KnowledgeAddAttachmentsPlannedAction(
                action_id="att-1",
                action_type="knowledge.add_attachments",
                workspace=_magazyn_target(),
                attachment_ids=("file-a", "file-b"),
            ),
            KnowledgeAddSourcesPlannedAction(
                action_id="sources-1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1", "local-1"),
            ),
        ),
    )
    request = ConversationPlanningRequest(
        message_text=message,
        attachments=(
            ConversationPlanningAttachment(attachment_id="file-a", file_name="a.pdf"),
            ConversationPlanningAttachment(attachment_id="file-b", file_name="b.pdf"),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[plan])
    planner = ConversationInteractionPlanner(adapter)
    result = await planner.plan(request)
    assert len(result.actions) == 2
    assert len(result.objects) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_unknown_attachment() -> None:
    bad_plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        actions=(
            KnowledgeAddAttachmentsPlannedAction(
                action_id="att-1",
                action_type="knowledge.add_attachments",
                workspace=_magazyn_target(),
                attachment_ids=("unknown-att",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_plan, bad_plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(
        message_text="dodaj plik do magazyn",
        attachments=(ConversationPlanningAttachment(attachment_id="real-att", file_name="x.pdf"),),
    )
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output
    assert adapter.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_source_object_id_rejected() -> None:
    message = "dodaj https://www.cenniki.pl do magazyn"
    bad_payload = {
        "plan_version": "2",
        "response_mode": "aggregate",
        "objects": [
            {
                "object_id": "url-1",
                "object_type": "web_url",
                "value": "https://www.cenniki.pl",
                "evidence": {
                    "source": "message_text",
                    "start": 6,
                    "end": 27,
                    "text": "https://www.cenniki.pl",
                },
            }
        ],
        "actions": [
            {
                "action_id": "a1",
                "action_type": "knowledge.add_sources",
                "workspace": {"kind": "name", "value": "magazyn"},
                "source_object_ids": ["missing-obj"],
            }
        ],
    }
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_payload, bad_payload])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output
    assert adapter.call_count == 2


@pytest.mark.unit
def test_evidence_span_out_of_range_rejected() -> None:
    message = "dodaj https://example.com do magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            WebUrlExtractedObject(
                object_id="url-1",
                object_type="web_url",
                value="https://example.com",
                evidence=MessageTextEvidenceSpan(
                    source="message_text",
                    start=0,
                    end=len(message) + 5,
                    text="https://example.com",
                ),
            ),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(PlanRequestValidationError, match="out of range"):
        validate_plan_against_request(plan, request)


@pytest.mark.unit
def test_evidence_text_mismatch_rejected() -> None:
    message = "dodaj https://example.com do magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            WebUrlExtractedObject(
                object_id="url-1",
                object_type="web_url",
                value="https://example.com",
                evidence=MessageTextEvidenceSpan(
                    source="message_text",
                    start=6,
                    end=25,
                    text="https://wrong.com",
                ),
            ),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(PlanRequestValidationError, match="does not match message slice"):
        validate_plan_against_request(plan, request)


@pytest.mark.unit
def test_value_evidence_mismatch_rejected() -> None:
    message = "dodaj https://example.com do magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            WebUrlExtractedObject(
                object_id="url-1",
                object_type="web_url",
                value="https://different.com",
                evidence=MessageTextEvidenceSpan(
                    source="message_text",
                    start=6,
                    end=25,
                    text="https://example.com",
                ),
            ),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(PlanRequestValidationError, match="value does not match evidence"):
        validate_plan_against_request(plan, request)


@pytest.mark.unit
def test_unused_object_rejected() -> None:
    message = "dodaj https://example.com i https://other.com do magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            _web_object("url-1", message, "https://example.com"),
            _web_object("url-2", message, "https://other.com"),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(PlanRequestValidationError, match="unused extracted object"):
        validate_plan_against_request(plan, request)


@pytest.mark.unit
def test_same_object_in_two_actions_allowed() -> None:
    message = "dodaj https://example.com do magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(_web_object("url-1", message, "https://example.com"),),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                source_object_ids=("url-1",),
            ),
            KnowledgeAddSourcesPlannedAction(
                action_id="a2",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                source_object_ids=("url-1",),
            ),
        ),
    )
    request = ConversationPlanningRequest(message_text=message)
    validate_plan_against_request(plan, request)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_target_workspace_without_activation() -> None:
    message = "dodaj https://example.com do workspace magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(_web_object("url-1", message, "https://example.com"),),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    result = await planner.plan(request)
    assert all(action.action_type != "workspace.activate" for action in result.actions)
    assert any(action.action_type == "knowledge.add_sources" for action in result.actions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_url_in_question_routes_to_workspace_ask() -> None:
    message = "co sądzisz o https://example.com?"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        actions=(
            WorkspaceAskPlannedAction(
                action_id="ask-1",
                action_type="workspace.ask",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.active),
                question=message,
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    result = await planner.plan(request)
    assert result.actions[0].action_type == "workspace.ask"
    assert not result.objects
    assert not any(action.action_type == "knowledge.add_sources" for action in result.actions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_unknown_evidence_quote() -> None:
    message = "dodaj https://www.cenniki.pl do magazyn"
    bad_plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(_web_object("url-1", message, "https://www.cenniki.pl"),),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
                evidence_quotes=("user never said this",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_plan, bad_plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_invalid_dependency() -> None:
    message = "dodaj https://www.cenniki.pl do magazyn"
    adapter = RecordingPlannerAdapter(
        structured_outputs=[
            {
                "plan_version": "2",
                "response_mode": "aggregate",
                "objects": [
                    {
                        "object_id": "url-1",
                        "object_type": "web_url",
                        "value": "https://www.cenniki.pl",
                        "evidence": {
                            "source": "message_text",
                            "start": 6,
                            "end": 27,
                            "text": "https://www.cenniki.pl",
                        },
                    }
                ],
                "actions": [
                    {
                        "action_id": "a1",
                        "action_type": "knowledge.add_sources",
                        "workspace": {"kind": "name", "value": "magazyn"},
                        "source_object_ids": ["url-1"],
                        "depends_on": ["missing"],
                    }
                ],
            },
            {
                "plan_version": "2",
                "response_mode": "aggregate",
                "objects": [
                    {
                        "object_id": "url-1",
                        "object_type": "web_url",
                        "value": "https://www.cenniki.pl",
                        "evidence": {
                            "source": "message_text",
                            "start": 6,
                            "end": 27,
                            "text": "https://www.cenniki.pl",
                        },
                    }
                ],
                "actions": [
                    {
                        "action_id": "a1",
                        "action_type": "knowledge.add_sources",
                        "workspace": {"kind": "name", "value": "magazyn"},
                        "source_object_ids": ["url-1"],
                        "depends_on": ["missing"],
                    }
                ],
            },
        ]
    )
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output
    assert adapter.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_structured_output_unsupported() -> None:
    adapter = RecordingPlannerAdapter(supports_structured=False)
    planner = ConversationInteractionPlanner(adapter)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(ConversationPlanningRequest(message_text="hello"))
    assert (
        exc_info.value.code
        == ConversationPlanningErrorCode.conversation_planner_structured_output_unsupported
    )
    assert adapter.call_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_attempt_success() -> None:
    message = MIXED_ROUTING_MESSAGE
    invalid = {
        "plan_version": "2",
        "response_mode": "aggregate",
        "objects": [
            {
                "object_id": "url-1",
                "object_type": "web_url",
                "value": "https://cenniki.pl",
                "evidence": {
                    "source": "message_text",
                    "start": 0,
                    "end": 5,
                    "text": "wrong",
                },
            }
        ],
        "actions": [
            {
                "action_id": "a1",
                "action_type": "knowledge.add_sources",
                "workspace": {"kind": "ordinal", "value": "1"},
                "source_object_ids": ["url-1"],
            }
        ],
    }
    adapter = RecordingPlannerAdapter(structured_outputs=[invalid, _mixed_routing_plan()])
    planner = ConversationInteractionPlanner(adapter)
    plan = await planner.plan(_mixed_routing_request())
    assert len(plan.objects) == 3
    assert adapter.call_count == 2
    repair_content = adapter.recorded_messages[1][1].content.lower()
    assert "previous response was invalid" in repair_content
    assert "plan_version 2" in repair_content or "plan_version" in repair_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_attempt_failure() -> None:
    message = MIXED_ROUTING_MESSAGE
    invalid = {
        "plan_version": "2",
        "response_mode": "aggregate",
        "objects": [
            {
                "object_id": "url-1",
                "object_type": "web_url",
                "value": "https://cenniki.pl",
                "evidence": {
                    "source": "message_text",
                    "start": 0,
                    "end": 5,
                    "text": "wrong",
                },
            }
        ],
        "actions": [
            {
                "action_id": "a1",
                "action_type": "knowledge.add_sources",
                "workspace": {"kind": "ordinal", "value": "1"},
                "source_object_ids": ["url-1"],
            }
        ],
    }
    adapter = RecordingPlannerAdapter(structured_outputs=[invalid, invalid])
    planner = ConversationInteractionPlanner(adapter)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(_mixed_routing_request())
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output
    assert adapter.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_failure_no_local_retry() -> None:
    adapter = RecordingPlannerAdapter(provider_error=RuntimeError("provider exploded with secret"))
    planner = ConversationInteractionPlanner(adapter)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(_mixed_routing_request())
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_provider_failed
    assert "secret" not in str(exc_info.value)
    assert adapter.call_count == 1


@pytest.mark.unit
def test_prompt_safety() -> None:
    assert system_prompt_contains_required_rules()
    request = ConversationPlanningRequest(
        message_text="załącz https://safe.example do magazyn",
        attachments=(ConversationPlanningAttachment(attachment_id="att-safe", file_name="doc.pdf"),),
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="ws-1", name="magazyn", is_active=True),
        ),
        active_workspace_id="ws-1",
    )
    messages = build_planning_messages(request)
    combined = "\n".join(message.content for message in messages)
    assert "załącz https://safe.example do magazyn" in combined
    assert "att-safe" in combined
    assert "magazyn" in combined
    assert "objects" in combined
    assert "source_object_ids" in combined
    assert "evidence" in combined
    forbidden = (
        "xoxb-slack-token",
        "https://files.slack.com/",
        "slack_token",
        "password",
        "credentials",
    )
    for token in forbidden:
        assert token not in combined
