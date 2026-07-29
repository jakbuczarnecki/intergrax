# © Artur Czarnecki. All rights reserved.

"""Tests for the conversational interaction planner."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult

from local_workspace_application.conversation.interaction_draft_models import (
    ActiveDraftWorkspaceReference,
    ConversationInteractionDraft,
    DraftLocalFileReferenceSource,
    DraftWebUrlSource,
    KnowledgeAddAttachmentsDraftAction,
    KnowledgeAddSourcesDraftAction,
    NameDraftWorkspaceReference,
    OrdinalDraftWorkspaceReference,
    WorkspaceActivateDraftAction,
    WorkspaceAskDraftAction,
)
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
from local_workspace_application.conversation.interaction_plan_compiler import compile_interaction_draft
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    ConversationPlanningError,
    ConversationPlanningErrorCode,
    PlanRequestValidationError,
    _repair_category_for_error,
    validate_plan_against_request,
)
from local_workspace_application.conversation.interaction_prompt import RepairCategory
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


def _mixed_routing_draft() -> ConversationInteractionDraft:
    return ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://cenniki.pl"),),
            ),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                sources=(
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value=r"C:\cenniki\hurt.xlsx",
                    ),
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value=r"C:\cenniki\detal.xlsx",
                    ),
                ),
            ),
        ),
    )


def _mixed_routing_plan() -> ConversationInteractionPlan:
    return compile_interaction_draft(_mixed_routing_draft(), _mixed_routing_request())


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
    adapter = RecordingPlannerAdapter(structured_outputs=[_mixed_routing_draft()])
    planner = ConversationInteractionPlanner(adapter)
    plan = await planner.plan(_mixed_routing_request())
    assert len(plan.objects) == 3
    assert len(plan.actions) == 2
    add_sources = [a for a in plan.actions if a.action_type == "knowledge.add_sources"]
    assert len(add_sources) == 2
    url_action = next(a for a in add_sources if a.workspace.value == "1")
    local_action = next(a for a in add_sources if a.workspace.value == "2")
    assert url_action.source_object_ids == ("object-1",)
    assert set(local_action.source_object_ids) == {"object-2", "object-3"}
    assert all(action.action_type != "workspace.activate" for action in plan.actions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_workspace_activation() -> None:
    activate_draft = ConversationInteractionDraft(
        actions=(
            WorkspaceActivateDraftAction(
                action_type="workspace.activate",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                evidence_quotes=("przełącz mnie na workspace magazyn",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[activate_draft])
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
    plan = ConversationInteractionDraft(
        actions=(
            KnowledgeAddAttachmentsDraftAction(
                action_type="knowledge.add_attachments",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                attachment_ids=("file-a", "file-b"),
            ),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(
                    DraftWebUrlSource(object_type="web_url", value="https://docs.example.com/page"),
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value=r"c:\data\report.docx",
                    ),
                ),
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
    bad_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddAttachmentsDraftAction(
                action_type="knowledge.add_attachments",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                attachment_ids=("unknown-att",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_draft, bad_draft])
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
async def test_invalid_draft_action_reference_rejected() -> None:
    bad_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://www.cenniki.pl"),),
                depends_on_action_numbers=(2,),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_draft, bad_draft])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text="dodaj https://www.cenniki.pl do magazyn")
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
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://example.com"),),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[draft])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    result = await planner.plan(request)
    assert all(action.action_type != "workspace.activate" for action in result.actions)
    assert any(action.action_type == "knowledge.add_sources" for action in result.actions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_url_in_question_routes_to_workspace_ask() -> None:
    message = "co sądzisz o https://example.com?"
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceAskDraftAction(
                action_type="workspace.ask",
                workspace=ActiveDraftWorkspaceReference(kind=WorkspaceReferenceKind.active),
                question=message,
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[draft])
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
    bad_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://www.cenniki.pl"),),
                evidence_quotes=("user never said this",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_draft, bad_draft])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text=message)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_invalid_dependency() -> None:
    message = "dodaj https://www.cenniki.pl do magazyn"
    invalid_dependency_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://www.cenniki.pl"),),
                depends_on_action_numbers=(9,),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(
        structured_outputs=[invalid_dependency_draft, invalid_dependency_draft]
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
    invalid = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://not-in-message.example"),),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[invalid, _mixed_routing_draft()])
    planner = ConversationInteractionPlanner(adapter)
    plan = await planner.plan(_mixed_routing_request())
    assert len(plan.objects) == 3
    assert adapter.call_count == 2
    repair_content = adapter.recorded_messages[1][1].content.lower()
    assert "source value" in repair_content or "semantic draft" in repair_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_attempt_failure() -> None:
    message = MIXED_ROUTING_MESSAGE
    invalid = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://not-in-message.example"),),
            ),
        ),
    )
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
    assert "message_text_segments" not in combined
    assert "action_id" not in combined
    assert "object_id" not in combined
    assert "semantic intent" in combined
    assert "occurrence" in combined
    forbidden = (
        "xoxb-slack-token",
        "https://files.slack.com/",
        "slack_token",
        "password",
        "credentials",
    )
    for token in forbidden:
        assert token not in combined


@pytest.mark.unit
@pytest.mark.asyncio
async def test_planner_uses_conversation_interaction_draft_as_output_model() -> None:
    captured: list[type] = []

    class CapturingAdapter(RecordingPlannerAdapter):
        def generate_structured(self, messages, output_model, *, temperature=None, max_tokens=None, run_id=None):
            captured.append(output_model)
            return super().generate_structured(
                messages,
                output_model,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )

    adapter = CapturingAdapter(structured_outputs=[_mixed_routing_draft()])
    planner = ConversationInteractionPlanner(adapter)
    result = await planner.plan(_mixed_routing_request())
    assert captured == [ConversationInteractionDraft]
    assert isinstance(result, ConversationInteractionPlan)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compiler_failure_triggers_safe_repair_category() -> None:
    invalid = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://missing.example"),),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[invalid, _mixed_routing_draft()])
    planner = ConversationInteractionPlanner(adapter)
    await planner.plan(_mixed_routing_request())
    repair_content = adapter.recorded_messages[1][1].content
    assert "https://missing.example" not in repair_content
    assert "Traceback" not in repair_content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_first_draft_single_adapter_call() -> None:
    adapter = RecordingPlannerAdapter(structured_outputs=[_mixed_routing_draft()])
    planner = ConversationInteractionPlanner(adapter)
    await planner.plan(_mixed_routing_request())
    assert adapter.call_count == 1


@pytest.mark.unit
def test_repair_category_compiler_source_not_found() -> None:
    from local_workspace_application.conversation.interaction_plan_compiler import (
        ConversationDraftCompilationError,
        ConversationDraftCompilationErrorCode,
    )

    exc = ConversationDraftCompilationError(
        ConversationDraftCompilationErrorCode.source_value_not_found
    )
    assert _repair_category_for_error(exc) == RepairCategory.source_value_not_grounded


@pytest.mark.unit
def test_repair_category_request_validation() -> None:
    exc = PlanRequestValidationError("evidence quote not found in user context")
    assert _repair_category_for_error(exc) == RepairCategory.canonical_request_grounding
