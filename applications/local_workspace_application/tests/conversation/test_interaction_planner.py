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
    KnowledgeAddLocalReferencesPlannedAction,
    KnowledgeAddWebUrlsPlannedAction,
    LocalReference,
    WorkspaceActivatePlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    ConversationPlanningError,
    ConversationPlanningErrorCode,
    PlanRequestValidationError,
    extract_user_url_candidates,
    validate_plan_against_request,
)
from local_workspace_application.conversation.interaction_prompt import (
    build_planning_messages,
    system_prompt_contains_required_rules,
)


def _magazyn_target() -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn")


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


def _mixed_polish_plan() -> ConversationInteractionPlan:
    return ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="url-1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://www.cenniki.pl",),
                evidence_quotes=("https://www.cenniki.pl",),
            ),
            KnowledgeAddLocalReferencesPlannedAction(
                action_id="local-1",
                action_type="knowledge.add_local_references",
                workspace=_magazyn_target(),
                references=(LocalReference(kind="file", value=r"c:\moje dokumenty\cenniki.xls"),),
                evidence_quotes=(r"c:\moje dokumenty\cenniki.xls",),
            ),
        ),
    )


def _mixed_polish_request() -> ConversationPlanningRequest:
    return ConversationPlanningRequest(
        message_text=(
            "dołącz informacje o cennikach ze strony https://www.cenniki.pl\n"
            "oraz dorzuć moją kopię lokalną cenników z\n"
            r"c:\moje dokumenty\cenniki.xls"
            "\n"
            "a to wszystko do workspace 'magazyn'"
        ),
        available_workspaces=(
            ConversationPlanningWorkspace(
                workspace_id="ws-1",
                name="default",
                is_active=True,
            ),
            ConversationPlanningWorkspace(
                workspace_id="ws-2",
                name="magazyn",
                is_active=False,
            ),
        ),
        active_workspace_id="ws-1",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_polish_message_plan() -> None:
    adapter = RecordingPlannerAdapter(structured_outputs=[_mixed_polish_plan()])
    planner = ConversationInteractionPlanner(adapter)
    plan = await planner.plan(_mixed_polish_request())
    assert len(plan.actions) == 2
    assert plan.actions[0].action_type == "knowledge.add_web_urls"
    assert plan.actions[1].action_type == "knowledge.add_local_references"
    for action in plan.actions:
        assert action.workspace.kind == WorkspaceReferenceKind.name
        assert action.workspace.value == "magazyn"
    assert all(action.action_type != "workspace.activate" for action in plan.actions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_workspace_activation() -> None:
    activate_plan = ConversationInteractionPlan(
        plan_version="1",
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
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddAttachmentsPlannedAction(
                action_id="att-1",
                action_type="knowledge.add_attachments",
                workspace=_magazyn_target(),
                attachment_ids=("file-a", "file-b"),
            ),
            KnowledgeAddWebUrlsPlannedAction(
                action_id="url-1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://docs.example.com/page",),
            ),
            KnowledgeAddLocalReferencesPlannedAction(
                action_id="local-1",
                action_type="knowledge.add_local_references",
                workspace=_magazyn_target(),
                references=(LocalReference(kind="file", value=r"c:\data\report.docx"),),
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
    assert len(result.actions) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_unknown_attachment() -> None:
    bad_plan = ConversationInteractionPlan(
        plan_version="1",
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
async def test_hallucination_protection_unknown_url() -> None:
    bad_plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="url-1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://hallucinated.example",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_plan, bad_plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text="dodaj stronę do magazyn")
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_unknown_local_path() -> None:
    bad_plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddLocalReferencesPlannedAction(
                action_id="local-1",
                action_type="knowledge.add_local_references",
                workspace=_magazyn_target(),
                references=(LocalReference(kind="file", value=r"c:\not\in\message.txt"),),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_plan, bad_plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(message_text="dodaj lokalny plik do magazyn")
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_unknown_evidence_quote() -> None:
    bad_plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="url-1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://www.cenniki.pl",),
                evidence_quotes=("user never said this",),
            ),
        ),
    )
    adapter = RecordingPlannerAdapter(structured_outputs=[bad_plan, bad_plan])
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(
        message_text="dodaj https://www.cenniki.pl do magazyn",
    )
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(request)
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_protection_invalid_dependency() -> None:
    adapter = RecordingPlannerAdapter(
        structured_outputs=[
            {
                "plan_version": "1",
                "response_mode": "aggregate",
                "actions": [
                    {
                        "action_id": "a1",
                        "action_type": "knowledge.add_web_urls",
                        "workspace": {"kind": "name", "value": "magazyn"},
                        "urls": ["https://www.cenniki.pl"],
                        "depends_on": ["missing"],
                    }
                ],
            },
            {
                "plan_version": "1",
                "response_mode": "aggregate",
                "actions": [
                    {
                        "action_id": "a1",
                        "action_type": "knowledge.add_web_urls",
                        "workspace": {"kind": "name", "value": "magazyn"},
                        "urls": ["https://www.cenniki.pl"],
                        "depends_on": ["missing"],
                    }
                ],
            },
        ]
    )
    planner = ConversationInteractionPlanner(adapter)
    request = ConversationPlanningRequest(
        message_text="dodaj https://www.cenniki.pl do magazyn",
    )
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
    invalid = {
        "plan_version": "1",
        "response_mode": "aggregate",
        "actions": [
            {
                "action_id": "a1",
                "action_type": "knowledge.add_web_urls",
                "workspace": {"kind": "name", "value": "magazyn"},
                "urls": ["https://hallucinated.example"],
            }
        ],
    }
    adapter = RecordingPlannerAdapter(structured_outputs=[invalid, _mixed_polish_plan()])
    planner = ConversationInteractionPlanner(adapter)
    plan = await planner.plan(_mixed_polish_request())
    assert len(plan.actions) == 2
    assert adapter.call_count == 2
    assert "previous response was invalid" in adapter.recorded_messages[1][1].content.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_attempt_failure() -> None:
    invalid = {
        "plan_version": "1",
        "response_mode": "aggregate",
        "actions": [
            {
                "action_id": "a1",
                "action_type": "knowledge.add_web_urls",
                "workspace": {"kind": "name", "value": "magazyn"},
                "urls": ["https://hallucinated.example"],
            }
        ],
    }
    adapter = RecordingPlannerAdapter(structured_outputs=[invalid, invalid])
    planner = ConversationInteractionPlanner(adapter)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(_mixed_polish_request())
    assert exc_info.value.code == ConversationPlanningErrorCode.conversation_planner_invalid_output
    assert adapter.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_failure_no_local_retry() -> None:
    adapter = RecordingPlannerAdapter(provider_error=RuntimeError("provider exploded with secret"))
    planner = ConversationInteractionPlanner(adapter)
    with pytest.raises(ConversationPlanningError) as exc_info:
        await planner.plan(_mixed_polish_request())
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
    assert "TARGET" in combined or "target" in combined.lower()
    assert "clarification" in combined.lower()
    forbidden = (
        "xoxb-slack-token",
        "https://files.slack.com/",
        "slack_token",
        "password",
        "credentials",
    )
    for token in forbidden:
        assert token not in combined


def _url_plan(url: str) -> ConversationInteractionPlan:
    return ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="url-1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=(url,),
            ),
        ),
    )


def _local_plan(path: str) -> ConversationInteractionPlan:
    return ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddLocalReferencesPlannedAction(
                action_id="local-1",
                action_type="knowledge.add_local_references",
                workspace=_magazyn_target(),
                references=(LocalReference(kind="file", value=path),),
            ),
        ),
    )


@pytest.mark.unit
def test_url_grounding_rejects_shortened_url() -> None:
    request = ConversationPlanningRequest(
        message_text="dodaj https://example.com/cennik?region=pl do magazyn",
    )
    with pytest.raises(PlanRequestValidationError):
        validate_plan_against_request(_url_plan("https://example.com"), request)


@pytest.mark.unit
def test_url_grounding_rejects_changed_host() -> None:
    request = ConversationPlanningRequest(
        message_text="dodaj https://safe.example.evil.com/page do magazyn",
    )
    with pytest.raises(PlanRequestValidationError):
        validate_plan_against_request(_url_plan("https://safe.example"), request)


@pytest.mark.unit
def test_url_grounding_accepts_trailing_sentence_period() -> None:
    request = ConversationPlanningRequest(
        message_text="dodaj https://example.com/path. do magazyn",
    )
    validate_plan_against_request(_url_plan("https://example.com/path"), request)


@pytest.mark.unit
def test_url_grounding_accepts_full_query_url() -> None:
    url = "https://example.com/path?x=1&y=2"
    request = ConversationPlanningRequest(message_text=f"dodaj {url} do magazyn")
    validate_plan_against_request(_url_plan(url), request)


@pytest.mark.unit
def test_extract_user_url_candidates_strips_sentence_punctuation() -> None:
    candidates = extract_user_url_candidates(
        ("Zobacz https://example.com/path.", "oraz https://docs.test/a?b=1).")
    )
    assert "https://example.com/path" in candidates
    assert "https://docs.test/a?b=1" in candidates


@pytest.mark.unit
def test_local_reference_rejects_shortened_directory() -> None:
    full_path = r"C:\dokumenty\cenniki\cennik-2026.xlsx"
    request = ConversationPlanningRequest(message_text=f"dodaj {full_path} do magazyn")
    with pytest.raises(PlanRequestValidationError):
        validate_plan_against_request(_local_plan(r"C:\dokumenty\cenniki"), request)


@pytest.mark.unit
def test_local_reference_rejects_shortened_filename_without_extension() -> None:
    full_path = r"C:\dokumenty\cenniki\cennik-2026.xlsx"
    request = ConversationPlanningRequest(message_text=f"dodaj {full_path} do magazyn")
    with pytest.raises(PlanRequestValidationError):
        validate_plan_against_request(_local_plan(r"C:\dokumenty\cenniki\cennik-2026"), request)


@pytest.mark.unit
def test_local_reference_accepts_windows_case_insensitive_match() -> None:
    full_path = r"C:\dokumenty\cenniki\cennik-2026.xlsx"
    request = ConversationPlanningRequest(message_text=f"dodaj {full_path} do magazyn")
    validate_plan_against_request(
        _local_plan(r"c:\DOKUMENTY\CENNIKI\CENNIK-2026.XLSX"),
        request,
    )


@pytest.mark.unit
def test_local_reference_accepts_unc_path() -> None:
    unc_path = r"\\server\share\folder\cennik.xlsx"
    request = ConversationPlanningRequest(message_text=f"dodaj {unc_path} do magazyn")
    validate_plan_against_request(_local_plan(unc_path), request)
