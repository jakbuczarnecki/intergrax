# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for conversational interaction prompt contract."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult

from local_workspace_application.conversation.interaction_draft_models import (
    ConversationInteractionDraft,
    DraftWebUrlSource,
    KnowledgeAddSourcesDraftAction,
    NameDraftWorkspaceReference,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningAttachment,
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    ConversationPlanningError,
    _repair_category_for_error,
)
from local_workspace_application.conversation.interaction_plan_compiler import (
    ConversationDraftCompilationError,
    ConversationDraftCompilationErrorCode,
)
from local_workspace_application.conversation.interaction_prompt import (
    RepairCategory,
    build_planning_messages,
    build_safe_planning_context,
    repair_message_for_category,
    system_prompt_contains_required_rules,
)
from pydantic import ValidationError


_E2E_FIXTURE_FRAGMENTS = (
    "https://cenniki.pl",
    r"C:\cenniki\hurt.xlsx",
    r"C:\cenniki\detal.xlsx",
    "https://example.com/docs",
)


@pytest.mark.unit
def test_schema_not_manually_added_to_prompt() -> None:
    request = ConversationPlanningRequest(message_text="hello world")
    combined = "\n".join(m.content for m in build_planning_messages(request))
    assert "JSON_SCHEMA" not in combined
    assert "model_json_schema" not in combined
    assert "$defs" not in combined
    assert "additionalProperties" not in combined


@pytest.mark.unit
def test_message_text_preserved_in_safe_context() -> None:
    message = "załącz https://safe.example do archiwum"
    request = ConversationPlanningRequest(message_text=message)
    context = build_safe_planning_context(request)
    assert context["message_text"] == message


@pytest.mark.unit
def test_no_message_text_segments_in_safe_context() -> None:
    request = ConversationPlanningRequest(
        message_text="dodaj https://portal.vendor.io i C:\\data\\file.pdf do projekty"
    )
    context = build_safe_planning_context(request)
    assert "message_text_segments" not in context
    assert "message_text_length" not in context


@pytest.mark.unit
def test_no_deterministic_source_extraction_in_context() -> None:
    request = ConversationPlanningRequest(
        message_text="dodaj https://portal.vendor.io i C:\\data\\file.pdf do projekty"
    )
    context = build_safe_planning_context(request)
    forbidden_keys = (
        "detected_urls",
        "detected_paths",
        "parsed_objects",
        "source_objects",
        "text_source_candidates",
    )
    for key in forbidden_keys:
        assert key not in context


@pytest.mark.unit
def test_activation_vs_target_rule_in_prompt() -> None:
    assert system_prompt_contains_required_rules()
    messages = build_planning_messages(ConversationPlanningRequest(message_text="test"))
    system = messages[0].content
    assert (
        "does not change the active workspace" in system.lower()
        or "not change the active workspace" in system.lower()
    )


@pytest.mark.unit
def test_prompt_provider_neutrality() -> None:
    messages = build_planning_messages(ConversationPlanningRequest(message_text="test"))
    system = messages[0].content
    for forbidden in ("Ollama", "Qwen", "Llama", "LangChain", "Pydantic", "pytest", "E2E"):
        assert forbidden not in system


@pytest.mark.unit
def test_semantic_examples_distinct_from_e2e_fixtures() -> None:
    messages = build_planning_messages(ConversationPlanningRequest(message_text="test"))
    system = messages[0].content
    for fragment in _E2E_FIXTURE_FRAGMENTS:
        assert fragment not in system


@pytest.mark.unit
def test_prompt_targets_semantic_draft_not_canonical_ids() -> None:
    messages = build_planning_messages(ConversationPlanningRequest(message_text="test"))
    system = messages[0].content
    assert "semantic intent" in system
    assert "Do not invent technical IDs" in system
    assert "depends_on_action_numbers" in system
    assert "plan_version" not in system
    assert "source_object_ids" not in system
    assert "evidence.start" not in system


@pytest.mark.unit
def test_repair_category_compiler_source_not_found() -> None:
    exc = ConversationDraftCompilationError(
        ConversationDraftCompilationErrorCode.source_value_not_found
    )
    assert _repair_category_for_error(exc) == RepairCategory.source_value_not_grounded


@pytest.mark.unit
def test_repair_category_draft_validation() -> None:
    exc = TypeError("structured output is not ConversationInteractionDraft")
    assert _repair_category_for_error(exc) == RepairCategory.draft_contract


@pytest.mark.unit
def test_repair_message_sanitization() -> None:
    for category in RepairCategory:
        message = repair_message_for_category(category)
        assert "Traceback" not in message
        assert "ValidationError" not in message
        assert "input_value" not in message


class _SingleCallAdapter(LLMAdapter):
    provider = "recording"
    model = "recording"

    def __init__(self, *, draft: ConversationInteractionDraft) -> None:
        super().__init__()
        self._draft = draft
        self.call_count = 0

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        return build_adapter_response(content="ok")

    def generate_structured(self, messages, output_model, *, temperature=None, max_tokens=None, run_id=None):
        self.call_count += 1
        return LLMStructuredResult(parsed=self._draft, response=build_adapter_response(content=""))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_first_result_single_adapter_call() -> None:
    message = "dodaj https://portal.vendor.io do projekty"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="projekty"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://portal.vendor.io"),),
            ),
        ),
    )
    adapter = _SingleCallAdapter(draft=draft)
    planner = ConversationInteractionPlanner(adapter)
    await planner.plan(ConversationPlanningRequest(message_text=message))
    assert adapter.call_count == 1


class _TwoCallAdapter(LLMAdapter):
    provider = "recording"
    model = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        return build_adapter_response(content="ok")

    def generate_structured(self, messages, output_model, *, temperature=None, max_tokens=None, run_id=None):
        self.call_count += 1
        raise ValidationError.from_exception_data(
            "ConversationInteractionDraft",
            [{"type": "value_error", "loc": (), "msg": "draft must contain at least one action", "input": {}}],
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exactly_one_repair_on_repeated_invalid_output() -> None:
    adapter = _TwoCallAdapter()
    planner = ConversationInteractionPlanner(adapter)
    with pytest.raises(ConversationPlanningError):
        await planner.plan(ConversationPlanningRequest(message_text="dodaj https://a.example do x"))
    assert adapter.call_count == 2


@pytest.mark.unit
def test_prompt_safety_no_failed_experiment_artifacts() -> None:
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
    assert "message_text_segments" not in combined
    assert "64-character" not in combined.lower()
    assert "evidence_grounding" not in combined
