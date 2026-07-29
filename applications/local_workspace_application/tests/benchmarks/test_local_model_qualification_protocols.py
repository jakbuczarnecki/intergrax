# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import pytest
from pydantic import ValidationError

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall

from local_workspace_application.benchmarks.local_model_qualification.config import BenchmarkConfig
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    StructuralFailureCategory,
)
from local_workspace_application.benchmarks.local_model_qualification.corpus import case_by_id
from local_workspace_application.benchmarks.local_model_qualification.protocols import (
    PROTOCOL_SINGLE_PLAN_TOOL,
    PROTOCOL_STRUCTURED_OUTPUT,
    SUBMIT_DRAFT_TOOL_NAME,
    run_protocol_attempt,
)
from local_workspace_application.conversation.interaction_draft_models import (
    ConversationInteractionDraft,
    DraftWebUrlSource,
    KnowledgeAddSourcesDraftAction,
    NameDraftWorkspaceReference,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import WorkspaceReferenceKind

_BENCHMARK = BenchmarkConfig(
    repetitions=1,
    warmup_runs=0,
    temperature=0.0,
    max_tokens=8192,
    randomize_case_order=False,
)


@dataclass
class FakeAdapter:
    structured_supported: bool = True
    tools_supported: bool = True
    structured_result: Any | None = None
    structured_error: Exception | None = None
    tools_result: LLMAdapterResponse | None = None
    tools_error: Exception | None = None
    tool_executed: bool = False

    def supports_structured_output(self) -> bool:
        return self.structured_supported

    def supports_tools(self) -> bool:
        return self.tools_supported

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        if self.structured_error is not None:
            raise self.structured_error
        draft = self.structured_result or ConversationInteractionDraft(
            actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
        )
        return LLMStructuredResult(
            parsed=draft,
            response=LLMAdapterResponse(content="ignored"),
        )

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        if self.tools_error is not None:
            raise self.tools_error
        if self.tools_result is not None:
            return self.tools_result
        draft = ConversationInteractionDraft(
            actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
        )
        return LLMAdapterResponse(
            content="",
            finish_reason=LLMFinishReason.TOOL_CALLS,
            tool_calls=(
                LLMToolCall(
                    id="1",
                    name=SUBMIT_DRAFT_TOOL_NAME,
                    arguments_json=draft.model_dump_json(),
                ),
            ),
        )


def test_structured_output_valid_draft_succeeds() -> None:
    request = case_by_id("planner.workspace_list").request
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(),
        protocol=PROTOCOL_STRUCTURED_OUTPUT,
        request=request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.ok
    assert attempt.plan is not None


def test_structured_output_unsupported_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(structured_supported=False),
        protocol=PROTOCOL_STRUCTURED_OUTPUT,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value


def test_structured_output_wrong_parsed_type_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(structured_result={"not": "draft"}),
        protocol=PROTOCOL_STRUCTURED_OUTPUT,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value


def test_structured_output_provider_exception_classified_safely() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(structured_error=RuntimeError("boom")),
        protocol=PROTOCOL_STRUCTURED_OUTPUT,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.PROVIDER_ERROR.value
    assert attempt.error_type == "RuntimeError"


def test_tool_exactly_one_correct_call_succeeds() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.ok


def test_tool_handler_not_used() -> None:
    from pathlib import Path

    import local_workspace_application.benchmarks.local_model_qualification.protocols as protocols_module

    source = Path(protocols_module.__file__).read_text(encoding="utf-8")
    assert "ToolRegistry" not in source
    run_protocol_attempt(
        adapter=FakeAdapter(),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )


def test_both_protocols_use_same_compiler_and_validator() -> None:
    request = case_by_id("planner.target_workspace_without_activation").request
    bad_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(
                    kind=WorkspaceReferenceKind.name,
                    value="finanse",
                ),
                sources=(
                    DraftWebUrlSource(
                        object_type="web_url",
                        value="https://not-in-message.example",
                    ),
                ),
            ),
        )
    )

    structured = run_protocol_attempt(
        adapter=FakeAdapter(structured_result=bad_draft),
        protocol=PROTOCOL_STRUCTURED_OUTPUT,
        request=request,
        benchmark=_BENCHMARK,
        run_id="test-structured",
    )
    tool = run_protocol_attempt(
        adapter=FakeAdapter(
            tools_result=LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="1",
                        name=SUBMIT_DRAFT_TOOL_NAME,
                        arguments_json=bad_draft.model_dump_json(),
                    ),
                ),
            )
        ),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=request,
        benchmark=_BENCHMARK,
        run_id="test-tool",
    )
    assert structured.failure_category == StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value
    assert tool.failure_category == StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value


def test_compilation_failure_classified() -> None:
    request = case_by_id("planner.workspace_list").request
    bad_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(
                    kind=WorkspaceReferenceKind.name,
                    value="missing",
                ),
                sources=(
                    DraftWebUrlSource(
                        object_type="web_url",
                        value="https://missing.example",
                    ),
                ),
            ),
        )
    )
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(structured_result=bad_draft),
        protocol=PROTOCOL_STRUCTURED_OUTPUT,
        request=request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value


def test_tool_no_call_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(
            tools_result=LLMAdapterResponse(content="", tool_calls=()),
        ),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.MISSING_PLAN_TOOL_CALL.value


def test_tool_multiple_calls_classified() -> None:
    draft = ConversationInteractionDraft(
        actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
    )
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(
            tools_result=LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall(id="1", name=SUBMIT_DRAFT_TOOL_NAME, arguments_json=draft.model_dump_json()),
                    LLMToolCall(id="2", name=SUBMIT_DRAFT_TOOL_NAME, arguments_json=draft.model_dump_json()),
                ),
            )
        ),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.MULTIPLE_PLAN_TOOL_CALLS.value


def test_tool_wrong_name_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(
            tools_result=LLMAdapterResponse(
                content="",
                tool_calls=(LLMToolCall(id="1", name="other_tool", arguments_json="{}"),),
            )
        ),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.UNEXPECTED_PLAN_TOOL.value


def test_tool_invalid_arguments_json_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(
            tools_result=LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall(id="1", name=SUBMIT_DRAFT_TOOL_NAME, arguments_json="{bad"),
                ),
            )
        ),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.INVALID_TOOL_ARGUMENTS.value


def test_tool_pydantic_invalid_arguments_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(
            tools_result=LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall(id="1", name=SUBMIT_DRAFT_TOOL_NAME, arguments_json='{"actions": "bad"}'),
                ),
            )
        ),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value


def test_tool_unsupported_capability_classified() -> None:
    attempt = run_protocol_attempt(
        adapter=FakeAdapter(tools_supported=False),
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=case_by_id("planner.workspace_list").request,
        benchmark=_BENCHMARK,
        run_id="test",
    )
    assert attempt.failure_category == StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value
