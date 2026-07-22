# © Artur Czarnecki. All rights reserved.

"""Unit tests for LKW AskAnswerAssembler (MVP-2)."""

from __future__ import annotations

import json
from typing import Optional, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.ask_answer_assembler import (
    AskAnswerAssembler,
    project_ask_citations,
)
from local_workspace_application.workspaces.ask_models import (
    AskAnswerAssemblyError,
    AskAnswerAssemblyStatus,
)

pytestmark = pytest.mark.unit


class RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text
        self.calls = 0
        self.last_messages: list[ChatMessage] = []

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id
        self.calls += 1
        self.last_messages = list(messages)
        return build_adapter_response(content=self._fixed_text)


def _hit(**overrides: object) -> WorkspaceSearchHitV1:
    payload = {
        "document_id": "doc-1",
        "source_id": "src-1",
        "workspace_id": "ws-1",
        "source_path": "C:/docs/a.txt",
        "file_name": "a.txt",
        "score": 0.9,
        "snippet": "The contract terminates on 30 June 2026.",
        "metadata": {"provider_vector_id": "vec-secret", "chunk_id": "chunk-1", "page": 3},
    }
    payload.update(overrides)
    return WorkspaceSearchHitV1.model_validate(payload)


def test_ask_answer_assembler_uses_only_verified_evidence() -> None:
    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": "The contract terminates on 30 June 2026.",
                "used_evidence_ids": ["E1"],
            }
        )
    )
    assembler = AskAnswerAssembler(llm)
    evidence = [_hit()]
    result = assembler.assemble(question="When does the contract terminate?", evidence=evidence)

    assert llm.calls == 1
    assert result.status == AskAnswerAssemblyStatus.COMPLETED
    assert result.answer is not None
    assert "30 June 2026" in result.answer
    assert result.used_evidence_ids == ["E1"]

    context = assembler.last_model_context
    assert context is not None
    assert len(context) == 1
    item = context[0]
    assert item["evidence_id"] == "E1"
    assert item["document_id"] == "doc-1"
    assert item["snippet"].startswith("The contract terminates")
    assert "provider_vector_id" not in item
    assert "metadata" not in item
    assert item["location"] == {"page": 3}

    user_payload = json.loads(llm.last_messages[-1].content or "")
    assert "provider_vector_id" not in json.dumps(user_payload)


def test_ask_answer_assembler_skips_model_when_evidence_is_empty() -> None:
    llm = RecordingFakeLLM(fixed_text="should-not-run")
    assembler = AskAnswerAssembler(llm)
    result = assembler.assemble(question="Anything?", evidence=[])

    assert llm.calls == 0
    assert result.status == AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    assert result.used_evidence_ids == []
    citations = project_ask_citations(used_evidence_ids=[], indexed_evidence={})
    assert citations == []


def test_ask_answer_assembler_returns_insufficient_evidence_without_answer() -> None:
    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "insufficient_evidence",
                "answer": "I will invent an answer anyway",
                "used_evidence_ids": ["E1"],
            }
        )
    )
    assembler = AskAnswerAssembler(llm)
    result = assembler.assemble(question="Unknown topic?", evidence=[_hit()])

    assert llm.calls == 1
    assert result.status == AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    assert result.used_evidence_ids == []


def test_ask_answer_assembler_rejects_unknown_evidence_reference() -> None:
    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": "Fabricated grounding",
                "used_evidence_ids": ["E99"],
            }
        )
    )
    assembler = AskAnswerAssembler(llm)
    with pytest.raises(AskAnswerAssemblyError) as exc_info:
        assembler.assemble(question="When?", evidence=[_hit()])
    assert exc_info.value.code == "unknown_evidence_reference"


def test_completed_answer_requires_at_least_one_verified_citation() -> None:
    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": "An answer without citations",
                "used_evidence_ids": [],
            }
        )
    )
    assembler = AskAnswerAssembler(llm)
    result = assembler.assemble(question="When?", evidence=[_hit()])
    assert result.status == AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    citations = project_ask_citations(
        used_evidence_ids=result.used_evidence_ids,
        indexed_evidence={"E1": _hit()},
    )
    assert citations == []
