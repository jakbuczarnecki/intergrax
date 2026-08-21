# © Artur Czarnecki. All rights reserved.

"""ENG-6 — investigation proof parser and validation."""

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.tools.investigation_proof import (
    InvestigationProofValidationError,
    build_investigation_proof_step,
    collect_available_evidence_ids,
    parse_public_decision_note,
    record_first_investigation_step,
    validate_follow_up_investigation_step,
)
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_first_round_allows_empty_basis() -> None:
    step = record_first_investigation_step(
        round_index=1,
        assistant_content="",
        next_tool_call_ids=("evidence-a",),
    )
    assert step.basis_tool_call_ids == ()
    assert step.next_tool_call_ids == ("evidence-a",)


def test_collect_available_evidence_ids_from_tool_messages() -> None:
    messages = [
        ChatMessage(role="assistant", content="x", tool_calls=[{"id": "evidence-a"}]),
        ChatMessage(role="tool", content="A", tool_call_id="evidence-a"),
        ChatMessage(role="tool", content="B", tool_call_id="evidence-b"),
    ]
    assert collect_available_evidence_ids(messages) == ("evidence-a", "evidence-b")


def test_parse_public_decision_note() -> None:
    parsed = parse_public_decision_note(
        "EVIDENCE_BASIS: evidence-a,evidence-b\n"
        "PURPOSE: verify normalized effect"
    )
    assert parsed.basis_tool_call_ids == ("evidence-a", "evidence-b")
    assert parsed.public_reason == "verify normalized effect"


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("PURPOSE: only purpose", "missing EVIDENCE_BASIS"),
        ("EVIDENCE_BASIS: evidence-a", "missing PURPOSE"),
        ("EVIDENCE_BASIS: evidence-a\nPURPOSE:", "empty PURPOSE"),
        ("EVIDENCE_BASIS: evidence-a,\nPURPOSE: x", "empty EVIDENCE_BASIS id segment"),
    ],
)
def test_malformed_public_decision_note_rejected(content: str, match: str) -> None:
    with pytest.raises(InvestigationProofValidationError, match=match):
        parse_public_decision_note(content)


def test_unknown_basis_id_rejected() -> None:
    with pytest.raises(InvestigationProofValidationError, match="unknown basis tool_call_id"):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS: missing-id\nPURPOSE: inspect subgroup",
            available_evidence_ids=frozenset({"evidence-a"}),
            next_tool_call_ids=("evidence-b",),
        )


def test_empty_follow_up_basis_rejected_when_prior_evidence_exists() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="follow-up tool round requires explicit evidence basis",
    ):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS:\nPURPOSE: inspect subgroup",
            available_evidence_ids=frozenset({"evidence-a"}),
            next_tool_call_ids=("evidence-b",),
        )


def test_duplicate_basis_id_rejected() -> None:
    with pytest.raises(InvestigationProofValidationError, match="duplicate basis tool_call_id"):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS: evidence-a,evidence-a\nPURPOSE: inspect subgroup",
            available_evidence_ids=frozenset({"evidence-a"}),
            next_tool_call_ids=("evidence-b",),
        )


def test_build_investigation_proof_step_uses_tool_call_ids_for_next() -> None:
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content="EVIDENCE_BASIS: evidence-a\nPURPOSE: inspect subgroup",
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="evidence-b",
                name="probe.b",
                arguments={"confirm": True},
            ),
        ),
        messages_before_round=[
            ChatMessage(role="tool", content="A", tool_call_id="evidence-a"),
        ],
    )
    assert step.basis_tool_call_ids == ("evidence-a",)
    assert step.next_tool_call_ids == ("evidence-b",)
