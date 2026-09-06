# © Artur Czarnecki. All rights reserved.

"""ENG-6 — investigation proof parser and validation."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from intergrax.contracts.model_visible_evidence import ModelVisibleEvidenceReference
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.investigation_proof import (
    InvestigationEvidenceBasis,
    InvestigationProofValidationError,
    build_completed_observation_reference_index,
    build_investigation_proof_step,
    collect_available_evidence_ids,
    format_investigation_follow_up_context,
    mint_runtime_observation_evidence_reference,
    parse_public_decision_note,
    prepare_native_planner_messages_with_follow_up_context,
    record_first_investigation_step,
    validate_follow_up_investigation_step,
)
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import (
    NativeToolPlanAlignmentError,
    validate_native_tool_plan_alignment,
)
from intergrax.tools.model_observation_format import format_tool_model_observation_content
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.execution_models import ToolModelObservation

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _EvidenceIn(BaseModel):
    label: str = "x"


def _tool_observation(*, tool_call_id: str, evidence_reference: str) -> ChatMessage:
    observation = ToolModelObservation(
        content=json.dumps({"payload": "x"}),
        evidence_reference=evidence_reference,
    )
    return ChatMessage(
        role="tool",
        content=format_tool_model_observation_content(observation),
        tool_call_id=tool_call_id,
    )


def _assistant_call(*, tool_call_id: str, tool_name: str = "probe.a") -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    )


def test_first_round_allows_empty_basis() -> None:
    step = record_first_investigation_step(
        round_index=1,
        assistant_content="",
        next_tool_call_ids=("call_abc",),
    )
    assert step.declared_basis_references == ()
    assert step.basis_bindings == ()
    assert step.basis_tool_call_ids == ()
    assert step.next_tool_call_ids == ("call_abc",)


def test_collect_available_evidence_ids_from_canonical_tool_messages() -> None:
    messages = [
        _assistant_call(tool_call_id="call_abc"),
        _tool_observation(tool_call_id="call_abc", evidence_reference="evidence.telemetry.x"),
    ]
    assert collect_available_evidence_ids(messages) == ("evidence.telemetry.x",)


def test_collect_available_evidence_ids_prefers_envelope_over_legacy_json() -> None:
    messages = [
        _assistant_call(tool_call_id="call_abc"),
        ChatMessage(
            role="tool",
            content=format_tool_model_observation_content(
                ToolModelObservation(
                    content=json.dumps(
                        {
                            "evidence_id": "evidence.legacy",
                            "evidence_reference": "observation.probe.a.step-1",
                        }
                    ),
                    evidence_reference="evidence.workload.line4.incident_window",
                )
            ),
            tool_call_id="call_abc",
        ),
    ]
    assert collect_available_evidence_ids(messages) == (
        "evidence.workload.line4.incident_window",
    )


def test_collect_available_evidence_ids_ignores_orphan_tools() -> None:
    messages = [
        ChatMessage(
            role="tool",
            content=json.dumps({"evidence_reference": "evidence.orphan"}),
            tool_call_id="fake-x",
        ),
        ChatMessage(role="user", content="objective"),
    ]
    assert collect_available_evidence_ids(messages) == ()


def test_collect_available_evidence_ids_ignores_incomplete_exchange() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content="x",
            tool_calls=[{"id": "call_abc"}],
        ),
        _tool_observation(tool_call_id="call_abc", evidence_reference="evidence.a"),
        ChatMessage(
            role="tool",
            content=json.dumps({"evidence_reference": "evidence.b"}),
            tool_call_id="call_def",
        ),
    ]
    assert collect_available_evidence_ids(messages) == ()


def test_build_completed_observation_reference_index() -> None:
    messages = [
        _assistant_call(tool_call_id="call_abc"),
        _tool_observation(tool_call_id="call_abc", evidence_reference="evidence.telemetry.x"),
    ]
    assert build_completed_observation_reference_index(messages) == {
        "evidence.telemetry.x": "call_abc",
    }


def test_parse_public_decision_note() -> None:
    parsed = parse_public_decision_note(
        "EVIDENCE_BASIS: evidence-a,evidence-b\n"
        "PURPOSE: verify normalized effect"
    )
    assert parsed.basis_evidence_references == ("evidence-a", "evidence-b")
    assert parsed.public_reason == "verify normalized effect"


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("PURPOSE: only purpose", "exactly two non-empty fields"),
        ("EVIDENCE_BASIS: evidence-a", "exactly two non-empty fields"),
        ("EVIDENCE_BASIS: evidence-a\nPURPOSE:", "empty PURPOSE"),
        ("EVIDENCE_BASIS: evidence-a,\nPURPOSE: x", "empty EVIDENCE_BASIS id segment"),
        (
            "commentary\nEVIDENCE_BASIS: evidence-a\nPURPOSE: inspect",
            "exactly two non-empty fields",
        ),
        (
            "EVIDENCE_BASIS: evidence-a\nPURPOSE: inspect\ncommentary",
            "exactly two non-empty fields",
        ),
        (
            "EVIDENCE_BASIS: evidence-a\nEVIDENCE_BASIS: evidence-b\nPURPOSE: inspect",
            "exactly two non-empty fields",
        ),
        (
            "EVIDENCE_BASIS: evidence-a\nPURPOSE: inspect\nPURPOSE: inspect again",
            "exactly two non-empty fields",
        ),
        (
            "EVIDENCE_BASIS: evidence-a\nPURPOSE: inspect telemetry\nand compare equipment behavior",
            "exactly two non-empty fields",
        ),
        (
            "PURPOSE: inspect telemetry\nand compare equipment behavior",
            "missing EVIDENCE_BASIS",
        ),
        (
            "EVIDENCE_BASIS: evidence-a,\nevidence-b\nPURPOSE: inspect",
            "exactly two non-empty fields",
        ),
        (
            "PURPOSE: inspect\nEVIDENCE_BASIS: evidence-a",
            "missing EVIDENCE_BASIS",
        ),
    ],
)
def test_malformed_public_decision_note_rejected(content: str, match: str) -> None:
    with pytest.raises(InvestigationProofValidationError, match=match):
        parse_public_decision_note(content)


@pytest.mark.parametrize(
    ("content", "basis", "purpose"),
    [
        ("EVIDENCE_BASIS: evidence-a\n\nPURPOSE: inspect", ("evidence-a",), "inspect"),
        (
            "\nEVIDENCE_BASIS: evidence-a\nPURPOSE: inspect\n",
            ("evidence-a",),
            "inspect",
        ),
        (
            "\n\nEVIDENCE_BASIS: evidence-a\n\n\nPURPOSE: inspect\n\n",
            ("evidence-a",),
            "inspect",
        ),
        (
            "   EVIDENCE_BASIS: evidence-a   \n   PURPOSE: inspect   ",
            ("evidence-a",),
            "inspect",
        ),
        ("EVIDENCE_BASIS: evidence-a\r\nPURPOSE: inspect", ("evidence-a",), "inspect"),
    ],
)
def test_parse_public_decision_note_accepts_layout_whitespace(
    content: str,
    basis: tuple[str, ...],
    purpose: str,
) -> None:
    parsed = parse_public_decision_note(content)
    assert parsed.basis_evidence_references == basis
    assert parsed.public_reason == purpose


def test_parse_public_decision_note_qwen_blank_line_fixture() -> None:
    """Regression for Qwen-style blank serialization lines between fields."""
    parsed = parse_public_decision_note(
        "EVIDENCE_BASIS: observation.probe.a.step-1\n"
        "\n"
        "PURPOSE: inspect equipment degradation"
    )
    assert parsed.basis_evidence_references == ("observation.probe.a.step-1",)
    assert parsed.public_reason == "inspect equipment degradation"


def test_unknown_basis_reference_rejected() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="unknown basis evidence reference",
    ):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS: missing-id\nPURPOSE: inspect subgroup",
            available_evidence_references=frozenset({"evidence.telemetry.x"}),
            reference_index={"evidence.telemetry.x": "call_abc"},
            next_tool_call_ids=("call_def",),
        )


def test_unknown_machine_alias_tool_response_0_rejected() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="unknown basis evidence reference: tool_response_0",
    ):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content=(
                "EVIDENCE_BASIS: tool_response_0\n"
                "PURPOSE: inspect equipment degradation"
            ),
            available_evidence_references=frozenset({"evidence.telemetry.x"}),
            reference_index={"evidence.telemetry.x": "call_abc"},
            next_tool_call_ids=("call_def",),
        )


def test_valid_semantic_basis_binds_canonical_tool_call_id() -> None:
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content=(
            "EVIDENCE_BASIS: evidence.telemetry.x\n"
            "PURPOSE: inspect equipment degradation"
        ),
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_def",
                name="probe.b",
                arguments={"confirm": True},
            ),
        ),
        messages_before_round=[
            _assistant_call(tool_call_id="call_abc"),
            _tool_observation(
                tool_call_id="call_abc",
                evidence_reference="evidence.telemetry.x",
            ),
        ],
    )
    assert step.declared_basis_references == ("evidence.telemetry.x",)
    assert step.basis_bindings == (
        InvestigationEvidenceBasis(
            declared_reference="evidence.telemetry.x",
            tool_call_id="call_abc",
        ),
    )
    assert step.basis_tool_call_ids == ("call_abc",)
    assert step.next_tool_call_ids == ("call_def",)


def test_multiple_basis_references_bind_in_declaration_order() -> None:
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content=(
            "EVIDENCE_BASIS: evidence.workload.x,evidence.throughput.x\n"
            "PURPOSE: compare workload and throughput"
        ),
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_c",
                name="probe.c",
                arguments={"confirm": True},
            ),
        ),
        messages_before_round=[
            _assistant_call(tool_call_id="call_a", tool_name="probe.a"),
            _tool_observation(tool_call_id="call_a", evidence_reference="evidence.workload.x"),
            _assistant_call(tool_call_id="call_b", tool_name="probe.b"),
            _tool_observation(
                tool_call_id="call_b",
                evidence_reference="evidence.throughput.x",
            ),
        ],
    )
    assert step.declared_basis_references == (
        "evidence.workload.x",
        "evidence.throughput.x",
    )
    assert step.basis_tool_call_ids == ("call_a", "call_b")


def test_available_but_not_declared_basis_is_not_auto_bound() -> None:
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content="EVIDENCE_BASIS: evidence.b\nPURPOSE: inspect subgroup",
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_d",
                name="probe.d",
                arguments={"confirm": True},
            ),
        ),
        messages_before_round=[
            _assistant_call(tool_call_id="call_a"),
            _tool_observation(tool_call_id="call_a", evidence_reference="evidence.a"),
            _assistant_call(tool_call_id="call_b"),
            _tool_observation(tool_call_id="call_b", evidence_reference="evidence.b"),
            _assistant_call(tool_call_id="call_c"),
            _tool_observation(tool_call_id="call_c", evidence_reference="evidence.c"),
        ],
    )
    assert step.declared_basis_references == ("evidence.b",)
    assert step.basis_tool_call_ids == ("call_b",)


def test_empty_follow_up_basis_rejected_when_prior_evidence_exists() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="follow-up tool round requires explicit evidence basis",
    ):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS:\nPURPOSE: inspect subgroup",
            available_evidence_references=frozenset({"evidence.a"}),
            reference_index={"evidence.a": "call_abc"},
            next_tool_call_ids=("call_def",),
        )


def test_empty_follow_up_basis_diagnostic_includes_counts() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match=(
            "round_index=2, available_evidence_count=2, basis_count=0"
        ),
    ):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS:\nPURPOSE: inspect staffing",
            available_evidence_references=frozenset({"evidence.a", "evidence.b"}),
            reference_index={"evidence.a": "call_a", "evidence.b": "call_b"},
            next_tool_call_ids=("call_def",),
        )


def test_format_investigation_follow_up_context_lists_available_refs() -> None:
    rendered = format_investigation_follow_up_context(
        round_index=2,
        available_evidence_references=(
            "evidence.workload.line4.incident_window",
            "evidence.throughput.line4.incident_window",
        ),
    )
    assert "ENG6_FOLLOW_UP_CONTEXT" in rendered
    assert "ROUND: 2" in rendered
    assert "- evidence.workload.line4.incident_window" in rendered
    assert "- evidence.throughput.line4.incident_window" in rendered
    assert "AVAILABLE_EVIDENCE_REFS" in rendered
    assert "An empty EVIDENCE_BASIS is invalid." in rendered
    assert "materially motivate this follow-up action" in rendered


def test_prepare_native_planner_messages_appends_follow_up_context() -> None:
    workload_ref = "evidence.workload.line4.incident_window"
    messages = [
        ChatMessage(role="user", content="investigate"),
    ]
    prepared = prepare_native_planner_messages_with_follow_up_context(
        messages,
        round_index=1,
        prior_model_visible_references=(
            ModelVisibleEvidenceReference(
                evidence_reference=workload_ref,
                acquisition_id="baseline_production_workload_read_0",
            ),
        ),
    )
    assert len(prepared) == 2
    assert prepared[-1].role == "system"
    assert workload_ref in prepared[-1].content
    assert "ENG6_FOLLOW_UP_CONTEXT" in prepared[-1].content


def test_prepare_native_planner_messages_skips_context_without_prior_evidence() -> None:
    messages = [ChatMessage(role="user", content="investigate")]
    prepared = prepare_native_planner_messages_with_follow_up_context(
        messages,
        round_index=1,
    )
    assert prepared == messages


def test_available_refs_in_context_match_validator_inventory() -> None:
    messages = [
        _assistant_call(tool_call_id="call_a"),
        _tool_observation(tool_call_id="call_a", evidence_reference="evidence.a"),
        _assistant_call(tool_call_id="call_b", tool_name="probe.b"),
        _tool_observation(tool_call_id="call_b", evidence_reference="evidence.b"),
    ]
    prepared = prepare_native_planner_messages_with_follow_up_context(
        messages,
        round_index=2,
    )
    index = build_completed_observation_reference_index(messages)
    for reference in index:
        assert f"- {reference}" in prepared[-1].content


def test_first_native_round_with_baseline_inventory_requires_basis() -> None:
    workload_ref = "evidence.workload.line4.incident_window"
    with pytest.raises(
        InvestigationProofValidationError,
        match="follow-up tool round requires explicit evidence basis",
    ):
        build_investigation_proof_step(
            round_index=1,
            assistant_content="EVIDENCE_BASIS:\nPURPOSE: inspect staffing",
            tool_calls=(
                LLMToolCall.from_openai_shape(
                    call_id="call_staffing",
                    name="production.staffing.schedule.read",
                    arguments={"line_id": "line4"},
                ),
            ),
            messages_before_round=[
                ChatMessage(role="user", content="investigate"),
            ],
            prior_model_visible_references=(
                ModelVisibleEvidenceReference(
                    evidence_reference=workload_ref,
                    acquisition_id="baseline_production_workload_read_0",
                ),
            ),
        )


def test_first_native_round_without_prior_evidence_allows_empty_basis() -> None:
    step = build_investigation_proof_step(
        round_index=1,
        assistant_content="PURPOSE: gather initial telemetry",
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_a",
                name="probe.a",
                arguments={"label": "a"},
            ),
        ),
        messages_before_round=[ChatMessage(role="user", content="investigate")],
    )
    assert step.declared_basis_references == ()
    assert step.public_reason == "gather initial telemetry"


def test_independent_hypothesis_with_explicit_basis_passes() -> None:
    workload_ref = "evidence.workload.line4.incident_window"
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content=(
            f"EVIDENCE_BASIS: {workload_ref}\n"
            "PURPOSE: test staffing explanation"
        ),
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_staffing",
                name="production.staffing.schedule.read",
                arguments={"line_id": "line4"},
            ),
        ),
        messages_before_round=[
            _assistant_call(tool_call_id="call_a"),
            _tool_observation(tool_call_id="call_a", evidence_reference=workload_ref),
        ],
    )
    assert step.declared_basis_references == (workload_ref,)
    assert step.public_reason == "test staffing explanation"


def test_generic_observation_reference_in_available_inventory_passes() -> None:
    observation_ref = mint_runtime_observation_evidence_reference(
        tool_id="probe.a",
        step_id="run:loop1:tool",
    )
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content=(
            f"EVIDENCE_BASIS: {observation_ref}\n"
            "PURPOSE: confirm observation"
        ),
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_b",
                name="probe.b",
                arguments={"confirm": True},
            ),
        ),
        messages_before_round=[
            _assistant_call(tool_call_id="call_a"),
            _tool_observation(tool_call_id="call_a", evidence_reference=observation_ref),
        ],
    )
    assert step.declared_basis_references == (observation_ref,)


def test_duplicate_basis_reference_rejected() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="duplicate basis evidence reference",
    ):
        validate_follow_up_investigation_step(
            round_index=2,
            assistant_content="EVIDENCE_BASIS: evidence-a,evidence-a\nPURPOSE: inspect subgroup",
            available_evidence_references=frozenset({"evidence-a"}),
            reference_index={"evidence-a": "call_abc"},
            next_tool_call_ids=("call_def",),
        )


def test_mint_runtime_observation_evidence_reference_is_stable() -> None:
    reference = mint_runtime_observation_evidence_reference(
        tool_id="probe.a",
        step_id="run:loop1:tool",
    )
    assert reference == "observation.probe.a.run:loop1:tool"


def test_validate_native_tool_plan_alignment_name_mismatch() -> None:
    with pytest.raises(NativeToolPlanAlignmentError, match="name mismatch"):
        validate_native_tool_plan_alignment(
            (
                LLMToolCall.from_openai_shape(
                    call_id="evidence-b",
                    name="probe.b",
                    arguments={"label": "b"},
                ),
            ),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.c",
                        input=_EvidenceIn(label="b"),
                    )
                ]
            ),
        )


def test_validate_native_tool_plan_alignment_count_mismatch() -> None:
    with pytest.raises(NativeToolPlanAlignmentError, match="count does not match"):
        validate_native_tool_plan_alignment(
            (
                LLMToolCall.from_openai_shape(
                    call_id="evidence-a",
                    name="probe.a",
                    arguments={"label": "a"},
                ),
                LLMToolCall.from_openai_shape(
                    call_id="evidence-b",
                    name="probe.b",
                    arguments={"label": "b"},
                ),
            ),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(label="a"),
                    )
                ]
            ),
        )


def test_validate_native_tool_plan_alignment_argument_mismatch() -> None:
    with pytest.raises(NativeToolPlanAlignmentError, match="arguments mismatch"):
        validate_native_tool_plan_alignment(
            (
                LLMToolCall.from_openai_shape(
                    call_id="evidence-a",
                    name="probe.a",
                    arguments={"label": "wrong"},
                ),
            ),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(label="expected"),
                    )
                ]
            ),
        )


def test_validate_native_tool_plan_alignment_valid() -> None:
    validate_native_tool_plan_alignment(
        (
            LLMToolCall.from_openai_shape(
                call_id="evidence-a",
                name="probe.a",
                arguments={"label": "expected"},
            ),
        ),
        ToolCallPlan(
            calls=[
                PlannedToolCall(
                    step_id="tool",
                    tool_id="probe.a",
                    input=_EvidenceIn(label="expected"),
                )
            ]
        ),
    )


def test_validate_native_tool_plan_alignment_malformed_json_rejected() -> None:
    with pytest.raises(NativeToolPlanAlignmentError, match="malformed"):
        validate_native_tool_plan_alignment(
            (
                LLMToolCall(
                    id="tc-1",
                    name="probe.a",
                    arguments_json="{BROKEN",
                ),
            ),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(),
                    )
                ]
            ),
        )


@pytest.mark.parametrize("arguments_json", ["[]", '"x"'])
def test_validate_native_tool_plan_alignment_non_object_json_rejected(
    arguments_json: str,
) -> None:
    with pytest.raises(NativeToolPlanAlignmentError, match="JSON object"):
        validate_native_tool_plan_alignment(
            (
                LLMToolCall(
                    id="tc-1",
                    name="probe.a",
                    arguments_json=arguments_json,
                ),
            ),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_EvidenceIn(),
                    )
                ]
            ),
        )


def test_validate_native_tool_plan_alignment_empty_json_with_defaults_valid() -> None:
    validate_native_tool_plan_alignment(
        (
            LLMToolCall(
                id="tc-1",
                name="probe.a",
                arguments_json="{}",
            ),
        ),
        ToolCallPlan(
            calls=[
                PlannedToolCall(
                    step_id="tool",
                    tool_id="probe.a",
                    input=_EvidenceIn(),
                )
            ]
        ),
    )


def test_baseline_workload_reference_binds_from_prior_inventory() -> None:
    workload_ref = "evidence.workload.line4.incident_window"
    step = build_investigation_proof_step(
        round_index=2,
        assistant_content=(
            f"EVIDENCE_BASIS: {workload_ref}\n"
            "PURPOSE: inspect staffing implications of workload pressure"
        ),
        tool_calls=(
            LLMToolCall.from_openai_shape(
                call_id="call_staffing",
                name="production.staffing.schedule.read",
                arguments={"line_id": "line4"},
            ),
        ),
        messages_before_round=[
            ChatMessage(role="system", content="Already gathered evidence IDs:\n- workload"),
            ChatMessage(role="user", content="investigate"),
        ],
        prior_model_visible_references=(
            ModelVisibleEvidenceReference(
                evidence_reference=workload_ref,
                acquisition_id="baseline_production_workload_read_0",
            ),
        ),
    )
    assert step.declared_basis_references == (workload_ref,)
    assert step.basis_tool_call_ids == ("baseline_production_workload_read_0",)


def test_evidence_known_but_not_observed_fails_closed() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="unknown basis evidence reference",
    ):
        build_investigation_proof_step(
            round_index=2,
            assistant_content=(
                "EVIDENCE_BASIS: evidence.workload.line4.incident_window\n"
                "PURPOSE: inspect subgroup"
            ),
            tool_calls=(
                LLMToolCall.from_openai_shape(
                    call_id="call_next",
                    name="probe.b",
                    arguments={"confirm": True},
                ),
            ),
            messages_before_round=[
                ChatMessage(role="user", content="investigate"),
            ],
        )


def test_duplicate_reference_with_conflicting_provenance_fails() -> None:
    with pytest.raises(
        InvestigationProofValidationError,
        match="ambiguous evidence reference provenance",
    ):
        build_completed_observation_reference_index(
            [],
            prior_references=(
                ModelVisibleEvidenceReference(
                    evidence_reference="evidence.workload.line4.incident_window",
                    acquisition_id="baseline_production_workload_read_0",
                ),
                ModelVisibleEvidenceReference(
                    evidence_reference="evidence.workload.line4.incident_window",
                    acquisition_id="baseline_production_workload_read_1",
                ),
            ),
        )


def test_domain_identity_wins_over_generic_observation_fallback() -> None:
    messages = [
        _assistant_call(tool_call_id="call_a"),
        _tool_observation(
            tool_call_id="call_a",
            evidence_reference="evidence.workload.line4.incident_window",
        ),
    ]
    available = collect_available_evidence_ids(messages)
    assert available == ("evidence.workload.line4.incident_window",)
    assert "observation.production.workload.read" not in available
