# © Artur Czarnecki. All rights reserved.

"""PP-3C-R1 — regression tests closing false-PASS evaluator gaps."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.investigation_proof import InvestigationProof, InvestigationProofStep
from platform_proofs.tools.iterative_sql_investigation.contracts import (
    PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
    SqlQueryInput,
)
from platform_proofs.tools.iterative_sql_investigation.evaluator import (
    build_execution_snapshot,
    evaluate_scenario,
    investigation_proof_passes_eng6_chain,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import ScenarioId

pytestmark = pytest.mark.unit


def _sql_traces(*sql_queries: str) -> tuple[ToolCallTrace, ...]:
    return tuple(
        ToolCallTrace(
            tool_name=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
            arguments=SqlQueryInput(sql=sql).model_dump(),
            output_preview="North express long_haul rate 0.68",
            success=True,
            error_message=None,
            raw_trace={},
        )
        for sql in sql_queries
    )


def _scenario_a_proof() -> InvestigationProof:
    return InvestigationProof(
        steps=(
            InvestigationProofStep(
                round_index=1,
                basis_tool_call_ids=(),
                next_tool_call_ids=("tc-1",),
                public_reason="compare regions",
            ),
            InvestigationProofStep(
                round_index=2,
                basis_tool_call_ids=("tc-1",),
                next_tool_call_ids=("tc-2",),
                public_reason="inspect segment",
            ),
            InvestigationProofStep(
                round_index=3,
                basis_tool_call_ids=("tc-1", "tc-2"),
                next_tool_call_ids=("tc-3",),
                public_reason="hub rates",
            ),
        ),
        final_available_evidence_ids=("tc-1", "tc-2", "tc-3"),
    )


def _scenario_a_sql() -> tuple[str, ...]:
    return (
        "SELECT region, AVG(delayed::int) FROM proof.parcel_events GROUP BY region",
        "SELECT origin_hub, AVG(delayed::int) FROM proof.parcel_events GROUP BY origin_hub",
        (
            "SELECT service_type, route_type, AVG(delayed::int) FROM proof.parcel_events "
            "WHERE region='North' GROUP BY service_type, route_type"
        ),
    )


def test_scenario_a1_correct_sql_wrong_final_conclusion_fails() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces(*_scenario_a_sql()),
        investigation_proof=_scenario_a_proof(),
        stop_reason="planner_final_answer",
        final_answer="South standard local segment drives North delays; volume is unrelated.",
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is False
    assert result.outcome_a is not None
    assert result.outcome_a.identifies_north_anomalous_segment is True
    assert result.outcome_a.conclusion_supported is False


def test_scenario_a2_hub_sql_volume_root_cause_final_fails() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces(*_scenario_a_sql()),
        investigation_proof=_scenario_a_proof(),
        stop_reason="planner_final_answer",
        final_answer=(
            "North-Volume is the primary root cause of elevated North delays; "
            "the express long_haul segment is secondary."
        ),
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is False
    assert result.outcome_a is not None
    assert result.outcome_a.rejects_volume_only_explanation is False


def test_scenario_a3_supported_evidence_and_conclusion_passes() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces(*_scenario_a_sql()),
        investigation_proof=_scenario_a_proof(),
        stop_reason="planner_final_answer",
        final_answer=(
            "North delays are driven by the North express long_haul segment; "
            "normalized hub rates falsify a volume-only explanation."
        ),
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is True
    assert result.outcome_a is not None
    assert result.outcome_a.conclusion_supported is True


def test_scenario_a_cannot_pass_from_sql_keywords_alone() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces(*_scenario_a_sql()),
        investigation_proof=_scenario_a_proof(),
        stop_reason="planner_final_answer",
        final_answer="Investigation incomplete; no supported operational explanation identified.",
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is False
    assert result.outcome_a is not None
    assert result.outcome_a.identifies_north_anomalous_segment is True
    assert result.outcome_a.conclusion_supported is False


def test_scenario_b1_global_association_only_fails() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces(
            "SELECT weight_kg, AVG(delayed::int) AS delay_rate FROM proof.parcel_events GROUP BY weight_kg",
        ),
        investigation_proof=None,
        stop_reason="planner_final_answer",
        final_answer=(
            "Heavier parcels correlate with delays globally, but direct causation is not established."
        ),
    )
    result = evaluate_scenario(ScenarioId.B, snapshot)
    assert result.passed is False
    assert result.outcome_b is not None
    assert result.outcome_b.detects_global_association is True
    assert result.outcome_b.verifies_segmented_evidence is False


def test_scenario_b2_global_plus_segmented_control_no_causation_passes() -> None:
    proof = InvestigationProof(
        steps=(
            InvestigationProofStep(
                round_index=1,
                basis_tool_call_ids=(),
                next_tool_call_ids=("tc-1",),
                public_reason="global weight-delay association",
            ),
            InvestigationProofStep(
                round_index=2,
                basis_tool_call_ids=("tc-1",),
                next_tool_call_ids=("tc-2",),
                public_reason="segmented control by service and route",
            ),
        ),
        final_available_evidence_ids=("tc-1", "tc-2"),
    )
    snapshot = build_execution_snapshot(
        traces=_sql_traces(
            "SELECT weight_kg, AVG(delayed::int) FROM proof.parcel_events GROUP BY weight_kg",
            (
                "SELECT route_type, service_type, weight_kg, AVG(delayed::int) "
                "FROM proof.parcel_events GROUP BY route_type, service_type, weight_kg"
            ),
        ),
        investigation_proof=proof,
        stop_reason="planner_final_answer",
        final_answer=(
            "Weight correlates with delay globally, but within service_type and route_type segments "
            "the association weakens — confounding, not direct causation."
        ),
    )
    result = evaluate_scenario(ScenarioId.B, snapshot)
    assert result.passed is True
    assert result.outcome_b is not None
    assert result.outcome_b.verifies_segmented_evidence is True
    assert result.outcome_b.claims_direct_causation is False


def test_max_iterations_without_final_answer_fails() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces(*_scenario_a_sql()),
        investigation_proof=_scenario_a_proof(),
        stop_reason="max_iterations",
        final_answer="",
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is False
    assert result.platform_invariants_pass is False
    assert any(reason.startswith("incomplete_termination:") for reason in result.failure_reasons)


def test_investigation_proof_invalid_basis_reference_fails() -> None:
    malformed = InvestigationProof(
        steps=(
            InvestigationProofStep(
                round_index=1,
                basis_tool_call_ids=(),
                next_tool_call_ids=("tc-1",),
                public_reason="first query",
            ),
            InvestigationProofStep(
                round_index=2,
                basis_tool_call_ids=("missing-id",),
                next_tool_call_ids=("tc-2",),
                public_reason="follow-up",
            ),
        ),
        final_available_evidence_ids=("tc-1", "tc-2"),
    )
    assert investigation_proof_passes_eng6_chain(malformed) is False
    snapshot = build_execution_snapshot(
        traces=_sql_traces(*_scenario_a_sql()),
        investigation_proof=malformed,
        stop_reason="planner_final_answer",
        final_answer=(
            "North delays are driven by the North express long_haul segment; "
            "volume alone does not explain the anomaly."
        ),
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is False
    assert "follow_up_missing_evidence_basis" in result.failure_reasons
