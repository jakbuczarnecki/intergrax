# © Artur Czarnecki. All rights reserved.

"""Deterministic orchestration and evaluator tests — fake LLM only."""

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


def test_scenario_a_passes_with_supported_evidence_chain() -> None:
    traces = _sql_traces(
        "SELECT region, AVG(delayed::int) FROM proof.parcel_events GROUP BY region",
        "SELECT origin_hub, AVG(delayed::int) FROM proof.parcel_events GROUP BY origin_hub",
        (
            "SELECT service_type, route_type, AVG(delayed::int) FROM proof.parcel_events "
            "WHERE region='North' GROUP BY service_type, route_type"
        ),
    )
    proof = InvestigationProof(
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
    snapshot = build_execution_snapshot(
        traces=traces,
        investigation_proof=proof,
        stop_reason="planner_final_answer",
        final_answer=(
            "North delays are driven by the North express long_haul segment; "
            "normalized hub rates falsify a volume-only explanation."
        ),
    )
    result = evaluate_scenario(ScenarioId.A, snapshot)
    assert result.passed is True
    assert result.outcome_a is not None
    assert result.outcome_a.identifies_north_anomalous_segment is True
    assert result.outcome_a.conclusion_supported is True


def test_scenario_b_rejects_direct_causation_claim() -> None:
    traces = _sql_traces(
        "SELECT weight_kg, AVG(delayed::int) FROM proof.parcel_events GROUP BY weight_kg",
        "SELECT route_type, service_type, weight_kg, delayed FROM proof.parcel_events",
    )
    snapshot = build_execution_snapshot(
        traces=traces,
        investigation_proof=None,
        stop_reason="planner_final_answer",
        final_answer="Heavier weight causes delays across the network.",
    )
    result = evaluate_scenario(ScenarioId.B, snapshot)
    assert result.passed is False
    assert result.outcome_b is not None
    assert result.outcome_b.claims_direct_causation is True


def test_scenario_c_requires_missing_evidence_acknowledgement() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces("SELECT column_name FROM information_schema.columns"),
        investigation_proof=None,
        stop_reason="planner_final_answer",
        final_answer="Staffing shortages definitely cause the delays.",
    )
    result = evaluate_scenario(ScenarioId.C, snapshot)
    assert result.passed is False
    assert result.outcome_c is not None
    assert result.outcome_c.claims_staffing_cause is True


def test_scenario_c_passes_with_bounded_limitation() -> None:
    snapshot = build_execution_snapshot(
        traces=_sql_traces("SELECT region, delayed FROM proof.parcel_events LIMIT 10"),
        investigation_proof=None,
        stop_reason="planner_final_answer",
        final_answer=(
            "Available data has no staffing fields, so staffing shortages cannot be established "
            "as the cause of delays from this dataset."
        ),
    )
    result = evaluate_scenario(ScenarioId.C, snapshot)
    assert result.outcome_c is not None
    assert result.outcome_c.reports_missing_evidence is True
    assert result.outcome_c.claims_staffing_cause is False
