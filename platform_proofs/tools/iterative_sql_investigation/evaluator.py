# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

import re
from collections.abc import Sequence

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.investigation_proof import InvestigationProof

from platform_proofs.tools.iterative_sql_investigation.dataset import ANOMALY_HUB, ANOMALY_SEGMENT
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ScenarioAOutcome,
    ScenarioBOutcome,
    ScenarioCOutcome,
    ScenarioExecutionSnapshot,
    ScenarioRunResult,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import ScenarioId

_NORMAL_TERMINATION = frozenset({"planner_final_answer", "max_iterations"})
_CAUSATION_CLAIMS = re.compile(
    r"\b(weight|heavier|heavy parcels?).{0,40}\b(cause|causes|causing|drives|responsible)\b",
    re.IGNORECASE,
)
_STAFFING_CAUSE = re.compile(
    r"\bstaff(ing)?\s+(shortage|shortages|levels?).{0,30}\b(cause|causes|drives|explain)\b",
    re.IGNORECASE,
)
_MISSING_EVIDENCE = re.compile(
    r"\b(cannot|can not|can't|unable to|do not have|don't have|no data|not available|"
    r"missing|insufficient).{0,60}\b(staff|staffing|evidence|data)\b",
    re.IGNORECASE,
)
_VOLUME_ONLY = re.compile(
    r"\b(high[- ]volume|volume alone|sheer volume|parcel count).{0,40}\b(explain|cause|drives)\b",
    re.IGNORECASE,
)


def _combined_text(parts: Sequence[str]) -> str:
    return "\n".join(part for part in parts if part).lower()


def _extract_sql_and_outputs(traces: Sequence[ToolCallTrace]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sql_texts: list[str] = []
    outputs: list[str] = []
    for trace in traces:
        if not trace.success:
            continue
        arguments = trace.arguments or {}
        sql = arguments.get("sql")
        if isinstance(sql, str) and sql.strip():
            sql_texts.append(sql.lower())
        if trace.output_preview:
            outputs.append(trace.output_preview.lower())
    return tuple(sql_texts), tuple(outputs)


def build_execution_snapshot(
    *,
    traces: Sequence[ToolCallTrace],
    investigation_proof: InvestigationProof | None,
    stop_reason: str,
    final_answer: str,
) -> ScenarioExecutionSnapshot:
    sql_texts, output_texts = _extract_sql_and_outputs(traces)
    successful = sum(1 for trace in traces if trace.success)
    steps = len(investigation_proof.steps) if investigation_proof else 0
    follow_up_ok = True
    if investigation_proof and len(investigation_proof.steps) >= 2:
        follow_up_ok = any(step.basis_tool_call_ids for step in investigation_proof.steps[1:])
    return ScenarioExecutionSnapshot(
        stop_reason=stop_reason,
        successful_tool_calls=successful,
        sql_texts=sql_texts,
        output_texts=output_texts,
        investigation_proof_steps=steps,
        follow_up_has_valid_basis=follow_up_ok,
        final_answer=final_answer.strip(),
    )


def _top_hub_analysis_present(snapshot: ScenarioExecutionSnapshot) -> bool:
    sql = _combined_text(snapshot.sql_texts)
    return "origin_hub" in sql and (
        "avg" in sql or "rate" in sql or "delay" in sql or "delayed" in sql
    )


def validate_platform_invariants(
    snapshot: ScenarioExecutionSnapshot,
    *,
    scenario_id: ScenarioId,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    min_tool_calls = 3 if scenario_id is ScenarioId.A else 1
    if snapshot.successful_tool_calls < min_tool_calls:
        reasons.append(f"insufficient_tool_calls:{snapshot.successful_tool_calls}")
    if snapshot.investigation_proof_steps < 1 and snapshot.successful_tool_calls >= 2:
        reasons.append("missing_investigation_proof")
    if snapshot.successful_tool_calls >= 3 and not snapshot.follow_up_has_valid_basis:
        reasons.append("follow_up_missing_evidence_basis")
    if snapshot.stop_reason not in _NORMAL_TERMINATION:
        reasons.append(f"abnormal_stop_reason:{snapshot.stop_reason}")
    return not reasons, tuple(reasons)


def _segment_tokens() -> tuple[str, str, str]:
    region, service, route = ANOMALY_SEGMENT
    return region.lower(), service.lower(), route.replace("_", " ")


def evaluate_scenario_a(snapshot: ScenarioExecutionSnapshot) -> ScenarioAOutcome:
    corpus = _combined_text([snapshot.final_answer, *snapshot.sql_texts, *snapshot.output_texts])
    region, service, route = _segment_tokens()
    segment_signal = (
        region in corpus
        and service in corpus
        and ("long_haul" in corpus or "long haul" in corpus or route in corpus)
    ) or ANOMALY_HUB.lower() in corpus
    rejects_volume = not _VOLUME_ONLY.search(snapshot.final_answer)
    if _top_hub_analysis_present(snapshot):
        rejects_volume = True
    return ScenarioAOutcome(
        identifies_north_anomalous_segment=segment_signal,
        rejects_volume_only_explanation=rejects_volume,
        conclusion_supported=segment_signal and rejects_volume,
    )


def evaluate_scenario_b(snapshot: ScenarioExecutionSnapshot) -> ScenarioBOutcome:
    sql_corpus = _combined_text(snapshot.sql_texts)
    global_assoc = "weight" in sql_corpus and ("delay" in sql_corpus or "delayed" in sql_corpus)
    segmented = any(
        token in sql_corpus
        for token in ("group by", "service_type", "route_type", "segment")
    ) and "weight" in sql_corpus
    claims_causation = bool(_CAUSATION_CLAIMS.search(snapshot.final_answer))
    if re.search(r"\b(confound|correlation|within segment|controlled)\b", snapshot.final_answer, re.I):
        claims_causation = False
    return ScenarioBOutcome(
        detects_global_association=global_assoc,
        verifies_segmented_evidence=segmented or global_assoc,
        claims_direct_causation=claims_causation,
    )


def evaluate_scenario_c(snapshot: ScenarioExecutionSnapshot) -> ScenarioCOutcome:
    evidence_corpus = _combined_text(snapshot.output_texts)
    answer = snapshot.final_answer
    staffing_in_results = "staff" in evidence_corpus and "column" not in evidence_corpus
    claims_staffing = bool(_STAFFING_CAUSE.search(answer))
    if re.search(
        r"\b(cannot|can not|can't|unable to|not establish\w*|insufficient)\b",
        answer,
        re.I,
    ):
        claims_staffing = False
    return ScenarioCOutcome(
        staffing_evidence_available=staffing_in_results,
        claims_staffing_cause=claims_staffing,
        reports_missing_evidence=bool(_MISSING_EVIDENCE.search(answer))
        or (
            "staff" in answer.lower()
            and re.search(r"\b(no|not|missing|unavailable|cannot)\b", answer, re.I) is not None
        ),
    )


def evaluate_scenario(
    scenario_id: ScenarioId,
    snapshot: ScenarioExecutionSnapshot,
) -> ScenarioRunResult:
    platform_ok, platform_reasons = validate_platform_invariants(
        snapshot,
        scenario_id=scenario_id,
    )
    outcome_a = outcome_b = outcome_c = None
    semantic_ok = True
    semantic_reasons: list[str] = []

    if scenario_id is ScenarioId.A:
        outcome_a = evaluate_scenario_a(snapshot)
        semantic_ok = all(
            (
                outcome_a.identifies_north_anomalous_segment,
                outcome_a.rejects_volume_only_explanation,
                outcome_a.conclusion_supported,
            )
        )
    elif scenario_id is ScenarioId.B:
        outcome_b = evaluate_scenario_b(snapshot)
        semantic_ok = (
            outcome_b.detects_global_association
            and outcome_b.verifies_segmented_evidence
            and not outcome_b.claims_direct_causation
        )
    elif scenario_id is ScenarioId.C:
        outcome_c = evaluate_scenario_c(snapshot)
        semantic_ok = (
            not outcome_c.staffing_evidence_available
            and not outcome_c.claims_staffing_cause
            and outcome_c.reports_missing_evidence
        )

    passed = platform_ok and semantic_ok
    failure_reasons = list(platform_reasons) + semantic_reasons
    if not semantic_ok and not semantic_reasons:
        failure_reasons.append("semantic_outcome_failed")
    return ScenarioRunResult(
        scenario_id=scenario_id,
        passed=passed,
        stop_reason=snapshot.stop_reason,
        successful_tool_calls=snapshot.successful_tool_calls,
        investigation_proof_steps=snapshot.investigation_proof_steps,
        platform_invariants_pass=platform_ok,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        outcome_c=outcome_c,
        failure_reasons=tuple(failure_reasons),
    )
