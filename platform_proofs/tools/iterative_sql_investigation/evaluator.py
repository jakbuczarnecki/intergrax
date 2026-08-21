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

_SUCCESSFUL_TERMINATION = frozenset({"planner_final_answer"})
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
    r"\b(high[- ]volume|volume alone|sheer volume|parcel count|north[- ]volume).{0,40}\b(explain|cause|drives|root)\b",
    re.IGNORECASE,
)
_NORTH_VOLUME_ROOT = re.compile(
    r"\b(north[- ]volume).{0,50}\b(cause|causes|drives|explain|root|primary|main)\b",
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


def investigation_proof_passes_eng6_chain(proof: InvestigationProof | None) -> bool:
    """Post-hoc audit of ENG-6 evidence basis invariants on a completed proof."""
    if proof is None or not proof.steps:
        return False
    available: set[str] = set()
    for step in proof.steps:
        if step.round_index > 1:
            if available and not step.basis_tool_call_ids:
                return False
            unknown = [basis_id for basis_id in step.basis_tool_call_ids if basis_id not in available]
            if unknown:
                return False
        available.update(step.next_tool_call_ids)
    if proof.final_available_evidence_ids:
        final = frozenset(proof.final_available_evidence_ids)
        for step in proof.steps[1:]:
            if any(basis_id not in final for basis_id in step.basis_tool_call_ids):
                return False
    return True


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
    follow_up_ok = investigation_proof_passes_eng6_chain(investigation_proof)
    return ScenarioExecutionSnapshot(
        stop_reason=stop_reason,
        successful_tool_calls=successful,
        sql_texts=sql_texts,
        output_texts=output_texts,
        investigation_proof_steps=steps,
        follow_up_has_valid_basis=follow_up_ok,
        final_answer=final_answer.strip(),
    )


def _segment_tokens() -> tuple[str, str, str]:
    region, service, route = ANOMALY_SEGMENT
    return region.lower(), service.lower(), route.replace("_", " ")


def _north_anomaly_evidence_investigation(sql_texts: Sequence[str], output_texts: Sequence[str]) -> bool:
    """Evidence path: SQL/output shows relevant North anomaly investigation."""
    evidence = _combined_text([*sql_texts, *output_texts])
    region, service, route = _segment_tokens()
    segment_probe = (
        region in evidence
        and service in evidence
        and ("long_haul" in evidence or "long haul" in evidence or route in evidence)
    )
    hub_probe = ANOMALY_HUB.lower() in evidence and (
        "origin_hub" in evidence or "hub" in evidence
    )
    return segment_probe or hub_probe


def _final_answer_identifies_north_segment(final_answer: str) -> bool:
    """Final conclusion attributes the anomaly to the North service/route segment."""
    answer = final_answer.lower()
    region, service, route = _segment_tokens()
    has_region = region in answer
    has_service_route = service in answer and (
        "long_haul" in answer or "long haul" in answer or route in answer
    )
    segment_named = has_region and has_service_route
    hub_named = ANOMALY_HUB.lower() in answer and region in answer
    return segment_named or hub_named


def _final_answer_rejects_volume_only(final_answer: str) -> bool:
    """Final conclusion must not treat sheer volume / North-Volume as the root explanation."""
    if _VOLUME_ONLY.search(final_answer):
        return False
    if _NORTH_VOLUME_ROOT.search(final_answer):
        return False
    if re.search(
        r"\bvolume\b.{0,30}\b(alone|only|primary|main|root)\b.{0,30}\b(explain|cause|driver)\b",
        final_answer,
        re.IGNORECASE,
    ):
        return False
    return True


def _global_weight_delay_association(sql_corpus: str) -> bool:
    return "weight" in sql_corpus and ("delay" in sql_corpus or "delayed" in sql_corpus)


def _confounder_control_segmentation(sql_corpus: str) -> bool:
    """Segmented/control analysis over confounder dimensions (service_type / route_type)."""
    if not _global_weight_delay_association(sql_corpus):
        return False
    has_confounder_axis = "service_type" in sql_corpus or "route_type" in sql_corpus
    has_segmentation = "group by" in sql_corpus or "segment" in sql_corpus
    return has_confounder_axis and has_segmentation


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
    if snapshot.stop_reason not in _SUCCESSFUL_TERMINATION:
        reasons.append(f"incomplete_termination:{snapshot.stop_reason}")
    return not reasons, tuple(reasons)


def evaluate_scenario_a(snapshot: ScenarioExecutionSnapshot) -> ScenarioAOutcome:
    evidence_path = _north_anomaly_evidence_investigation(snapshot.sql_texts, snapshot.output_texts)
    final_segment = _final_answer_identifies_north_segment(snapshot.final_answer)
    rejects_volume = _final_answer_rejects_volume_only(snapshot.final_answer)
    return ScenarioAOutcome(
        identifies_north_anomalous_segment=evidence_path,
        rejects_volume_only_explanation=rejects_volume,
        conclusion_supported=evidence_path and final_segment and rejects_volume,
    )


def evaluate_scenario_b(snapshot: ScenarioExecutionSnapshot) -> ScenarioBOutcome:
    sql_corpus = _combined_text(snapshot.sql_texts)
    global_assoc = _global_weight_delay_association(sql_corpus)
    segmented = _confounder_control_segmentation(sql_corpus)
    claims_causation = bool(_CAUSATION_CLAIMS.search(snapshot.final_answer))
    if re.search(r"\b(confound|correlation|within segment|controlled)\b", snapshot.final_answer, re.I):
        claims_causation = False
    return ScenarioBOutcome(
        detects_global_association=global_assoc,
        verifies_segmented_evidence=global_assoc and segmented,
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
