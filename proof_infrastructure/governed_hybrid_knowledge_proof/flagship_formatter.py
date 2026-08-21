# © Artur Czarnecki. All rights reserved.

"""Format persisted flagship runs into compact CTO-demo proof sections."""

from __future__ import annotations

from datetime import datetime

from intergrax.runtime.evidence.obligation_derivation_contracts import (
    MaxAgeTemporalConstraintV1,
    PolicyEvidenceBasisV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    LiveCallFailureV1,
    PersistedLiveEvidenceProvenanceV2,
    RequiredEvidenceEvaluationV1,
    WorkspaceAskRunV2,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    LiveEvidenceRequirementV1,
    RequiredEvidenceObligationV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_models import (
    FlagshipRequirementProofV1,
    FlagshipScenarioProofV1,
    FlagshipScenarioIdV1,
)


def _obligation_for_requirement(
    run: WorkspaceAskRunV2,
    requirement_id: str,
) -> RequiredEvidenceObligationV1 | None:
    for obligation in run.required_evidence_obligations:
        if obligation.requirement_id == requirement_id:
            return obligation
    return None


def _evaluation_for_requirement(
    run: WorkspaceAskRunV2,
    requirement_id: str,
) -> RequiredEvidenceEvaluationV1 | None:
    if run.evidence_admissibility is None:
        return None
    for evaluation in run.evidence_admissibility.requirement_evaluations:
        if evaluation.requirement_id == requirement_id:
            return evaluation
    return None


def _live_provenance_for_call(
    run: WorkspaceAskRunV2,
    call_id: str,
) -> PersistedLiveEvidenceProvenanceV2 | None:
    for item in run.persisted_evidence:
        if isinstance(item, PersistedLiveEvidenceProvenanceV2) and item.call_id == call_id:
            return item
    return None


def _format_temporal_facts(
  run: WorkspaceAskRunV2,
  obligation: LiveEvidenceRequirementV1,
) -> tuple[str | None, str | None]:
    provenance = _live_provenance_for_call(run, obligation.call_id)
    evaluated_at = (
        run.evidence_admissibility.evaluated_at.isoformat()
        if run.evidence_admissibility is not None
        else None
    )
    effective_at = None
    if provenance is not None and provenance.temporal is not None:
        if provenance.temporal.kind == "point_in_time":
            effective_at = provenance.temporal.effective_at.isoformat()
        elif provenance.temporal.kind == "validity_interval":
            effective_at = provenance.temporal.valid_from.isoformat()
    return effective_at, evaluated_at


def build_requirement_proof(
    run: WorkspaceAskRunV2,
    requirement_id: str,
) -> FlagshipRequirementProofV1:
    obligation = _obligation_for_requirement(run, requirement_id)
    evaluation = _evaluation_for_requirement(run, requirement_id)
    if obligation is None or not isinstance(obligation, LiveEvidenceRequirementV1):
        return FlagshipRequirementProofV1(requirement_id=requirement_id)
    provenance = _live_provenance_for_call(run, obligation.call_id)
    effective_at, evaluated_at = _format_temporal_facts(run, obligation)
    origin = obligation.policy_origin
    return FlagshipRequirementProofV1(
        requirement_id=requirement_id,
        source_connection_ref=provenance.connection_ref if provenance is not None else None,
        capability_id=provenance.capability_id if provenance is not None else None,
        call_id=obligation.call_id,
        policy_document_id=origin.policy_document_id if origin is not None else None,
        policy_revision_id=origin.revision_id if origin is not None else None,
        policy_rule_id=origin.rule_id if origin is not None else None,
        temporal_constraint=obligation.temporal_constraint,
        temporal_effective_at=effective_at,
        temporal_evaluated_at=evaluated_at,
        outcome=evaluation.status if evaluation is not None else None,
        reason=evaluation.reason_code if evaluation is not None else None,
    )


def build_scenario_proof(
    *,
    scenario_id: FlagshipScenarioIdV1,
    run: WorkspaceAskRunV2,
    llm_calls: int,
    passed: bool,
    detail: str,
) -> FlagshipScenarioProofV1:
    requirements = tuple(
        build_requirement_proof(run, obligation.requirement_id)
        for obligation in run.required_evidence_obligations
    )
    basis = run.policy_basis
    return FlagshipScenarioProofV1(
        scenario_id=scenario_id,
        policy_basis=basis,
        derivation_snapshot_id=basis.derivation_snapshot_id if basis is not None else None,
        requirements=requirements,
        overall_admissibility=(
            run.evidence_admissibility.overall_status
            if run.evidence_admissibility is not None
            else None
        ),
        llm_calls=llm_calls,
        answer=run.answer,
        run_id=run.run_id,
        passed=passed,
        detail=detail,
    )


def format_scenario_section(proof: FlagshipScenarioProofV1) -> str:
    lines = [
        f"SCENARIO: {proof.scenario_id.value}",
        "POLICY BASIS:",
    ]
    if proof.policy_basis is not None:
        for revision in proof.policy_basis.policy_revisions:
            lines.append(
                f"  {revision.policy_document_id} revision={revision.revision_id}"
            )
    else:
        lines.append("  none")
    lines.append(f"DERIVATION SNAPSHOT: {proof.derivation_snapshot_id or 'none'}")
    lines.append("REQUIREMENTS:")
    for requirement in proof.requirements:
        lines.append(f"REQ-{requirement.requirement_id}")
        lines.append(f"  source: {requirement.source_connection_ref or 'none'}")
        lines.append(f"  capability: {requirement.capability_id or 'none'}")
        lines.append(f"  call_id: {requirement.call_id or 'none'}")
        if requirement.policy_document_id is not None:
            lines.append(
                "  policy_origin: "
                f"{requirement.policy_document_id}:"
                f"{requirement.policy_revision_id}:"
                f"{requirement.policy_rule_id}"
            )
        if requirement.temporal_constraint is not None:
            if isinstance(requirement.temporal_constraint, MaxAgeTemporalConstraintV1):
                lines.append(
                    "  temporal_constraint: max_age="
                    f"{requirement.temporal_constraint.max_age_seconds}s"
                )
            else:
                lines.append(
                    f"  temporal_constraint: {requirement.temporal_constraint.kind}"
                )
        if requirement.temporal_effective_at is not None:
            lines.append(f"  effective_at: {requirement.temporal_effective_at}")
        if requirement.temporal_evaluated_at is not None:
            lines.append(f"  evaluated_at: {requirement.temporal_evaluated_at}")
        lines.append(
            f"  outcome: {requirement.outcome.value if requirement.outcome else 'none'}"
        )
        lines.append(
            f"  reason: {requirement.reason.value if requirement.reason else 'none'}"
        )
    lines.append(
        "OVERALL ADMISSIBILITY: "
        f"{proof.overall_admissibility.value if proof.overall_admissibility else 'none'}"
    )
    lines.append(f"LLM CALLS: {proof.llm_calls}")
    lines.append(f"ANSWER: {proof.answer or 'none'}")
    lines.append(f"RUN ID: {proof.run_id}")
    lines.append(f"RESULT: {'PASS' if proof.passed else 'FAIL'} — {proof.detail}")
    return "\n".join(lines)


def build_history_comparison(
    *,
    rev17: FlagshipScenarioProofV1,
    rev18: FlagshipScenarioProofV1,
) -> str:
    rev17_security = next(
        (
            item
            for item in rev17.requirements
            if item.requirement_id.endswith(":security")
        ),
        None,
    )
    rev18_security = next(
        (
            item
            for item in rev18.requirements
            if item.requirement_id.endswith(":security")
        ),
        None,
    )
    lines = [
        "STRUCTURAL HISTORY COMPARISON",
        "REV17:",
        f"  policy basis revision security={rev17_security.policy_revision_id if rev17_security else 'none'}",
        f"  derivation snapshot={rev17.derivation_snapshot_id}",
        f"  security temporal={_temporal_label(rev17_security)}",
        f"  result={rev17.overall_admissibility.value if rev17.overall_admissibility else 'none'}",
        "REV18:",
        f"  policy basis revision security={rev18_security.policy_revision_id if rev18_security else 'none'}",
        f"  derivation snapshot={rev18.derivation_snapshot_id}",
        f"  security temporal={_temporal_label(rev18_security)}",
        f"  result={rev18.overall_admissibility.value if rev18.overall_admissibility else 'none'}",
        "KEY: same requirement_id, different revision/snapshot, different admissibility",
    ]
    return "\n".join(lines)


def _temporal_label(requirement: FlagshipRequirementProofV1 | None) -> str:
    if requirement is None or requirement.temporal_constraint is None:
        return "none"
    if isinstance(requirement.temporal_constraint, MaxAgeTemporalConstraintV1):
        return f"max_age={requirement.temporal_constraint.max_age_seconds}s"
    return requirement.temporal_constraint.kind


def live_call_failure_for_suffix(
    run: WorkspaceAskRunV2,
    suffix: str,
) -> LiveCallFailureV1 | None:
    for failure in run.live_call_failures:
        if failure.call_id.endswith(suffix):
            return failure
    return None


def distinct_live_identities(run: WorkspaceAskRunV2) -> tuple[set[str], set[str], set[str], set[str]]:
    providers: set[str] = set()
    connections: set[str] = set()
    capabilities: set[str] = set()
    call_ids: set[str] = set()
    for item in run.persisted_evidence:
        if isinstance(item, PersistedLiveEvidenceProvenanceV2):
            providers.add(item.provider_id)
            connections.add(item.connection_ref)
            capabilities.add(item.capability_id)
            call_ids.add(item.call_id)
    return providers, connections, capabilities, call_ids


def summary_label_for_admissibility(
    status: EvidenceAdmissibilityStatusV1 | None,
) -> str:
    if status is EvidenceAdmissibilityStatusV1.SATISFIED:
        return "SATISFIED"
    if status is EvidenceAdmissibilityStatusV1.UNSATISFIED:
        return "UNSATISFIED"
    return "UNKNOWN"
