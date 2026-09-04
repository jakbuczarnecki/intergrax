# © Artur Czarnecki. All rights reserved.

"""Thin adapter: legacy verification failure → GAP-1A EvidenceChallenge."""

from __future__ import annotations

from intergrax.contracts.evidence_claims import (
    ChallengeDefectFamily,
    ChallengeResolution,
    DefectCode,
    EvidenceChallenge,
    EvidenceChallengeId,
    EvidenceClaimId,
    EvidenceReferenceId,
    mint_evidence_challenge_id,
    validate_defect_code,
    validate_evidence_reference_id,
)
from intergrax.runtime.migration.legacy_critic_contracts import (
    LegacyCriticAction,
    LegacyCriticLayer,
    LegacyCriticScope,
    LegacyCriticVerdict,
    LegacyLayerVerdict,
)
from intergrax.runtime.migration.legacy_critic_trace import (
    LEGACY_CRITIC_STEP_EVALUATOR_LOOP,
    LEGACY_CRITIC_STEP_L0_FAILED,
    LegacyCriticVerdictDiagV1,
)
from intergrax.runtime.nexus.execution.evaluator_loop_metadata import (
    LEGACY_CRITIC_EVALUATOR_LOOP_STEP,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    UNSUPPORTED_INFERENCE_ERROR,
)

UNSUPPORTED_INFERENCE_DEFECT = validate_defect_code("incident.unsupported_inference")


def map_legacy_verdict_to_challenge(
    verdict: LegacyCriticVerdict,
    *,
    claim_id: EvidenceClaimId,
    evidence_ids: tuple[EvidenceReferenceId, ...] = (),
) -> EvidenceChallenge | None:
    if verdict.passed:
        return None

    defect_family = ChallengeDefectFamily.UNSUPPORTED_INFERENCE
    defect_code = UNSUPPORTED_INFERENCE_DEFECT
    for reason in verdict.failure_reasons:
        if reason == UNSUPPORTED_INFERENCE_ERROR:
            defect_family = ChallengeDefectFamily.UNSUPPORTED_INFERENCE
            defect_code = UNSUPPORTED_INFERENCE_DEFECT
            break
        if "missing_claim_set" in reason or "missing_diagnosis" in reason:
            defect_family = ChallengeDefectFamily.MISSING_EVIDENCE
            defect_code = validate_defect_code("incident.missing_claim_material")
            break

    description = "; ".join(verdict.failure_reasons) or "verification_rejected_material_claim"
    return EvidenceChallenge(
        challenge_id=mint_evidence_challenge_id(),
        claim_id=claim_id,
        defect_family=defect_family,
        defect_code=defect_code,
        evidence_ids=evidence_ids,
        description=description,
        resolution=ChallengeResolution.OPEN,
    )


def legacy_verdict_from_validation_errors(
    errors: list[str],
    *,
    node_id: str,
) -> LegacyCriticVerdict | None:
    if not errors:
        return None
    return LegacyCriticVerdict(
        scope=LegacyCriticScope.NODE_PARTIAL,
        passed=False,
        layers=[
            LegacyLayerVerdict(
                layer=LegacyCriticLayer.L0_DETERMINISTIC,
                passed=False,
                errors=list(errors),
            ),
        ],
        recommended_action=LegacyCriticAction.REVISE,
        failure_reasons=list(errors),
    )


def first_failed_node_partial_verdict_from_persisted_trace(
    events: list[dict[str, object]],
    *,
    node_id: str,
) -> LegacyCriticVerdict | None:
    """Return first failed NODE_PARTIAL legacy verdict from persisted trace events."""
    for raw_event in events:
        if raw_event.get("step") != LEGACY_CRITIC_STEP_L0_FAILED:
            continue
        payload = raw_event.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            diag = LegacyCriticVerdictDiagV1(
                scope=str(payload.get("scope", "")),
                passed=bool(payload.get("passed", False)),
                recommended_action=str(payload.get("recommended_action", "")),
                layer=str(payload.get("layer", "")),
                score=payload.get("score"),
                failure_reasons=tuple(str(item) for item in payload.get("failure_reasons", [])),
                agent_id=str(payload.get("agent_id", "")),
                node_id=str(payload.get("node_id")) if payload.get("node_id") else None,
            )
        except (TypeError, ValueError):
            continue
        if diag.passed:
            continue
        if diag.node_id != node_id:
            continue
        if diag.scope != LegacyCriticScope.NODE_PARTIAL.value:
            continue
        reasons = list(diag.failure_reasons)
        if not reasons:
            continue
        return LegacyCriticVerdict(
            scope=LegacyCriticScope.NODE_PARTIAL,
            passed=False,
            layers=[
                LegacyLayerVerdict(
                    layer=LegacyCriticLayer.L0_DETERMINISTIC,
                    passed=False,
                    errors=reasons,
                )
            ],
            recommended_action=LegacyCriticAction(diag.recommended_action),
            failure_reasons=reasons,
        )
    return None


def count_evaluator_loop_iterations_from_persisted_trace(
    events: list[dict[str, object]],
    *,
    node_id: str,
) -> int:
    count = 0
    for raw_event in events:
        step = raw_event.get("step")
        if step not in {LEGACY_CRITIC_STEP_EVALUATOR_LOOP, LEGACY_CRITIC_EVALUATOR_LOOP_STEP}:
            continue
        payload = raw_event.get("payload")
        if isinstance(payload, dict) and payload.get("node_id") == node_id:
            count += 1
    return count


def apply_challenge_lifecycle(
    claim_set: dict[str, object],
    failed_verdict: LegacyCriticVerdict,
    *,
    claim_id: EvidenceClaimId,
    initial_evidence_ids: tuple[EvidenceReferenceId, ...],
    resolving_evidence_ids: tuple[EvidenceReferenceId, ...] = (),
    resolved: bool,
    satisfied_description: str = "Follow-up telemetry gathered via platform tools",
) -> tuple[dict[str, object], EvidenceChallenge | None]:
    """Project verification failure into GAP-1A challenge lifecycle without synthesizing verdicts."""
    open_challenge = map_legacy_verdict_to_challenge(
        failed_verdict,
        claim_id=claim_id,
        evidence_ids=initial_evidence_ids,
    )
    if open_challenge is None:
        return claim_set, None

    from intergrax.contracts.evidence_claims import EvidenceClaimSet

    claim_set_model = EvidenceClaimSet.model_validate(claim_set)
    if not resolved:
        updated = EvidenceClaimSet(
            claims=claim_set_model.claims,
            challenges=(open_challenge,),
        )
        return updated.model_dump(mode="json"), open_challenge

    satisfied_evidence_ids = initial_evidence_ids + resolving_evidence_ids
    satisfied = build_satisfied_challenge(
        open_challenge.challenge_id,
        claim_id=claim_id,
        evidence_ids=satisfied_evidence_ids,
        description=satisfied_description,
    )
    updated = EvidenceClaimSet(
        claims=claim_set_model.claims,
        challenges=(satisfied,),
    )
    return updated.model_dump(mode="json"), satisfied


def build_satisfied_challenge(
    challenge_id: EvidenceChallengeId,
    *,
    claim_id: EvidenceClaimId,
    evidence_ids: tuple[EvidenceReferenceId, ...],
    description: str,
) -> EvidenceChallenge:
    return EvidenceChallenge(
        challenge_id=challenge_id,
        claim_id=claim_id,
        defect_family=ChallengeDefectFamily.UNSUPPORTED_INFERENCE,
        defect_code=UNSUPPORTED_INFERENCE_DEFECT,
        evidence_ids=evidence_ids,
        description=description,
        resolution=ChallengeResolution.SATISFIED,
    )


# Historical aliases retained for platform proof scenario contracts.
map_critic_verdict_to_challenge = map_legacy_verdict_to_challenge
