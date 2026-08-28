# © Artur Czarnecki. All rights reserved.

"""Thin adapter: platform CriticVerdict → GAP-1A EvidenceChallenge (no lifecycle ownership)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.evidence_claims import (
    EvidenceClaimSet,
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
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
from intergrax.runtime.critic.trace import CriticTraceEmitter, CriticVerdictDiagV1
from intergrax.runtime.critic.trace_steps import CRITIC_STEP_L0_FAILED
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    UNSUPPORTED_INFERENCE_ERROR,
)

UNSUPPORTED_INFERENCE_DEFECT = validate_defect_code("incident.unsupported_inference")


def map_critic_verdict_to_challenge(
    verdict: CriticVerdict,
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

    description = "; ".join(verdict.failure_reasons) or "critic_rejected_material_claim"
    return EvidenceChallenge(
        challenge_id=mint_evidence_challenge_id(),
        claim_id=claim_id,
        defect_family=defect_family,
        defect_code=defect_code,
        evidence_ids=evidence_ids,
        description=description,
        resolution=ChallengeResolution.OPEN,
    )


def first_failed_node_partial_verdict_from_trace(
    emitter: CriticTraceEmitter,
    *,
    node_id: str,
) -> CriticVerdict | None:
    """Return the first failed NODE_PARTIAL critic verdict emitted during graph execution."""
    for event in emitter.events:
        if event.step != CRITIC_STEP_L0_FAILED:
            continue
        payload = event.payload
        if not isinstance(payload, CriticVerdictDiagV1):
            continue
        if payload.passed:
            continue
        if payload.node_id != node_id:
            continue
        if payload.scope != CriticScope.NODE_PARTIAL.value:
            continue
        reasons = list(payload.failure_reasons)
        if not reasons:
            continue
        recommended_action = CriticAction(payload.recommended_action)
        return CriticVerdict(
            scope=CriticScope.NODE_PARTIAL,
            passed=False,
            layers=[
                LayerVerdict(
                    layer=CriticLayer.L0_DETERMINISTIC,
                    passed=False,
                    errors=reasons,
                )
            ],
            recommended_action=recommended_action,
            failure_reasons=reasons,
        )
    return None


def first_failed_node_partial_verdict_from_persisted_trace(
    events: list[dict[str, object]],
    *,
    node_id: str,
) -> CriticVerdict | None:
    """Return first failed NODE_PARTIAL critic verdict from persisted trace events."""
    for raw_event in events:
        if raw_event.get("step") != CRITIC_STEP_L0_FAILED:
            continue
        payload = raw_event.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            diag = CriticVerdictDiagV1(
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
        if diag.scope != CriticScope.NODE_PARTIAL.value:
            continue
        reasons = list(diag.failure_reasons)
        if not reasons:
            continue
        return CriticVerdict(
            scope=CriticScope.NODE_PARTIAL,
            passed=False,
            layers=[
                LayerVerdict(
                    layer=CriticLayer.L0_DETERMINISTIC,
                    passed=False,
                    errors=reasons,
                )
            ],
            recommended_action=CriticAction(diag.recommended_action),
            failure_reasons=reasons,
        )
    return None


def count_evaluator_loop_iterations_from_persisted_trace(
    events: list[dict[str, object]],
    *,
    node_id: str,
) -> int:
    from intergrax.runtime.critic.trace_steps import CRITIC_STEP_EVALUATOR_LOOP

    count = 0
    for raw_event in events:
        if raw_event.get("step") != CRITIC_STEP_EVALUATOR_LOOP:
            continue
        payload = raw_event.get("payload")
        if isinstance(payload, dict) and payload.get("node_id") == node_id:
            count += 1
    return count


def apply_challenge_lifecycle(
    claim_set: dict[str, Any],
    failed_verdict: CriticVerdict,
    *,
    claim_id: EvidenceClaimId,
    initial_evidence_ids: tuple[EvidenceReferenceId, ...],
    resolving_evidence_ids: tuple[EvidenceReferenceId, ...] = (),
    resolved: bool,
    satisfied_description: str = "Follow-up telemetry gathered via platform tools",
) -> tuple[dict[str, Any], EvidenceChallenge | None]:
    """Project critic failure into GAP-1A challenge lifecycle without synthesizing verdicts."""
    open_challenge = map_critic_verdict_to_challenge(
        failed_verdict,
        claim_id=claim_id,
        evidence_ids=initial_evidence_ids,
    )
    if open_challenge is None:
        return claim_set, None

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
