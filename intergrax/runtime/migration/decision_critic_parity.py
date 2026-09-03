# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision/Critic dual-run parity — migration-only observational comparison (DS-MIG parity).

This module exists solely to compare canonical Decision authority outcomes with
legacy Critic shadow observations during CriticOrchestrator retirement qualification.
It must not be imported by Decision core contracts or runtime decision modules.

Scheduled for deletion together with ``intergrax/runtime/critic/**``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, NewType, Protocol, Sequence, TypeVar

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_record import DecisionProposalRef, candidate_decision_ref
from intergrax.contracts.decision_revision import DecisionRevisionDisposition
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
)
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
)
from intergrax.runtime.decision_flow import (
    DecisionFlowHostAction,
    DecisionFlowResult,
    DecisionFlowScope,
)

T = TypeVar("T")

DecisionCriticParityDifferenceCode = NewType("DecisionCriticParityDifferenceCode", str)


def parity_difference_code(value: str) -> DecisionCriticParityDifferenceCode:
    if type(value) is not str:
        raise TypeError("parity difference code must be str")
    if not value or not value.strip():
        raise ValueError("parity difference code must be non-empty")
    return DecisionCriticParityDifferenceCode(value)


DECISION_ACCEPT_CRITIC_CHALLENGE = parity_difference_code(
    "decision_accept_critic_challenge",
)
DECISION_CHALLENGE_CRITIC_ACCEPT = parity_difference_code(
    "decision_challenge_critic_accept",
)
DECISION_SUPERSET_CAPABILITY = parity_difference_code("decision_superset_capability")
CRITIC_CAPABILITY_NOT_EXERCISED_BY_DECISION = parity_difference_code(
    "critic_capability_not_exercised_by_decision",
)
LEGACY_L2_NOT_DECISION_VERIFICATION = parity_difference_code(
    "legacy_l2_not_decision_verification",
)
LEGACY_RETRY_IS_EXECUTION_RETRY = parity_difference_code(
    "legacy_retry_is_execution_retry",
)
LEGACY_REVISE_IS_DECISION_REVISION = parity_difference_code(
    "legacy_revise_is_decision_revision",
)
LEGACY_HITL_IS_DECISION_HUMAN_REVIEW = parity_difference_code(
    "legacy_hitl_is_decision_human_review",
)
SHADOW_PROVIDER_UNAVAILABLE = parity_difference_code("shadow_provider_unavailable")
SHADOW_EXECUTION_ERROR = parity_difference_code("shadow_execution_error")


class ParityHostScope(str, Enum):
    """Parity observation scopes aligned with migrated authority points."""

    GRAPH_FINAL = "graph_final"
    UAEP_STEP = "uaep_step"


class NormalizedParityOutcome(str, Enum):
    """Normalized semantic verification outcome for cross-system comparison."""

    ACCEPTABLE = "acceptable"
    CHALLENGED = "challenged"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class DecisionCriticParityClassification(str, Enum):
    """Explicit parity classification — not a boolean."""

    MATCH = "match"
    EXPECTED_DIFFERENCE = "expected_difference"
    MISMATCH = "mismatch"
    CAPABILITY_GAP = "capability_gap"
    SHADOW_UNAVAILABLE = "shadow_unavailable"
    SHADOW_ERROR = "shadow_error"


class ParityVerificationCapability(str, Enum):
    """Verification capability classes exercised during qualification."""

    STRUCTURAL = "structural"
    DETERMINISTIC_GUARDRAIL = "deterministic_guardrail"
    EVIDENCE = "evidence"
    SEMANTIC = "semantic"
    TRAJECTORY = "trajectory"
    DOMAIN = "domain"
    HUMAN_HITL = "human_hitl"


class ParityCapabilityRequirementMode(str, Enum):
    """How retirement evidence must be proven for one verification capability."""

    CROSS_SYSTEM = "cross_system"
    DECISION_SUPERSET = "decision_superset"
    ARCHITECTURAL_MAPPING = "architectural_mapping"


@dataclass(frozen=True, slots=True)
class ParityCapabilityRequirement:
    """Typed retirement requirement for one verification capability."""

    capability: ParityVerificationCapability
    mode: ParityCapabilityRequirementMode


DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS: tuple[ParityCapabilityRequirement, ...] = (
    ParityCapabilityRequirement(
        ParityVerificationCapability.STRUCTURAL,
        ParityCapabilityRequirementMode.CROSS_SYSTEM,
    ),
    ParityCapabilityRequirement(
        ParityVerificationCapability.DETERMINISTIC_GUARDRAIL,
        ParityCapabilityRequirementMode.CROSS_SYSTEM,
    ),
    ParityCapabilityRequirement(
        ParityVerificationCapability.SEMANTIC,
        ParityCapabilityRequirementMode.CROSS_SYSTEM,
    ),
    ParityCapabilityRequirement(
        ParityVerificationCapability.TRAJECTORY,
        ParityCapabilityRequirementMode.CROSS_SYSTEM,
    ),
    ParityCapabilityRequirement(
        ParityVerificationCapability.EVIDENCE,
        ParityCapabilityRequirementMode.DECISION_SUPERSET,
    ),
    ParityCapabilityRequirement(
        ParityVerificationCapability.DOMAIN,
        ParityCapabilityRequirementMode.DECISION_SUPERSET,
    ),
    ParityCapabilityRequirement(
        ParityVerificationCapability.HUMAN_HITL,
        ParityCapabilityRequirementMode.ARCHITECTURAL_MAPPING,
    ),
)


class CriticRetirementReadiness(str, Enum):
    """Retirement gate outcome derived from accumulated parity evidence."""

    READY = "ready"
    NOT_READY = "not_ready"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class DecisionCriticParityIdentity:
    """Bind Decision and Critic observations to the same host input."""

    host_scope: ParityHostScope
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str | None
    tenant_id: str
    agent_id: str
    subject: str
    proposal_ref: DecisionProposalRef | None = None


@dataclass(frozen=True, slots=True)
class DecisionParityObservation:
    """Normalized Decision authority observation."""

    outcome: NormalizedParityOutcome
    host_action: DecisionFlowHostAction
    verification_disposition: VerificationDisposition | None
    revision_disposition: DecisionRevisionDisposition | None
    human_review_pending: bool
    capabilities: frozenset[ParityVerificationCapability]


@dataclass(frozen=True, slots=True)
class CriticParityObservation:
    """Normalized legacy Critic shadow observation."""

    outcome: NormalizedParityOutcome
    passed: bool | None
    failed_layer: CriticLayer | None
    recommended_action: CriticAction | None
    capabilities: frozenset[ParityVerificationCapability]
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionCriticParityDifference:
    """One typed parity difference record."""

    code: DecisionCriticParityDifferenceCode
    detail: str


@dataclass(frozen=True, slots=True)
class DecisionCriticParityResult:
    """Immutable comparison result for one host evaluation."""

    identity: DecisionCriticParityIdentity
    decision_observation: DecisionParityObservation
    critic_observation: CriticParityObservation
    outcome_match: bool
    capability_match: bool
    classification: DecisionCriticParityClassification
    differences: tuple[DecisionCriticParityDifference, ...]
    retirement_blocking: bool


@dataclass(frozen=True, slots=True)
class DecisionCriticParityMetrics:
    """Aggregate parity metrics for qualification reporting."""

    total_comparisons: int
    matches: int
    expected_differences: int
    mismatches: int
    shadow_unavailable: int
    shadow_errors: int
    retirement_blocking_mismatches: int
    outcome_agreement_rate: float
    retirement_blocking_rate: float
    by_scope: tuple[tuple[ParityHostScope, int], ...]


@dataclass(frozen=True, slots=True)
class CriticRetirementReadinessReport:
    """Retirement readiness with explicit evidence summary."""

    readiness: CriticRetirementReadiness
    blocking_mismatch_count: int
    shadow_error_count: int
    shadow_unavailable_count: int
    scopes_exercised: frozenset[ParityHostScope]
    decision_capabilities_exercised: frozenset[ParityVerificationCapability]
    critic_capabilities_exercised: frozenset[ParityVerificationCapability]
    cross_system_capabilities_qualified: frozenset[ParityVerificationCapability]
    decision_superset_capabilities_qualified: frozenset[ParityVerificationCapability]
    architectural_mappings_qualified: frozenset[ParityVerificationCapability]
    missing_scopes: frozenset[ParityHostScope]
    missing_capabilities: frozenset[ParityVerificationCapability]


@dataclass(frozen=True, slots=True)
class _CriticRetirementReadinessEvidence:
    blocking_mismatch_count: int
    shadow_error_count: int
    shadow_unavailable_count: int
    scopes_exercised: frozenset[ParityHostScope]
    decision_capabilities_exercised: frozenset[ParityVerificationCapability]
    critic_capabilities_exercised: frozenset[ParityVerificationCapability]
    cross_system_capabilities_qualified: frozenset[ParityVerificationCapability]
    decision_superset_capabilities_qualified: frozenset[ParityVerificationCapability]
    architectural_mappings_qualified: frozenset[ParityVerificationCapability]
    missing_scopes: frozenset[ParityHostScope]
    missing_capabilities: frozenset[ParityVerificationCapability]


class DecisionCriticParityObserver(Protocol):
    """Replaceable sink for parity observations without host coupling."""

    def record(self, result: DecisionCriticParityResult) -> None:
        """Record one parity comparison result."""
        ...


def parity_host_scope_from_flow_scope(flow_scope: DecisionFlowScope) -> ParityHostScope:
    if flow_scope is DecisionFlowScope.GRAPH_FINAL:
        return ParityHostScope.GRAPH_FINAL
    if flow_scope is DecisionFlowScope.UAEP_STEP:
        return ParityHostScope.UAEP_STEP
    raise ValueError(f"unsupported decision flow scope for parity: {flow_scope.value!r}")


def _decision_capabilities_from_verification(
    verification_result: VerificationResult,
) -> frozenset[ParityVerificationCapability]:
    capabilities: set[ParityVerificationCapability] = set()
    for record in verification_result.stage_records:
        stage_value = str(record.stage)
        if "structural" in stage_value:
            capabilities.add(ParityVerificationCapability.STRUCTURAL)
        if "guardrail" in stage_value:
            capabilities.add(ParityVerificationCapability.DETERMINISTIC_GUARDRAIL)
        if "evidence" in stage_value:
            capabilities.add(ParityVerificationCapability.EVIDENCE)
        if "semantic" in stage_value:
            capabilities.add(ParityVerificationCapability.SEMANTIC)
        if "trajectory" in stage_value:
            capabilities.add(ParityVerificationCapability.TRAJECTORY)
        if "domain" in stage_value:
            capabilities.add(ParityVerificationCapability.DOMAIN)
    if not capabilities and verification_result.disposition is VerificationDisposition.PASSED:
        capabilities.add(ParityVerificationCapability.STRUCTURAL)
    return frozenset(capabilities)


def project_decision_observation(
    result: DecisionFlowResult[T],
) -> DecisionParityObservation:
    """Project one Decision flow result into normalized parity semantics."""
    if type(result) is not DecisionFlowResult:
        raise TypeError("result must be DecisionFlowResult")
    verification = result.verification_result
    revision_disposition = (
        result.revision_decision.disposition
        if result.revision_decision is not None
        else None
    )
    human_review_pending = result.human_review_pending is not None
    capabilities = _decision_capabilities_from_verification(verification)
    if human_review_pending:
        capabilities = frozenset(
            set(capabilities) | {ParityVerificationCapability.HUMAN_HITL},
        )
    if verification.disposition is VerificationDisposition.PASSED:
        if result.host_action is DecisionFlowHostAction.CONTINUE:
            outcome = NormalizedParityOutcome.ACCEPTABLE
        else:
            outcome = NormalizedParityOutcome.CHALLENGED
    else:
        outcome = NormalizedParityOutcome.CHALLENGED
    return DecisionParityObservation(
        outcome=outcome,
        host_action=result.host_action,
        verification_disposition=verification.disposition,
        revision_disposition=revision_disposition,
        human_review_pending=human_review_pending,
        capabilities=capabilities,
    )


def _critic_failed_layer(verdict: CriticVerdict) -> CriticLayer | None:
    for layer in verdict.layers:
        if not layer.passed:
            return layer.layer
    return None


def _critic_capabilities_from_verdict(verdict: CriticVerdict) -> frozenset[ParityVerificationCapability]:
    capabilities: set[ParityVerificationCapability] = set()
    for layer in verdict.layers:
        if layer.layer is CriticLayer.L0_DETERMINISTIC:
            capabilities.add(ParityVerificationCapability.STRUCTURAL)
            capabilities.add(ParityVerificationCapability.DETERMINISTIC_GUARDRAIL)
        elif layer.layer is CriticLayer.L1_SEMANTIC:
            capabilities.add(ParityVerificationCapability.SEMANTIC)
        elif layer.layer is CriticLayer.L1_TRAJECTORY:
            capabilities.add(ParityVerificationCapability.TRAJECTORY)
        elif layer.layer is CriticLayer.L2_HUMAN:
            capabilities.add(ParityVerificationCapability.HUMAN_HITL)
    return frozenset(capabilities)


def project_critic_observation(
    verdict: CriticVerdict | None,
    *,
    shadow_unavailable: bool = False,
    shadow_error: str | None = None,
) -> CriticParityObservation:
    """Project one Critic shadow verdict into normalized parity semantics."""
    if shadow_error is not None:
        return CriticParityObservation(
            outcome=NormalizedParityOutcome.ERROR,
            passed=None,
            failed_layer=None,
            recommended_action=None,
            capabilities=frozenset(),
            failure_reasons=(shadow_error,),
        )
    if shadow_unavailable or verdict is None:
        return CriticParityObservation(
            outcome=NormalizedParityOutcome.UNAVAILABLE,
            passed=None,
            failed_layer=None,
            recommended_action=None,
            capabilities=frozenset(),
        )
    failed_layer = _critic_failed_layer(verdict)
    outcome = (
        NormalizedParityOutcome.ACCEPTABLE
        if verdict.passed
        else NormalizedParityOutcome.CHALLENGED
    )
    return CriticParityObservation(
        outcome=outcome,
        passed=verdict.passed,
        failed_layer=failed_layer,
        recommended_action=verdict.recommended_action,
        capabilities=_critic_capabilities_from_verdict(verdict),
        failure_reasons=tuple(verdict.failure_reasons),
    )


def _expected_difference_codes(
  *,
    decision: DecisionParityObservation,
    critic: CriticParityObservation,
) -> tuple[DecisionCriticParityDifference, ...]:
    differences: list[DecisionCriticParityDifference] = []
    action = critic.recommended_action
    if action is CriticAction.RETRY:
        differences.append(
            DecisionCriticParityDifference(
                code=LEGACY_RETRY_IS_EXECUTION_RETRY,
                detail="legacy retry belongs to execution reliability, not verification",
            ),
        )
    if action is CriticAction.REVISE:
        if (
            decision.revision_disposition is DecisionRevisionDisposition.ALLOWED
            or decision.verification_disposition is VerificationDisposition.CHALLENGED
        ):
            differences.append(
                DecisionCriticParityDifference(
                    code=LEGACY_REVISE_IS_DECISION_REVISION,
                    detail="legacy revise maps to decision revision lifecycle",
                ),
            )
    if critic.failed_layer is CriticLayer.L2_HUMAN or action is CriticAction.ESCALATE_HITL:
        differences.append(
            DecisionCriticParityDifference(
                code=LEGACY_L2_NOT_DECISION_VERIFICATION,
                detail="legacy L2 human escalation is outside decision verification",
            ),
        )
        if decision.human_review_pending:
            differences.append(
                DecisionCriticParityDifference(
                    code=LEGACY_HITL_IS_DECISION_HUMAN_REVIEW,
                    detail="decision human review pending maps to legacy HITL escalation",
                ),
            )
    return tuple(differences)


def _classify_outcome_mismatch(
    *,
    decision: DecisionParityObservation,
    critic: CriticParityObservation,
    expected: tuple[DecisionCriticParityDifference, ...],
) -> tuple[DecisionCriticParityClassification, tuple[DecisionCriticParityDifference, ...], bool]:
    if critic.outcome is NormalizedParityOutcome.UNAVAILABLE:
        return (
            DecisionCriticParityClassification.SHADOW_UNAVAILABLE,
            (DecisionCriticParityDifference(
                code=SHADOW_PROVIDER_UNAVAILABLE,
                detail="critic shadow unavailable",
            ),),
            False,
        )
    if critic.outcome is NormalizedParityOutcome.ERROR:
        return (
            DecisionCriticParityClassification.SHADOW_ERROR,
            (DecisionCriticParityDifference(
                code=SHADOW_EXECUTION_ERROR,
                detail=critic.failure_reasons[0] if critic.failure_reasons else "shadow error",
            ),),
            False,
        )
    if decision.outcome == critic.outcome:
        return DecisionCriticParityClassification.MATCH, (), False
    if expected:
        return DecisionCriticParityClassification.EXPECTED_DIFFERENCE, expected, False
    differences: list[DecisionCriticParityDifference] = []
    blocking = False
    if (
        decision.outcome is NormalizedParityOutcome.ACCEPTABLE
        and critic.outcome is NormalizedParityOutcome.CHALLENGED
    ):
        differences.append(
            DecisionCriticParityDifference(
                code=DECISION_ACCEPT_CRITIC_CHALLENGE,
                detail="decision acceptable while critic challenged",
            ),
        )
        blocking = True
    elif (
        decision.outcome is NormalizedParityOutcome.CHALLENGED
        and critic.outcome is NormalizedParityOutcome.ACCEPTABLE
    ):
        differences.append(
            DecisionCriticParityDifference(
                code=DECISION_CHALLENGE_CRITIC_ACCEPT,
                detail="decision challenged while critic acceptable",
            ),
        )
        blocking = True
    return DecisionCriticParityClassification.MISMATCH, tuple(differences), blocking


def _capability_match(
    *,
    decision: DecisionParityObservation,
    critic: CriticParityObservation,
) -> bool:
    if critic.outcome in (
        NormalizedParityOutcome.UNAVAILABLE,
        NormalizedParityOutcome.ERROR,
    ):
        return True
    critic_required = {
        capability
        for capability in critic.capabilities
        if capability is not ParityVerificationCapability.HUMAN_HITL
    }
    if not critic_required:
        return True
    return critic_required.issubset(decision.capabilities)


def compare_decision_critic_parity(
    *,
    identity: DecisionCriticParityIdentity,
    decision_result: DecisionFlowResult[T],
    critic_verdict: CriticVerdict | None = None,
    shadow_unavailable: bool = False,
    shadow_error: str | None = None,
) -> DecisionCriticParityResult:
    """Compare normalized Decision and Critic observations for one host input."""
    decision_observation = project_decision_observation(decision_result)
    critic_observation = project_critic_observation(
        critic_verdict,
        shadow_unavailable=shadow_unavailable,
        shadow_error=shadow_error,
    )
    expected = _expected_difference_codes(
        decision=decision_observation,
        critic=critic_observation,
    )
    classification, differences, blocking = _classify_outcome_mismatch(
        decision=decision_observation,
        critic=critic_observation,
        expected=expected,
    )
    if classification is DecisionCriticParityClassification.MATCH and expected:
        classification = DecisionCriticParityClassification.EXPECTED_DIFFERENCE
        differences = expected
    outcome_match = decision_observation.outcome == critic_observation.outcome
    capability_match = _capability_match(
        decision=decision_observation,
        critic=critic_observation,
    )
    if (
        outcome_match
        and not capability_match
        and decision_observation.outcome is NormalizedParityOutcome.ACCEPTABLE
        and critic_observation.outcome is NormalizedParityOutcome.ACCEPTABLE
        and classification is DecisionCriticParityClassification.MATCH
    ):
        classification = DecisionCriticParityClassification.CAPABILITY_GAP
        differences = differences + (
            DecisionCriticParityDifference(
                code=CRITIC_CAPABILITY_NOT_EXERCISED_BY_DECISION,
                detail=(
                    "critic exercised verification capability without decision "
                    "equivalent on same input"
                ),
            ),
        )
    if (
        not capability_match
        and critic_observation.outcome is NormalizedParityOutcome.CHALLENGED
        and decision_observation.outcome is NormalizedParityOutcome.ACCEPTABLE
    ):
        blocking = True
        extra = (
            DecisionCriticParityDifference(
                code=DECISION_ACCEPT_CRITIC_CHALLENGE,
                detail="critic capability exercised without decision equivalent",
            ),
        )
        differences = differences + extra
    if (
        decision_observation.outcome is NormalizedParityOutcome.CHALLENGED
        and critic_observation.outcome is NormalizedParityOutcome.ACCEPTABLE
        and decision_observation.capabilities - critic_observation.capabilities
    ):
        differences = differences + (
            DecisionCriticParityDifference(
                code=DECISION_SUPERSET_CAPABILITY,
                detail="decision exercised superset verification capability",
            ),
        )
        blocking = False
    return DecisionCriticParityResult(
        identity=identity,
        decision_observation=decision_observation,
        critic_observation=critic_observation,
        outcome_match=outcome_match,
        capability_match=capability_match,
        classification=classification,
        differences=differences,
        retirement_blocking=blocking,
    )


def build_parity_identity(
    *,
    flow_scope: DecisionFlowScope,
    task_id: str,
    run_id: str,
    attempt_id: str,
    tenant_id: str,
    agent_id: str,
    subject: str,
    execution_id: str | None = None,
    decision_result: DecisionFlowResult[T] | None = None,
) -> DecisionCriticParityIdentity:
    proposal_ref = None
    if decision_result is not None:
        proposal_ref = candidate_decision_ref(decision_result.candidate)
    return DecisionCriticParityIdentity(
        host_scope=parity_host_scope_from_flow_scope(flow_scope),
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        subject=subject,
        proposal_ref=proposal_ref,
    )


def aggregate_parity_metrics(
    results: Sequence[DecisionCriticParityResult],
) -> DecisionCriticParityMetrics:
    """Aggregate parity metrics across one qualification run."""
    total = len(results)
    if total == 0:
        return DecisionCriticParityMetrics(
            total_comparisons=0,
            matches=0,
            expected_differences=0,
            mismatches=0,
            shadow_unavailable=0,
            shadow_errors=0,
            retirement_blocking_mismatches=0,
            outcome_agreement_rate=0.0,
            retirement_blocking_rate=0.0,
            by_scope=(),
        )
    matches = sum(
        1
        for item in results
        if item.classification is DecisionCriticParityClassification.MATCH
    )
    expected = sum(
        1
        for item in results
        if item.classification is DecisionCriticParityClassification.EXPECTED_DIFFERENCE
    )
    mismatches = sum(
        1
        for item in results
        if item.classification is DecisionCriticParityClassification.MISMATCH
    )
    unavailable = sum(
        1
        for item in results
        if item.classification is DecisionCriticParityClassification.SHADOW_UNAVAILABLE
    )
    errors = sum(
        1
        for item in results
        if item.classification is DecisionCriticParityClassification.SHADOW_ERROR
    )
    blocking = sum(1 for item in results if item.retirement_blocking)
    agreements = sum(1 for item in results if item.outcome_match)
    scope_counts: dict[ParityHostScope, int] = {}
    for item in results:
        scope_counts[item.identity.host_scope] = scope_counts.get(item.identity.host_scope, 0) + 1
    return DecisionCriticParityMetrics(
        total_comparisons=total,
        matches=matches,
        expected_differences=expected,
        mismatches=mismatches,
        shadow_unavailable=unavailable,
        shadow_errors=errors,
        retirement_blocking_mismatches=blocking,
        outcome_agreement_rate=agreements / total,
        retirement_blocking_rate=blocking / total,
        by_scope=tuple(sorted(scope_counts.items(), key=lambda pair: pair[0].value)),
    )


def _paired_capabilities(
    result: DecisionCriticParityResult,
) -> frozenset[ParityVerificationCapability]:
    if result.classification in (
        DecisionCriticParityClassification.SHADOW_UNAVAILABLE,
        DecisionCriticParityClassification.SHADOW_ERROR,
    ):
        return frozenset()
    return (
        result.decision_observation.capabilities
        & result.critic_observation.capabilities
    )


def _cross_system_capability_qualified(
    parity_results: Sequence[DecisionCriticParityResult],
    capability: ParityVerificationCapability,
) -> bool:
    for result in parity_results:
        if result.classification in (
            DecisionCriticParityClassification.SHADOW_UNAVAILABLE,
            DecisionCriticParityClassification.SHADOW_ERROR,
        ):
            continue
        if capability not in _paired_capabilities(result):
            continue
        return True
    return False


def _decision_superset_capability_qualified(
    parity_results: Sequence[DecisionCriticParityResult],
    capability: ParityVerificationCapability,
) -> bool:
    for result in parity_results:
        if result.classification in (
            DecisionCriticParityClassification.SHADOW_UNAVAILABLE,
            DecisionCriticParityClassification.SHADOW_ERROR,
        ):
            continue
        if capability in result.decision_observation.capabilities:
            return True
    return False


def _architectural_mapping_qualified(
    parity_results: Sequence[DecisionCriticParityResult],
    capability: ParityVerificationCapability,
) -> bool:
    if capability is not ParityVerificationCapability.HUMAN_HITL:
        return False
    for result in parity_results:
        if result.classification in (
            DecisionCriticParityClassification.SHADOW_UNAVAILABLE,
            DecisionCriticParityClassification.SHADOW_ERROR,
        ):
            continue
        codes = {difference.code for difference in result.differences}
        if LEGACY_HITL_IS_DECISION_HUMAN_REVIEW in codes:
            return True
    return False


def _qualify_capability_requirement(
    parity_results: Sequence[DecisionCriticParityResult],
    requirement: ParityCapabilityRequirement,
) -> bool:
    if requirement.mode is ParityCapabilityRequirementMode.CROSS_SYSTEM:
        return _cross_system_capability_qualified(
            parity_results,
            requirement.capability,
        )
    if requirement.mode is ParityCapabilityRequirementMode.DECISION_SUPERSET:
        return _decision_superset_capability_qualified(
            parity_results,
            requirement.capability,
        )
    if requirement.mode is ParityCapabilityRequirementMode.ARCHITECTURAL_MAPPING:
        return _architectural_mapping_qualified(
            parity_results,
            requirement.capability,
        )
    raise ValueError(f"unsupported requirement mode: {requirement.mode.value!r}")


def _critic_retirement_readiness_report(
    *,
    readiness: CriticRetirementReadiness,
    evidence: _CriticRetirementReadinessEvidence,
) -> CriticRetirementReadinessReport:
    return CriticRetirementReadinessReport(
        readiness=readiness,
        blocking_mismatch_count=evidence.blocking_mismatch_count,
        shadow_error_count=evidence.shadow_error_count,
        shadow_unavailable_count=evidence.shadow_unavailable_count,
        scopes_exercised=evidence.scopes_exercised,
        decision_capabilities_exercised=evidence.decision_capabilities_exercised,
        critic_capabilities_exercised=evidence.critic_capabilities_exercised,
        cross_system_capabilities_qualified=evidence.cross_system_capabilities_qualified,
        decision_superset_capabilities_qualified=evidence.decision_superset_capabilities_qualified,
        architectural_mappings_qualified=evidence.architectural_mappings_qualified,
        missing_scopes=evidence.missing_scopes,
        missing_capabilities=evidence.missing_capabilities,
    )


def evaluate_critic_retirement_readiness(
    parity_results: Sequence[DecisionCriticParityResult],
    *,
    required_scopes: frozenset[ParityHostScope],
    capability_requirements: tuple[ParityCapabilityRequirement, ...],
) -> CriticRetirementReadinessReport:
    """Evaluate whether accumulated parity evidence supports Critic retirement."""
    scopes_exercised: set[ParityHostScope] = set()
    decision_capabilities_exercised: set[ParityVerificationCapability] = set()
    critic_capabilities_exercised: set[ParityVerificationCapability] = set()
    cross_system_qualified: set[ParityVerificationCapability] = set()
    decision_superset_qualified: set[ParityVerificationCapability] = set()
    architectural_qualified: set[ParityVerificationCapability] = set()
    blocking_count = 0
    shadow_error_count = 0
    shadow_unavailable_count = 0
    for result in parity_results:
        scopes_exercised.add(result.identity.host_scope)
        decision_capabilities_exercised.update(result.decision_observation.capabilities)
        critic_capabilities_exercised.update(result.critic_observation.capabilities)
        if result.retirement_blocking:
            blocking_count += 1
        if result.classification is DecisionCriticParityClassification.SHADOW_ERROR:
            shadow_error_count += 1
        if result.classification is DecisionCriticParityClassification.SHADOW_UNAVAILABLE:
            shadow_unavailable_count += 1
    for requirement in capability_requirements:
        if not _qualify_capability_requirement(parity_results, requirement):
            continue
        if requirement.mode is ParityCapabilityRequirementMode.CROSS_SYSTEM:
            cross_system_qualified.add(requirement.capability)
        elif requirement.mode is ParityCapabilityRequirementMode.DECISION_SUPERSET:
            decision_superset_qualified.add(requirement.capability)
        elif requirement.mode is ParityCapabilityRequirementMode.ARCHITECTURAL_MAPPING:
            architectural_qualified.add(requirement.capability)
    required_capabilities = frozenset(
        requirement.capability for requirement in capability_requirements
    )
    qualified_capabilities = (
        frozenset(cross_system_qualified)
        | frozenset(decision_superset_qualified)
        | frozenset(architectural_qualified)
    )
    missing_scopes = required_scopes - frozenset(scopes_exercised)
    missing_capabilities = required_capabilities - qualified_capabilities
    evidence = _CriticRetirementReadinessEvidence(
        blocking_mismatch_count=blocking_count,
        shadow_error_count=shadow_error_count,
        shadow_unavailable_count=shadow_unavailable_count,
        scopes_exercised=frozenset(scopes_exercised),
        decision_capabilities_exercised=frozenset(decision_capabilities_exercised),
        critic_capabilities_exercised=frozenset(critic_capabilities_exercised),
        cross_system_capabilities_qualified=frozenset(cross_system_qualified),
        decision_superset_capabilities_qualified=frozenset(decision_superset_qualified),
        architectural_mappings_qualified=frozenset(architectural_qualified),
        missing_scopes=frozenset(missing_scopes),
        missing_capabilities=frozenset(missing_capabilities),
    )
    if missing_scopes or missing_capabilities:
        return _critic_retirement_readiness_report(
            readiness=CriticRetirementReadiness.INSUFFICIENT_EVIDENCE,
            evidence=evidence,
        )
    if blocking_count > 0 or shadow_error_count > 0:
        return _critic_retirement_readiness_report(
            readiness=CriticRetirementReadiness.NOT_READY,
            evidence=evidence,
        )
    return _critic_retirement_readiness_report(
        readiness=CriticRetirementReadiness.READY,
        evidence=evidence,
    )


def parity_identity_from_decision_identity(
    *,
    identity: DecisionIdentity,
    host_scope: ParityHostScope,
    agent_id: str,
    subject: str,
    proposal_ref: DecisionProposalRef | None = None,
) -> DecisionCriticParityIdentity:
    execution = identity.execution
    return DecisionCriticParityIdentity(
        host_scope=host_scope,
        task_id=str(execution.task_id),
        run_id=str(execution.run_id),
        attempt_id=str(execution.attempt_id),
        execution_id=str(execution.execution_id) if execution.execution_id is not None else None,
        tenant_id=identity.tenant_id,
        agent_id=agent_id,
        subject=subject,
        proposal_ref=proposal_ref,
    )


def critic_scope_for_parity_host_scope(host_scope: ParityHostScope) -> CriticScope:
    if host_scope is ParityHostScope.GRAPH_FINAL:
        return CriticScope.GRAPH_FINAL
    if host_scope is ParityHostScope.UAEP_STEP:
        return CriticScope.UAEP_STEP
    raise ValueError(f"unsupported parity host scope: {host_scope.value!r}")
