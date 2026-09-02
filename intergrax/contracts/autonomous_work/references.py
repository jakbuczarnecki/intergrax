# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed opaque references for Autonomous Work composition and continuity (AW-1A)."""

from __future__ import annotations

from typing import Callable, NewType, TypeVar

from intergrax.contracts.autonomous_work._validation import require_opaque_ref

ResponsibilityTemplateRef = NewType("ResponsibilityTemplateRef", str)
DefaultGoalPolicyRef = NewType("DefaultGoalPolicyRef", str)
PrincipalBindingPolicyRef = NewType("PrincipalBindingPolicyRef", str)
WorkspaceScopeRef = NewType("WorkspaceScopeRef", str)
GovernanceProfileRef = NewType("GovernanceProfileRef", str)
BudgetProfileRef = NewType("BudgetProfileRef", str)
MemoryProfileRef = NewType("MemoryProfileRef", str)
CapabilityProfileRef = NewType("CapabilityProfileRef", str)
CodecraftProfileRef = NewType("CodecraftProfileRef", str)
RiskProfileRef = NewType("RiskProfileRef", str)
ScheduleProfileRef = NewType("ScheduleProfileRef", str)
EscalationPolicyRef = NewType("EscalationPolicyRef", str)
CollaborationProfileRef = NewType("CollaborationProfileRef", str)
ObservabilityProfileRef = NewType("ObservabilityProfileRef", str)

PrincipalBindingRef = NewType("PrincipalBindingRef", str)
WorkspaceContextRef = NewType("WorkspaceContextRef", str)
ResponsibilityScopeRef = NewType("ResponsibilityScopeRef", str)

SuccessCriteriaRef = NewType("SuccessCriteriaRef", str)
MetricRef = NewType("MetricRef", str)
SlaSloRef = NewType("SlaSloRef", str)
DeadlineOrCadenceRef = NewType("DeadlineOrCadenceRef", str)
ProgressProjectionRef = NewType("ProgressProjectionRef", str)
EvaluationCadenceRef = NewType("EvaluationCadenceRef", str)

WorkReference = NewType("WorkReference", str)
ArtifactReference = NewType("ArtifactReference", str)
ProblemReference = NewType("ProblemReference", str)
ContextAnchorReference = NewType("ContextAnchorReference", str)
ProgressCheckpointRef = NewType("ProgressCheckpointRef", str)
ExternalDependencyReference = NewType("ExternalDependencyReference", str)
HumanPendingReference = NewType("HumanPendingReference", str)

_TRef = TypeVar("_TRef", bound=str)


def _make_ref_validator(
    label: str,
    ref_type: Callable[[str], _TRef],
) -> Callable[[object], _TRef]:
    def validate(value: object) -> _TRef:
        return ref_type(require_opaque_ref(value, label=label))

    return validate


validate_responsibility_template_ref = _make_ref_validator(
    "ResponsibilityTemplateRef",
    ResponsibilityTemplateRef,
)
validate_default_goal_policy_ref = _make_ref_validator(
    "DefaultGoalPolicyRef",
    DefaultGoalPolicyRef,
)
validate_principal_binding_policy_ref = _make_ref_validator(
    "PrincipalBindingPolicyRef",
    PrincipalBindingPolicyRef,
)
validate_workspace_scope_ref = _make_ref_validator(
    "WorkspaceScopeRef", WorkspaceScopeRef
)
validate_governance_profile_ref = _make_ref_validator(
    "GovernanceProfileRef",
    GovernanceProfileRef,
)
validate_budget_profile_ref = _make_ref_validator("BudgetProfileRef", BudgetProfileRef)
validate_memory_profile_ref = _make_ref_validator("MemoryProfileRef", MemoryProfileRef)
validate_capability_profile_ref = _make_ref_validator(
    "CapabilityProfileRef",
    CapabilityProfileRef,
)
validate_codecraft_profile_ref = _make_ref_validator(
    "CodecraftProfileRef",
    CodecraftProfileRef,
)
validate_risk_profile_ref = _make_ref_validator("RiskProfileRef", RiskProfileRef)
validate_schedule_profile_ref = _make_ref_validator(
    "ScheduleProfileRef", ScheduleProfileRef
)
validate_escalation_policy_ref = _make_ref_validator(
    "EscalationPolicyRef",
    EscalationPolicyRef,
)
validate_collaboration_profile_ref = _make_ref_validator(
    "CollaborationProfileRef",
    CollaborationProfileRef,
)
validate_observability_profile_ref = _make_ref_validator(
    "ObservabilityProfileRef",
    ObservabilityProfileRef,
)
validate_principal_binding_ref = _make_ref_validator(
    "PrincipalBindingRef",
    PrincipalBindingRef,
)
validate_workspace_context_ref = _make_ref_validator(
    "WorkspaceContextRef",
    WorkspaceContextRef,
)
validate_responsibility_scope_ref = _make_ref_validator(
    "ResponsibilityScopeRef",
    ResponsibilityScopeRef,
)
validate_success_criteria_ref = _make_ref_validator(
    "SuccessCriteriaRef", SuccessCriteriaRef
)
validate_metric_ref = _make_ref_validator("MetricRef", MetricRef)
validate_sla_slo_ref = _make_ref_validator("SlaSloRef", SlaSloRef)
validate_deadline_or_cadence_ref = _make_ref_validator(
    "DeadlineOrCadenceRef",
    DeadlineOrCadenceRef,
)
validate_progress_projection_ref = _make_ref_validator(
    "ProgressProjectionRef",
    ProgressProjectionRef,
)
validate_evaluation_cadence_ref = _make_ref_validator(
    "EvaluationCadenceRef",
    EvaluationCadenceRef,
)
validate_work_reference = _make_ref_validator("WorkReference", WorkReference)
validate_artifact_reference = _make_ref_validator(
    "ArtifactReference", ArtifactReference
)
validate_problem_reference = _make_ref_validator("ProblemReference", ProblemReference)
validate_context_anchor_reference = _make_ref_validator(
    "ContextAnchorReference",
    ContextAnchorReference,
)
validate_progress_checkpoint_ref = _make_ref_validator(
    "ProgressCheckpointRef",
    ProgressCheckpointRef,
)
validate_external_dependency_reference = _make_ref_validator(
    "ExternalDependencyReference",
    ExternalDependencyReference,
)
validate_human_pending_reference = _make_ref_validator(
    "HumanPendingReference",
    HumanPendingReference,
)
