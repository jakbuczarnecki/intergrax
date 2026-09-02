# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""WorkerDefinition and WorkerInstance semantic contracts (AW-1A)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerDefinitionId,
    WorkerGoalId,
    WorkerInstanceId,
    validate_responsibility_id,
    validate_worker_definition_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    CollaborationProfileRef,
    EscalationPolicyRef,
    GovernanceProfileRef,
    MemoryProfileRef,
    ObservabilityProfileRef,
    RiskProfileRef,
    ScheduleProfileRef,
    validate_budget_profile_ref,
    validate_capability_profile_ref,
    validate_codecraft_profile_ref,
    validate_collaboration_profile_ref,
    validate_escalation_policy_ref,
    validate_governance_profile_ref,
    validate_memory_profile_ref,
    validate_observability_profile_ref,
    validate_risk_profile_ref,
    validate_schedule_profile_ref,
)
from intergrax.contracts.autonomous_work.references import (
    DefaultGoalPolicyRef,
    PrincipalBindingPolicyRef,
    PrincipalBindingRef,
    ResponsibilityTemplateRef,
    WorkspaceContextRef,
    WorkspaceScopeRef,
    validate_default_goal_policy_ref,
    validate_principal_binding_policy_ref,
    validate_principal_binding_ref,
    validate_responsibility_template_ref,
    validate_workspace_context_ref,
    validate_workspace_scope_ref,
)
from intergrax.contracts.autonomous_work.revision import (
    DefinitionRevision,
    Revision,
    validate_definition_revision,
    validate_revision,
)


@dataclass(frozen=True, slots=True)
class WorkerDefinition:
    """Reusable worker role definition — descriptive role does not grant authority."""

    worker_definition_id: WorkerDefinitionId
    display_name: str
    role: str
    revision: DefinitionRevision
    responsibility_template_refs: tuple[ResponsibilityTemplateRef, ...]
    default_goal_policy_ref: DefaultGoalPolicyRef
    principal_binding_policy_ref: PrincipalBindingPolicyRef
    workspace_scope_ref: WorkspaceScopeRef
    governance_profile_ref: GovernanceProfileRef
    budget_profile_ref: BudgetProfileRef
    memory_profile_ref: MemoryProfileRef
    capability_profile_ref: CapabilityProfileRef
    codecraft_profile_ref: CodecraftProfileRef
    risk_profile_ref: RiskProfileRef
    schedule_profile_ref: ScheduleProfileRef
    escalation_policy_ref: EscalationPolicyRef
    collaboration_profile_ref: CollaborationProfileRef
    observability_profile_ref: ObservabilityProfileRef

    def __post_init__(self) -> None:
        validate_worker_definition_id(self.worker_definition_id)
        object.__setattr__(
            self,
            "display_name",
            require_non_empty_text(self.display_name, label="display_name"),
        )
        object.__setattr__(
            self,
            "role",
            require_non_empty_text(self.role, label="role"),
        )
        if type(self.revision) is not DefinitionRevision:
            raise TypeError("revision must be DefinitionRevision")
        validate_definition_revision(self.revision)
        object.__setattr__(
            self,
            "responsibility_template_refs",
            freeze_tuple(
                self.responsibility_template_refs,
                label="responsibility_template_refs",
            ),
        )
        for template_ref in self.responsibility_template_refs:
            validate_responsibility_template_ref(template_ref)
        validate_default_goal_policy_ref(self.default_goal_policy_ref)
        validate_principal_binding_policy_ref(self.principal_binding_policy_ref)
        validate_workspace_scope_ref(self.workspace_scope_ref)
        validate_governance_profile_ref(self.governance_profile_ref)
        validate_budget_profile_ref(self.budget_profile_ref)
        validate_memory_profile_ref(self.memory_profile_ref)
        validate_capability_profile_ref(self.capability_profile_ref)
        validate_codecraft_profile_ref(self.codecraft_profile_ref)
        validate_risk_profile_ref(self.risk_profile_ref)
        validate_schedule_profile_ref(self.schedule_profile_ref)
        validate_escalation_policy_ref(self.escalation_policy_ref)
        validate_collaboration_profile_ref(self.collaboration_profile_ref)
        validate_observability_profile_ref(self.observability_profile_ref)


@dataclass(frozen=True, slots=True)
class WorkerInstance:
    """Durable instantiated worker — survives executions and host restarts."""

    worker_instance_id: WorkerInstanceId
    worker_definition_id: WorkerDefinitionId
    definition_revision: DefinitionRevision
    lifecycle_state: WorkerLifecycleState
    principal_binding_ref: PrincipalBindingRef
    workspace_context_ref: WorkspaceContextRef
    active_responsibility_refs: tuple[ResponsibilityId, ...]
    active_goal_refs: tuple[WorkerGoalId, ...]
    created_at: datetime
    updated_at: datetime
    revision: Revision

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        validate_worker_definition_id(self.worker_definition_id)
        if type(self.definition_revision) is not DefinitionRevision:
            raise TypeError("definition_revision must be DefinitionRevision")
        validate_definition_revision(self.definition_revision)
        if type(self.lifecycle_state) is not WorkerLifecycleState:
            raise TypeError("lifecycle_state must be WorkerLifecycleState")
        validate_principal_binding_ref(self.principal_binding_ref)
        validate_workspace_context_ref(self.workspace_context_ref)
        object.__setattr__(
            self,
            "active_responsibility_refs",
            freeze_tuple(
                self.active_responsibility_refs, label="active_responsibility_refs"
            ),
        )
        object.__setattr__(
            self,
            "active_goal_refs",
            freeze_tuple(self.active_goal_refs, label="active_goal_refs"),
        )
        for responsibility_id in self.active_responsibility_refs:
            validate_responsibility_id(responsibility_id)
        for goal_id in self.active_goal_refs:
            validate_worker_goal_id(goal_id)
        created_at = require_aware_utc(self.created_at, label="created_at")
        updated_at = require_aware_utc(self.updated_at, label="updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        if type(self.revision) is not Revision:
            raise TypeError("revision must be Revision")
        validate_revision(self.revision)
