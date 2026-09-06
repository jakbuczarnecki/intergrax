# © Artur Czarnecki. All rights reserved.

"""Canonical host execution admission for effective profile revisions (P1.2A)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionConflictError,
    MissingPinnedEffectiveProfileRevisionError,
)
from intergrax.applications._shared.profile_resolution.activation_service import (
    resolve_active_effective_profile_revision,
)
from intergrax.applications._shared.profile_resolution.execution_pinning import (
    attach_revision_checkpoint_evidence_to_task,
    pin_effective_profile_revision_for_execution,
    require_execution_pinned_revision,
    resolve_revision_for_execution,
    revision_id_from_checkpoint,
    verify_checkpoint_revision_consistency,
)
from intergrax.applications.contracts.profile_resolution.activation import (
    ActiveEffectiveProfileRevisionStore,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionPinningStore,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class EffectiveProfileExecutionPinningDependencies:
    """Immutable dependency bundle for one host effective profile revision."""

    revision_store: EffectiveProfileRevisionStore
    pinning_store: EffectiveProfileExecutionPinningStore
    active_store: ActiveEffectiveProfileRevisionStore
    scope: EffectiveProfileRevisionScope


@dataclass(frozen=True, slots=True)
class EffectiveProfileRevisionAdmission(EffectiveProfileRevisionAdmissionPort):
    """Application-owned admission gate for canonical host task execution."""

    _dependencies: EffectiveProfileExecutionPinningDependencies

    def admit_root_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        task: Task,
        resume_checkpoint: TaskCheckpoint | None = None,
        restore_existing_execution: bool = False,
    ) -> Task:
        """Pin or verify revision binding; return task with checkpoint evidence.

        Re-entry is signaled by ``restore_existing_execution`` or a non-null
        ``resume_checkpoint``. Re-entry without an existing binding fails closed.
        """
        deps = self._dependencies
        is_reentry = restore_existing_execution or resume_checkpoint is not None
        existing = deps.pinning_store.get(tenant_id=tenant_id, execution_id=execution_id)
        if existing is None:
            if is_reentry:
                raise MissingPinnedEffectiveProfileRevisionError(
                    tenant_id=tenant_id,
                    execution_id=str(execution_id),
                )
            admitted_revision = resolve_active_effective_profile_revision(
                active_store=deps.active_store,
                revision_store=deps.revision_store,
                scope=deps.scope,
            )
            pin_effective_profile_revision_for_execution(
                revision=admitted_revision,
                tenant_id=tenant_id,
                execution_id=execution_id,
                pinning_store=deps.pinning_store,
                revision_store=deps.revision_store,
            )
        else:
            admitted_revision = resolve_revision_for_execution(
                tenant_id=tenant_id,
                execution_id=execution_id,
                pinning_store=deps.pinning_store,
                revision_store=deps.revision_store,
                scope_application_id=deps.scope.application_id,
                scope_tenant_id=deps.scope.tenant_id,
            )

        binding = require_execution_pinned_revision(
            tenant_id=tenant_id,
            execution_id=execution_id,
            pinning_store=deps.pinning_store,
        )
        if resume_checkpoint is not None:
            verify_checkpoint_revision_consistency(
                checkpoint=resume_checkpoint,
                binding=binding,
            )
            checkpoint_revision_id = revision_id_from_checkpoint(resume_checkpoint)
            if checkpoint_revision_id != admitted_revision.revision_id:
                raise EffectiveProfileRevisionConflictError(
                    "checkpoint revision conflicts with admitted execution revision",
                )

        return attach_revision_checkpoint_evidence_to_task(task, admitted_revision)


def build_effective_profile_revision_admission(
    dependencies: EffectiveProfileExecutionPinningDependencies,
) -> EffectiveProfileRevisionAdmission:
    """Composition helper for host task execution wiring."""
    return EffectiveProfileRevisionAdmission(_dependencies=dependencies)
