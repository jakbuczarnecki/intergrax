# © Artur Czarnecki. All rights reserved.

"""Execution admission pinning for effective profile revisions (P1.2)."""

from __future__ import annotations

from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionConflictError,
    EffectiveProfileRevisionError,
    MissingPinnedEffectiveProfileRevisionError,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EFFECTIVE_PROFILE_REVISION_METADATA_KEY,
    EffectiveProfileExecutionBinding,
    EffectiveProfileExecutionPinningStore,
    EffectiveProfileRevisionCheckpointEvidence,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task


class InMemoryEffectiveProfileExecutionPinningStore:
    """Test adapter for execution revision pinning."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], EffectiveProfileExecutionBinding] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def pin(self, binding: EffectiveProfileExecutionBinding) -> None:
        key = (binding.tenant_id, binding.execution_id)
        if key in self._bindings and self._bindings[key] != binding:
            raise EffectiveProfileRevisionConflictError(
                f"execution already pinned: {binding.execution_id}"
            )
        self._bindings[key] = binding

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> EffectiveProfileExecutionBinding | None:
        return self._bindings.get((tenant_id, execution_id))


def pin_effective_profile_revision_for_execution(
    *,
    revision: EffectiveProfileRevision,
    tenant_id: str,
    execution_id: ExecutionId,
    pinning_store: EffectiveProfileExecutionPinningStore,
    revision_store: EffectiveProfileRevisionStore | None = None,
) -> EffectiveProfileExecutionBinding:
    """Bind one admitted execution to an immutable effective profile revision."""
    if not tenant_id or tenant_id != tenant_id.strip():
        raise EffectiveProfileRevisionError("tenant_id must be non-empty")
    validated_execution_id = validate_execution_id(execution_id)
    if revision_store is not None:
        stored = revision_store.get(revision.revision_id, scope=revision.scope)
        if stored is None:
            raise MissingPinnedEffectiveProfileRevisionError(
                tenant_id=tenant_id,
                execution_id=validated_execution_id,
            )
    binding = EffectiveProfileExecutionBinding(
        tenant_id=tenant_id,
        execution_id=validated_execution_id,
        revision_id=revision.revision_id,
        fingerprint=revision.fingerprint,
    )
    pinning_store.pin(binding)
    return binding


def require_execution_pinned_revision(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    pinning_store: EffectiveProfileExecutionPinningStore,
) -> EffectiveProfileExecutionBinding:
    """Fail closed when execution pinning evidence is missing."""
    binding = pinning_store.get(tenant_id=tenant_id, execution_id=execution_id)
    if binding is None:
        raise MissingPinnedEffectiveProfileRevisionError(
            tenant_id=tenant_id,
            execution_id=str(execution_id),
        )
    return binding


def inherit_child_execution_pinned_revision(
    *,
    tenant_id: str,
    parent_execution_id: ExecutionId,
    child_execution_id: ExecutionId,
    pinning_store: EffectiveProfileExecutionPinningStore,
) -> EffectiveProfileExecutionBinding:
    """Default child semantics: inherit parent's pinned effective revision."""
    parent_binding = require_execution_pinned_revision(
        tenant_id=tenant_id,
        execution_id=parent_execution_id,
        pinning_store=pinning_store,
    )
    child_binding = EffectiveProfileExecutionBinding(
        tenant_id=tenant_id,
        execution_id=validate_execution_id(child_execution_id),
        revision_id=parent_binding.revision_id,
        fingerprint=parent_binding.fingerprint,
    )
    pinning_store.pin(child_binding)
    return child_binding


def checkpoint_evidence_for_revision(
    revision: EffectiveProfileRevision,
) -> EffectiveProfileRevisionCheckpointEvidence:
    """Build checkpoint evidence reference for resume workflows."""
    return EffectiveProfileRevisionCheckpointEvidence(
        revision_id=revision.revision_id,
        fingerprint=revision.fingerprint,
    )


def attach_revision_checkpoint_evidence_to_task(
    task: Task,
    revision: EffectiveProfileRevision,
) -> Task:
    """Persist revision evidence on task metadata for checkpoint/resume."""
    evidence = checkpoint_evidence_for_revision(revision)
    metadata = dict(task.metadata)
    metadata[EFFECTIVE_PROFILE_REVISION_METADATA_KEY] = {
        "schema_version": evidence.schema_version,
        "revision_id": evidence.revision_id.value,
        "fingerprint": evidence.fingerprint,
    }
    return task.model_copy(update={"metadata": metadata})


def verify_checkpoint_revision_consistency(
    *,
    checkpoint: TaskCheckpoint,
    binding: EffectiveProfileExecutionBinding,
) -> None:
    """Fail closed when checkpoint evidence conflicts with canonical binding."""
    checkpoint_revision_id = revision_id_from_checkpoint(checkpoint)
    raw = checkpoint.task_snapshot.get("metadata", {}).get(EFFECTIVE_PROFILE_REVISION_METADATA_KEY)
    if raw is None:
        raise MissingPinnedEffectiveProfileRevisionError(
            tenant_id=binding.tenant_id,
            execution_id=str(binding.execution_id),
        )
    evidence = EffectiveProfileRevisionCheckpointEvidence.model_validate(raw)
    if checkpoint_revision_id != binding.revision_id:
        raise EffectiveProfileRevisionConflictError(
            "checkpoint revision conflicts with execution binding",
        )
    if evidence.fingerprint != binding.fingerprint:
        raise EffectiveProfileRevisionConflictError(
            "checkpoint fingerprint conflicts with execution binding",
        )


def revision_id_from_checkpoint(checkpoint: TaskCheckpoint) -> EffectiveProfileRevisionId:
    """Resolve pinned revision identity from checkpoint task metadata."""
    raw = checkpoint.task_snapshot.get("metadata", {}).get(EFFECTIVE_PROFILE_REVISION_METADATA_KEY)
    if raw is None:
        raise MissingPinnedEffectiveProfileRevisionError(
            tenant_id=str(checkpoint.tenant_id),
            execution_id="checkpoint",
        )
    evidence = EffectiveProfileRevisionCheckpointEvidence.model_validate(raw)
    return evidence.revision_id


def resolve_revision_for_execution(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    pinning_store: EffectiveProfileExecutionPinningStore,
    revision_store: EffectiveProfileRevisionStore,
    scope_application_id: str,
    scope_tenant_id: str | None = None,
) -> EffectiveProfileRevision:
    """Resolve pinned revision to immutable snapshot — fail closed on missing data."""
    binding = require_execution_pinned_revision(
        tenant_id=tenant_id,
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    if binding.tenant_id != tenant_id:
        raise EffectiveProfileRevisionError("execution binding tenant mismatch")

    revision = revision_store.get(
        binding.revision_id,
        scope=EffectiveProfileRevisionScope(
            application_id=scope_application_id,
            tenant_id=scope_tenant_id,
        ),
    )
    if revision is None:
        raise MissingPinnedEffectiveProfileRevisionError(
            tenant_id=tenant_id,
            execution_id=str(execution_id),
        )
    if revision.fingerprint != binding.fingerprint:
        raise EffectiveProfileRevisionError("pinned revision fingerprint mismatch")
    if revision.scope.application_id != scope_application_id:
        raise EffectiveProfileRevisionError("pinned revision application scope mismatch")
    if scope_tenant_id is not None and revision.scope.tenant_id != scope_tenant_id:
        raise EffectiveProfileRevisionError("pinned revision tenant scope mismatch")
    return revision
