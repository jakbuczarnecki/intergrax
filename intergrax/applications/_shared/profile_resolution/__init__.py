# © Artur Czarnecki. All rights reserved.

"""Application-owned profile resolution entrypoints (P1.1/P1.2)."""

from intergrax.applications._shared.profile_resolution.diff_engine import (
    ProfileFieldDiffer,
    diff_effective_profile_revisions,
)
from intergrax.applications._shared.profile_resolution.engine import resolve_profile
from intergrax.applications._shared.profile_resolution.execution_pinning import (
    InMemoryEffectiveProfileExecutionPinningStore,
    attach_revision_checkpoint_evidence_to_task,
    checkpoint_evidence_for_revision,
    inherit_child_execution_pinned_revision,
    pin_effective_profile_revision_for_execution,
    require_execution_pinned_revision,
    resolve_revision_for_execution,
    revision_id_from_checkpoint,
)
from intergrax.applications._shared.profile_resolution.field_resolvers import (
    DEFAULT_FIELD_RESOLVERS,
    ProfileFieldResolver,
    ProfileFieldResolveResult,
)
from intergrax.applications._shared.profile_resolution.fingerprint import (
    compute_effective_profile_fingerprint,
)
from intergrax.applications._shared.profile_resolution.materialize import (
    materialize_effective_profile_revision,
)
from intergrax.applications._shared.profile_resolution.store import (
    InMemoryEffectiveProfileRevisionStore,
)

__all__ = [
    "DEFAULT_FIELD_RESOLVERS",
    "InMemoryEffectiveProfileExecutionPinningStore",
    "InMemoryEffectiveProfileRevisionStore",
    "ProfileFieldDiffer",
    "ProfileFieldResolveResult",
    "ProfileFieldResolver",
    "attach_revision_checkpoint_evidence_to_task",
    "checkpoint_evidence_for_revision",
    "compute_effective_profile_fingerprint",
    "diff_effective_profile_revisions",
    "inherit_child_execution_pinned_revision",
    "materialize_effective_profile_revision",
    "pin_effective_profile_revision_for_execution",
    "require_execution_pinned_revision",
    "resolve_profile",
    "resolve_revision_for_execution",
    "revision_id_from_checkpoint",
]
