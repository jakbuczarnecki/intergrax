# © Artur Czarnecki. All rights reserved.

"""Production revision-bound ``RegistryProjectionInputBundle`` assembly (AC-3-FINAL).

Builds frozen AP-10 projection inputs from canonical lifecycle stores keyed by
``runtime_revision_id``. Caller authority is limited to ``EffectiveRoster`` until
Phase 4; revision, lock, artifact locator, and digest are derived from stores.

Does **not** synthesize roster/lock/revision identity from ``ApplicationManifest``,
does **not** accept caller factory maps, and does **not** use host-side in-memory
factory resolver shortcuts.
"""

from __future__ import annotations

from pathlib import Path

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.errors import RuntimeMaterializationConflict
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
    validate_runtime_materialization_record_matches_revision,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    AgentPlatformRuntimeStores,
)
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionError,
    RegistryProjectionInputBundle,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    build_production_runtime_agent_factory_resolver,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest

_PROJECTION_ELIGIBLE_REVISION_STATES = frozenset(
    {
        RuntimeRevisionState.VALIDATED,
        RuntimeRevisionState.ACTIVE,
        RuntimeRevisionState.SUPERSEDED,
    }
)


class ProductionRegistryProjectionInputError(ValueError):
    """Canonical production projection input assembly failed."""


def production_test_artifact_locator(artifact_root: Path) -> str:
    """Deterministic filesystem artifact locator for production proofs/tests."""
    resolved = artifact_root.resolve()
    if not resolved.is_dir():
        raise ProductionRegistryProjectionInputError(
            f"artifact root must be an existing directory: {resolved!s}"
        )
    return f"test://{resolved.as_posix()}"


def resolve_production_artifact_root(artifact_locator: str) -> Path:
    """Resolve one immutable VENV_BUNDLE artifact directory from a canonical locator."""
    normalized = artifact_locator.strip()
    if not normalized:
        raise ProductionRegistryProjectionInputError(
            "artifact_locator must be non-empty"
        )
    if normalized.startswith("reference://"):
        raise ProductionRegistryProjectionInputError(
            "reference:// artifact locators are forbidden for production authority"
        )
    if normalized.startswith("test://"):
        candidate = Path(normalized.removeprefix("test://"))
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise ProductionRegistryProjectionInputError(
                f"test artifact locator does not resolve to a directory: {artifact_locator!r}"
            )
        return resolved
    if normalized.startswith("file://"):
        candidate = Path(normalized.removeprefix("file://"))
        if not candidate.is_absolute():
            candidate = Path(normalized.removeprefix("file://").lstrip("/"))
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise ProductionRegistryProjectionInputError(
                f"file artifact locator does not resolve to a directory: {artifact_locator!r}"
            )
        return resolved
    raise ProductionRegistryProjectionInputError(
        f"unsupported production artifact_locator scheme: {artifact_locator!r}"
    )


def _validated_lock_identity(lock: MaterializedRuntimeLock) -> MaterializedRuntimeLock:
    if lock.lock_id is None or lock.lock_digest is None:
        return lock.with_content_identity()
    return lock


def _validate_revision_roster_lock_authority(
    *,
    runtime_revision: RuntimeRevision,
    effective_roster: EffectiveRoster,
    materialized_runtime_lock: MaterializedRuntimeLock,
    materialization_artifact_digest: str,
) -> None:
    if runtime_revision.application_id != effective_roster.application_id:
        raise ProductionRegistryProjectionInputError(
            "runtime revision application_id mismatch with effective roster"
        )
    if (
        runtime_revision.application_environment_id
        != effective_roster.application_environment_id
    ):
        raise ProductionRegistryProjectionInputError(
            "runtime revision application_environment_id mismatch with effective roster"
        )
    if runtime_revision.application_release_id != effective_roster.manifest_release_id:
        raise ProductionRegistryProjectionInputError(
            "runtime revision application_release_id mismatch with effective roster"
        )
    if effective_roster.effective_roster_revision_id is None:
        raise ProductionRegistryProjectionInputError(
            "effective roster requires effective_roster_revision_id"
        )
    if (
        runtime_revision.effective_roster_revision_id
        != effective_roster.effective_roster_revision_id
    ):
        raise ProductionRegistryProjectionInputError(
            "runtime revision effective_roster_revision_id mismatch with effective roster"
        )
    if runtime_revision.materialization_artifact_digest is None:
        raise ProductionRegistryProjectionInputError(
            "runtime revision requires materialization_artifact_digest"
        )
    if (
        runtime_revision.materialization_artifact_digest
        != materialization_artifact_digest
    ):
        raise ProductionRegistryProjectionInputError(
            "materialization_artifact_digest mismatch with runtime revision"
        )
    if (
        runtime_revision.materialization_topology
        is not MaterializationTopology.VENV_BUNDLE
    ):
        raise ProductionRegistryProjectionInputError(
            "production registry projection requires VENV_BUNDLE topology"
        )

    lock = _validated_lock_identity(materialized_runtime_lock)
    if runtime_revision.materialized_runtime_lock_id is not None:
        if lock.lock_id != runtime_revision.materialized_runtime_lock_id:
            raise ProductionRegistryProjectionInputError(
                "materialized runtime lock id mismatch with runtime revision"
            )
    if runtime_revision.materialized_runtime_lock_digest is not None:
        if lock.lock_digest != runtime_revision.materialized_runtime_lock_digest:
            raise ProductionRegistryProjectionInputError(
                "materialized runtime lock digest mismatch with runtime revision"
            )


def _validate_canonical_lock_authority(
    *,
    runtime_revision: RuntimeRevision,
    materialization: RuntimeMaterializationRecord,
    materialized_runtime_lock: MaterializedRuntimeLock,
) -> MaterializedRuntimeLock:
    lock = _validated_lock_identity(materialized_runtime_lock)
    lock_id = runtime_revision.materialized_runtime_lock_id
    lock_digest = runtime_revision.materialized_runtime_lock_digest
    if lock_id is None or lock_digest is None:
        raise ProductionRegistryProjectionInputError(
            "runtime revision requires materialized runtime lock identity"
        )
    if lock.lock_id != lock_id:
        raise ProductionRegistryProjectionInputError(
            "canonical materialized runtime lock id mismatch with runtime revision"
        )
    if lock.lock_digest != lock_digest:
        raise ProductionRegistryProjectionInputError(
            "canonical materialized runtime lock digest mismatch with runtime revision"
        )
    if materialization.materialized_runtime_lock_id != lock_id:
        raise ProductionRegistryProjectionInputError(
            "canonical materialized runtime lock id mismatch with materialization record"
        )
    if materialization.materialized_runtime_lock_digest != lock_digest:
        raise ProductionRegistryProjectionInputError(
            "canonical materialized runtime lock digest mismatch with materialization record"
        )
    return lock


def _assemble_production_registry_projection_input_bundle(
    *,
    runtime_revision: RuntimeRevision,
    effective_roster: EffectiveRoster,
    materialized_runtime_lock: MaterializedRuntimeLock,
    manifest: ApplicationManifest,
    build_context: ApplicationBuildContext,
    artifact_locator: str,
    materialization_artifact_digest: str,
) -> RegistryProjectionInputBundle:
    """Internal assembly from already-resolved canonical lifecycle authority."""
    _validate_revision_roster_lock_authority(
        runtime_revision=runtime_revision,
        effective_roster=effective_roster,
        materialized_runtime_lock=materialized_runtime_lock,
        materialization_artifact_digest=materialization_artifact_digest,
    )
    if manifest.app_id != runtime_revision.application_id:
        raise ProductionRegistryProjectionInputError(
            "manifest application_id mismatch with runtime revision"
        )

    artifact_root = resolve_production_artifact_root(artifact_locator)
    try:
        factory_resolver = build_production_runtime_agent_factory_resolver(
            runtime_revision=runtime_revision,
            artifact_root=artifact_root,
            expected_artifact_digest=materialization_artifact_digest,
        )
    except RuntimeAgentFactoryResolutionError as exc:
        raise ProductionRegistryProjectionInputError(str(exc)) from exc

    lock = factory_resolver.materialized_runtime_lock
    _validate_revision_roster_lock_authority(
        runtime_revision=runtime_revision,
        effective_roster=effective_roster,
        materialized_runtime_lock=lock,
        materialization_artifact_digest=materialization_artifact_digest,
    )

    return RegistryProjectionInputBundle(
        runtime_revision=runtime_revision,
        effective_roster=effective_roster,
        manifest=manifest,
        build_context=build_context,
        factory_resolver=factory_resolver,
        builders=None,
        materialization_artifact_digest=materialization_artifact_digest,
    )


def build_production_registry_projection_input_bundle_for_revision(
    *,
    application_id: str,
    application_environment_id: str,
    runtime_revision_id: str,
    effective_roster: EffectiveRoster,
    manifest: ApplicationManifest,
    build_context: ApplicationBuildContext,
    stores: AgentPlatformRuntimeStores,
) -> RegistryProjectionInputBundle:
    """Assemble one production projection input bundle from canonical lifecycle stores."""
    normalized_revision_id = runtime_revision_id.strip()
    if not normalized_revision_id:
        raise ProductionRegistryProjectionInputError(
            "runtime_revision_id must be non-empty"
        )

    revision = stores.revision_store.get_revision(normalized_revision_id)
    if revision is None:
        raise ProductionRegistryProjectionInputError(
            f"runtime revision not found: {normalized_revision_id!r}"
        )
    if revision.runtime_revision_id != normalized_revision_id:
        raise ProductionRegistryProjectionInputError(
            "runtime revision id mismatch with canonical store record"
        )
    if revision.application_id != application_id:
        raise ProductionRegistryProjectionInputError(
            "runtime revision application_id mismatch with projection scope"
        )
    if revision.application_environment_id != application_environment_id:
        raise ProductionRegistryProjectionInputError(
            "runtime revision application_environment_id mismatch with projection scope"
        )
    if revision.revision_state not in _PROJECTION_ELIGIBLE_REVISION_STATES:
        raise ProductionRegistryProjectionInputError(
            f"runtime revision state {revision.revision_state.value!r} is not projection-eligible"
        )

    materialization = stores.materialization_store.get_by_revision(
        normalized_revision_id
    )
    if materialization is None:
        raise ProductionRegistryProjectionInputError(
            f"missing canonical materialization record for {normalized_revision_id!r}"
        )
    try:
        validate_runtime_materialization_record_matches_revision(
            revision, materialization
        )
    except RuntimeMaterializationConflict as exc:
        raise ProductionRegistryProjectionInputError(str(exc)) from exc

    artifact_digest = revision.materialization_artifact_digest
    if artifact_digest is None:
        raise ProductionRegistryProjectionInputError(
            "runtime revision requires materialization_artifact_digest"
        )
    if artifact_digest != materialization.materialization_artifact_digest:
        raise ProductionRegistryProjectionInputError(
            "materialization_artifact_digest mismatch with canonical materialization record"
        )

    lock_id = revision.materialized_runtime_lock_id
    if lock_id is None:
        raise ProductionRegistryProjectionInputError(
            "runtime revision requires materialized_runtime_lock_id"
        )
    lock = stores.lock_store.get_lock(lock_id)
    if lock is None:
        raise ProductionRegistryProjectionInputError(
            f"canonical materialized runtime lock not found: {lock_id!r}"
        )
    lock = _validate_canonical_lock_authority(
        runtime_revision=revision,
        materialization=materialization,
        materialized_runtime_lock=lock,
    )

    return _assemble_production_registry_projection_input_bundle(
        runtime_revision=revision,
        effective_roster=effective_roster,
        materialized_runtime_lock=lock,
        manifest=manifest,
        build_context=build_context,
        artifact_locator=materialization.artifact_locator,
        materialization_artifact_digest=artifact_digest,
    )


def build_production_registry_projection_for_revision(
    *,
    application_id: str,
    application_environment_id: str,
    runtime_revision_id: str,
    effective_roster: EffectiveRoster,
    manifest: ApplicationManifest,
    build_context: ApplicationBuildContext,
    stores: AgentPlatformRuntimeStores,
):
    """Build and project one production registry from canonical lifecycle stores."""
    from intergrax.applications._shared.registry_projection import (
        build_registry_projection,
    )

    bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        effective_roster=effective_roster,
        manifest=manifest,
        build_context=build_context,
        stores=stores,
    )
    try:
        return build_registry_projection(bundle)
    except RegistryProjectionError as exc:
        raise ProductionRegistryProjectionInputError(str(exc)) from exc


__all__ = [
    "ProductionRegistryProjectionInputError",
    "build_production_registry_projection_for_revision",
    "build_production_registry_projection_input_bundle_for_revision",
    "production_test_artifact_locator",
    "resolve_production_artifact_root",
]
