# © Artur Czarnecki. All rights reserved.

"""Canonical reference runtime materialization authority (AC-3 reference lifecycle).

Composes ``RuntimeMaterializationService`` and ``RuntimeMaterializationStore`` for
explicit reference/demo production bootstrap. Reference activation remains
fail-closed without a persisted ``RuntimeMaterializationRecord``.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.dependency import (
    LockPackageRole,
    MaterializedAgentClosureEntry,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.errors import (
    MaterializationInputConflict,
    RuntimeMaterializationConflict,
)
from intergrax.agent_distribution.materialization import (
    ApplicationBuildContext,
    MaterializationInput,
    MaterializationOutput,
)
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_context_staging import RUNTIME_GRAPH_MANIFEST_FILENAME
from intergrax.agent_distribution.runtime_graph import (
    CandidateApplicationRuntimeGraph,
    GRAPH_SCHEMA_VERSION_V3,
    RuntimeGraphAgentRef,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
    validate_runtime_materialization_record_matches_revision,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology, RuntimeRevision
from intergrax.applications._shared.production_agent_platform_runtime import AgentPlatformRuntimeStores
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications.contracts.manifest import ApplicationManifest

REFERENCE_RESOLVER_ALGORITHM_ID = "intergrax.reference-process-local"
REFERENCE_RESOLVER_ALGORITHM_VERSION = "1.0.0"
REFERENCE_MATERIALIZER_ID = "intergrax.reference-process-local-materializer"
REFERENCE_MATERIALIZER_VERSION = "1.0.0"
REFERENCE_SOURCE_CONTEXT_ROOT = "/reference/process-local"
REFERENCE_OUTPUT_ROOT = "/reference/process-local/output"
_REFERENCE_ARTIFACT_LOCATOR_PREFIX = "reference://process-local/venv-bundle"


@dataclass(frozen=True, slots=True)
class ReferenceRuntimeIdentityDigests:
    """Content-derived lock and graph identities for one reference roster."""

    materialized_runtime_lock_id: str
    materialized_runtime_lock_digest: str
    runtime_graph_digest: str


@dataclass(frozen=True, slots=True)
class ReferenceProcessLocalVenvBundleMaterializationAdapter:
    """Process-local reference adapter — no filesystem artifact build."""

    artifact_locator: str
    materializer_id: str = REFERENCE_MATERIALIZER_ID
    materializer_version: str = REFERENCE_MATERIALIZER_VERSION

    @property
    def topology(self) -> MaterializationTopology:
        return MaterializationTopology.VENV_BUNDLE

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        revision = materialization_input.runtime_revision
        artifact_digest = revision.materialization_artifact_digest
        if artifact_digest is None:
            raise MaterializationInputConflict(
                "reference runtime revision requires materialization_artifact_digest"
            )
        return MaterializationOutput(
            materialization_artifact_digest=artifact_digest,
            artifact_locator=self.artifact_locator,
            health_check_evidence_ref=(
                f"reference://materialization/{revision.runtime_revision_id}"
            ),
            runtime_graph_manifest_path=RUNTIME_GRAPH_MANIFEST_FILENAME,
            topology=MaterializationTopology.VENV_BUNDLE,
        )


def build_reference_runtime_identity_digests(
    roster: EffectiveRoster,
) -> ReferenceRuntimeIdentityDigests:
    """Derive canonical lock/graph identities for one reference effective roster."""
    lock = build_reference_materialized_runtime_lock(roster)
    graph = build_reference_candidate_runtime_graph(roster, lock)
    if lock.lock_id is None or lock.lock_digest is None:
        raise MaterializationInputConflict("reference runtime lock requires content identity")
    if graph.runtime_graph_digest is None:
        raise MaterializationInputConflict("reference runtime graph requires content identity")
    return ReferenceRuntimeIdentityDigests(
        materialized_runtime_lock_id=lock.lock_id,
        materialized_runtime_lock_digest=lock.lock_digest,
        runtime_graph_digest=graph.runtime_graph_digest,
    )


def build_reference_materialized_runtime_lock(
    roster: EffectiveRoster,
) -> MaterializedRuntimeLock:
    """Build one content-identified lock artifact for reference lifecycle proofs."""
    enabled_entries = tuple(entry for entry in roster.entries if entry.effective_enablement)
    if not enabled_entries:
        raise MaterializationInputConflict("reference roster requires enabled entries")
    roster_revision_id = roster.effective_roster_revision_id
    if roster_revision_id is None:
        raise MaterializationInputConflict("reference roster requires revision identity")
    agent_closure = tuple(
        MaterializedAgentClosureEntry(
            distribution_package_id=entry.distribution_package_id,
            package_digest=entry.package_digest,
            role=LockPackageRole.DIRECT,
        )
        for entry in enabled_entries
    )
    return MaterializedRuntimeLock(
        resolver_algorithm_id=REFERENCE_RESOLVER_ALGORITHM_ID,
        resolver_algorithm_version=REFERENCE_RESOLVER_ALGORITHM_VERSION,
        inputs_digest=(
            f"reference:{roster.application_id}:{roster_revision_id}"
        ),
        intergrax_version="0.1.0",
        python_version="3.12",
        packages=(),
        agent_closure=agent_closure,
    ).with_content_identity()


def build_reference_candidate_runtime_graph(
    roster: EffectiveRoster,
    lock: MaterializedRuntimeLock,
) -> CandidateApplicationRuntimeGraph:
    """Build one content-identified runtime graph for reference lifecycle proofs."""
    if lock.lock_id is None:
        raise MaterializationInputConflict("reference runtime lock requires lock_id")
    enabled_entries = tuple(entry for entry in roster.entries if entry.effective_enablement)
    if not enabled_entries:
        raise MaterializationInputConflict("reference roster requires enabled entries")
    direct_agents = tuple(
        RuntimeGraphAgentRef(
            logical_agent_id=entry.logical_agent_id,
            distribution_package_id=entry.distribution_package_id,
            package_digest=entry.package_digest,
        )
        for entry in enabled_entries
    )
    return CandidateApplicationRuntimeGraph(
        graph_schema_version=GRAPH_SCHEMA_VERSION_V3,
        application_id=roster.application_id,
        materialized_runtime_lock_id=lock.lock_id,
        direct_agents=direct_agents,
    ).with_content_identity()


def build_reference_distribution_build_context(
    revision: RuntimeRevision,
    manifest: ApplicationManifest,
) -> ApplicationBuildContext:
    """Build neutral distribution build context for reference materialization."""
    return ApplicationBuildContext(
        application_id=revision.application_id,
        application_release_id=revision.application_release_id,
        application_environment_id=revision.application_environment_id,
        source_context_root=REFERENCE_SOURCE_CONTEXT_ROOT,
        platform_version=revision.platform_version,
        python_version="3.12",
        output_root=REFERENCE_OUTPUT_ROOT,
        application_source_root=f"applications/{manifest.app_id}",
    )


def build_reference_materialization_input(
    projection_input: RegistryProjectionInputBundle,
) -> MaterializationInput:
    """Assemble canonical materialization input from one reference projection bundle."""
    revision = projection_input.runtime_revision
    roster = projection_input.effective_roster
    lock = build_reference_materialized_runtime_lock(roster)
    graph = build_reference_candidate_runtime_graph(roster, lock)
    _validate_reference_revision_identity(revision, lock, graph)
    return MaterializationInput(
        runtime_revision=revision,
        materialized_runtime_lock=lock,
        candidate_runtime_graph=graph,
        effective_roster=roster,
        application_build_context=build_reference_distribution_build_context(
            revision,
            projection_input.manifest,
        ),
    )


def prepare_reference_runtime_materialization(
    stores: AgentPlatformRuntimeStores,
    projection_input: RegistryProjectionInputBundle,
    *,
    artifact_locator: str | None = None,
) -> RuntimeMaterializationRecord:
    """Persist canonical materialization authority for one reference runtime revision."""
    revision = projection_input.runtime_revision
    runtime_revision_id = revision.runtime_revision_id
    normalized_locator = (
        artifact_locator.strip()
        if artifact_locator is not None
        else _default_reference_artifact_locator(runtime_revision_id)
    )
    if not normalized_locator:
        raise MaterializationInputConflict("reference materialization requires artifact_locator")

    existing = stores.materialization_store.get_by_revision(runtime_revision_id)
    if existing is not None:
        validate_runtime_materialization_record_matches_revision(revision, existing)
        if existing.artifact_locator != normalized_locator:
            raise RuntimeMaterializationConflict(
                "runtime materialization artifact locator mismatch"
            )
        return existing

    materialization_input = build_reference_materialization_input(projection_input)
    lock = materialization_input.materialized_runtime_lock
    stores.lock_store.persist_lock(lock)
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.VENV_BUNDLE: ReferenceProcessLocalVenvBundleMaterializationAdapter(
                artifact_locator=normalized_locator,
            ),
        }
    )
    output = service.materialize(materialization_input)
    record = _build_runtime_materialization_record(
        revision,
        artifact_locator=output.artifact_locator,
    )
    validate_runtime_materialization_record_matches_revision(revision, record)
    return stores.materialization_store.persist(record)


def _default_reference_artifact_locator(runtime_revision_id: str) -> str:
    normalized = runtime_revision_id.strip()
    if not normalized:
        raise ValueError("runtime_revision_id must be non-empty")
    return f"{_REFERENCE_ARTIFACT_LOCATOR_PREFIX}/{normalized}"


def _build_runtime_materialization_record(
    revision: RuntimeRevision,
    *,
    artifact_locator: str,
) -> RuntimeMaterializationRecord:
    if revision.materialization_topology is None:
        raise MaterializationInputConflict(
            "reference runtime revision requires materialization_topology"
        )
    if revision.materialization_artifact_digest is None:
        raise MaterializationInputConflict(
            "reference runtime revision requires materialization_artifact_digest"
        )
    if revision.materialized_runtime_lock_id is None:
        raise MaterializationInputConflict(
            "reference runtime revision requires materialized_runtime_lock_id"
        )
    if revision.materialized_runtime_lock_digest is None:
        raise MaterializationInputConflict(
            "reference runtime revision requires materialized_runtime_lock_digest"
        )
    return RuntimeMaterializationRecord(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        materialization_topology=revision.materialization_topology,
        artifact_locator=artifact_locator,
        materialization_artifact_digest=revision.materialization_artifact_digest,
        materialized_runtime_lock_id=revision.materialized_runtime_lock_id,
        materialized_runtime_lock_digest=revision.materialized_runtime_lock_digest,
    )


def _validate_reference_revision_identity(
    revision: RuntimeRevision,
    lock: MaterializedRuntimeLock,
    graph: CandidateApplicationRuntimeGraph,
) -> None:
    if lock.lock_id is None or lock.lock_digest is None:
        raise MaterializationInputConflict("reference runtime lock requires content identity")
    if graph.runtime_graph_digest is None:
        raise MaterializationInputConflict("reference runtime graph requires content identity")
    if revision.materialized_runtime_lock_id != lock.lock_id:
        raise MaterializationInputConflict("reference runtime revision lock id mismatch")
    if revision.materialized_runtime_lock_digest != lock.lock_digest:
        raise MaterializationInputConflict("reference runtime revision lock digest mismatch")
    if revision.runtime_graph_digest != graph.runtime_graph_digest:
        raise MaterializationInputConflict("reference runtime revision graph digest mismatch")


__all__ = [
    "ReferenceProcessLocalVenvBundleMaterializationAdapter",
    "ReferenceRuntimeIdentityDigests",
    "build_reference_candidate_runtime_graph",
    "build_reference_distribution_build_context",
    "build_reference_materialization_input",
    "build_reference_materialized_runtime_lock",
    "build_reference_runtime_identity_digests",
    "prepare_reference_runtime_materialization",
]
