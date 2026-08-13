# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime revision registry projection (AGENT_DISTRIBUTION AP-10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import content_digest_for_model
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.stores import RuntimeRevisionStore
from intergrax.applications._shared.registry_snapshot import resolve_registry_snapshot
from intergrax.applications._shared.wiring import (
    BuilderMap,
    build_application_registry,
    binding_from_roster_entry,
    _index_manifest_bindings,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.harness_snapshot import HarnessRegistrySnapshot

_NON_EMPTY = Field(min_length=1)
SCHEMA_REGISTRY_PROJECTION_EVIDENCE_V1: Final = "registry_projection_evidence.v1"
SCHEMA_RUNTIME_REGISTRY_PROJECTION_SNAPSHOT_V1: Final = (
    "runtime_registry_projection_snapshot.v1"
)


class RegistryProjectionError(ValueError):
    """Frozen revision registry projection failed."""


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class RegistryProjectionEvidence(BaseModel):
    """Audit evidence for one traffic-serving registry projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_REGISTRY_PROJECTION_EVIDENCE_V1
    runtime_revision_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    effective_roster_revision_id: str = _NON_EMPTY
    materialized_runtime_lock_id: str | None = None
    materialized_runtime_lock_digest: str | None = None
    runtime_graph_digest: str | None = None
    registered_agent_ids: tuple[str, ...] = ()
    readiness_token: str = _NON_EMPTY

    @field_validator(
        "runtime_revision_id",
        "application_id",
        "application_environment_id",
        "effective_roster_revision_id",
        "materialized_runtime_lock_id",
        "materialized_runtime_lock_digest",
        "runtime_graph_digest",
        "readiness_token",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class RuntimeRegistryProjectionSnapshot(BaseModel):
    """Audit snapshot: revision identity + routable agent contract ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RUNTIME_REGISTRY_PROJECTION_SNAPSHOT_V1
    evidence: RegistryProjectionEvidence
    agent_contract_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MaterializedRegistryProjection:
    """Immutable registry projection bound to one runtime revision."""

    evidence: RegistryProjectionEvidence
    agent_registry: AgentRegistry
    harness_snapshot: HarnessRegistrySnapshot


@dataclass(frozen=True, slots=True)
class RegistryProjectionInputBundle:
    """Frozen revision inputs used to build a registry projection."""

    runtime_revision: RuntimeRevision
    effective_roster: EffectiveRoster
    manifest: ApplicationManifest
    build_context: ApplicationBuildContext
    builders: BuilderMap | None = None


class RegistryProjectionInputStore(Protocol):
    """Port for frozen projection inputs keyed by runtime_revision_id."""

    def get(self, runtime_revision_id: str) -> RegistryProjectionInputBundle | None:
        """Load frozen projection inputs for one runtime revision."""

    def register(self, bundle: RegistryProjectionInputBundle) -> None:
        """Persist frozen projection inputs for one runtime revision."""


class RuntimeRegistryProjectionStore(Protocol):
    """Port for materialized registry projections keyed by runtime_revision_id."""

    def get(self, runtime_revision_id: str) -> MaterializedRegistryProjection | None:
        """Load a prepared projection."""

    def put(self, projection: MaterializedRegistryProjection) -> None:
        """Store a prepared projection without mutating prior revisions."""


class InMemoryRegistryProjectionInputStore:
    """Deterministic in-memory projection input store for tests."""

    def __init__(self) -> None:
        self._bundles: dict[str, RegistryProjectionInputBundle] = {}

    def get(self, runtime_revision_id: str) -> RegistryProjectionInputBundle | None:
        return self._bundles.get(runtime_revision_id)

    def register(self, bundle: RegistryProjectionInputBundle) -> None:
        self._bundles[bundle.runtime_revision.runtime_revision_id] = bundle


class InMemoryRuntimeRegistryProjectionStore:
    """Deterministic in-memory projection cache keyed by runtime_revision_id."""

    def __init__(self) -> None:
        self._projections: dict[str, MaterializedRegistryProjection] = {}

    def get(self, runtime_revision_id: str) -> MaterializedRegistryProjection | None:
        return self._projections.get(runtime_revision_id)

    def put(self, projection: MaterializedRegistryProjection) -> None:
        revision_id = projection.evidence.runtime_revision_id
        if revision_id in self._projections:
            raise RegistryProjectionError(
                f"registry projection already materialized for {revision_id!r}"
            )
        self._projections[revision_id] = projection


def _expected_contract_ids(
    bundle: RegistryProjectionInputBundle,
) -> tuple[str, ...]:
    manifest_bindings = _index_manifest_bindings(bundle.manifest)
    contract_ids: list[str] = []
    for entry in bundle.effective_roster.entries:
        if not entry.effective_enablement:
            continue
        binding = binding_from_roster_entry(entry, manifest_bindings)
        contract_id = binding.contract_id
        if contract_id is None:
            raise RegistryProjectionError(
                f"enabled roster entry {entry.logical_agent_id!r} lacks contract id"
            )
        contract_ids.append(contract_id)
    return tuple(sorted(contract_ids))


def _validate_revision_bundle(bundle: RegistryProjectionInputBundle) -> None:
    revision = bundle.runtime_revision
    roster = bundle.effective_roster
    manifest = bundle.manifest

    if revision.application_environment_id != roster.application_environment_id:
        raise RegistryProjectionError("runtime revision environment mismatch with roster")
    if roster.application_id != manifest.app_id:
        raise RegistryProjectionError("effective roster application_id mismatch with manifest")
    if roster.effective_roster_revision_id is None:
        raise RegistryProjectionError("effective roster requires revision identity")
    if revision.effective_roster_revision_id != roster.effective_roster_revision_id:
        raise RegistryProjectionError("runtime revision roster revision mismatch")


def _validate_revision_identity(
    revision: RuntimeRevision,
    bundle: RegistryProjectionInputBundle,
) -> None:
    frozen_revision = bundle.runtime_revision
    roster = bundle.effective_roster
    if revision.runtime_revision_id != frozen_revision.runtime_revision_id:
        raise RegistryProjectionError("runtime revision id mismatch")
    if revision.application_environment_id != roster.application_environment_id:
        raise RegistryProjectionError("runtime revision environment mismatch")
    if revision.effective_roster_revision_id != roster.effective_roster_revision_id:
        raise RegistryProjectionError("effective roster revision mismatch")
    if frozen_revision.materialized_runtime_lock_id is not None:
        if revision.materialized_runtime_lock_id != frozen_revision.materialized_runtime_lock_id:
            raise RegistryProjectionError("materialized runtime lock id mismatch")
    if frozen_revision.materialized_runtime_lock_digest is not None:
        if (
            revision.materialized_runtime_lock_digest
            != frozen_revision.materialized_runtime_lock_digest
        ):
            raise RegistryProjectionError("materialized runtime lock digest mismatch")
    if frozen_revision.runtime_graph_digest is not None:
        if revision.runtime_graph_digest != frozen_revision.runtime_graph_digest:
            raise RegistryProjectionError("runtime graph digest mismatch")


def build_registry_projection(
    bundle: RegistryProjectionInputBundle,
) -> MaterializedRegistryProjection:
    """Build one immutable registry projection from frozen revision inputs."""
    _validate_revision_bundle(bundle)
    registry = build_application_registry(
        bundle.manifest,
        bundle.build_context,
        builders=bundle.builders,
        effective_roster=bundle.effective_roster,
    )
    expected_ids = _expected_contract_ids(bundle)
    registered_ids = tuple(sorted(registry.list_agent_ids()))
    if registered_ids != expected_ids:
        raise RegistryProjectionError(
            f"registry agent ids {registered_ids!r} do not match roster {expected_ids!r}"
        )

    revision = bundle.runtime_revision
    roster = bundle.effective_roster
    seed = RegistryProjectionEvidence(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=roster.application_id,
        application_environment_id=roster.application_environment_id,
        effective_roster_revision_id=roster.effective_roster_revision_id or "",
        materialized_runtime_lock_id=revision.materialized_runtime_lock_id,
        materialized_runtime_lock_digest=revision.materialized_runtime_lock_digest,
        runtime_graph_digest=revision.runtime_graph_digest,
        registered_agent_ids=registered_ids,
        readiness_token="projection-seed",
    )
    readiness_token = (
        f"projection-ready:{revision.runtime_revision_id}:"
        f"{content_digest_for_model(seed)}"
    )
    evidence = seed.model_copy(update={"readiness_token": readiness_token})
    harness_snapshot = resolve_registry_snapshot(
        bundle.build_context,
        agent_registry=registry,
    )
    return MaterializedRegistryProjection(
        evidence=evidence,
        agent_registry=registry,
        harness_snapshot=harness_snapshot,
    )


def projection_audit_snapshot(
    projection: MaterializedRegistryProjection,
) -> RuntimeRegistryProjectionSnapshot:
    """Typed audit snapshot for one materialized projection."""
    return RuntimeRegistryProjectionSnapshot(
        evidence=projection.evidence,
        agent_contract_ids=tuple(sorted(projection.agent_registry.list_agent_ids())),
    )


class ApplicationRegistryProjectionCoordinator:
    """AP-10 implementation of RuntimeServingProjectionCoordinator."""

    def __init__(
        self,
        revision_store: RuntimeRevisionStore,
        input_store: RegistryProjectionInputStore,
        projection_store: RuntimeRegistryProjectionStore,
    ) -> None:
        self._revision_store = revision_store
        self._input_store = input_store
        self._projection_store = projection_store

    def prepare_projection(self, runtime_revision_id: str) -> str:
        existing = self._projection_store.get(runtime_revision_id)
        if existing is not None:
            return existing.evidence.readiness_token

        revision = self._revision_store.get_revision(runtime_revision_id)
        if revision is None:
            raise RegistryProjectionError(
                f"runtime revision not found: {runtime_revision_id!r}"
            )

        bundle = self._input_store.get(runtime_revision_id)
        if bundle is None:
            raise RegistryProjectionError(
                f"missing frozen projection inputs for {runtime_revision_id!r}"
            )

        _validate_revision_identity(revision, bundle)
        projection = build_registry_projection(bundle)
        self._projection_store.put(projection)
        return projection.evidence.readiness_token

    def rollback_projection(self, runtime_revision_id: str) -> None:
        """Ensure rollback target projection is available without rebuilding from desired state."""
        if self._projection_store.get(runtime_revision_id) is not None:
            return
        self.prepare_projection(runtime_revision_id)

    def get_projection(self, runtime_revision_id: str) -> MaterializedRegistryProjection | None:
        return self._projection_store.get(runtime_revision_id)


__all__ = [
    "ApplicationRegistryProjectionCoordinator",
    "InMemoryRegistryProjectionInputStore",
    "InMemoryRuntimeRegistryProjectionStore",
    "MaterializedRegistryProjection",
    "RegistryProjectionEvidence",
    "RegistryProjectionError",
    "RegistryProjectionInputBundle",
    "RegistryProjectionInputStore",
    "RuntimeRegistryProjectionSnapshot",
    "RuntimeRegistryProjectionStore",
    "build_registry_projection",
    "projection_audit_snapshot",
]
