# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime revision registry projection (AGENT_DISTRIBUTION AP-10)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import content_digest_for_model
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.stores import RuntimeRevisionStore
from intergrax.applications._shared.registry_snapshot import resolve_registry_snapshot
from intergrax.applications._shared.wiring import (
    BuilderMap,
    build_application_registry,
    binding_from_roster_entry,
    _index_manifest_bindings,
    _resolve_manifest_binding_for_entry,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.attestation.canonical_json import stable_payload_hash
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
    application_release_id: str = _NON_EMPTY
    effective_roster_revision_id: str = _NON_EMPTY
    materialized_runtime_lock_id: str | None = None
    materialized_runtime_lock_digest: str | None = None
    runtime_graph_digest: str | None = None
    materialization_artifact_digest: str | None = None
    registry_factory_wiring_digest: str = _NON_EMPTY
    registered_agent_ids: tuple[str, ...] = ()
    readiness_token: str = _NON_EMPTY

    @field_validator(
        "runtime_revision_id",
        "application_id",
        "application_environment_id",
        "application_release_id",
        "effective_roster_revision_id",
        "materialized_runtime_lock_id",
        "materialized_runtime_lock_digest",
        "runtime_graph_digest",
        "materialization_artifact_digest",
        "registry_factory_wiring_digest",
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
    materialization_artifact_digest: str | None = None


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


def _manifest_factory_entrypoint(binding: AgentBinding) -> dict[str, str]:
    if binding.factory_path is not None:
        return {"manifest_factory_path": binding.factory_path}
    if binding.builder_key is not None:
        return {"manifest_builder_key": binding.builder_key}
    import_path = binding.import_path
    if import_path is None:
        raise RegistryProjectionError("manifest binding lacks factory entrypoint")
    return {"manifest_import_path": import_path}


def _entry_factory_authority_payload(
    entry: EffectiveRosterEntry,
    manifest_bindings: Mapping[str, AgentBinding],
) -> dict[str, object]:
    manifest_binding = _resolve_manifest_binding_for_entry(entry, manifest_bindings)
    payload: dict[str, object] = {
        "logical_agent_id": entry.logical_agent_id,
        "package_digest": entry.package_digest,
        "distribution_package_id": entry.distribution_package_id,
    }
    factory_reference = entry.factory_reference
    if factory_reference is not None:
        payload["factory_reference"] = factory_reference.model_dump(mode="json")
    elif manifest_binding is not None:
        payload.update(_manifest_factory_entrypoint(manifest_binding))
    else:
        raise RegistryProjectionError(
            f"enabled roster entry {entry.logical_agent_id!r} lacks factory authority"
        )
    return payload


def _factory_authority_fingerprint(
    bundle: RegistryProjectionInputBundle,
) -> tuple[dict[str, object], ...]:
    manifest_bindings = _index_manifest_bindings(bundle.manifest)
    authorities: list[dict[str, object]] = []
    for entry in bundle.effective_roster.entries:
        if not entry.effective_enablement:
            continue
        authorities.append(_entry_factory_authority_payload(entry, manifest_bindings))
    return tuple(sorted(authorities, key=lambda item: str(item["logical_agent_id"])))


def _build_context_fingerprint(ctx: ApplicationBuildContext) -> dict[str, object]:
    manifest = ctx.manifest
    app_id = manifest.app_id if isinstance(manifest, ApplicationManifest) else None
    return {
        "manifest_app_id": app_id,
        "skill_profile": (
            ctx.skill_profile.model_dump(mode="json") if ctx.skill_profile is not None else None
        ),
        "tool_profile": (
            ctx.tool_profile.model_dump(mode="json") if ctx.tool_profile is not None else None
        ),
        "strict_harness": ctx.strict_harness,
    }


def _bundle_semantic_fingerprint(bundle: RegistryProjectionInputBundle) -> str:
    payload = {
        "runtime_revision": bundle.runtime_revision.model_dump(mode="json"),
        "effective_roster": bundle.effective_roster.model_dump(mode="json"),
        "manifest": bundle.manifest.model_dump(mode="json"),
        "materialization_artifact_digest": bundle.materialization_artifact_digest,
        "factory_authorities": _factory_authority_fingerprint(bundle),
        "build_context": _build_context_fingerprint(bundle.build_context),
    }
    return stable_payload_hash(payload)


def _registry_factory_wiring_digest(bundle: RegistryProjectionInputBundle) -> str:
    revision = bundle.runtime_revision
    seed = {
        "application_release_id": revision.application_release_id,
        "materialization_artifact_digest": bundle.materialization_artifact_digest,
        "manifest": bundle.manifest.model_dump(mode="json"),
        "factory_authorities": _factory_authority_fingerprint(bundle),
        "build_context": _build_context_fingerprint(bundle.build_context),
    }
    return stable_payload_hash(seed)


class InMemoryRegistryProjectionInputStore:
    """Deterministic in-memory projection input store for tests."""

    def __init__(self) -> None:
        self._bundles: dict[str, RegistryProjectionInputBundle] = {}
        self._fingerprints: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, runtime_revision_id: str) -> RegistryProjectionInputBundle | None:
        with self._lock:
            return self._bundles.get(runtime_revision_id)

    def register(self, bundle: RegistryProjectionInputBundle) -> None:
        revision_id = bundle.runtime_revision.runtime_revision_id
        fingerprint = _bundle_semantic_fingerprint(bundle)
        with self._lock:
            existing = self._fingerprints.get(revision_id)
            if existing is not None:
                if existing != fingerprint:
                    raise RegistryProjectionError(
                        f"conflicting frozen projection inputs for {revision_id!r}"
                    )
                return
            self._bundles[revision_id] = bundle
            self._fingerprints[revision_id] = fingerprint


def _projection_semantic_fingerprint(projection: MaterializedRegistryProjection) -> str:
    evidence = projection.evidence.model_copy(update={"readiness_token": "projection-seed"})
    return content_digest_for_model(evidence)


class InMemoryRuntimeRegistryProjectionStore:
    """Deterministic in-memory projection cache keyed by runtime_revision_id."""

    def __init__(self) -> None:
        self._projections: dict[str, MaterializedRegistryProjection] = {}
        self._fingerprints: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, runtime_revision_id: str) -> MaterializedRegistryProjection | None:
        with self._lock:
            return self._projections.get(runtime_revision_id)

    def put(self, projection: MaterializedRegistryProjection) -> None:
        revision_id = projection.evidence.runtime_revision_id
        fingerprint = _projection_semantic_fingerprint(projection)
        with self._lock:
            existing = self._projections.get(revision_id)
            if existing is not None:
                if self._fingerprints[revision_id] != fingerprint:
                    raise RegistryProjectionError(
                        f"conflicting registry projection for {revision_id!r}"
                    )
                return
            self._projections[revision_id] = projection
            self._fingerprints[revision_id] = fingerprint


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


def _validate_factory_reference_against_builders(
    *,
    logical_agent_id: str,
    factory_reference: AgentBindingFactoryReference,
    builders: Mapping[object, object],
) -> None:
    builder_key = factory_reference.builder_key
    if builder_key is not None and builder_key not in builders:
        raise RegistryProjectionError(
            f"roster entry {logical_agent_id!r} references builder_key "
            f"{builder_key!r} missing from frozen release builders"
        )


def _validate_entry_package_digest_trust(
    *,
    logical_agent_id: str,
    package_digest: str,
    trusted_package_digests: frozenset[str],
) -> None:
    if package_digest not in trusted_package_digests:
        raise RegistryProjectionError(
            f"roster entry {logical_agent_id!r} package_digest "
            f"{package_digest!r} is not trusted by runtime revision"
        )


def _validate_factory_authority(bundle: RegistryProjectionInputBundle) -> None:
    builders: Mapping[object, object] = bundle.builders or {}
    manifest_bindings = _index_manifest_bindings(bundle.manifest)
    revision = bundle.runtime_revision
    trusted_packages = frozenset(revision.installed_agent_package_digests)
    require_trusted_packages = bool(trusted_packages)

    for entry in bundle.effective_roster.entries:
        if not entry.effective_enablement:
            continue

        manifest_binding = _resolve_manifest_binding_for_entry(entry, manifest_bindings)
        factory_reference = entry.factory_reference

        if manifest_binding is None:
            if factory_reference is None:
                raise RegistryProjectionError(
                    f"operator-added agent {entry.logical_agent_id!r} requires "
                    "immutable factory_reference authority"
                )
            if require_trusted_packages:
                _validate_entry_package_digest_trust(
                    logical_agent_id=entry.logical_agent_id,
                    package_digest=entry.package_digest,
                    trusted_package_digests=trusted_packages,
                )
            _validate_factory_reference_against_builders(
                logical_agent_id=entry.logical_agent_id,
                factory_reference=factory_reference,
                builders=builders,
            )
            continue

        if require_trusted_packages:
            _validate_entry_package_digest_trust(
                logical_agent_id=entry.logical_agent_id,
                package_digest=entry.package_digest,
                trusted_package_digests=trusted_packages,
            )

        if factory_reference is not None:
            _validate_factory_reference_against_builders(
                logical_agent_id=entry.logical_agent_id,
                factory_reference=factory_reference,
                builders=builders,
            )


def _validate_release_authority(bundle: RegistryProjectionInputBundle) -> None:
    """Mirror MaterializationInput release binding for registry factory wiring."""
    revision = bundle.runtime_revision
    roster = bundle.effective_roster
    manifest = bundle.manifest

    if revision.application_environment_id != roster.application_environment_id:
        raise RegistryProjectionError("runtime revision environment mismatch with roster")
    if revision.application_release_id != roster.manifest_release_id:
        raise RegistryProjectionError("runtime revision release mismatch with roster")
    if roster.application_id != manifest.app_id:
        raise RegistryProjectionError("effective roster application_id mismatch with manifest")
    if roster.effective_roster_revision_id is None:
        raise RegistryProjectionError("effective roster requires revision identity")
    if revision.effective_roster_revision_id != roster.effective_roster_revision_id:
        raise RegistryProjectionError("runtime revision roster revision mismatch")

    artifact_digest = bundle.materialization_artifact_digest
    if revision.materialization_artifact_digest is not None:
        if artifact_digest is None:
            raise RegistryProjectionError(
                "projection inputs require materialization artifact identity"
            )
        if artifact_digest != revision.materialization_artifact_digest:
            raise RegistryProjectionError("materialization artifact digest mismatch")


def _validate_revision_bundle(bundle: RegistryProjectionInputBundle) -> None:
    _validate_release_authority(bundle)
    _validate_factory_authority(bundle)


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
    if revision.application_release_id != roster.manifest_release_id:
        raise RegistryProjectionError("runtime revision release mismatch")
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
    if revision.materialization_artifact_digest is not None:
        artifact_digest = bundle.materialization_artifact_digest
        if artifact_digest is None:
            raise RegistryProjectionError(
                "projection inputs require materialization artifact identity"
            )
        if revision.materialization_artifact_digest != artifact_digest:
            raise RegistryProjectionError("materialization artifact digest mismatch")


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
    wiring_digest = _registry_factory_wiring_digest(bundle)
    seed = RegistryProjectionEvidence(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=roster.application_id,
        application_environment_id=roster.application_environment_id,
        application_release_id=revision.application_release_id,
        effective_roster_revision_id=roster.effective_roster_revision_id or "",
        materialized_runtime_lock_id=revision.materialized_runtime_lock_id,
        materialized_runtime_lock_digest=revision.materialized_runtime_lock_digest,
        runtime_graph_digest=revision.runtime_graph_digest,
        materialization_artifact_digest=bundle.materialization_artifact_digest,
        registry_factory_wiring_digest=wiring_digest,
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
        self._prepare_lock = threading.Lock()

    def prepare_projection(self, runtime_revision_id: str) -> str:
        existing = self._projection_store.get(runtime_revision_id)
        if existing is not None:
            return existing.evidence.readiness_token

        with self._prepare_lock:
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
