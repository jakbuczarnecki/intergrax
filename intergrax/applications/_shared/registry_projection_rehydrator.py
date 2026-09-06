# © Artur Czarnecki. All rights reserved.

"""Deterministic process-local registry projection rehydration from durable authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from intergrax.agent_distribution.stores import ApplicationEnvironmentServingStore
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    assemble_production_registry_projection_input_bundle,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionError,
    RuntimeRegistryProjectionStore,
    build_registry_projection,
)
from intergrax.applications._shared.registry_projection_authority_resolver import (
    RegistryProjectionAuthorityError,
    RegistryProjectionAuthorityResolver,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    PROJECTION_DESCRIPTOR_CONTRACT_VERSION,
    RegistryProjectionDescriptorError,
    RuntimeRegistryProjectionDescriptor,
    RuntimeRegistryProjectionDescriptorStore,
)

REHYDRATION_OUTCOME_READY: Final = "rehydration_ready"


class RegistryProjectionRehydrationError(Exception):
    """Serving revision projection rehydration failed."""


@dataclass(frozen=True, slots=True)
class RegistryProjectionRehydrationEvidence:
    """Inspectable evidence for one successful rehydration."""

    application_id: str
    application_environment_id: str
    runtime_revision_id: str
    descriptor_version: str
    materialization_artifact_digest: str
    effective_roster_revision_id: str
    outcome: str = REHYDRATION_OUTCOME_READY


@dataclass(frozen=True, slots=True)
class RegistryProjectionRehydrationResult:
    """Outcome of one serving projection rehydration."""

    projection: MaterializedRegistryProjection
    evidence: RegistryProjectionRehydrationEvidence


def _validate_descriptor_against_authority(
    descriptor: RuntimeRegistryProjectionDescriptor,
    *,
    serving_revision_id: str,
    resolved_revision_id: str,
    resolved_roster_revision_id: str,
    resolved_lock_id: str,
    resolved_lock_digest: str,
    resolved_artifact_digest: str,
    resolved_artifact_locator: str,
) -> None:
    if descriptor.runtime_revision_id != serving_revision_id:
        raise RegistryProjectionRehydrationError(
            "descriptor runtime_revision_id mismatch with serving pointer"
        )
    if descriptor.runtime_revision_id != resolved_revision_id:
        raise RegistryProjectionRehydrationError(
            "descriptor runtime_revision_id mismatch with canonical revision authority"
        )
    if descriptor.effective_roster_revision_id != resolved_roster_revision_id:
        raise RegistryProjectionRehydrationError(
            "descriptor effective_roster_revision_id mismatch with canonical roster authority"
        )
    if descriptor.materialized_runtime_lock_id != resolved_lock_id:
        raise RegistryProjectionRehydrationError(
            "descriptor materialized_runtime_lock_id mismatch with canonical lock authority"
        )
    if descriptor.materialized_runtime_lock_digest != resolved_lock_digest:
        raise RegistryProjectionRehydrationError(
            "descriptor materialized_runtime_lock_digest mismatch with canonical lock authority"
        )
    if descriptor.materialization_artifact_digest != resolved_artifact_digest:
        raise RegistryProjectionRehydrationError(
            "descriptor materialization_artifact_digest mismatch with canonical materialization"
        )
    if descriptor.materialization_artifact_locator != resolved_artifact_locator:
        raise RegistryProjectionRehydrationError(
            "descriptor materialization_artifact_locator mismatch with canonical materialization"
        )


class RuntimeRegistryProjectionRehydrator:
    """Reconstruct process-local projection from durable revision-bound authority."""

    def __init__(
        self,
        *,
        serving_store: ApplicationEnvironmentServingStore,
        descriptor_store: RuntimeRegistryProjectionDescriptorStore,
        authority: RegistryProjectionAuthorityResolver,
        projection_store: RuntimeRegistryProjectionStore,
    ) -> None:
        self._serving_store = serving_store
        self._descriptor_store = descriptor_store
        self._authority = authority
        self._projection_store = projection_store

    def rehydrate_serving_registry_projection(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> RegistryProjectionRehydrationResult:
        """Load serving revision descriptor, validate authority, and materialize projection."""
        serving = self._serving_store.get_serving_record(
            application_id,
            application_environment_id,
        )
        if serving is None or serving.traffic_serving_revision_id is None:
            raise RegistryProjectionRehydrationError(
                "no traffic-serving runtime revision for rehydration"
            )

        serving_revision_id = serving.traffic_serving_revision_id
        existing = self._projection_store.get(serving_revision_id)
        if existing is not None:
            return RegistryProjectionRehydrationResult(
                projection=existing,
                evidence=RegistryProjectionRehydrationEvidence(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                    runtime_revision_id=serving_revision_id,
                    descriptor_version=PROJECTION_DESCRIPTOR_CONTRACT_VERSION,
                    materialization_artifact_digest=(
                        existing.evidence.materialization_artifact_digest or ""
                    ),
                    effective_roster_revision_id=(
                        existing.evidence.effective_roster_revision_id
                    ),
                ),
            )

        try:
            descriptor = self._descriptor_store.get_for_revision(
                application_id,
                application_environment_id,
                serving_revision_id,
            )
        except RegistryProjectionDescriptorError as exc:
            raise RegistryProjectionRehydrationError(str(exc)) from exc
        if descriptor is None:
            raise RegistryProjectionRehydrationError(
                f"missing durable projection descriptor for serving revision "
                f"{serving_revision_id!r}"
            )

        try:
            resolved = self._authority.require_for_revision(
                application_id=application_id,
                application_environment_id=application_environment_id,
                runtime_revision_id=serving_revision_id,
            )
        except RegistryProjectionAuthorityError as exc:
            raise RegistryProjectionRehydrationError(str(exc)) from exc

        revision = resolved.runtime_revision
        materialization = resolved.runtime_materialization
        lock = resolved.materialized_runtime_lock
        roster_revision_id = revision.effective_roster_revision_id or ""
        _validate_descriptor_against_authority(
            descriptor,
            serving_revision_id=serving_revision_id,
            resolved_revision_id=revision.runtime_revision_id,
            resolved_roster_revision_id=roster_revision_id,
            resolved_lock_id=lock.lock_id or "",
            resolved_lock_digest=lock.lock_digest or "",
            resolved_artifact_digest=materialization.materialization_artifact_digest,
            resolved_artifact_locator=materialization.artifact_locator,
        )
        if descriptor.application_release_id != revision.application_release_id:
            raise RegistryProjectionRehydrationError(
                "descriptor application_release_id mismatch with runtime revision"
            )
        if descriptor.application_id != revision.application_id:
            raise RegistryProjectionRehydrationError(
                "descriptor application_id mismatch with runtime revision"
            )
        if descriptor.application_environment_id != revision.application_environment_id:
            raise RegistryProjectionRehydrationError(
                "descriptor application_environment_id mismatch with runtime revision"
            )

        manifest = descriptor.manifest
        if manifest.app_id != revision.application_id:
            raise RegistryProjectionRehydrationError(
                "descriptor manifest application_id mismatch with runtime revision"
            )
        build_context = descriptor.build_context_snapshot.to_build_context(manifest)
        try:
            bundle = assemble_production_registry_projection_input_bundle(
                runtime_revision=revision,
                effective_roster=resolved.effective_roster,
                materialized_runtime_lock=lock,
                manifest=manifest,
                build_context=build_context,
                artifact_locator=descriptor.materialization_artifact_locator,
                materialization_artifact_digest=descriptor.materialization_artifact_digest,
            )
            projection = build_registry_projection(bundle)
        except (ProductionRegistryProjectionInputError, RegistryProjectionError) as exc:
            raise RegistryProjectionRehydrationError(str(exc)) from exc

        self._projection_store.put(projection)
        return RegistryProjectionRehydrationResult(
            projection=projection,
            evidence=RegistryProjectionRehydrationEvidence(
                application_id=application_id,
                application_environment_id=application_environment_id,
                runtime_revision_id=serving_revision_id,
                descriptor_version=descriptor.descriptor_version,
                materialization_artifact_digest=descriptor.materialization_artifact_digest,
                effective_roster_revision_id=descriptor.effective_roster_revision_id,
            ),
        )


def rehydrate_serving_registry_projection(
    *,
    application_id: str,
    application_environment_id: str,
    rehydrator: RuntimeRegistryProjectionRehydrator,
) -> RegistryProjectionRehydrationResult:
    """Convenience entrypoint for composition roots."""
    return rehydrator.rehydrate_serving_registry_projection(
        application_id=application_id,
        application_environment_id=application_environment_id,
    )


__all__ = [
    "REHYDRATION_OUTCOME_READY",
    "RegistryProjectionRehydrationError",
    "RegistryProjectionRehydrationEvidence",
    "RegistryProjectionRehydrationResult",
    "RuntimeRegistryProjectionRehydrator",
    "rehydrate_serving_registry_projection",
]
