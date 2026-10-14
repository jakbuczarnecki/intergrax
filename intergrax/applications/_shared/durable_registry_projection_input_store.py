# © Artur Czarnecki. All rights reserved.

"""Durable ``RegistryProjectionInputStore`` backed by projection descriptors (PROD-Q-R1-R7-D3)."""

from __future__ import annotations

import threading

from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.stores import RuntimeRevisionStore
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    assemble_production_registry_projection_input_bundle,
)
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionError,
    RegistryProjectionInputBundle,
    registry_projection_input_semantic_fingerprint,
)
from intergrax.applications._shared.registry_projection_authority_resolver import (
    RegistryProjectionAuthorityError,
    RegistryProjectionAuthorityResolver,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    RegistryProjectionDescriptorError,
    RuntimeRegistryProjectionDescriptorStore,
)
from intergrax.applications.contracts.manifest import ApplicationManifest


class DescriptorBackedRegistryProjectionInputStore:
    """Reconstruct projection inputs from durable descriptors and AD authority."""

    def __init__(
        self,
        *,
        revision_store: RuntimeRevisionStore,
        descriptor_store: RuntimeRegistryProjectionDescriptorStore,
        authority: RegistryProjectionAuthorityResolver,
    ) -> None:
        self._revision_store = revision_store
        self._descriptor_store = descriptor_store
        self._authority = authority
        self._session_bundles: dict[str, RegistryProjectionInputBundle] = {}
        self._fingerprints: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, runtime_revision_id: str) -> RegistryProjectionInputBundle | None:
        revision = self._revision_store.get_revision(runtime_revision_id)
        if revision is None:
            return None
        application_id = revision.application_id
        application_environment_id = revision.application_environment_id
        try:
            descriptor = self._descriptor_store.get_for_revision(
                application_id,
                application_environment_id,
                runtime_revision_id,
            )
        except RegistryProjectionDescriptorError as exc:
            raise RegistryProjectionError(str(exc)) from exc
        if descriptor is not None:
            return self._reconstruct_from_durable_authority(revision)
        with self._lock:
            return self._session_bundles.get(runtime_revision_id)

    def register(self, bundle: RegistryProjectionInputBundle) -> None:
        revision_id = bundle.runtime_revision.runtime_revision_id
        fingerprint = registry_projection_input_semantic_fingerprint(bundle)
        revision = bundle.runtime_revision
        with self._lock:
            existing = self._fingerprints.get(revision_id)
            if existing is not None:
                if existing != fingerprint:
                    raise RegistryProjectionError(
                        f"conflicting frozen projection inputs for {revision_id!r}"
                    )
                return
        try:
            durable = self._descriptor_store.get_for_revision(
                revision.application_id,
                revision.application_environment_id,
                revision_id,
            )
        except RegistryProjectionDescriptorError as exc:
            raise RegistryProjectionError(str(exc)) from exc
        if durable is not None:
            reconstructed = self._reconstruct_from_durable_authority(revision)
            if reconstructed is None:
                raise RegistryProjectionError(
                    f"durable descriptor present but authority incomplete for "
                    f"{revision_id!r}",
                )
            durable_fingerprint = registry_projection_input_semantic_fingerprint(
                reconstructed,
            )
            if durable_fingerprint != fingerprint:
                raise RegistryProjectionError(
                    f"durable descriptor authority mismatch for {revision_id!r}"
                )
        with self._lock:
            existing = self._fingerprints.get(revision_id)
            if existing is not None:
                if existing != fingerprint:
                    raise RegistryProjectionError(
                        f"conflicting frozen projection inputs for {revision_id!r}"
                    )
                return
            self._session_bundles[revision_id] = bundle
            self._fingerprints[revision_id] = fingerprint

    def _reconstruct_from_durable_authority(
        self,
        revision: RuntimeRevision,
    ) -> RegistryProjectionInputBundle | None:
        application_id = revision.application_id
        application_environment_id = revision.application_environment_id
        runtime_revision_id = revision.runtime_revision_id
        try:
            descriptor = self._descriptor_store.get_for_revision(
                application_id,
                application_environment_id,
                runtime_revision_id,
            )
        except RegistryProjectionDescriptorError as exc:
            raise RegistryProjectionError(str(exc)) from exc
        if descriptor is None:
            return None
        try:
            resolved = self._authority.require_for_revision(
                application_id=application_id,
                application_environment_id=application_environment_id,
                runtime_revision_id=runtime_revision_id,
            )
        except RegistryProjectionAuthorityError as exc:
            raise RegistryProjectionError(str(exc)) from exc
        manifest = descriptor.manifest
        if not isinstance(manifest, ApplicationManifest):
            raise RegistryProjectionError(
                "projection descriptor requires ApplicationManifest authority"
            )
        roster_contract_authority = self._authority.roster_contract_authority_for(
            resolved=resolved,
            manifest=manifest,
        )
        build_context = descriptor.build_context_snapshot.to_build_context(manifest)
        try:
            return assemble_production_registry_projection_input_bundle(
                runtime_revision=resolved.runtime_revision,
                effective_roster=resolved.effective_roster,
                materialized_runtime_lock=resolved.materialized_runtime_lock,
                manifest=manifest,
                build_context=build_context,
                artifact_locator=descriptor.materialization_artifact_locator,
                materialization_artifact_digest=descriptor.materialization_artifact_digest,
                roster_contract_authority=roster_contract_authority,
            )
        except ProductionRegistryProjectionInputError as exc:
            raise RegistryProjectionError(str(exc)) from exc


__all__ = ["DescriptorBackedRegistryProjectionInputStore"]
