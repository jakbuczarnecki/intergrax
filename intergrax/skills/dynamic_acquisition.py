# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dynamic skill acquisition — exact release resolution and host profile binding."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.skills.catalog import (
    SkillCatalogEntry,
    SkillCatalogProviderRegistry,
    SkillPackageResolution,
)
from intergrax.skills.errors import (
    DynamicSkillAcquisitionBindingError,
    DynamicSkillAcquisitionConflictError,
    DynamicSkillAcquisitionResolutionError,
)
from intergrax.skills.identity import (
    SkillDiscoveryCandidateIdentity,
    SkillPackageIdentity,
)
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.read import SkillRegistryRead

_NON_EMPTY = Field(min_length=1)

SCHEMA_DYNAMIC_SKILL_ACQUISITION_REQUEST_V1: Final = (
    "dynamic_skill_acquisition_request.v1"
)
SCHEMA_DYNAMIC_SKILL_ACQUISITION_RESULT_V1: Final = (
    "dynamic_skill_acquisition_result.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class DynamicSkillAcquisitionOutcome(StrEnum):
    BOUND = "bound"
    ALREADY_BOUND = "already_bound"


class DynamicSkillAcquisitionRequest(BaseModel):
    """Typed immutable acquisition request for one selected skill release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_SKILL_ACQUISITION_REQUEST_V1
    operation_id: str = _NON_EMPTY
    host_profile_id: str = _NON_EMPTY
    capability_identity_key: CapabilityIdentityKey
    selected_identity: SkillDiscoveryCandidateIdentity
    catalog_entry_id: str | None = None

    @field_validator("operation_id", "host_profile_id", "catalog_entry_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class DynamicSkillAcquisitionResult(BaseModel):
    """Acquisition outcome — composition/binding semantics, not skill execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_SKILL_ACQUISITION_RESULT_V1
    outcome: DynamicSkillAcquisitionOutcome
    operation_id: str = _NON_EMPTY
    host_profile_id: str = _NON_EMPTY
    selected_identity: SkillDiscoveryCandidateIdentity
    resolved_package_identity: SkillPackageIdentity
    registry_skill_id: str = _NON_EMPTY
    binding: SkillRuntimeBindingMetadata
    domain_reference: str = _NON_EMPTY

    @field_validator("operation_id", "host_profile_id", "registry_skill_id", "domain_reference")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class DynamicSkillAcquisitionPort(Protocol):
    """Public Skill domain acquisition boundary."""

    def acquire(
        self,
        request: DynamicSkillAcquisitionRequest,
    ) -> DynamicSkillAcquisitionResult: ...


class SkillHostBindingMaterializer(Protocol):
    """Materialize resolved package into registry bindings and profile enablement."""

    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        """Return ``(registry_skill_id, binding_metadata)``."""


class SkillHostBindingPort(Protocol):
    """Host-profile scoped binding authority over ``SkillRegistryRead``."""

    host_profile_id: str

    def registry_read(self) -> SkillRegistryRead: ...

    def is_bound(self, logical_skill_id: str) -> bool: ...

    def binding_metadata(self, logical_skill_id: str) -> SkillRuntimeBindingMetadata | None: ...

    def bind(
        self,
        *,
        operation_id: str,
        host_profile_id: str,
        resolved: SkillPackageResolution,
        materializer: SkillHostBindingMaterializer,
    ) -> DomainLifecycleHandoffAck: ...


def assert_exact_discovery_candidate_match(
    *,
    identity: SkillDiscoveryCandidateIdentity,
    resolution: SkillPackageResolution,
) -> None:
    entry = resolution.entry
    if entry.catalog_source_id != identity.catalog_source_id:
        raise DynamicSkillAcquisitionResolutionError(
            "resolved catalog source id does not match selected identity",
        )
    resolved = resolution.package_candidate
    expected = identity.package
    if resolved.logical_skill_id != expected.logical_skill_id:
        raise DynamicSkillAcquisitionResolutionError(
            "resolved logical skill id does not match selected candidate",
        )
    if resolved.package_reference != expected.package_reference:
        raise DynamicSkillAcquisitionResolutionError(
            "resolved package reference does not match selected candidate",
        )
    if resolved.package_version != expected.package_version:
        raise DynamicSkillAcquisitionResolutionError(
            "resolved package version does not match selected candidate",
        )
    if (
        expected.package_digest is not None
        and resolved.package_digest != expected.package_digest
    ):
        raise DynamicSkillAcquisitionResolutionError(
            "resolved package digest does not match selected candidate",
        )


def resolve_discovery_candidate_exact(
    *,
    identity: SkillDiscoveryCandidateIdentity,
    catalog_entry_id: str | None,
    registry: SkillCatalogProviderRegistry,
) -> SkillPackageResolution:
    provider = registry.require(identity.catalog_source_id)
    matched_entry: SkillCatalogEntry | None = None
    for entry in provider.list_entries():
        if catalog_entry_id is not None and entry.catalog_entry_id != catalog_entry_id:
            continue
        if entry.logical_skill_id != identity.package.logical_skill_id:
            continue
        if entry.package_reference != identity.package.package_reference:
            continue
        if entry.catalog_source_id != identity.catalog_source_id:
            continue
        matched_entry = entry
        break
    if matched_entry is None:
        raise DynamicSkillAcquisitionResolutionError(
            "no catalog entry matches selected skill discovery candidate",
        )
    resolution = provider.resolve_package(
        matched_entry,
        version_selector=identity.package.package_version,
    )
    assert_exact_discovery_candidate_match(identity=identity, resolution=resolution)
    return resolution


def _domain_reference(identity: SkillPackageIdentity) -> str:
    return (
        f"skill:{identity.logical_skill_id}@"
        f"{identity.package_version}:{identity.package_digest}"
    )


class DynamicSkillAcquisitionService:
    """Resolve exact skill release and bind on host profile registry/profile."""

    def __init__(
        self,
        *,
        catalog_registry: SkillCatalogProviderRegistry,
        binding: SkillHostBindingPort,
        materializer: SkillHostBindingMaterializer,
    ) -> None:
        self._catalog_registry = catalog_registry
        self._binding = binding
        self._materializer = materializer
        self._completed: dict[str, DynamicSkillAcquisitionResult] = {}

    def acquire(
        self,
        request: DynamicSkillAcquisitionRequest,
    ) -> DynamicSkillAcquisitionResult:
        if request.host_profile_id != self._binding.host_profile_id:
            raise DynamicSkillAcquisitionResolutionError("host_profile_id mismatch")

        prior = self._completed.get(request.operation_id)
        if prior is not None:
            if prior.selected_identity != request.selected_identity:
                raise DynamicSkillAcquisitionConflictError(
                    "operation_id replay with different selected identity",
                )
            return prior

        resolution = resolve_discovery_candidate_exact(
            identity=request.selected_identity,
            catalog_entry_id=request.catalog_entry_id,
            registry=self._catalog_registry,
        )
        if resolution.package_candidate.package_digest is None:
            raise DynamicSkillAcquisitionResolutionError(
                "resolved package candidate lacks digest required for binding",
            )
        package_identity = SkillPackageIdentity.from_candidate(
            resolution.package_candidate,
        )

        ack = self._binding.bind(
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            resolved=resolution,
            materializer=self._materializer,
        )
        if ack.disposition is not DomainLifecycleHandoffDisposition.ACCEPTED:
            raise DynamicSkillAcquisitionBindingError(
                ack.reason_detail or "skill host binding rejected acquisition",
            )

        binding_meta = self._binding.binding_metadata(package_identity.logical_skill_id)
        if binding_meta is None:
            raise DynamicSkillAcquisitionBindingError(
                "bound skill missing registry binding metadata",
            )

        outcome = (
            DynamicSkillAcquisitionOutcome.ALREADY_BOUND
            if "idempotent" in (ack.reason_detail or "")
            else DynamicSkillAcquisitionOutcome.BOUND
        )
        result = DynamicSkillAcquisitionResult(
            outcome=outcome,
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            selected_identity=request.selected_identity,
            resolved_package_identity=package_identity,
            registry_skill_id=package_identity.logical_skill_id,
            binding=binding_meta,
            domain_reference=ack.domain_reference or _domain_reference(package_identity),
        )
        self._completed[request.operation_id] = result
        return result


__all__ = [
    "DynamicSkillAcquisitionOutcome",
    "DynamicSkillAcquisitionPort",
    "DynamicSkillAcquisitionRequest",
    "DynamicSkillAcquisitionResult",
    "DynamicSkillAcquisitionService",
    "SkillHostBindingMaterializer",
    "SkillHostBindingPort",
    "assert_exact_discovery_candidate_match",
    "resolve_discovery_candidate_exact",
]
