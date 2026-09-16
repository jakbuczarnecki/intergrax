# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dynamic tool acquisition — exact release resolution and host activation."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.tools.catalog import (
    ToolCatalogEntry,
    ToolCatalogProviderRegistry,
    ToolPackageResolution,
)
from intergrax.tools.errors import (
    DynamicToolAcquisitionActivationError,
    DynamicToolAcquisitionConflictError,
    DynamicToolAcquisitionResolutionError,
)
from intergrax.tools.identity import (
    ToolDiscoveryCandidateIdentity,
    ToolPackageIdentity,
)
from intergrax.tools.registry.provenance import ToolRuntimeActivationMetadata
from intergrax.tools.registry.read import ToolRegistryRead

_NON_EMPTY = Field(min_length=1)

SCHEMA_DYNAMIC_TOOL_ACQUISITION_REQUEST_V1: Final = (
    "dynamic_tool_acquisition_request.v1"
)
SCHEMA_DYNAMIC_TOOL_ACQUISITION_RESULT_V1: Final = (
    "dynamic_tool_acquisition_result.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class DynamicToolAcquisitionOutcome(StrEnum):
    ACTIVATED = "activated"
    ALREADY_ACTIVE = "already_active"


class DynamicToolAcquisitionRequest(BaseModel):
    """Typed immutable acquisition request for one selected tool release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_TOOL_ACQUISITION_REQUEST_V1
    operation_id: str = _NON_EMPTY
    host_profile_id: str = _NON_EMPTY
    capability_identity_key: CapabilityIdentityKey
    selected_identity: ToolDiscoveryCandidateIdentity
    catalog_entry_id: str | None = None

    @field_validator("operation_id", "host_profile_id", "catalog_entry_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class DynamicToolAcquisitionResult(BaseModel):
    """Acquisition outcome — lifecycle semantics, not tool execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_TOOL_ACQUISITION_RESULT_V1
    outcome: DynamicToolAcquisitionOutcome
    operation_id: str = _NON_EMPTY
    host_profile_id: str = _NON_EMPTY
    selected_identity: ToolDiscoveryCandidateIdentity
    resolved_package_identity: ToolPackageIdentity
    registry_tool_id: str = _NON_EMPTY
    activation: ToolRuntimeActivationMetadata
    domain_reference: str = _NON_EMPTY

    @field_validator("operation_id", "host_profile_id", "registry_tool_id", "domain_reference")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class DynamicToolAcquisitionPort(Protocol):
    """Public Tool domain acquisition boundary."""

    def acquire(
        self,
        request: DynamicToolAcquisitionRequest,
    ) -> DynamicToolAcquisitionResult: ...


class ToolHostActivationMaterializer(Protocol):
    """Materialize resolved package into registry bindings."""

    def materialize(
        self,
        resolution: ToolPackageResolution,
    ) -> tuple[str, ToolRuntimeActivationMetadata]:
        """Return ``(registry_tool_id, activation_metadata)``."""


class ToolHostActivationPort(Protocol):
    """Host-profile scoped activation authority over ``ToolRegistryRead``."""

    host_profile_id: str

    def registry_read(self) -> ToolRegistryRead: ...

    def is_active(self, logical_tool_id: str) -> bool: ...

    def activation_metadata(self, logical_tool_id: str) -> ToolRuntimeActivationMetadata | None: ...

    def activate(
        self,
        *,
        operation_id: str,
        host_profile_id: str,
        resolved: ToolPackageResolution,
        materializer: ToolHostActivationMaterializer,
    ) -> DomainLifecycleHandoffAck: ...


def assert_exact_discovery_candidate_match(
    *,
    identity: ToolDiscoveryCandidateIdentity,
    resolution: ToolPackageResolution,
) -> None:
    entry = resolution.entry
    if entry.catalog_source_id != identity.catalog_source_id:
        raise DynamicToolAcquisitionResolutionError(
            "resolved catalog source id does not match selected identity",
        )
    resolved = resolution.package_candidate
    expected = identity.package
    if resolved.logical_tool_id != expected.logical_tool_id:
        raise DynamicToolAcquisitionResolutionError(
            "resolved logical tool id does not match selected candidate",
        )
    if resolved.package_reference != expected.package_reference:
        raise DynamicToolAcquisitionResolutionError(
            "resolved package reference does not match selected candidate",
        )
    if resolved.package_version != expected.package_version:
        raise DynamicToolAcquisitionResolutionError(
            "resolved package version does not match selected candidate",
        )
    if (
        expected.package_digest is not None
        and resolved.package_digest != expected.package_digest
    ):
        raise DynamicToolAcquisitionResolutionError(
            "resolved package digest does not match selected candidate",
        )


def resolve_discovery_candidate_exact(
    *,
    identity: ToolDiscoveryCandidateIdentity,
    catalog_entry_id: str | None,
    registry: ToolCatalogProviderRegistry,
) -> ToolPackageResolution:
    provider = registry.require(identity.catalog_source_id)
    matched_entry: ToolCatalogEntry | None = None
    for entry in provider.list_entries():
        if catalog_entry_id is not None and entry.catalog_entry_id != catalog_entry_id:
            continue
        if entry.logical_tool_id != identity.package.logical_tool_id:
            continue
        if entry.package_reference != identity.package.package_reference:
            continue
        if entry.catalog_source_id != identity.catalog_source_id:
            continue
        matched_entry = entry
        break
    if matched_entry is None:
        raise DynamicToolAcquisitionResolutionError(
            "no catalog entry matches selected tool discovery candidate",
        )
    resolution = provider.resolve_package(
        matched_entry,
        version_selector=identity.package.package_version,
    )
    assert_exact_discovery_candidate_match(identity=identity, resolution=resolution)
    return resolution


def _domain_reference(identity: ToolPackageIdentity) -> str:
    return (
        f"tool:{identity.logical_tool_id}@"
        f"{identity.package_version}:{identity.package_digest}"
    )


class DynamicToolAcquisitionService:
    """Resolve exact tool release and activate on host profile registry."""

    def __init__(
        self,
        *,
        catalog_registry: ToolCatalogProviderRegistry,
        activation: ToolHostActivationPort,
        materializer: ToolHostActivationMaterializer,
    ) -> None:
        self._catalog_registry = catalog_registry
        self._activation = activation
        self._materializer = materializer
        self._completed: dict[str, DynamicToolAcquisitionResult] = {}

    def acquire(
        self,
        request: DynamicToolAcquisitionRequest,
    ) -> DynamicToolAcquisitionResult:
        if request.host_profile_id != self._activation.host_profile_id:
            raise DynamicToolAcquisitionResolutionError("host_profile_id mismatch")

        prior = self._completed.get(request.operation_id)
        if prior is not None:
            if prior.selected_identity != request.selected_identity:
                raise DynamicToolAcquisitionConflictError(
                    "operation_id replay with different selected identity",
                )
            return prior

        resolution = resolve_discovery_candidate_exact(
            identity=request.selected_identity,
            catalog_entry_id=request.catalog_entry_id,
            registry=self._catalog_registry,
        )
        if resolution.package_candidate.package_digest is None:
            raise DynamicToolAcquisitionResolutionError(
                "resolved package candidate lacks digest required for activation",
            )
        package_identity = ToolPackageIdentity.from_candidate(
            resolution.package_candidate,
        )

        ack = self._activation.activate(
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            resolved=resolution,
            materializer=self._materializer,
        )
        if ack.disposition is not DomainLifecycleHandoffDisposition.ACCEPTED:
            raise DynamicToolAcquisitionActivationError(
                ack.reason_detail or "tool host activation rejected acquisition",
            )

        activation_meta = self._activation.activation_metadata(
            package_identity.logical_tool_id,
        )
        if activation_meta is None:
            raise DynamicToolAcquisitionActivationError(
                "activated tool missing registry activation metadata",
            )

        outcome = (
            DynamicToolAcquisitionOutcome.ALREADY_ACTIVE
            if "idempotent" in (ack.reason_detail or "")
            else DynamicToolAcquisitionOutcome.ACTIVATED
        )
        result = DynamicToolAcquisitionResult(
            outcome=outcome,
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            selected_identity=request.selected_identity,
            resolved_package_identity=package_identity,
            registry_tool_id=package_identity.logical_tool_id,
            activation=activation_meta,
            domain_reference=ack.domain_reference or _domain_reference(package_identity),
        )
        self._completed[request.operation_id] = result
        return result


__all__ = [
    "DynamicToolAcquisitionOutcome",
    "DynamicToolAcquisitionPort",
    "DynamicToolAcquisitionRequest",
    "DynamicToolAcquisitionResult",
    "DynamicToolAcquisitionService",
    "ToolHostActivationMaterializer",
    "ToolHostActivationPort",
    "assert_exact_discovery_candidate_match",
    "resolve_discovery_candidate_exact",
]
