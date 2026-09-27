# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ensure exact qualified Marketplace Tool release is active on host profile (S24-GAP-02-P3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.release_identity import CapabilityReleaseIdentity
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionPort,
    DynamicToolAcquisitionRequest,
    ToolHostActivationPort,
)
from intergrax.tools.errors import (
    DynamicToolAcquisitionActivationError,
    DynamicToolAcquisitionConflictError,
    DynamicToolAcquisitionResolutionError,
)
from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
from intergrax.tools.registry.provenance import ToolRuntimeActivationMetadata


class QualifiedMarketplaceToolActivationOutcome(StrEnum):
    ALREADY_ACTIVE_EXACT = "already_active_exact"
    ACTIVATED_EXACT = "activated_exact"
    RELEASE_CONFLICT = "release_conflict"
    RESOLUTION_FAILURE = "resolution_failure"
    ACTIVATION_FAILURE = "activation_failure"
    UNAVAILABLE = "unavailable"
    INTEGRITY_FAILURE = "integrity_failure"


@dataclass(frozen=True, slots=True)
class QualifiedMarketplaceToolActivationResult:
    outcome: QualifiedMarketplaceToolActivationOutcome
    registry_tool_id: str = ""
    reason_detail: str = ""


def _expected_activation_metadata(
    release: CapabilityReleaseIdentity,
) -> ToolRuntimeActivationMetadata | None:
    if (
        release.version_label is None
        or release.content_digest is None
        or release.package_reference is None
    ):
        return None
    return ToolRuntimeActivationMetadata(
        catalog_source_id=release.discovery.source.source_id,
        logical_tool_id=release.discovery.logical.logical_id,
        package_reference=release.package_reference,
        version_label=release.version_label,
        content_digest=release.content_digest,
    )


def _metadata_matches_release(
    active: ToolRuntimeActivationMetadata,
    expected: ToolRuntimeActivationMetadata,
) -> bool:
    return (
        active.catalog_source_id == expected.catalog_source_id
        and active.logical_tool_id == expected.logical_tool_id
        and active.package_reference == expected.package_reference
        and active.version_label == expected.version_label
        and active.content_digest == expected.content_digest
    )


def _dynamic_acquisition_request_from_stage(
    *,
    stage: MarketplaceQualifiedToolStage,
    execution_request_id: str,
    host_profile_id: str,
) -> DynamicToolAcquisitionRequest | None:
    release = stage.selected_release
    expected = _expected_activation_metadata(release)
    if expected is None:
        return None
    identity_key = CapabilityIdentityKey.from_discovery_identity(release.discovery)
    selected_identity = ToolDiscoveryCandidateIdentity(
        catalog_source_id=expected.catalog_source_id,
        package=ToolPackageCandidate(
            logical_tool_id=expected.logical_tool_id,
            package_reference=expected.package_reference,
            package_version=expected.version_label,
            package_digest=expected.content_digest,
        ),
    )
    return DynamicToolAcquisitionRequest(
        operation_id=f"marketplace-qualified-tool-activation:{execution_request_id}",
        host_profile_id=host_profile_id,
        capability_identity_key=identity_key,
        selected_identity=selected_identity,
    )


class QualifiedMarketplaceToolActivationResolver:
    """Reuse exact active release or acquire once — never silent replacement."""

    def __init__(
        self,
        *,
        activation_read: ToolHostActivationPort,
        acquisition: DynamicToolAcquisitionPort,
        host_profile_id: str,
    ) -> None:
        self._activation_read = activation_read
        self._acquisition = acquisition
        self._host_profile_id = host_profile_id

    def ensure_exact_active(
        self,
        *,
        stage: MarketplaceQualifiedToolStage,
        execution_request_id: str,
    ) -> QualifiedMarketplaceToolActivationResult:
        if self._host_profile_id != self._activation_read.host_profile_id:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE,
                reason_detail="host_profile_id mismatch",
            )
        release = stage.selected_release
        expected = _expected_activation_metadata(release)
        if expected is None:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE,
                reason_detail="staged release missing material identity",
            )
        logical_tool_id = expected.logical_tool_id
        if self._activation_read.is_active(logical_tool_id):
            active_meta = self._activation_read.activation_metadata(logical_tool_id)
            if active_meta is None:
                return QualifiedMarketplaceToolActivationResult(
                    outcome=QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE,
                    reason_detail="active tool missing activation metadata",
                )
            if not _metadata_matches_release(active_meta, expected):
                return QualifiedMarketplaceToolActivationResult(
                    outcome=QualifiedMarketplaceToolActivationOutcome.RELEASE_CONFLICT,
                    reason_detail="release_identity_conflict",
                )
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
                registry_tool_id=logical_tool_id,
            )

        request = _dynamic_acquisition_request_from_stage(
            stage=stage,
            execution_request_id=execution_request_id,
            host_profile_id=self._host_profile_id,
        )
        if request is None:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE,
                reason_detail="cannot build acquisition request",
            )
        try:
            acquisition_result = self._acquisition.acquire(request)
        except DynamicToolAcquisitionConflictError as exc:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.RELEASE_CONFLICT,
                reason_detail=str(exc),
            )
        except DynamicToolAcquisitionResolutionError as exc:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.RESOLUTION_FAILURE,
                reason_detail=str(exc),
            )
        except DynamicToolAcquisitionActivationError as exc:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATION_FAILURE,
                reason_detail=str(exc),
            )

        active_meta = acquisition_result.activation
        if not _metadata_matches_release(active_meta, expected):
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATION_FAILURE,
                reason_detail="activated release does not match staged release",
            )
        if acquisition_result.registry_tool_id != logical_tool_id:
            return QualifiedMarketplaceToolActivationResult(
                outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATION_FAILURE,
                reason_detail="registry_tool_id mismatch",
            )
        return QualifiedMarketplaceToolActivationResult(
            outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
            registry_tool_id=acquisition_result.registry_tool_id,
        )


__all__ = [
    "QualifiedMarketplaceToolActivationOutcome",
    "QualifiedMarketplaceToolActivationResolver",
    "QualifiedMarketplaceToolActivationResult",
]
