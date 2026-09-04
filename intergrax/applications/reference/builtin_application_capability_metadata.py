# © Artur Czarnecki. All rights reserved.

"""Reference application capability metadata from harness manifests — non-executable."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications.contracts.application_capability_projection import (
    application_capability_descriptor_from_manifest,
)
from intergrax.contracts.application_capability_metadata import (
    ApplicationCapabilityDescriptor,
    merge_application_capability_descriptors,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications.reference.harness_manifest_catalog import build_harness_reference_manifests


class HarnessReferenceApplicationCapabilityMetadataProvider:
    """Aggregate application descriptors from harness reference manifests."""

    def __init__(self, *, manifests: Sequence[ApplicationManifest] | None = None) -> None:
        self._manifests = (
            tuple(manifests) if manifests is not None else build_harness_reference_manifests()
        )

    def list_application_capability_descriptors(self) -> Sequence[ApplicationCapabilityDescriptor]:
        return merge_application_capability_descriptors(
            application_capability_descriptor_from_manifest(manifest)
            for manifest in self._manifests
        )
