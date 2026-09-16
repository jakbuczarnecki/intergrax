# © Artur Czarnecki. All rights reserved.

"""ME-15 fixture materializer — registers resolved package into host skill registry."""

from __future__ import annotations

from intergrax.skills.catalog import SkillPackageResolution
from intergrax.skills.dynamic_acquisition import SkillHostBindingMaterializer
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.runtime import SkillRegistry
from testing_support.canonical_me15_reference_skill import me15_manifest_for_release
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)


class Me15SkillHostBindingMaterializer(SkillHostBindingMaterializer):
    def __init__(self, registry: SkillRegistry, *, catalog_source_id: str) -> None:
        self._registry = registry
        self._catalog_source_id = catalog_source_id

    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        candidate = resolution.package_candidate
        if candidate.package_digest is None:
            raise ValueError("package digest required")
        release = CapabilityReleaseIdentity(
            discovery=CapabilityDiscoveryIdentity(
                kind=CapabilityKind.SKILL,
                source=CapabilitySourceIdentity(
                    source_id=self._catalog_source_id,
                    source_kind=CapabilitySourceKind.OFFICIAL,
                ),
                logical=CapabilityLogicalIdentity(
                    kind=CapabilityKind.SKILL,
                    logical_id=candidate.logical_skill_id,
                ),
            ),
            version_label=candidate.package_version,
            content_digest=candidate.package_digest,
            package_reference=candidate.package_reference,
        )
        manifest = me15_manifest_for_release(release)
        binding = SkillRuntimeBindingMetadata(
            catalog_source_id=self._catalog_source_id,
            logical_skill_id=candidate.logical_skill_id,
            package_reference=candidate.package_reference,
            version_label=candidate.package_version,
            content_digest=candidate.package_digest,
        )
        self._registry.register(manifest, binding=binding)
        return candidate.logical_skill_id, binding


__all__ = ["Me15SkillHostBindingMaterializer"]
