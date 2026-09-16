# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Versioned release identity parallel to Stage-3 discovery keys (ME-7)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity

if TYPE_CHECKING:
    from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry

SCHEMA_CAPABILITY_RELEASE_IDENTITY_V1: Final = "capability_release_identity.v1"


class CapabilityReleaseIdentity(BaseModel):
    """Exact immutable release descriptor for audit, reference, and integrity handoff.

    This contract identifies an exact release for audit/reference. It does not
    redefine federation or Stage-3 discovery identity.

    ``CapabilityIdentityKey`` remains the source-qualified logical discovery key
    (kind + source + logical_id) and intentionally excludes publisher, version,
    digest, and package_reference. ``release_sort_key`` is deterministic ordering
    only — not a federation merge key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_release_identity.v1"] = (
        SCHEMA_CAPABILITY_RELEASE_IDENTITY_V1
    )
    discovery: CapabilityDiscoveryIdentity
    publisher: str | None = None
    version_label: str | None = None
    content_digest: str | None = None
    package_reference: str | None = None

    @classmethod
    def from_catalog_entry(cls, entry: CapabilityCatalogEntry) -> CapabilityReleaseIdentity:
        provenance = entry.provenance
        return cls(
            discovery=entry.identity,
            publisher=provenance.publisher,
            version_label=provenance.version_label,
            content_digest=provenance.content_digest,
            package_reference=provenance.package_reference,
        )

    @property
    def release_sort_key(self) -> tuple[str, ...]:
        return (
            *self.discovery.sort_key,
            self.publisher or "",
            self.version_label or "",
            self.content_digest or "",
            self.package_reference or "",
        )
