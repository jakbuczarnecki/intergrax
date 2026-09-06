# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Enterprise-private Tool catalog → capability catalog entry adapter (Stage 7)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.adapters._private_validation import (
    validate_enterprise_private_source,
    validate_unique_source_logical_records,
)
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance


@dataclass(frozen=True, slots=True)
class PrivateToolCatalogRecord:
    """Read-only Tool catalog metadata supplied by an enterprise-private source."""

    logical_id: str
    version_label: str | None = None
    package_reference: str | None = None
    content_digest: str | None = None
    publisher: str | None = None
    display_label: str | None = None


def project_private_tool_record(
    source: CapabilitySourceIdentity,
    record: PrivateToolCatalogRecord,
) -> CapabilityCatalogEntry:
    """Map one private Tool catalog row to a federated catalog entry."""
    logical_id = record.logical_id.strip()
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=logical_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=record.version_label,
            package_reference=record.package_reference,
            content_digest=record.content_digest,
            publisher=record.publisher,
        ),
        display_label=record.display_label or logical_id,
    )


class PrivateToolCapabilityCatalogSource:
    """Read-only adapter over in-memory enterprise-private Tool catalog metadata."""

    def __init__(
        self,
        *,
        source: CapabilitySourceIdentity,
        records: tuple[PrivateToolCatalogRecord, ...],
    ) -> None:
        validate_enterprise_private_source(source)
        validate_unique_source_logical_records(
            records=tuple((record.logical_id, record.version_label) for record in records),
            record_label="tool",
        )
        self._source = source
        self._records = records

    @property
    def source_id(self) -> str:
        return self._source.source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        entries = [
            project_private_tool_record(self._source, record) for record in self._records
        ]
        return tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))
