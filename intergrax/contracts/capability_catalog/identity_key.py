# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Source-qualified identity keys for discovery evidence (Stage 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog.identity import (
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind

if TYPE_CHECKING:
    from intergrax.contracts.capability_catalog.identity import (
        CapabilityDiscoveryIdentity,
    )

SCHEMA_CAPABILITY_IDENTITY_KEY_V1: Final = "capability_identity_key.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityIdentityKey(BaseModel):
    """Stable identity key aligned with Stage-1 ``sort_key`` semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_identity_key.v1"] = SCHEMA_CAPABILITY_IDENTITY_KEY_V1
    kind: CapabilityKind
    source_id: str = _NON_EMPTY
    source_kind: CapabilitySourceKind = CapabilitySourceKind.UNKNOWN
    logical_id: str = _NON_EMPTY

    @classmethod
    def from_discovery_identity(
        cls,
        identity: CapabilityDiscoveryIdentity,
    ) -> CapabilityIdentityKey:
        return cls(
            kind=identity.kind,
            source_id=identity.source.source_id,
            source_kind=identity.source.source_kind,
            logical_id=identity.logical.logical_id,
        )

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.kind.value,
            self.source_id,
            self.source_kind.value,
            self.logical_id,
        )
