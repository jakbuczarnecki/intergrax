# © Artur Czarnecki. All rights reserved.

"""Caller-held claim authority snapshot for suspended work re-entry (UCA-6C-R6)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)


class SuspendedOperationClaimAuthority(BaseModel):
    """Immutable proof of claim authority held by the re-entry caller (not store SoT)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str
    fence: int = Field(ge=0)
    materialization_revision: int = Field(ge=0)
    pause_generation: int = Field(ge=1)

    @field_validator("owner_id")
    @classmethod
    def _strip_owner_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("owner_id must be non-empty")
        return normalized

    @classmethod
    def from_claimed_descriptor(
        cls,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationClaimAuthority:
        """Build authority from a descriptor after claim or reclaim (store snapshot)."""
        ownership = descriptor.claim_ownership
        if ownership is None:
            raise ValueError("descriptor has no claim ownership")
        return cls(
            owner_id=ownership.owner_id,
            fence=ownership.fence,
            materialization_revision=descriptor.materialization_revision,
            pause_generation=descriptor.pause_generation,
        )

    @classmethod
    def for_host_pending_claim(
        cls,
        *,
        host_owner_id: str,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationClaimAuthority:
        """Authority before first claim when materialization is not yet CLAIMED."""
        if (
            descriptor.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
        ):
            raise ValueError("descriptor already claimed; use from_claimed_descriptor")
        if descriptor.claim_ownership is not None:
            raise ValueError("descriptor claim ownership must be absent before claim")
        return cls(
            owner_id=host_owner_id,
            fence=0,
            materialization_revision=descriptor.materialization_revision,
            pause_generation=descriptor.pause_generation,
        )


__all__ = ["SuspendedOperationClaimAuthority"]
