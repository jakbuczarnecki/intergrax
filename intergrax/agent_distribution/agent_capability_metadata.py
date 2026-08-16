# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Non-executable agent capability metadata projection (AGENT-CONSOLIDATION-2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)

SCHEMA_AGENT_CAPABILITY_DESCRIPTOR_V1: Final = "agent_capability_descriptor.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentCapabilityDescriptor(BaseModel):
    """Declared agent contract/capability metadata for architecture discovery only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_CAPABILITY_DESCRIPTOR_V1
    contract_id: str = _NON_EMPTY
    agent_version: str = _NON_EMPTY
    capabilities: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()

    @field_validator("contract_id", "agent_version")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("capabilities", "skill_ids", "tool_ids")
    @classmethod
    def _strip_tuple_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)


class AgentCapabilityDescriptorConflictError(ValueError):
    """Raised when two descriptors share contract_id but disagree on metadata."""


class AgentCapabilityMetadataProvider(Protocol):
    """Read-only port for non-executable agent contract/capability metadata."""

    def list_agent_capability_descriptors(self) -> Sequence[AgentCapabilityDescriptor]:
        """List declared agent metadata for architecture/discovery surfaces."""


def merge_agent_capability_descriptors(
    descriptors: Sequence[AgentCapabilityDescriptor],
) -> tuple[AgentCapabilityDescriptor, ...]:
    """Merge descriptors with deterministic conflict semantics (identical rows dedupe)."""
    merged: dict[str, AgentCapabilityDescriptor] = {}
    for descriptor in descriptors:
        existing = merged.get(descriptor.contract_id)
        if existing is None:
            merged[descriptor.contract_id] = descriptor
            continue
        if existing == descriptor:
            continue
        raise AgentCapabilityDescriptorConflictError(
            f"conflicting agent capability metadata for contract_id={descriptor.contract_id!r}: "
            f"existing={existing!r} incoming={descriptor!r}",
        )
    return tuple(sorted(merged.values(), key=lambda item: item.contract_id))
