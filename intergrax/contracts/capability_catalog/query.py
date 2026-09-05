# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed capability discovery query contract (CAPABILITY-CATALOG-1 Stage 3)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.kind import CapabilityKind, V1_CAPABILITY_KINDS
from intergrax.contracts.capability_catalog.scope import CapabilityDiscoveryScope

SCHEMA_CAPABILITY_DISCOVERY_QUERY_V1: Final = "capability_discovery_query.v1"
_NON_EMPTY = Field(min_length=1)


class LogicalIdentityFilter(BaseModel):
    """Exact or prefix logical identity constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exact_logical_ids: tuple[str, ...] = ()
    logical_id_prefixes: tuple[str, ...] = ()

    @field_validator("exact_logical_ids", "logical_id_prefixes")
    @classmethod
    def _validate_non_empty_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("logical identity filter values must be non-empty")
            if stripped != item:
                raise ValueError(
                    "logical identity filter values must not have surrounding whitespace",
                )
            normalized.append(stripped)
        return tuple(normalized)

    @model_validator(mode="after")
    def _validate_has_constraint(self) -> LogicalIdentityFilter:
        if not self.exact_logical_ids and not self.logical_id_prefixes:
            raise ValueError(
                "logical identity filter requires exact_logical_ids or logical_id_prefixes",
            )
        return self


class SourceFilter(BaseModel):
    """Typed source identity constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ids: tuple[str, ...] = ()
    source_kinds: tuple[CapabilitySourceKind, ...] = ()

    @field_validator("source_ids")
    @classmethod
    def _validate_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("source_ids values must be non-empty")
            if stripped != item:
                raise ValueError("source_ids values must not have surrounding whitespace")
            normalized.append(stripped)
        return tuple(normalized)

    @model_validator(mode="after")
    def _validate_has_constraint(self) -> SourceFilter:
        if not self.source_ids and not self.source_kinds:
            raise ValueError("source filter requires source_ids or source_kinds")
        return self


class CapabilityDiscoveryQuery(BaseModel):
    """Typed, deterministic discovery query over a federated catalog snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_discovery_query.v1"] = (
        SCHEMA_CAPABILITY_DISCOVERY_QUERY_V1
    )
    scope: CapabilityDiscoveryScope
    kinds: tuple[CapabilityKind, ...] = ()
    logical_identity: LogicalIdentityFilter | None = None
    source: SourceFilter | None = None
    availability_constraints: tuple[AvailabilityDisposition, ...] = ()

    @field_validator("kinds")
    @classmethod
    def _validate_kinds(cls, value: tuple[CapabilityKind, ...]) -> tuple[CapabilityKind, ...]:
        if not value:
            return value
        unknown = frozenset(value) - V1_CAPABILITY_KINDS
        if unknown:
            raise ValueError(f"unsupported capability kinds: {sorted(item.value for item in unknown)}")
        return tuple(dict.fromkeys(value))

    @field_validator("availability_constraints")
    @classmethod
    def _validate_availability_constraints(
        cls,
        value: tuple[AvailabilityDisposition, ...],
    ) -> tuple[AvailabilityDisposition, ...]:
        return tuple(dict.fromkeys(value))
