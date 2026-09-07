# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic capability usage projection (CAPABILITY-CATALOG-1 Stage 13)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.capability_metering.in_memory import InMemoryCapabilityUsageConsumer
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_metering import (
    CapabilityUsageEvent,
    CapabilityUsageOutcome,
)

SCHEMA_CAPABILITY_USAGE_SUMMARY_V1: Final = "capability_usage_summary.v1"


class CapabilityUsageIdentityRollup(BaseModel):
    """Usage counts keyed by source-qualified capability identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: CapabilityIdentityKey
    total_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)


class CapabilityUsagePublisherRollup(BaseModel):
    """Publisher attribution rollup — absent publisher remains None."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publisher: str | None
    total_count: int = Field(ge=0)


class CapabilityUsageSummaryReport(BaseModel):
    """Tenant-scoped usage projection — no monetary fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_usage_summary.v1"] = SCHEMA_CAPABILITY_USAGE_SUMMARY_V1
    tenant_id: str
    identity_rollups: tuple[CapabilityUsageIdentityRollup, ...]
    publisher_rollups: tuple[CapabilityUsagePublisherRollup, ...]

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        if not value or not value.strip():
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != value.strip():
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return value


def _rollup_outcome_counts(
    events: tuple[CapabilityUsageEvent, ...],
) -> tuple[int, int, int, int, int]:
    total = len(events)
    succeeded = sum(1 for event in events if event.outcome is CapabilityUsageOutcome.SUCCEEDED)
    failed = sum(1 for event in events if event.outcome is CapabilityUsageOutcome.FAILED)
    cancelled = sum(1 for event in events if event.outcome is CapabilityUsageOutcome.CANCELLED)
    timeout = sum(1 for event in events if event.outcome is CapabilityUsageOutcome.TIMEOUT)
    return total, succeeded, failed, cancelled, timeout


def project_capability_usage_summary(
    consumer: InMemoryCapabilityUsageConsumer,
    *,
    tenant_id: str,
) -> CapabilityUsageSummaryReport:
    """Aggregate usage for one tenant from a reference consumer snapshot."""
    tenant_events = tuple(
        event for event in consumer.snapshot() if event.tenant_id == tenant_id
    )

    identity_groups: dict[tuple[str, str, str, str], list[CapabilityUsageEvent]] = {}
    for event in tenant_events:
        key = event.identity.sort_key
        identity_groups.setdefault(key, []).append(event)

    identity_rollups: list[CapabilityUsageIdentityRollup] = []
    for key in sorted(identity_groups):
        grouped = tuple(identity_groups[key])
        total, succeeded, failed, cancelled, timeout = _rollup_outcome_counts(grouped)
        identity_rollups.append(
            CapabilityUsageIdentityRollup(
                identity=grouped[0].identity,
                total_count=total,
                succeeded_count=succeeded,
                failed_count=failed,
                cancelled_count=cancelled,
                timeout_count=timeout,
            ),
        )

    publisher_groups: dict[str | None, list[CapabilityUsageEvent]] = {}
    for event in tenant_events:
        publisher_groups.setdefault(event.provenance.publisher, []).append(event)

    publisher_rollups: list[CapabilityUsagePublisherRollup] = []
    for publisher in sorted(
        publisher_groups,
        key=lambda value: (value is None, value or ""),
    ):
        publisher_rollups.append(
            CapabilityUsagePublisherRollup(
                publisher=publisher,
                total_count=len(publisher_groups[publisher]),
            ),
        )

    return CapabilityUsageSummaryReport(
        tenant_id=tenant_id,
        identity_rollups=tuple(identity_rollups),
        publisher_rollups=tuple(publisher_rollups),
    )
