# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit WorkerCapabilityNeed → canonical CapabilityNeed projection (UCA-6B)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityNeed,
    derive_worker_capability_need_id,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed


def _capability_kinds_for_worker_need(
    need: WorkerCapabilityNeed,
) -> tuple[CapabilityKind, ...]:
    if need.need_kind is CapabilityNeedKind.TOOL_OPERATION:
        return (CapabilityKind.TOOL,)
    if need.need_kind in {
        CapabilityNeedKind.DATA_TRANSFORMATION,
        CapabilityNeedKind.WORKFLOW_ALTERNATIVE,
        CapabilityNeedKind.GENERAL_CAPABILITY,
    }:
        return (CapabilityKind.TOOL, CapabilityKind.SKILL)
    if need.need_kind in {
        CapabilityNeedKind.EXTERNAL_INTEGRATION,
        CapabilityNeedKind.PROTOCOL_ADAPTATION,
        CapabilityNeedKind.SCHEMA_ADAPTATION,
    }:
        return (CapabilityKind.SKILL, CapabilityKind.TOOL)
    return (CapabilityKind.TOOL, CapabilityKind.SKILL)


def _intent_summary_for_worker_need(need: WorkerCapabilityNeed) -> str:
    if need.required_operations:
        return "|".join(need.required_operations)
    return need.need_kind.value


def project_worker_capability_need_to_capability_need(
    need: WorkerCapabilityNeed,
) -> CapabilityNeed:
    """Typed projection — no model_dump passthrough; AW-only fields stay on worker need."""
    need_id = derive_worker_capability_need_id(need)
    return CapabilityNeed(
        need_id=need_id,
        kinds=_capability_kinds_for_worker_need(need),
        intent_summary=_intent_summary_for_worker_need(need),
    )


@runtime_checkable
class WorkerCapabilityNeedProjection(Protocol):
    """Replaceable worker need projection for tenant-specific semantics."""

    def project(self, need: WorkerCapabilityNeed) -> CapabilityNeed:
        """Project worker need to canonical CapabilityNeed."""
        ...


class DefaultWorkerCapabilityNeedProjection:
    """Stable default projection used by AW recovery coordinator."""

    def project(self, need: WorkerCapabilityNeed) -> CapabilityNeed:
        return project_worker_capability_need_to_capability_need(need)


__all__ = [
    "DefaultWorkerCapabilityNeedProjection",
    "WorkerCapabilityNeedProjection",
    "project_worker_capability_need_to_capability_need",
]
