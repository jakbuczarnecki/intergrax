# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed durability semantics for canonical RuntimeEvent persistence (NPSC-5F/R1)."""

from __future__ import annotations

from enum import Enum

from intergrax.runtime.events.event_catalog import get_catalog_entry, should_persist_event
from intergrax.runtime.events.event_taxonomy import RetentionClass
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.spine_consolidation import get_platform_kind_entry


class EvidencePersistenceRequirement(str, Enum):
    """What a persistence failure means for a runtime signal."""

    NOT_PERSISTED = "not_persisted"
    MANDATORY = "mandatory"
    BEST_EFFORT = "best_effort"


def retention_class_for_runtime_event(event: RuntimeEvent) -> RetentionClass:
    if (
        event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        and (event.event_kind or "").startswith("platform.")
    ):
        platform_entry = get_platform_kind_entry(event.event_kind)
        if platform_entry is not None:
            return platform_entry.retention_class
    catalog_entry = get_catalog_entry(event.event_type)
    if catalog_entry is not None:
        return catalog_entry.retention_class
    return RetentionClass.OPERATIONAL


def evidence_persistence_requirement(event: RuntimeEvent) -> EvidencePersistenceRequirement:
    """
    Classify durability for ``event`` when ``should_persist_event`` is true.

    ``should_persist_event`` decides whether persistence applies; this enum decides
    whether a persistence failure must fail-closed at the bus boundary.
    """
    if not should_persist_event(event):
        return EvidencePersistenceRequirement.NOT_PERSISTED
    retention = retention_class_for_runtime_event(event)
    if retention is RetentionClass.DEBUG:
        return EvidencePersistenceRequirement.BEST_EFFORT
    return EvidencePersistenceRequirement.MANDATORY
