# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.event_catalog import (
    EVENT_CATALOG,
    EVENT_OPS_FILTER_HINTS,
    EVENT_PHASE_COVERAGE,
    EventCategory,
    RetentionClass,
    category_for_event_kind,
    category_for_spine_type,
    get_catalog_entry,
    list_unmapped_event_types,
    sample_rate_for_spine_type,
)
from intergrax.runtime.events.payload_registry import EVENT_TYPE_PREFERRED_SCHEMA
from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.spine_consolidation import PLATFORM_KIND_CATALOG

pytestmark = pytest.mark.gate


def test_event_catalog_covers_all_spine_types() -> None:
    unmapped = list_unmapped_event_types()
    assert unmapped == [], f"missing catalog entries: {[u.name for u in unmapped]}"


def test_phase_coverage_view_matches_catalog() -> None:
    for event_type, entry in EVENT_CATALOG.items():
        assert EVENT_PHASE_COVERAGE[event_type] == entry.phase
        assert phase_for_event(event_type) == entry.phase


def test_ops_hints_view_matches_catalog() -> None:
    for event_type, entry in EVENT_CATALOG.items():
        assert EVENT_OPS_FILTER_HINTS[event_type] == entry.ops_hint


def test_preferred_schema_ids_match_payload_registry() -> None:
    for event_type, schema_id in EVENT_TYPE_PREFERRED_SCHEMA.items():
        entry = get_catalog_entry(event_type)
        assert entry is not None
        assert entry.preferred_payload_schema_id == schema_id


def test_sample_rate_reduced_for_high_volume_types() -> None:
    assert sample_rate_for_spine_type(RuntimeEventType.TASK_PROGRESS) < 1.0
    assert sample_rate_for_spine_type(RuntimeEventType.TASK_CREATED) == 1.0


def test_consolidated_platform_kinds_removed_from_spine() -> None:
    spine_values = {member.value for member in RuntimeEventType}
    for entry in PLATFORM_KIND_CATALOG.values():
        _, _, flat_name = entry.kind.rpartition(".")
        assert flat_name not in spine_values
    assert len(PLATFORM_KIND_CATALOG) == 22


def test_retention_class_audit_for_tool_events() -> None:
    entry = get_catalog_entry(RuntimeEventType.TOOL_COMPLETED)
    assert entry is not None
    assert entry.retention_class == RetentionClass.AUDIT


def test_category_for_event_kind_namespaces() -> None:
    assert category_for_event_kind("agents.legal.clause_flagged") == EventCategory.AGENT
    assert category_for_event_kind("platform.adaptive.signal") == EventCategory.PLATFORM
    assert category_for_event_kind("intergrax.llm.stream.delta") == EventCategory.AGENT


def test_category_for_spine_type_task_and_tool() -> None:
    assert category_for_spine_type(RuntimeEventType.TASK_CREATED) == EventCategory.TASK
    assert category_for_spine_type(RuntimeEventType.TOOL_REQUESTED) == EventCategory.TOOL
    assert category_for_spine_type(RuntimeEventType.AGENT_SELECTED) == EventCategory.AGENT
