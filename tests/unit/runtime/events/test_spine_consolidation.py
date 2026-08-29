# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import dataclasses

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEventType, parse_runtime_event_payload
from intergrax.runtime.events.spine_consolidation import (
    PLATFORM_KIND_CATALOG,
    PlatformKindEntry,
    assert_publication_spine_budget,
    build_platform_signal_event,
    publication_spine_type_count,
    should_persist_platform_kind,
)

pytestmark = pytest.mark.gate


def _legacy_flat_event_payload(event_type: str) -> dict[str, object]:
    return {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "schema_version": "runtime_event.v2",
        "event_type": event_type,
        "phase": ExecutionPhase.STEP_EXECUTION,
        "payload": {"reason": "denied"},
    }


def test_publication_spine_budget_within_target() -> None:
    assert publication_spine_type_count() <= 56
    assert_publication_spine_budget()


def test_platform_kind_entry_has_no_legacy_spine_value_field() -> None:
    fields = {field.name for field in dataclasses.fields(PlatformKindEntry)}
    assert "legacy_spine_value" not in fields
    assert fields == {"kind", "phase", "ops_hint", "sample_rate", "retention_class"}


def test_build_platform_signal_event_emits_domain_signal() -> None:
    run_id = mint_run_id()
    task_id = mint_task_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        event = build_platform_signal_event(
            kind="platform.hook.hook_timeout",
            task_id=task_id,
            run_id=run_id,
            phase=ExecutionPhase.INTAKE,
            payload={"hook_name": "demo"},
        )
    finally:
        reset_active_execution_identity(token)
    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == "platform.hook.hook_timeout"
    assert event.ops_hint == "ops:alert"
    assert "legacy_spine_type" not in event.payload
    assert event.payload["hook_name"] == "demo"


@pytest.mark.parametrize("legacy_event_type", ["scale_requested", "hook_error"])
def test_parse_runtime_event_payload_rejects_legacy_flat_event_types(
    legacy_event_type: str,
) -> None:
    with pytest.raises(Exception):
        parse_runtime_event_payload(_legacy_flat_event_payload(legacy_event_type))


def test_platform_kind_sampling_deterministic() -> None:
    kind = "platform.capacity.capacity_signal_collected"
    first = should_persist_platform_kind(kind, "evt-a")
    second = should_persist_platform_kind(kind, "evt-a")
    assert first == second


def test_platform_kind_catalog_contains_only_canonical_fields() -> None:
    for entry in PLATFORM_KIND_CATALOG.values():
        assert entry.kind.startswith("platform.")
        assert entry.ops_hint.startswith("ops:")
