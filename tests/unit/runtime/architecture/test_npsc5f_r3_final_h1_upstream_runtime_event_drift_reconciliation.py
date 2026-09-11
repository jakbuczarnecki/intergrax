# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R3 Final H1 — upstream RuntimeEvent / event-surface drift reconciliation."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import get_catalog_entry, should_persist_event
from intergrax.runtime.events.persistence_contract import (
    EvidenceTenantRoutingMismatchError,
    MandatoryEvidencePersistenceError,
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
    TaskRuntimeEventRuns,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import load_complete_run_journal, read_run_journal_page
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    envelope_from_runtime_event,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.journal_export import serialize_runtime_event
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.npsc5f_r1_protected_drift import (
    R1_POST_R2_QUALIFIED_BASELINE_SHA,
    collect_r1_protected_production_drift,
    git_changed_paths,
)
from testing_support.npsc5f_r2_protected_drift import R2_IMPLEMENTATION_SHA, collect_r2_protected_production_drift
from testing_support.npsc5f_r3_h1_upstream_event_drift import (
    EXECUTION_FAILED_RUNTIME_EVENT_QUALIFIED_SHA,
    NPSC5F_R3_H1_QUALIFIED_INTEGRATED_SHA,
    classify_post_r3_event_surface_change,
    collect_post_r3_event_surface_paths,
)
from testing_support.npsc5f_r3_protected_drift import R3_IMPLEMENTATION_SHA, collect_r3_protected_production_drift

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-h1"


def _execution_failed_event(tenant_id: str = _TENANT) -> RuntimeEvent:
    return sample_runtime_event(tenant_id=tenant_id).model_copy(
        update={
            "event_type": RuntimeEventType.EXECUTION_FAILED,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "severity": EventSeverity.ERROR,
            "timestamp": datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc),
            "payload": {
                "schema_id": "execution_failure.v1",
                "failure_kind": ExecutionFailureKind.DELEGATE_EXCEPTION.value,
                "safe_summary": "bounded failure summary",
                "failure_code": "E_H1",
            },
        },
    )


def test_npsc5f_r3_h1_runtime_event_drift_preserves_frozen_identity_contract() -> None:
    """Post-R3 enum extension does not alter RuntimeEvent field identity or equality semantics."""
    baseline = sample_runtime_event(tenant_id=_TENANT)
    fields = set(RuntimeEvent.model_fields)
    assert {
        "tenant_id",
        "event_id",
        "event_type",
        "phase",
        "severity",
        "timestamp",
        "payload",
        "schema_version",
        "task_id",
        "run_id",
        "traceparent",
        "tracestate",
    }.issubset(fields)
    clone = baseline.model_copy(deep=True)
    assert baseline == clone
    assert RuntimeEventType.EXECUTION_FAILED.value == "execution_failed"
    assert RuntimeEventType("execution_failed") is RuntimeEventType.EXECUTION_FAILED


def test_npsc5f_r3_h1_post_r3_event_surface_classification_recorded() -> None:
    changed = collect_post_r3_event_surface_paths(_REPO_ROOT, to_ref="origin/development")
    buckets = {path: classify_post_r3_event_surface_change(path) for path in changed}
    assert buckets.get("intergrax/runtime/events/runtime_event.py") == "A"
    assert buckets.get("intergrax/runtime/events/event_catalog.py") == "G"
    assert buckets.get("intergrax/runtime/events/payload_registry.py") == "H"
    assert buckets.get("intergrax/runtime/events/payloads/canonical.py") == "I"
    assert buckets.get("intergrax/runtime/events/spine_consolidation.py") == "J"


def test_npsc5f_r3_h1_r1_mandatory_evidence_and_tenant_invariants() -> None:
    class _Failing(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            raise RuntimeError("sink down")

        def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None, after=None):
            return []

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return []

        def list_positioned_for_task_grouped_by_run(self, task_id, *, tenant_id: str, limit: int = 1000):
            return TaskRuntimeEventRuns(runs=())

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return None

    bus = RuntimeEventBus(persistence=_Failing(), record_history=True)
    notified: list[str] = []
    bus.subscribe(lambda evt: notified.append(evt.event_id))
    event = sample_runtime_event(tenant_id=_TENANT)
    with pytest.raises(MandatoryEvidencePersistenceError, match="mandatory"):
        bus.record(event, tenant_id=_TENANT)
    assert notified == []

    store = InMemoryRuntimeEventStore()
    original = sample_runtime_event(tenant_id=_TENANT)
    store.append(original, tenant_id=_TENANT)
    duplicate = original.model_copy(deep=True)
    store.append(duplicate, tenant_id=_TENANT)
    assert len(store.list_for_run(original.run_id, tenant_id=_TENANT)) == 1

    conflicting = original.model_copy(update={"payload": {"tool_id": "other"}})
    with pytest.raises(RuntimeEventPersistenceIntegrityError):
        store.append(conflicting, tenant_id=_TENANT)

    with pytest.raises(EvidenceTenantRoutingMismatchError):
        store.append(original, tenant_id="other-tenant")


def test_npsc5f_r3_h1_r2_journal_semantics_with_execution_failed_event(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "h1_journal.db")
    run_id = mint_run_id()
    events = [
        sample_runtime_event(tenant_id=_TENANT, run_id=run_id),
        _execution_failed_event().model_copy(update={"run_id": run_id}),
    ]
    for event in events:
        store.append(event, tenant_id=_TENANT)
    page = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=10, cursor=None)
    assert page.is_complete
    complete = load_complete_run_journal(store, tenant_id=_TENANT, run_id=run_id)
    assert len(complete) == 2
    assert complete[1].event_type is RuntimeEventType.EXECUTION_FAILED


def test_npsc5f_r3_h1_r3_execution_failure_payload_not_export_allowlisted() -> None:
    event = _execution_failed_event()
    catalog = get_catalog_entry(RuntimeEventType.EXECUTION_FAILED)
    assert catalog is not None
    assert should_persist_event(event)
    envelope = serialize_runtime_event(event)
    blob = envelope.model_dump_json()
    assert "safe_summary" not in blob
    assert "failure_kind" not in blob
    assert "failure_code" not in blob
    assert envelope_is_content_safe(envelope)
    direct = envelope_from_runtime_event(event)
    for forbidden in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert forbidden not in direct.model_dump_json()
    parsed = json.loads(blob)
    payload = parsed.get("payload") or {}
    assert "failure_kind" not in payload
    assert "safe_summary" not in payload


def test_npsc5f_r3_h1_sentinel_baselines_after_qualified_drift() -> None:
    assert R3_IMPLEMENTATION_SHA == "0346face3ef68d8f21504822a26f8f45f2384cf9"
    assert EXECUTION_FAILED_RUNTIME_EVENT_QUALIFIED_SHA == "40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8"
    r1_drift = collect_r1_protected_production_drift(
        _REPO_ROOT,
        from_sha=R1_POST_R2_QUALIFIED_BASELINE_SHA,
    )
    assert r1_drift == []
    assert collect_r2_protected_production_drift(_REPO_ROOT) == []
    assert collect_r3_protected_production_drift(_REPO_ROOT) == []


def test_npsc5f_r3_h1_runtime_event_drift_was_captured_before_baseline_advancement() -> None:
    """Prove the R1 sentinel window would have fired between R2 impl and the qualified enum commit."""
    historical = git_changed_paths(
        _REPO_ROOT,
        from_sha=R2_IMPLEMENTATION_SHA,
        to_ref=EXECUTION_FAILED_RUNTIME_EVENT_QUALIFIED_SHA,
    )
    assert "intergrax/runtime/events/runtime_event.py" in historical


def test_npsc5f_r3_h1_r1_sentinel_clean_since_qualified_enum_baseline() -> None:
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            f"{R1_POST_R2_QUALIFIED_BASELINE_SHA}..origin/development",
            "--",
            "intergrax/runtime/events/runtime_event.py",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_npsc5f_r3_h1_integrated_head_pin_recorded() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "origin/development"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == NPSC5F_R3_H1_QUALIFIED_INTEGRATED_SHA
