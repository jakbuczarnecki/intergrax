# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R1 Final — durable evidence commit & tenant integrity qualification and freeze."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.evidence_durability import (
    EvidencePersistenceRequirement,
    evidence_persistence_requirement,
    retention_class_for_runtime_event,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import should_persist_event
from intergrax.runtime.events.event_taxonomy import RetentionClass
from intergrax.runtime.events.persistence_contract import (
    EvidenceTenantRoutingMismatchError,
    MandatoryEvidencePersistenceError,
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
    resolve_event_tenant_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

R1_IMPLEMENTATION_SHA = "455d3b216f0ad56ea9cdf9db6e0f760b50063a81"
NPSC_5F_P0_SHA = "7811371da1069b661987b050a4c9bf42c02bda69"
NPSC_5E_FINAL_SHA = "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"

_R1_PRODUCTION_SURFACE = (
    "intergrax/runtime/events/evidence_durability.py",
    "intergrax/runtime/events/persistence_contract.py",
    "intergrax/runtime/events/event_bus.py",
)

_FORBIDDEN_CONTROL_PLANE_SYMBOLS = frozenset(
    {
        "ChildExecutionRunner",
        "AttemptLifecycleService",
        "LongRunningCoordinator",
        "FanOutPartialRecoveryService",
    },
)

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R1 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_commit_tenant_integrity.py"],
    ),
    (
        "NPSC-5F P0 gate",
        ["tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"],
    ),
    ("Runtime events suites", ["tests/unit/runtime/events/"]),
    (
        "Runtime observability suites",
        ["tests/unit/runtime/observability/"],
    ),
    (
        "NPSC-5E Final",
        ["tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
)


def _run_pytest(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *targets, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_canonical_predecessor_shas_recorded() -> None:
    assert R1_IMPLEMENTATION_SHA.startswith("455d3b2")
    assert NPSC_5F_P0_SHA.startswith("7811371")
    assert NPSC_5E_FINAL_SHA.startswith("fabdcfe")


class _FailingPersistence(RuntimeEventPersistence):
    def append(self, event, *, tenant_id: str):
        raise RuntimeError("sink down")

    def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None):
        return []

    def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
        return []

    def get_by_event_id(self, *, tenant_id: str, event_id):
        return None


class _CountingPersistence(RuntimeEventPersistence):
    def __init__(self, inner: RuntimeEventPersistence) -> None:
        self._inner = inner
        self.append_calls = 0

    def append(self, event, *, tenant_id: str):
        self.append_calls += 1
        return self._inner.append(event, tenant_id=tenant_id)

    def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None):
        return self._inner.list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
        )

    def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
        return self._inner.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

    def get_by_event_id(self, *, tenant_id: str, event_id):
        return self._inner.get_by_event_id(tenant_id=tenant_id, event_id=event_id)

    def close(self) -> None:
        self._inner.close()


def test_r1_final_mandatory_success_persist_before_subscriber() -> None:
    order: list[str] = []
    inner = InMemoryRuntimeEventStore()

    class _Ordered(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            order.append("persist")
            return inner.append(event, tenant_id=tenant_id)

        def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None):
            return inner.list_positioned_for_run(run_id, tenant_id=tenant_id, limit=limit, through=through)

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return inner.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return inner.get_by_event_id(tenant_id=tenant_id, event_id=event_id)

    bus = RuntimeEventBus(persistence=_Ordered(), record_history=True)
    bus.subscribe(lambda _evt: order.append("subscriber"))
    event = sample_runtime_event(tenant_id="tenant-order-r1f")
    bus.record(event, tenant_id="tenant-order-r1f")
    assert order == ["persist", "subscriber"]
    assert len(bus.history) == 1


def test_r1_final_mandatory_failure_zero_history_and_subscribers() -> None:
    bus = RuntimeEventBus(persistence=_FailingPersistence(), record_history=True)
    notified: list[str] = []
    bus.subscribe(lambda evt: notified.append(evt.event_id))
    event = sample_runtime_event(tenant_id="tenant-fail-r1f")
    with pytest.raises(MandatoryEvidencePersistenceError):
        bus.record(event, tenant_id="tenant-fail-r1f")
    assert bus.history == []
    assert notified == []


def test_r1_final_not_persisted_skips_append() -> None:
    inner = InMemoryRuntimeEventStore()
    counter = _CountingPersistence(inner)
    bus = RuntimeEventBus(persistence=counter, record_history=True)
    event = sample_runtime_event(tenant_id="tenant-skip").model_copy(
        update={"event_type": RuntimeEventType.TASK_PROGRESS, "phase": ExecutionPhase.STEP_EXECUTION},
    )
    event_id = mint_event_id()
    while should_persist_event(event.model_copy(update={"event_id": event_id})):
        event_id = mint_event_id()
    skipped = event.model_copy(update={"event_id": event_id})
    assert evidence_persistence_requirement(skipped) is EvidencePersistenceRequirement.NOT_PERSISTED
    bus.record(skipped, tenant_id="tenant-skip")
    assert counter.append_calls == 0


def test_r1_final_uncatalogued_persisted_event_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    event = sample_runtime_event(tenant_id="tenant-uncat")
    monkeypatch.setattr(
        "intergrax.runtime.events.evidence_durability.get_catalog_entry",
        lambda _event_type: None,
    )
    assert retention_class_for_runtime_event(event) is RetentionClass.OPERATIONAL
    assert evidence_persistence_requirement(event) is EvidencePersistenceRequirement.MANDATORY


def test_r1_final_tenant_whitespace_fail_closed() -> None:
    event = sample_runtime_event(tenant_id="T1")
    with pytest.raises(ValueError, match="whitespace"):
        resolve_event_tenant_id(event, " T1 ")
    with pytest.raises(ValueError, match="whitespace"):
        resolve_event_tenant_id(sample_runtime_event(tenant_id=" T1 "), None)


def test_r1_final_non_string_persistence_tenant_rejected() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id="T1")
    with pytest.raises(TypeError, match="tenant_id must be str"):
        store.get_by_event_id(tenant_id=1, event_id=event.event_id)  # type: ignore[arg-type]


def _store_factories(tmp_path: Path) -> list[tuple[str, Callable[[], RuntimeEventPersistence]]]:
    suffix = mint_run_id()
    return [
        ("memory", lambda: InMemoryRuntimeEventStore()),
        ("sqlite", lambda: SQLiteRuntimeEventStore(db_path=tmp_path / f"r1f_{suffix}.db")),
        ("document", lambda: DocumentBackedRuntimeEventStore(InMemoryDocumentStore())),
        (
            "validating",
            lambda: ValidatingRuntimeEventPersistence(InMemoryRuntimeEventStore()),
        ),
    ]


@pytest.mark.parametrize("label", ["memory", "sqlite", "document", "validating"])
def test_r1_final_tenant_mismatch_zero_write_all_adapters(
    tmp_path: Path,
    label: str,
) -> None:
    factories = dict(_store_factories(tmp_path))
    store = factories[label]()
    event = sample_runtime_event(tenant_id=f"{label}-event-tenant")
    route = f"{label}-route-tenant"
    with pytest.raises(EvidenceTenantRoutingMismatchError):
        store.append(event, tenant_id=route)
    assert store.get_by_event_id(tenant_id=f"{label}-event-tenant", event_id=event.event_id) is None
    assert store.get_by_event_id(tenant_id=route, event_id=event.event_id) is None
    store.close()


def test_r1_final_idempotent_same_content(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "idempotent.db")
    tenant_id = "tenant-idem"
    event = sample_runtime_event(tenant_id=tenant_id)
    first = store.append(event, tenant_id=tenant_id)
    second = store.append(event, tenant_id=tenant_id)
    assert second.position == first.position
    store.close()


def test_r1_final_duplicate_event_id_content_conflict_blocked(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    tenant_id = "tenant-conflict"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_id = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    )
    store.append(original, tenant_id=tenant_id)
    conflicting = original.model_copy(update={"event_type": RuntimeEventType.STEP_COMPLETED})
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="conflicts"):
        store.append(conflicting, tenant_id=tenant_id)


def test_r1_final_duplicate_event_id_different_route_tenant_blocked() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id="tenant-a")
    store.append(event, tenant_id="tenant-a")
    with pytest.raises(EvidenceTenantRoutingMismatchError):
        store.append(event, tenant_id="tenant-b")


def test_r1_final_subscriber_failure_does_not_erase_evidence() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)

    def _boom(_evt: RuntimeEvent) -> None:
        raise RuntimeError("subscriber failed")

    bus.subscribe(_boom)
    event = sample_runtime_event(tenant_id="tenant-sub-fail")
    bus.record(event, tenant_id="tenant-sub-fail")
    assert store.get_by_event_id(tenant_id="tenant-sub-fail", event_id=event.event_id) is not None


def test_r1_final_mandatory_exception_chaining_and_no_payload_leak() -> None:
    bus = RuntimeEventBus(persistence=_FailingPersistence(), record_history=True)
    event = sample_runtime_event(tenant_id="tenant-chain")
    with pytest.raises(MandatoryEvidencePersistenceError) as exc_info:
        bus.record(event, tenant_id="tenant-chain")
    err = exc_info.value
    assert err.__cause__ is not None
    message = str(err)
    assert event.run_id not in message
    assert "sink down" not in message


def test_r1_final_event_tenant_not_mutated_on_mismatch() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id="immutable-tenant")
    original_tenant = event.tenant_id
    with pytest.raises(EvidenceTenantRoutingMismatchError):
        store.append(event, tenant_id="other-tenant")
    assert event.tenant_id == original_tenant


def test_r1_final_no_magic_durability_strings_outside_typed_contract() -> None:
    violations: list[str] = []
    allowed_paths = {
        _REPO_ROOT / rel for rel in _R1_PRODUCTION_SURFACE
    }
    pattern = re.compile(r"""metadata\s*\[\s*["']critical["']\s*\]""")
    for path in allowed_paths:
        source = path.read_text(encoding="utf-8")
        if pattern.search(source):
            violations.append(path.name)
    assert violations == []


def test_r1_final_no_reflection_on_r1_production_surface() -> None:
    hits: list[str] = []
    for rel in _R1_PRODUCTION_SURFACE:
        path = _REPO_ROOT / rel
        source = path.read_text(encoding="utf-8")
        for match in _REFLECTION_PATTERN.finditer(source):
            hits.append(f"{rel}:{match.group(0)}")
    assert hits == []


def test_r1_final_no_execution_control_from_r1_production_surface() -> None:
    violations: list[str] = []
    for rel in _R1_PRODUCTION_SURFACE:
        path = _REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            symbol = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if symbol in _FORBIDDEN_CONTROL_PLANE_SYMBOLS:
                violations.append(f"{rel}:{node.lineno}:{symbol}")
    assert violations == []


def test_r1_final_no_evidence_queue_or_second_bus() -> None:
    bus_path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "event_bus.py"
    source = bus_path.read_text(encoding="utf-8")
    assert "asyncio.Queue" not in source
    events_root = _REPO_ROOT / "intergrax" / "runtime" / "events"
    second_bus_defs = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in events_root.rglob("*.py")
        if path.name != "event_bus.py" and "class RuntimeEventBus" in path.read_text(encoding="utf-8")
    ]
    assert second_bus_defs == []


@pytest.mark.gate
def test_npsc5f_r1_final_qualification_gate() -> None:
    assert evidence_persistence_requirement(sample_runtime_event()) is (
        EvidencePersistenceRequirement.MANDATORY
    )
