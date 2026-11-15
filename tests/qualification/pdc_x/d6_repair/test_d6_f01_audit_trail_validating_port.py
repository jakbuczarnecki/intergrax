# © Artur Czarnecki. All rights reserved.

"""PDC-D6 F-D6-01 — the evidence-plane security audit trail writes only through a validating port.

The F-35 COMPAT-X rule classifies ``EvidencePlaneSecurityAuditTrail.append`` as a canonical validating
path. The trail used to store whatever port it was given, so a port that does not validate received the
events as built. The trail now normalizes its port with ``as_evidence_persistence_port``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from intergrax.applications._shared.security_audit_trail_wiring import resolve_security_audit_trail_wiring
from intergrax.runtime.events import validating_evidence_persistence_port as validating_module
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.validating_evidence_persistence_port import (
    CanonicalRuntimeEventWriteValidatedPort,
    ValidatingEvidencePersistencePort,
)
from intergrax.runtime.security.security_audit_trail import EvidencePlaneSecurityAuditTrail
from tests.qualification.pdc_x.d6._support import open_evidence_plane, product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _RecordingPort:
    """A hostile stand-in: stores whatever it receives and is not a validating port."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def append(self, event: Any, *, tenant_id: str) -> Any:
        self.events.append(event)
        return event

    def list_for_task(self, task_id: str, *, tenant_id: str, limit: int = 1000) -> list[Any]:
        return [event for event in self.events if event.task_id == task_id and event.tenant_id == tenant_id][:limit]


@pytest.fixture()
def spies(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"prepare": 0, "schema": 0}
    real_prepare = validating_module.prepare_canonical_production_write_event
    real_schema = validating_module.assert_runtime_event_schema

    def _prepare(event: Any) -> Any:
        calls["prepare"] += 1
        return real_prepare(event)

    def _schema(event: Any) -> Any:
        calls["schema"] += 1
        return real_schema(event)

    monkeypatch.setattr(validating_module, "prepare_canonical_production_write_event", _prepare)
    monkeypatch.setattr(validating_module, "assert_runtime_event_schema", _schema)
    return calls


def _append(trail: EvidencePlaneSecurityAuditTrail) -> None:
    trail.append(tenant_id="tenant-a", action="policy.changed", actor_id="admin-1", resource="policy/egress")


def test_the_trail_port_is_always_a_validating_port() -> None:
    trail = EvidencePlaneSecurityAuditTrail(_RecordingPort())  # type: ignore[arg-type]
    assert isinstance(trail._evidence, CanonicalRuntimeEventWriteValidatedPort)


def test_events_are_prepared_and_schema_checked_before_a_non_validating_port_sees_them(spies: dict[str, int]) -> None:
    port = _RecordingPort()
    _append(EvidencePlaneSecurityAuditTrail(port))  # type: ignore[arg-type]

    assert spies == {"prepare": 1, "schema": 1}
    assert len(port.events) == 1


def test_a_schema_violation_never_reaches_the_non_validating_port(monkeypatch: pytest.MonkeyPatch) -> None:
    def _reject(event: Any) -> None:
        raise RuntimeEventSchemaError("rejected by the canonical schema guard")

    monkeypatch.setattr(validating_module, "assert_runtime_event_schema", _reject)
    port = _RecordingPort()
    trail = EvidencePlaneSecurityAuditTrail(port)  # type: ignore[arg-type]

    with pytest.raises(RuntimeEventSchemaError):
        _append(trail)

    assert port.events == []


def test_the_product_wiring_validates_both_bootstrap_events_before_a_non_validating_port(
    spies: dict[str, int],
) -> None:
    port = _RecordingPort()

    wiring = resolve_security_audit_trail_wiring(product_env(), evidence_persistence=port)  # type: ignore[arg-type]

    assert wiring.enabled is True
    assert spies == {"prepare": 2, "schema": 2}
    assert len(port.events) == 2


def test_an_already_validating_port_is_used_as_it_is() -> None:
    port = ValidatingEvidencePersistencePort(_RecordingPort())  # type: ignore[arg-type]
    trail = EvidencePlaneSecurityAuditTrail(port)
    assert trail._evidence is port


def test_a_legacy_runtime_event_persistence_is_accepted_and_listed_back() -> None:
    trail = EvidencePlaneSecurityAuditTrail(InMemoryRuntimeEventStore())  # type: ignore[arg-type]
    _append(trail)
    entries = trail.list_entries(tenant_id="tenant-a")
    assert [(entry.action, entry.actor_id, entry.resource) for entry in entries] == [
        ("policy.changed", "admin-1", "policy/egress")
    ]


def test_a_missing_port_is_rejected_at_construction() -> None:
    with pytest.raises(TypeError):
        EvidencePlaneSecurityAuditTrail(None)  # type: ignore[arg-type]


def test_the_durable_plane_port_still_round_trips_entries(tmp_path: Path) -> None:
    trail = EvidencePlaneSecurityAuditTrail(open_evidence_plane(tmp_path / "evidence.db"))
    _append(trail)
    restarted = EvidencePlaneSecurityAuditTrail(open_evidence_plane(tmp_path / "evidence.db"))
    assert [entry.action for entry in restarted.list_entries(tenant_id="tenant-a")] == ["policy.changed"]
